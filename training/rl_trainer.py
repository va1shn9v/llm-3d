"""
GRPO RL training loop on Tinker + Modal (Section 8.2 of spec).

Uses Tinker for model sampling + policy updates and Modal for reward
computation via Blender execution + metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from config import ProjectConfig, load_config
from environments.blender_3d.harness import Blender3DHarness
from environments.blender_3d.rubric import Blender3DRubric

log = logging.getLogger(__name__)


class RLPromptSampler:
    """Samples prompts for RL training from the dataset."""

    def __init__(self, jsonl_path: str | Path, seed: int = 42):
        self.items: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    meta = record.get("metadata", {})
                    self.items.append({
                        "object_id": meta.get("object_id", ""),
                        "category": meta.get("category", "unknown"),
                        "messages": record.get("messages", []),
                    })
        self.rng = np.random.default_rng(seed)
        log.info(f"Loaded {len(self.items)} RL prompts")

    def sample(self, n: int) -> list[dict]:
        indices = self.rng.choice(len(self.items), size=min(n, len(self.items)), replace=False)
        return [self.items[i] for i in indices]


class GRPOTrainer:
    """GRPO RL training loop."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        self.harness = Blender3DHarness(
            modal_endpoint=self.cfg.modal.endpoint,
            auth_token=self.cfg.modal.auth_token,
        )
        self.rubric = Blender3DRubric(self.cfg.reward)

    async def train(self, tinker_client: Any = None):
        """Run GRPO training loop.

        If tinker_client is None, runs in dry-run mode.
        """
        rl = self.cfg.rl
        sft = self.cfg.sft

        log.info(f"Starting GRPO training: {rl.steps} steps, "
                 f"bs={rl.batch_size}, n_comp={rl.num_completions}, "
                 f"temp={rl.temperature}")

        prompt_sampler = RLPromptSampler(sft.train_path, self.cfg.seed)

        training_client = None
        sampling_client = None
        if tinker_client is not None:
            training_client = tinker_client.create_lora_training_client(
                base_model=sft.base_model,
                rank=sft.lora_rank,
                target_modules=sft.target_modules,
            )
            training_client.load_state(rl.sft_checkpoint)
            sampling_client = tinker_client.create_sampling_client(
                base_model=sft.base_model,
                lora_state=rl.sft_checkpoint,
            )

        all_rewards_history: list[float] = []

        for step in range(rl.steps):
            t0 = time.monotonic()

            prompts = prompt_sampler.sample(rl.batch_size)

            all_completions = []
            for prompt in prompts:
                if sampling_client is not None:
                    completions = await asyncio.wrap_future(
                        sampling_client.sample(
                            prompt["messages"],
                            n=rl.num_completions,
                            temperature=rl.temperature,
                            max_tokens=rl.max_new_tokens,
                        ).result()
                    )
                    for c in completions:
                        all_completions.append({
                            "object_id": prompt["object_id"],
                            "code": c.text,
                            "seed": 42,
                        })
                else:
                    for _ in range(rl.num_completions):
                        all_completions.append({
                            "object_id": prompt["object_id"],
                            "code": f"from bpy_lib import *\n# dry run step {step}\nexport_scene()",
                            "seed": 42,
                        })

            exec_results = await self.harness.execute_batch(all_completions)

            rewards = [r.get("reward", 0.0) for r in exec_results]
            all_rewards_history.extend(rewards)

            advantages = self._compute_advantages(rewards, rl.batch_size, rl.num_completions)

            if training_client is not None:
                rl_data = self._prepare_grpo_data(
                    all_completions, advantages, rl.clip_ratio, rl.kl_coeff
                )
                fwdbwd = training_client.forward_backward(rl_data, loss_type="grpo")
                await asyncio.wrap_future(fwdbwd.result())

                optim = training_client.optim_step({
                    "optimizer": "adamw",
                    "lr": rl.learning_rate,
                    "weight_decay": 0.01,
                })
                await asyncio.wrap_future(optim.result())

            elapsed = time.monotonic() - t0

            if step % rl.log_every == 0:
                exec_rate = sum(1 for r in rewards if r > 0.05) / max(len(rewards), 1)
                mean_r = sum(rewards) / max(len(rewards), 1)
                log.info(
                    f"Step {step}/{rl.steps} | "
                    f"exec_rate={exec_rate:.2f} | "
                    f"mean_reward={mean_r:.3f} | "
                    f"elapsed={elapsed:.1f}s"
                )

            if step > 0 and step % rl.checkpoint_every == 0:
                if training_client is not None:
                    training_client.save_state(f"rl-step-{step}")
                log.info(f"Checkpoint saved: rl-step-{step}")

        log.info("GRPO training complete.")

    def _compute_advantages(
        self,
        rewards: list[float],
        batch_size: int,
        num_completions: int,
    ) -> list[float]:
        """Compute normalized advantages within each prompt group."""
        advantages = []
        for i in range(batch_size):
            group = rewards[i * num_completions:(i + 1) * num_completions]
            if not group:
                continue
            mean_r = sum(group) / len(group)
            std_r = max(float(np.std(group)), 1e-6)
            group_adv = [(r - mean_r) / std_r for r in group]
            advantages.extend(group_adv)
        return advantages

    def _prepare_grpo_data(
        self,
        completions: list[dict],
        advantages: list[float],
        clip_ratio: float,
        kl_coeff: float,
    ) -> list[dict]:
        """Package completions + advantages for Tinker's GRPO loss."""
        data = []
        for comp, adv in zip(completions, advantages):
            data.append({
                "code": comp["code"],
                "object_id": comp["object_id"],
                "advantage": adv,
                "clip_ratio": clip_ratio,
                "kl_coeff": kl_coeff,
            })
        return data


def run_rl(config_path: str | None = None):
    """Entry point for RL training."""
    cfg = load_config(config_path)
    trainer = GRPOTrainer(cfg)
    asyncio.run(trainer.train())
