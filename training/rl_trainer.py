"""
GRPO RL training loop with hard prompt mining.

Uses Tinker for model sampling + policy updates and Modal for reward
computation via Blender execution + metrics. Oversamples hard prompts
(captions where teacher LLM failed during synthetic data generation)
to focus RL exploration where it has the most room to improve.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tinker
    from tinker import types as tinker_types
except ImportError:
    tinker = None  # type: ignore[assignment]
    tinker_types = None  # type: ignore[assignment]

from config import ProjectConfig, load_config
from data.hard_prompts import HardPromptSampler
from data.storage import open_read
from environments.blender_3d.harness import Blender3DHarness
from training.wandb_logger import WandbLogger

log = logging.getLogger(__name__)


class RLPromptSampler:
    """Samples prompts for RL training, with optional hard mining."""

    def __init__(
        self,
        jsonl_path: str | Path,
        cfg: ProjectConfig,
    ):
        self.items: list[dict] = []
        with open_read(str(jsonl_path), cfg.storage) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    meta = record.get("metadata", {})
                    self.items.append({
                        "object_id": meta.get("object_id", ""),
                        "caption": meta.get("caption", ""),
                        "gt_mesh_path": meta.get("gt_mesh_path", ""),
                        "messages": record.get("messages", []),
                    })

        self._hard_sampler: HardPromptSampler | None = None
        hm = cfg.hard_mining
        if hm.enabled and Path(hm.hard_prompts_csv).exists():
            self._hard_sampler = HardPromptSampler(
                hard_csv_path=hm.hard_prompts_csv,
                all_prompts=self.items,
                cfg=hm,
                seed=cfg.seed,
            )
            log.info(
                f"Hard mining enabled: {self._hard_sampler.num_hard} hard / "
                f"{self._hard_sampler.num_normal} normal prompts"
            )
        else:
            self._rng = np.random.default_rng(cfg.seed)
            if hm.enabled:
                log.warning(
                    f"Hard mining enabled but CSV not found at {hm.hard_prompts_csv}. "
                    f"Falling back to uniform sampling."
                )

        log.info(f"Loaded {len(self.items)} RL prompts")

    def sample(self, n: int) -> list[dict]:
        if self._hard_sampler is not None:
            return self._hard_sampler.sample(n)

        indices = self._rng.choice(len(self.items), size=min(n, len(self.items)), replace=False)
        return [self.items[i] for i in indices]


def _render_prompt_to_model_input(
    messages: list[dict],
    tokenizer: Any,
) -> "tinker_types.ModelInput":
    """Convert chat-format messages into a Tinker ModelInput for sampling."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role != "assistant":
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    prompt_text = "".join(parts)
    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    return tinker_types.ModelInput.from_ints(tokens=tokens)


def _build_grpo_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    advantage: float,
) -> "tinker_types.Datum":
    """Package a single prompt+completion with its advantage as a Tinker Datum."""
    tokens = prompt_tokens + completion_tokens
    prompt_weights = [0.0] * len(prompt_tokens)
    completion_weights = [1.0] * len(completion_tokens)
    weights = (prompt_weights + completion_weights)[1:]

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
            advantage=advantage,
        ),
    )


class GRPOTrainer:
    """GRPO RL training loop with hard prompt mining."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        self.harness = Blender3DHarness(
            modal_endpoint=self.cfg.modal.endpoint,
            auth_token=self.cfg.modal.auth_token,
            reward_cfg=self.cfg.reward,
        )
        self.wb = WandbLogger(
            self.cfg.logging,
            run_name="grpo",
            tags=["rl", "grpo"],
            extra_config=self.cfg.rl.model_dump(),
        )
        self.wb.define_metric("rl/mean_reward", step_metric="step")
        self.wb.define_metric("rl/exec_rate", step_metric="step")
        self.wb.define_metric("rl/max_reward", step_metric="step")
        self.wb.define_metric("rl/min_reward", step_metric="step")
        self.wb.define_metric("rl/mean_advantage", step_metric="step")
        self.wb.define_metric("rl/step_time_s", step_metric="step")
        self.wb.define_metric("rl/hard_prompt_ratio_actual", step_metric="step")

    async def train(self, service_client: Any = None):
        """Run GRPO training loop with hard mining."""
        rl = self.cfg.rl
        sft = self.cfg.sft

        log.info(
            f"Starting GRPO training: {rl.steps} steps, "
            f"bs={rl.batch_size}, n_comp={rl.num_completions}, "
            f"temp={rl.temperature}, "
            f"hard_mining={self.cfg.hard_mining.enabled}"
        )

        prompt_sampler = RLPromptSampler(rl.prompt_path, self.cfg)

        training_client = None
        sampling_client = None
        tokenizer = None

        if service_client is not None:
            training_client = service_client.create_lora_training_client(
                base_model=sft.base_model,
                rank=sft.lora_rank,
                train_mlp=sft.train_mlp,
                train_attn=sft.train_attn,
                train_unembed=sft.train_unembed,
            )
            training_client.load_state(rl.sft_checkpoint).result()

            sampling_client = training_client.save_weights_and_get_sampling_client(
                name="rl-sampler",
            )
            tokenizer = training_client.get_tokenizer()

        all_rewards_history: list[float] = []

        for step in range(rl.steps):
            t0 = time.monotonic()

            prompts = prompt_sampler.sample(rl.batch_size)

            all_completions: list[dict] = []
            all_prompt_tokens: list[list[int]] = []
            all_completion_tokens: list[list[int]] = []

            for prompt in prompts:
                if sampling_client is not None:
                    model_input = _render_prompt_to_model_input(
                        prompt["messages"], tokenizer
                    )
                    params = tinker_types.SamplingParams(
                        max_tokens=rl.max_new_tokens,
                        temperature=rl.temperature,
                        stop=["\n\n"],
                    )
                    sample_result = sampling_client.sample(
                        prompt=model_input,
                        num_samples=rl.num_completions,
                        sampling_params=params,
                    ).result()

                    prompt_token_ids = model_input.to_ints()
                    for seq in sample_result.sequences:
                        code_text = tokenizer.decode(seq.tokens)
                        all_completions.append({
                            "object_id": prompt["object_id"],
                            "code": code_text,
                            "text_description": prompt.get("caption", ""),
                            "seed": 42,
                        })
                        all_prompt_tokens.append(prompt_token_ids)
                        all_completion_tokens.append(list(seq.tokens))
                else:
                    for _ in range(rl.num_completions):
                        all_completions.append({
                            "object_id": prompt["object_id"],
                            "code": (
                                "import bpy\n"
                                "bpy.ops.object.select_all(action='SELECT')\n"
                                "bpy.ops.object.delete()\n"
                                f"# dry run step {step}\n"
                            ),
                            "text_description": prompt.get("caption", ""),
                            "seed": 42,
                        })
                        all_prompt_tokens.append([])
                        all_completion_tokens.append([])

            exec_results = await self.harness.execute_batch(all_completions)

            rewards = [r.get("reward", 0.0) for r in exec_results]
            all_rewards_history.extend(rewards)

            advantages = self._compute_advantages(rewards, rl.batch_size, rl.num_completions)

            if training_client is not None:
                grpo_data = [
                    _build_grpo_datum(pt, ct, adv)
                    for pt, ct, adv in zip(
                        all_prompt_tokens, all_completion_tokens, advantages
                    )
                ]

                fwdbwd_future = training_client.forward_backward(
                    grpo_data, "cross_entropy"
                )
                optim_future = training_client.optim_step(
                    tinker_types.AdamParams(
                        learning_rate=rl.learning_rate,
                        weight_decay=0.01,
                    )
                )
                fwdbwd_future.result()
                optim_future.result()

                sampling_client = training_client.save_weights_and_get_sampling_client(
                    name=f"rl-sampler-step-{step}",
                )

            elapsed = time.monotonic() - t0

            exec_rate = sum(1 for r in rewards if r > 0.05) / max(len(rewards), 1)
            mean_r = sum(rewards) / max(len(rewards), 1)

            self.wb.log(
                {
                    "rl/mean_reward": mean_r,
                    "rl/exec_rate": exec_rate,
                    "rl/max_reward": max(rewards) if rewards else 0.0,
                    "rl/min_reward": min(rewards) if rewards else 0.0,
                    "rl/mean_advantage": float(np.mean(advantages)) if advantages else 0.0,
                    "rl/step_time_s": elapsed,
                },
                step=step,
            )

            if step % rl.log_every == 0:
                log.info(
                    f"Step {step}/{rl.steps} | "
                    f"exec_rate={exec_rate:.2f} | "
                    f"mean_reward={mean_r:.3f} | "
                    f"elapsed={elapsed:.1f}s"
                )

            if step > 0 and step % rl.checkpoint_every == 0:
                if training_client is not None:
                    training_client.save_state(f"rl-step-{step}").result()
                log.info(f"Checkpoint saved: rl-step-{step}")

        self.wb.log_summary({
            "rl/final_mean_reward": sum(all_rewards_history)
            / max(len(all_rewards_history), 1),
            "rl/total_steps": rl.steps,
        })
        self.wb.finish()
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
            group = rewards[i * num_completions : (i + 1) * num_completions]
            if not group:
                continue
            mean_r = sum(group) / len(group)
            std_r = max(float(np.std(group)), 1e-6)
            group_adv = [(r - mean_r) / std_r for r in group]
            advantages.extend(group_adv)
        return advantages


def run_rl(cfg: ProjectConfig | None = None):
    """Entry point for RL training.

    When called programmatically, pass a ProjectConfig directly or ``None``
    to use Pydantic defaults.  For CLI usage with overrides and sweeps,
    run via ``python -m training.rl_trainer`` (Hydra handles config).
    """
    if cfg is None:
        cfg = load_config()
    trainer = GRPOTrainer(cfg)
    asyncio.run(trainer.train())


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from config import _load_env_file
    from configs.structured import register_configs

    register_configs()
    _load_env_file()

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def _hydra_main(hydra_cfg: DictConfig) -> None:
        raw = OmegaConf.to_container(hydra_cfg, resolve=True)
        cfg = ProjectConfig(**raw)
        run_rl(cfg)

    _hydra_main()
