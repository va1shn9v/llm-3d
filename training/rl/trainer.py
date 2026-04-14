"""
Direct RL training loop for Blender code generation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Sequence

import numpy as np

from config import ProjectConfig, load_config, load_config_from_cli
from environments.blender_3d.harness import Blender3DHarness
from training.common.tinker import (
    build_tinker_service_client,
    clean_generated_code,
    render_prompt_to_model_input,
    require_tinker_types,
)
from training.common.tracking import WandbLogger
from training.rl.dataset import RLPromptSampler

log = logging.getLogger(__name__)


def _build_grpo_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    advantage: float,
) -> Any:
    """Package a single prompt+completion with its advantage as a Tinker datum."""
    types = require_tinker_types()
    tokens = prompt_tokens + completion_tokens
    prompt_weights = [0.0] * len(prompt_tokens)
    completion_weights = [1.0] * len(completion_tokens)
    weights = (prompt_weights + completion_weights)[1:]

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
            advantage=advantage,
        ),
    )


class GRPOTrainer:
    """GRPO RL training loop for direct instruct-model fine-tuning."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        if not self.cfg.modal.endpoint:
            raise ValueError("modal.endpoint must be configured before running RL")

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
        self.wb.define_metric("rl/execution_rate", step_metric="step")
        self.wb.define_metric("rl/max_reward", step_metric="step")
        self.wb.define_metric("rl/min_reward", step_metric="step")
        self.wb.define_metric("rl/mean_advantage", step_metric="step")
        self.wb.define_metric("rl/step_time_s", step_metric="step")

    async def train(self, service_client: Any | None = None) -> None:
        rl = self.cfg.rl
        prompt_sampler = RLPromptSampler(rl.prompt_path, self.cfg)

        if service_client is None:
            service_client = build_tinker_service_client()

        types = require_tinker_types()
        training_client = service_client.create_lora_training_client(
            base_model=rl.base_model,
            rank=rl.lora_rank,
            train_mlp=rl.train_mlp,
            train_attn=rl.train_attn,
            train_unembed=rl.train_unembed,
        )
        if rl.init_state_path:
            training_client.load_state(rl.init_state_path).result()

        tokenizer = training_client.get_tokenizer()
        sampling_client = training_client.save_weights_and_get_sampling_client(name="rl-sampler")

        all_rewards_history: list[float] = []
        log.info(
            "Starting direct RL: model=%s steps=%d batch_size=%d num_completions=%d",
            rl.base_model,
            rl.steps,
            rl.batch_size,
            rl.num_completions,
        )

        for step in range(rl.steps):
            step_start = time.monotonic()
            prompts = prompt_sampler.sample(rl.batch_size)

            all_completions: list[dict[str, Any]] = []
            all_prompt_tokens: list[list[int]] = []
            all_completion_tokens: list[list[int]] = []

            for prompt in prompts:
                model_input = render_prompt_to_model_input(prompt["messages"], tokenizer)
                params = types.SamplingParams(
                    max_tokens=rl.max_new_tokens,
                    temperature=rl.temperature,
                    stop=list(rl.stop),
                )
                sample_result = sampling_client.sample(
                    prompt=model_input,
                    num_samples=rl.num_completions,
                    sampling_params=params,
                ).result()

                prompt_token_ids = model_input.to_ints()
                for seq in sample_result.sequences:
                    code_text = clean_generated_code(tokenizer.decode(seq.tokens))
                    all_completions.append(
                        {
                            "object_id": prompt["object_id"],
                            "code": code_text,
                            "text_description": prompt["caption"],
                            "seed": self.cfg.seed,
                        }
                    )
                    all_prompt_tokens.append(prompt_token_ids)
                    all_completion_tokens.append(list(seq.tokens))

            exec_results = await self.harness.execute_batch(all_completions)
            rewards = [float(item.get("reward", 0.0)) for item in exec_results]
            all_rewards_history.extend(rewards)

            advantages = self._compute_advantages(rewards, len(prompts), rl.num_completions)
            grpo_data = [
                _build_grpo_datum(prompt_tokens, completion_tokens, advantage)
                for prompt_tokens, completion_tokens, advantage in zip(
                    all_prompt_tokens,
                    all_completion_tokens,
                    advantages,
                )
            ]

            fwdbwd_future = training_client.forward_backward(grpo_data, "cross_entropy")
            optim_future = training_client.optim_step(
                types.AdamParams(
                    learning_rate=rl.learning_rate,
                    weight_decay=0.01,
                )
            )
            fwdbwd_future.result()
            optim_future.result()

            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"rl-sampler-step-{step}",
            )

            elapsed = time.monotonic() - step_start
            execution_rate = sum(1 for item in exec_results if item.get("success")) / max(
                len(exec_results),
                1,
            )
            mean_reward = sum(rewards) / max(len(rewards), 1)

            self.wb.log(
                {
                    "rl/mean_reward": mean_reward,
                    "rl/execution_rate": execution_rate,
                    "rl/max_reward": max(rewards) if rewards else 0.0,
                    "rl/min_reward": min(rewards) if rewards else 0.0,
                    "rl/mean_advantage": float(np.mean(advantages)) if advantages else 0.0,
                    "rl/step_time_s": elapsed,
                },
                step=step,
            )

            if step % rl.log_every == 0:
                log.info(
                    "Step %d/%d | execution_rate=%.2f | mean_reward=%.3f | elapsed=%.1fs",
                    step,
                    rl.steps,
                    execution_rate,
                    mean_reward,
                    elapsed,
                )

            if step > 0 and step % rl.checkpoint_every == 0:
                training_client.save_state(f"rl-step-{step}").result()
                log.info("Checkpoint saved: rl-step-%d", step)

        self.wb.log_summary(
            {
                "rl/final_mean_reward": sum(all_rewards_history) / max(len(all_rewards_history), 1),
                "rl/total_steps": rl.steps,
            }
        )
        self.wb.finish()
        log.info("Direct RL complete.")

    def _compute_advantages(
        self,
        rewards: list[float],
        num_prompts: int,
        num_completions: int,
    ) -> list[float]:
        """Compute normalized advantages within each prompt group."""
        advantages: list[float] = []
        for index in range(num_prompts):
            group = rewards[index * num_completions : (index + 1) * num_completions]
            if not group:
                continue
            mean_reward = sum(group) / len(group)
            std_reward = max(float(np.std(group)), 1e-6)
            advantages.extend((reward - mean_reward) / std_reward for reward in group)
        return advantages


def run_rl(cfg: ProjectConfig | None = None) -> None:
    trainer = GRPOTrainer(cfg)
    asyncio.run(trainer.train())


def main(argv: Sequence[str] | None = None) -> None:
    cfg = load_config_from_cli(
        description="Run direct GRPO RL training against the prompt dataset.",
        argv=argv,
    )
    run_rl(cfg)


if __name__ == "__main__":
    main()
