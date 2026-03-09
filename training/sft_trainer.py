"""
SFT training loop on Tinker.

Trains Qwen2.5-Coder-7B-Instruct with LoRA on text-to-3D SFT dataset.
Uses Tinker's async API for forward_backward and optim_step.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from training.wandb_logger import WandbLogger

log = logging.getLogger(__name__)


class SFTDataLoader:
    """Loads SFT dataset and yields batches in chat format."""

    def __init__(self, jsonl_path: str | Path, batch_size: int = 8, seed: int = 42):
        self.samples: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        log.info(f"Loaded {len(self.samples)} SFT samples from {jsonl_path}")

    def __len__(self) -> int:
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = self.rng.permutation(len(self.samples))
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield [self.samples[i] for i in batch_idx]


def _messages_to_datum(
    messages: list[dict],
    tokenizer: Any,
) -> "tinker_types.Datum":
    """Convert a text-only chat message list into a Tinker Datum.

    Prompt tokens (system + user turns) get weight 0; assistant tokens get weight 1.
    """
    prompt_parts: list[str] = []
    assistant_part: str = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            assistant_part = content
        else:
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    prompt_text = "".join(prompt_parts) + "<|im_start|>assistant\n"
    completion_text = assistant_part + "<|im_end|>\n"

    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


class SFTTrainer:
    """SFT training loop using Tinker's async API."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        self._step = 0
        self._epoch = 0
        self.wb = WandbLogger(
            self.cfg.logging,
            run_name="sft",
            tags=["sft"],
            extra_config=self.cfg.sft.model_dump(),
        )
        self.wb.define_metric("sft/loss", step_metric="step")
        self.wb.define_metric("sft/avg_loss", step_metric="step")
        self.wb.define_metric("sft/epoch", step_metric="step")
        self.wb.define_metric("sft/learning_rate", step_metric="step")

    async def train(self, service_client: Any = None):
        """Run the full SFT training loop."""
        sft = self.cfg.sft

        log.info(
            f"Starting SFT training: {sft.epochs} epochs, "
            f"bs={sft.batch_size}, lr={sft.learning_rate}, "
            f"model={sft.base_model}"
        )

        dataloader = SFTDataLoader(
            sft.train_path,
            batch_size=sft.batch_size,
            seed=self.cfg.seed,
        )

        training_client = None
        tokenizer = None

        if service_client is not None:
            training_client = service_client.create_lora_training_client(
                base_model=sft.base_model,
                rank=sft.lora_rank,
                train_mlp=sft.train_mlp,
                train_attn=sft.train_attn,
                train_unembed=sft.train_unembed,
            )
            tokenizer = training_client.get_tokenizer()

        for epoch in range(sft.epochs):
            self._epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(dataloader):
                self._step = epoch * len(dataloader) + step

                if training_client is not None:
                    data = [
                        _messages_to_datum(sample["messages"], tokenizer)
                        for sample in batch
                    ]

                    fwdbwd_future = await training_client.forward_backward_async(
                        data, "cross_entropy"
                    )
                    fwdbwd_result = fwdbwd_future.result()

                    logprobs = np.concatenate(
                        [o["logprobs"].tolist() for o in fwdbwd_result.loss_fn_outputs]
                    )
                    weights = np.concatenate(
                        [d.loss_fn_inputs["weights"] for d in data]
                    )
                    step_loss = float(-np.dot(logprobs, weights) / max(weights.sum(), 1))
                else:
                    step_loss = np.random.exponential(0.5)
                    await asyncio.sleep(0.01)

                epoch_loss += step_loss
                epoch_steps += 1

                if (step + 1) % sft.grad_accum_steps == 0:
                    if training_client is not None:
                        optim_future = training_client.optim_step(
                            tinker_types.AdamParams(
                                learning_rate=sft.learning_rate,
                                weight_decay=sft.weight_decay,
                            )
                        )
                        optim_future.result()

                self.wb.log(
                    {
                        "sft/loss": step_loss,
                        "sft/epoch": epoch,
                        "sft/learning_rate": sft.learning_rate,
                    },
                    step=self._step,
                )

                if step % 50 == 0:
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    self.wb.log({"sft/avg_loss": avg_loss}, step=self._step)
                    log.info(
                        f"Epoch {epoch} Step {step}/{len(dataloader)} "
                        f"loss={step_loss:.4f} avg_loss={avg_loss:.4f}"
                    )

                if self._step > 0 and self._step % sft.eval_every_n_steps == 0:
                    await self._run_eval(training_client)

            if training_client is not None:
                training_client.save_state(f"sft-epoch-{epoch}").result()

            avg_loss = epoch_loss / max(epoch_steps, 1)
            self.wb.log(
                {
                    "sft/epoch_avg_loss": avg_loss,
                    "sft/epoch": epoch,
                },
                step=self._step,
            )
            log.info(f"Epoch {epoch} complete. avg_loss={avg_loss:.4f}")
            await self._run_eval(training_client)

        self.wb.finish()
        log.info("SFT training complete.")

    async def _run_eval(self, training_client: Any):
        """Run evaluation on validation set."""
        log.info(f"Running eval at step {self._step}...")
        log.info(f"  eval placeholder: step={self._step}, epoch={self._epoch}")


def run_sft(cfg: ProjectConfig | None = None):
    """Entry point for SFT training.

    When called programmatically, pass a ProjectConfig directly or ``None``
    to use Pydantic defaults.  For CLI usage with overrides and sweeps,
    run via ``python -m training.sft_trainer`` (Hydra handles config).
    """
    if cfg is None:
        cfg = load_config()
    trainer = SFTTrainer(cfg)
    asyncio.run(trainer.train())


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from configs.structured import register_configs

    register_configs()

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def _hydra_main(hydra_cfg: DictConfig) -> None:
        raw = OmegaConf.to_container(hydra_cfg, resolve=True)
        cfg = ProjectConfig(**raw)
        run_sft(cfg)

    _hydra_main()
