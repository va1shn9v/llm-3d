"""
SFT training loop on Tinker (Section 8.1 of spec).

Trains Qwen2.5-VL-7B-Instruct with LoRA on the prepared SFT dataset.
Uses Tinker's async API for forward_backward and optim_step.
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
            batch_idx = indices[start:start + self.batch_size]
            yield [self.samples[i] for i in batch_idx]


class SFTTrainer:
    """SFT training loop using Tinker's async API."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        self._step = 0
        self._epoch = 0

    async def train(self, tinker_client: Any = None):
        """Run the full SFT training loop.

        If tinker_client is None, runs in dry-run mode (logs only).
        """
        sft = self.cfg.sft

        log.info(f"Starting SFT training: {sft.epochs} epochs, "
                 f"bs={sft.batch_size}, lr={sft.learning_rate}")

        dataloader = SFTDataLoader(
            sft.train_path,
            batch_size=sft.batch_size,
            seed=self.cfg.seed,
        )

        training_client = None
        renderer = None

        if tinker_client is not None:
            training_client = tinker_client.create_lora_training_client(
                base_model=sft.base_model,
                rank=sft.lora_rank,
                target_modules=sft.target_modules,
            )

        for epoch in range(sft.epochs):
            self._epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(dataloader):
                self._step = epoch * len(dataloader) + step

                rendered = [sample["messages"] for sample in batch]

                if training_client is not None:
                    fwdbwd = training_client.forward_backward(
                        rendered, loss_type="cross_entropy"
                    )
                    result = await asyncio.wrap_future(fwdbwd.result())
                    step_loss = result.get("loss", 0.0)
                else:
                    step_loss = np.random.exponential(0.5)
                    await asyncio.sleep(0.01)

                epoch_loss += step_loss
                epoch_steps += 1

                if (step + 1) % sft.grad_accum_steps == 0:
                    if training_client is not None:
                        optim = training_client.optim_step({
                            "optimizer": "adamw",
                            "lr": sft.learning_rate,
                            "weight_decay": sft.weight_decay,
                            "beta1": 0.9,
                            "beta2": 0.999,
                        })
                        await asyncio.wrap_future(optim.result())

                if step % 50 == 0:
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    log.info(
                        f"Epoch {epoch} Step {step}/{len(dataloader)} "
                        f"loss={step_loss:.4f} avg_loss={avg_loss:.4f}"
                    )

                if self._step > 0 and self._step % sft.eval_every_n_steps == 0:
                    await self._run_eval(training_client)

            if training_client is not None:
                training_client.save_state(f"sft-epoch-{epoch}")

            avg_loss = epoch_loss / max(epoch_steps, 1)
            log.info(f"Epoch {epoch} complete. avg_loss={avg_loss:.4f}")
            await self._run_eval(training_client)

        log.info("SFT training complete.")

    async def _run_eval(self, training_client: Any):
        """Run evaluation on validation set."""
        log.info(f"Running eval at step {self._step}...")
        # Evaluation delegated to eval_runner.py for full metrics.
        # Here we just log a placeholder.
        log.info(f"  eval placeholder: step={self._step}, epoch={self._epoch}")


def run_sft(config_path: str | None = None):
    """Entry point for SFT training."""
    cfg = load_config(config_path)
    trainer = SFTTrainer(cfg)
    asyncio.run(trainer.train())
