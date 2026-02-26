"""
Blender3DEnvironment — complete environment: Dataset + Harness + Rubric.

Provides the standard interface expected by training frameworks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import ProjectConfig, load_config
from environments.blender_3d.dataset import Blender3DDataset
from environments.blender_3d.harness import Blender3DHarness
from environments.blender_3d.rubric import Blender3DRubric

log = logging.getLogger(__name__)


class Blender3DEnvironment:
    """Complete environment: Dataset + Harness + Rubric.

    Usage:
        env = Blender3DEnvironment(
            dataset_path="datasets/sft_train.jsonl",
            modal_endpoint="https://your-workspace--reward-api-web.modal.run",
            auth_token="...",
        )

        # For RL training:
        for batch in env.iter_batches(batch_size=16):
            prompts = [env.dataset.format_prompt(item) for item in batch]
            completions = model.generate(prompts)
            exec_results = await env.harness.execute_batch(completions)
            rewards = env.rubric.score_batch(exec_results)
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        images_base_dir: str | Path = ".",
        modal_endpoint: str | None = None,
        auth_token: str = "",
        num_views: int = 4,
        cfg: ProjectConfig | None = None,
    ):
        if cfg is None:
            cfg = load_config()

        dp = dataset_path or cfg.sft.train_path
        ep = modal_endpoint or cfg.modal.endpoint
        tk = auth_token or cfg.modal.auth_token

        self.dataset = Blender3DDataset(
            jsonl_path=dp,
            images_base_dir=images_base_dir,
            num_views=num_views,
        )
        self.harness = Blender3DHarness(
            modal_endpoint=ep,
            auth_token=tk,
        )
        self.rubric = Blender3DRubric(cfg.reward)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.dataset)

    def iter_batches(self, batch_size: int = 16):
        """Yield batches of dataset items."""
        items = []
        for i in range(len(self.dataset)):
            items.append(self.dataset[i])
            if len(items) == batch_size:
                yield items
                items = []
        if items:
            yield items

    async def step(
        self,
        completions: list[dict[str, str]],
    ) -> list[float]:
        """Execute completions and return rewards.

        completions: [{"object_id": str, "code": str}]
        """
        exec_results = await self.harness.execute_batch(completions)
        rewards = self.rubric.score_batch(exec_results)
        return rewards
