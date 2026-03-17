"""
Blender3DEnvironment — complete environment: Dataset + Harness.

Provides the standard interface expected by training frameworks.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import ProjectConfig, load_config
from environments.blender_3d.dataset import Blender3DDataset
from environments.blender_3d.harness import Blender3DHarness

log = logging.getLogger(__name__)


class Blender3DEnvironment:
    """Complete environment: Dataset + Harness.

    Rewards are computed server-side by the Modal reward API.

    Usage:
        env = Blender3DEnvironment(
            dataset_path="datasets/sft_train.jsonl",
            modal_endpoint="https://your-workspace--reward-api-web.modal.run",
            auth_token="...",
        )

        for batch in env.iter_batches(batch_size=16):
            prompts = [env.dataset.format_prompt(item) for item in batch]
            completions = model.generate(prompts)
            rewards = await env.step(completions)
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        modal_endpoint: str | None = None,
        auth_token: str = "",
        cfg: ProjectConfig | None = None,
    ):
        if cfg is None:
            cfg = load_config()

        dp = dataset_path or cfg.sft.train_path
        ep = modal_endpoint or cfg.modal.endpoint
        tk = auth_token or cfg.modal.auth_token

        self.dataset = Blender3DDataset(jsonl_path=dp, storage_cfg=cfg.storage)
        self.harness = Blender3DHarness(
            modal_endpoint=ep,
            auth_token=tk,
            reward_cfg=cfg.reward,
        )
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
        """Execute completions and return server-computed rewards.

        completions: [{"object_id": str, "code": str}]
        """
        exec_results = await self.harness.execute_batch(completions)
        return [r.get("reward", 0.0) for r in exec_results]
