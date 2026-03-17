"""
Blender3DHarness — executes generated code in Modal Blender sandbox.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from config import RewardConfig

log = logging.getLogger(__name__)


class Blender3DHarness:
    """Executes generated code in Modal Blender sandbox.

    Connects to the Modal reward API endpoint.
    Handles batching and parallelism.
    """

    def __init__(
        self,
        modal_endpoint: str,
        auth_token: str = "",
        timeout: float = 120.0,
        max_retries: int = 1,
        reward_cfg: RewardConfig | None = None,
    ):
        self.endpoint = modal_endpoint.rstrip("/")
        self.token = auth_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.reward_cfg = reward_cfg

    async def execute_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Send batch of code samples to Modal for execution + reward.

        Input:  [{"object_id": str, "code": str, "seed": int}]
        Output: [{"reward", "success", "metrics", ...}]
        """
        payload = {"items": items}
        if self.reward_cfg is not None:
            payload["reward_config"] = self.reward_cfg.model_dump(mode="json")
        params = {"token": self.token} if self.token else {}

        for attempt in range(1 + self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.endpoint}/reward/batch",
                        json=payload,
                        params=params,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    return result.get("rewards", [])
            except Exception as e:
                if attempt < self.max_retries:
                    log.warning(f"Batch exec attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    log.error(f"Batch exec failed after {1 + self.max_retries} attempts: {e}")
                    return [
                        {"object_id": item.get("object_id", ""), "reward": 0.0,
                         "success": False, "metrics": None, "error": str(e)}
                        for item in items
                    ]
