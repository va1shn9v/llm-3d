"""
Blender3DRubric — gated sparse + dense reward computation (Section 6 of spec).
"""

from __future__ import annotations

import logging
from typing import Any

from config import RewardConfig

log = logging.getLogger(__name__)


class Blender3DRubric:
    """Computes the gated reward from execution results.

    Reward has discrete gates at low quality and continuous bonus at high quality.
    This gives GRPO clear contrast within groups at every training phase.
    """

    def __init__(self, cfg: RewardConfig | None = None):
        if cfg is None:
            from config import load_config
            cfg = load_config().reward
        self.cfg = cfg

    def score(
        self,
        code: str,
        exec_result: dict[str, Any],
        gt_mesh_bytes: bytes | None = None,  # noqa: ARG002 - reserved for future per-object thresholds
    ) -> float:
        """Compute gated sparse + dense reward."""
        base = self._base_reward(code, exec_result)
        fmt = self._format_reward(code)
        return (1 - self.cfg.format_reward_weight) * base + self.cfg.format_reward_weight * fmt

    def _base_reward(self, code: str, exec_result: dict[str, Any]) -> float:
        cfg = self.cfg

        if not code or not code.strip():
            return cfg.gate_empty

        if "from bpy_lib import" not in code and "import bpy" not in code:
            return cfg.gate_no_import

        if not exec_result.get("success", False):
            return cfg.gate_exec_fail

        stats = exec_result.get("mesh_stats")
        if not stats or stats.get("faces", 0) < 4:
            return cfg.gate_no_geometry

        if stats.get("vertices", 0) > 100_000:
            return cfg.gate_degenerate

        metrics = exec_result.get("metrics")
        if metrics is None or metrics.get("error"):
            return cfg.gate_metrics_fail

        f_score = metrics.get("f_score_005", 0.0)
        if f_score < 0.05:
            return cfg.gate_no_resemblance

        quality = cfg.quality_floor + (cfg.quality_ceil - cfg.quality_floor) * min(
            1.0, f_score / cfg.f_score_target
        )
        return quality

    def _format_reward(self, code: str) -> float:
        score = 0.0
        if code.startswith("from bpy_lib import"):
            score += 0.25
        if "# object name:" in code:
            score += 0.25
        if "# part_" in code:
            score += 0.25
        if "export_scene()" in code:
            score += 0.25
        return score

    def score_batch(self, items: list[dict[str, Any]]) -> list[float]:
        """Score a batch of (code, exec_result) pairs."""
        return [
            self.score(
                item.get("code", ""),
                item,
                item.get("gt_mesh_bytes"),
            )
            for item in items
        ]
