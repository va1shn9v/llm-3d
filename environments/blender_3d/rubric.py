"""
Blender3DRubric — gated sparse + dense reward with geometric, CLIP, and format components.

Reward = geometric_weight * base + text_alignment_weight * clip + format_weight * format

The geometric reward uses a gated structure for clear GRPO contrast.
CLIP reward measures text-3D alignment via rendered views (only if geometric gate passes).
Format reward encourages good code structure.
"""

from __future__ import annotations

import logging
from typing import Any

from config import RewardConfig

log = logging.getLogger(__name__)


class Blender3DRubric:
    """Computes the combined reward from execution results.

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
        text_description: str = "",
        clip_score: float | None = None,
    ) -> float:
        """Compute combined reward: geometric + CLIP + format."""
        base = self._base_reward(code, exec_result)
        fmt = self._format_reward(code)

        text_align = 0.0
        if clip_score is not None and base > self.cfg.gate_no_resemblance:
            text_align = clip_score

        return (
            self.cfg.geometric_weight * base
            + self.cfg.text_alignment_weight * text_align
            + self.cfg.format_reward_weight * fmt
        )

    def _base_reward(self, code: str, exec_result: dict[str, Any]) -> float:
        """Gated geometric reward adapted for raw bpy code."""
        cfg = self.cfg

        if not code or not code.strip():
            return cfg.gate_empty

        if "import bpy" not in code:
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
        """Format reward for raw bpy code structure."""
        score = 0.0
        if code.strip().startswith("import bpy"):
            score += 0.25

        has_comments = any(
            line.strip().startswith("# ") and len(line.strip()) > 4
            for line in code.splitlines()
        )
        if has_comments:
            score += 0.25

        clears_scene = (
            "bpy.ops.object.select_all" in code
            or "bpy.data.objects.remove" in code
            or "bpy.ops.wm.read_factory_settings" in code
        )
        if clears_scene:
            score += 0.25

        if "EXPORT_PATH" in code or "export" in code.lower():
            score += 0.25
        return score

    def score_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[float]:
        """Score a batch of (code, exec_result, text, clip_score) dicts."""
        return [
            self.score(
                item.get("code", ""),
                item,
                text_description=item.get("text_description", ""),
                clip_score=item.get("clip_score"),
            )
            for item in items
        ]
