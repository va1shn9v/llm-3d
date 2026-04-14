"""
Binary-threshold reward with geometry and format components.
"""

from __future__ import annotations

import logging
from typing import Any

from config import BinaryRewardConfig, NumericBinaryRewardConfig, RewardConfig

log = logging.getLogger(__name__)


class Blender3DRubric:
    """Computes the combined reward from execution results.

    Each configured sub-reward is binary. Thresholds and weights are defined in ``RewardConfig``.
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
    ) -> float:
        """Compute combined reward: geometric + format."""
        return self.evaluate(
            code,
            exec_result,
            text_description=text_description,
        )["reward"]

    def evaluate(
        self,
        code: str,
        exec_result: dict[str, Any],
        text_description: str = "",
    ) -> dict[str, Any]:
        """Compute weighted binary rewards and return component details."""
        del text_description

        geometry_score, geometry_checks = self._base_reward(code, exec_result)
        format_score, format_checks = self._format_reward(code)

        total = (
            self.cfg.geometric_weight * geometry_score
            + self.cfg.format_reward_weight * format_score
        )

        return {
            "reward": total,
            "base_reward": geometry_score,
            "format_reward": format_score,
            "sub_rewards": {
                "geometry": geometry_checks,
                "format": format_checks,
            },
        }

    def _base_reward(self, code: str, exec_result: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """Binary geometric reward adapted for raw bpy code."""
        cfg = self.cfg
        stats = exec_result.get("mesh_stats")

        metrics = exec_result.get("metrics")
        checks = {
            "non_empty": self._binary(code.strip() != "", cfg.geometry.non_empty),
            "import_bpy": self._binary("import bpy" in code, cfg.geometry.import_bpy),
            "exec_success": self._binary(
                bool(exec_result.get("success", False)),
                cfg.geometry.exec_success,
            ),
            "min_faces": self._numeric_binary(
                self._coerce_number(stats, "faces"),
                cfg.geometry.min_faces,
                operator="ge",
            ),
            "max_vertices": self._numeric_binary(
                self._coerce_number(stats, "vertices"),
                cfg.geometry.max_vertices,
                operator="le",
            ),
            "metrics_available": self._binary(
                metrics is not None and not metrics.get("error"),
                cfg.geometry.metrics_available,
            ),
            "resemblance": self._numeric_binary(
                self._coerce_number(metrics, "f_score_005"),
                cfg.geometry.resemblance,
                operator="ge",
            ),
        }
        return self._aggregate(checks, cfg.geometry)

    def _format_reward(self, code: str) -> tuple[float, dict[str, float]]:
        """Binary format reward for raw bpy code structure."""
        has_comments = any(
            line.strip().startswith("# ") and len(line.strip()) > 4
            for line in code.splitlines()
        )

        clears_scene = (
            "bpy.ops.object.select_all" in code
            or "bpy.data.objects.remove" in code
            or "bpy.ops.wm.read_factory_settings" in code
        )
        has_export = "EXPORT_PATH" in code or "export" in code.lower()

        checks = {
            "import_first": self._binary(
                code.strip().startswith("import bpy"),
                self.cfg.format.import_first,
            ),
            "has_comments": self._binary(has_comments, self.cfg.format.has_comments),
            "clears_scene": self._binary(clears_scene, self.cfg.format.clears_scene),
            "has_export": self._binary(has_export, self.cfg.format.has_export),
        }
        return self._aggregate(checks, self.cfg.format)

    def _aggregate(
        self,
        checks: dict[str, tuple[float, float]],
        config_group: Any,
    ) -> tuple[float, dict[str, float]]:
        """Return weighted average and plain 0/1 check results."""
        total_weight = 0.0
        weighted_sum = 0.0
        values: dict[str, float] = {}

        for name, (value, weight) in checks.items():
            if weight <= 0:
                continue
            total_weight += weight
            weighted_sum += value * weight
            values[name] = value

        if total_weight == 0:
            log.warning("No enabled reward checks in %s", type(config_group).__name__)
            return 0.0, values
        return weighted_sum / total_weight, values

    @staticmethod
    def _binary(passed: bool, cfg: BinaryRewardConfig) -> tuple[float, float]:
        if not cfg.enabled:
            return 0.0, 0.0
        return (1.0 if passed else 0.0, cfg.weight)

    @staticmethod
    def _numeric_binary(
        value: float | None,
        cfg: NumericBinaryRewardConfig,
        *,
        operator: str,
    ) -> tuple[float, float]:
        if not cfg.enabled:
            return 0.0, 0.0
        if value is None:
            return 0.0, cfg.weight
        if operator == "ge":
            passed = value >= cfg.threshold
        elif operator == "le":
            passed = value <= cfg.threshold
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        return (1.0 if passed else 0.0, cfg.weight)

    @staticmethod
    def _coerce_number(data: dict[str, Any] | None, key: str) -> float | None:
        if not data:
            return None
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def score_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[float]:
        """Score a batch of (code, exec_result, text) dicts."""
        return [
            self.score(
                item.get("code", ""),
                item,
                text_description=item.get("text_description", ""),
            )
            for item in items
        ]
