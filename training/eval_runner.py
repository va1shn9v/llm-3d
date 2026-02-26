"""
Evaluation pipeline (Section 9 of spec).

Runs all experimental conditions across test sets and view counts.
Computes full metrics with confidence intervals via paired bootstrap.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import ProjectConfig, load_config
from environments.blender_3d.dataset import Blender3DDataset
from environments.blender_3d.harness import Blender3DHarness

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    object_id: str
    category: str
    condition: str
    num_views: int
    success: bool
    has_geometry: bool
    metrics: dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    code: str = ""
    error: str = ""


@dataclass
class AggregatedMetrics:
    condition: str
    test_set: str
    num_views: int | str
    n_total: int
    execution_rate: float
    geometry_rate: float
    mean_f_score_005: float
    mean_chamfer: float
    mean_hausdorff_90: float
    mean_normal_consistency: float
    mean_reward: float
    ci_f_score_005: tuple[float, float] = (0.0, 0.0)
    ci_chamfer: tuple[float, float] = (0.0, 0.0)


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    means.sort()
    lo = means[int(n_bootstrap * alpha / 2)]
    hi = means[int(n_bootstrap * (1 - alpha / 2))]
    return (round(lo, 6), round(hi, 6))


class EvalRunner:
    """Run full evaluation across conditions, test sets, and view counts."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        self.harness = Blender3DHarness(
            modal_endpoint=self.cfg.modal.endpoint,
            auth_token=self.cfg.modal.auth_token,
        )

    async def evaluate_condition(
        self,
        condition: str,
        test_dataset: Blender3DDataset,
        num_views: int,
        generate_fn: Any = None,
    ) -> list[EvalResult]:
        """Evaluate a single condition on a test set with given view count.

        generate_fn: async callable that takes messages and returns code string.
                     If None, uses a placeholder.
        """
        results: list[EvalResult] = []

        for i in range(len(test_dataset)):
            item = test_dataset[i]
            images = item["images"][:num_views]

            if generate_fn is not None:
                prompt = test_dataset.format_prompt({**item, "images": images})
                code = await generate_fn(prompt)
            else:
                code = "from bpy_lib import *\n# eval placeholder\nexport_scene()"

            exec_items = [{"object_id": item["object_id"], "code": code, "seed": 42}]
            exec_results = await self.harness.execute_batch(exec_items)

            er = exec_results[0] if exec_results else {}
            success = er.get("success", False)
            metrics = er.get("metrics") or {}
            has_geom = success and (er.get("mesh_stats", {}).get("faces", 0) >= 4)

            results.append(EvalResult(
                object_id=item["object_id"],
                category=item["category"],
                condition=condition,
                num_views=num_views,
                success=success,
                has_geometry=has_geom,
                metrics=metrics,
                reward=er.get("reward", 0.0),
                code=code,
                error=er.get("error", ""),
            ))

        return results

    def aggregate(
        self,
        results: list[EvalResult],
        condition: str,
        test_set: str,
        num_views: int | str = "all",
    ) -> AggregatedMetrics:
        """Aggregate results into summary metrics with CIs."""
        n = len(results)
        if n == 0:
            return AggregatedMetrics(
                condition=condition, test_set=test_set, num_views=num_views,
                n_total=0, execution_rate=0, geometry_rate=0,
                mean_f_score_005=0, mean_chamfer=0, mean_hausdorff_90=0,
                mean_normal_consistency=0, mean_reward=0,
            )

        exec_rate = sum(1 for r in results if r.success) / n
        geom_rate = sum(1 for r in results if r.has_geometry) / n

        f_scores = [r.metrics.get("f_score_005", 0.0) for r in results if r.success]
        chamfers = [r.metrics.get("chamfer", 1.0) for r in results if r.success]
        hausdorffs = [r.metrics.get("hausdorff_90", 1.0) for r in results if r.success]
        normals = [r.metrics.get("normal_consistency", 0.0) for r in results if r.success]
        rewards = [r.reward for r in results]

        bs = self.cfg.eval.bootstrap_samples
        ci_f = bootstrap_ci(f_scores, bs) if f_scores else (0.0, 0.0)
        ci_c = bootstrap_ci(chamfers, bs) if chamfers else (0.0, 0.0)

        return AggregatedMetrics(
            condition=condition,
            test_set=test_set,
            num_views=num_views,
            n_total=n,
            execution_rate=round(exec_rate, 4),
            geometry_rate=round(geom_rate, 4),
            mean_f_score_005=round(np.mean(f_scores).item(), 4) if f_scores else 0.0,
            mean_chamfer=round(np.mean(chamfers).item(), 6) if chamfers else 0.0,
            mean_hausdorff_90=round(np.mean(hausdorffs).item(), 4) if hausdorffs else 0.0,
            mean_normal_consistency=round(np.mean(normals).item(), 4) if normals else 0.0,
            mean_reward=round(np.mean(rewards).item(), 4) if rewards else 0.0,
            ci_f_score_005=ci_f,
            ci_chamfer=ci_c,
        )

    async def run_full_evaluation(
        self,
        conditions: dict[str, Any] | None = None,
        test_sets: dict[str, str] | None = None,
    ) -> list[AggregatedMetrics]:
        """Run the full 4-condition evaluation matrix.

        conditions: {name: generate_fn} — if None, uses dry-run.
        test_sets:  {name: jsonl_path}  — if None, uses config defaults.
        """
        if conditions is None:
            conditions = {
                "baseline": None,
                "sft_only": None,
                "sft_rl": None,
                "rl_only": None,
            }

        if test_sets is None:
            test_sets = {
                "id": self.cfg.sft.val_path,
            }

        all_agg: list[AggregatedMetrics] = []

        for ts_name, ts_path in test_sets.items():
            dataset = Blender3DDataset(ts_path)

            for cond_name, gen_fn in conditions.items():
                for nv in self.cfg.eval.view_counts:
                    log.info(f"Evaluating {cond_name} on {ts_name} with {nv} views...")
                    results = await self.evaluate_condition(cond_name, dataset, nv, gen_fn)
                    agg = self.aggregate(results, cond_name, ts_name, nv)
                    all_agg.append(agg)

                    log.info(
                        f"  {cond_name}/{ts_name}/{nv}v: "
                        f"exec={agg.execution_rate:.2f} "
                        f"f005={agg.mean_f_score_005:.3f} "
                        f"cd={agg.mean_chamfer:.5f} "
                        f"reward={agg.mean_reward:.3f}"
                    )

        return all_agg

    def save_results(
        self,
        aggregated: list[AggregatedMetrics],
        output_path: str | Path,
    ):
        """Save aggregated results as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for a in aggregated:
            data.append({
                "condition": a.condition,
                "test_set": a.test_set,
                "num_views": a.num_views,
                "n_total": a.n_total,
                "execution_rate": a.execution_rate,
                "geometry_rate": a.geometry_rate,
                "mean_f_score_005": a.mean_f_score_005,
                "mean_chamfer": a.mean_chamfer,
                "mean_hausdorff_90": a.mean_hausdorff_90,
                "mean_normal_consistency": a.mean_normal_consistency,
                "mean_reward": a.mean_reward,
                "ci_f_score_005": list(a.ci_f_score_005),
                "ci_chamfer": list(a.ci_chamfer),
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        log.info(f"Saved {len(data)} aggregated results to {output_path}")


def run_eval(config_path: str | None = None):
    """Entry point for evaluation."""
    cfg = load_config(config_path)
    runner = EvalRunner(cfg)
    results = asyncio.run(runner.run_full_evaluation())
    runner.save_results(results, Path(cfg.output_dir) / "eval_results.json")
