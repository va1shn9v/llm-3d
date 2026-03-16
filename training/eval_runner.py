"""
Evaluation pipeline (Section 9 of spec).

Runs all experimental conditions across test sets.
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
from training.wandb_logger import WandbLogger

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    object_id: str
    condition: str
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
    """Compute bootstrap confidence interval (vectorized)."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    samples = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True)
    means = np.sort(samples.mean(axis=1))
    lo = float(means[int(n_bootstrap * alpha / 2)])
    hi = float(means[int(n_bootstrap * (1 - alpha / 2))])
    return (round(lo, 6), round(hi, 6))


class EvalRunner:
    """Run full evaluation across conditions and test sets."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        self.harness = Blender3DHarness(
            modal_endpoint=self.cfg.modal.endpoint,
            auth_token=self.cfg.modal.auth_token,
        )
        self.wb = WandbLogger(
            self.cfg.logging,
            run_name="eval",
            tags=["eval"],
            extra_config=self.cfg.eval.model_dump(),
        )

    async def evaluate_condition(
        self,
        condition: str,
        test_dataset: Blender3DDataset,
        generate_fn: Any = None,
    ) -> list[EvalResult]:
        """Evaluate a single condition on a test set.

        generate_fn: async callable that takes messages and returns code string.
                     If None, uses a placeholder.
        """
        results: list[EvalResult] = []

        for i in range(len(test_dataset)):
            item = test_dataset[i]

            if generate_fn is not None:
                prompt = test_dataset.format_prompt(item)
                code = await generate_fn(prompt)
            else:
                code = "import bpy\n# eval placeholder — no-op\n"

            exec_items = [{"object_id": item["object_id"], "code": code, "seed": 42}]
            exec_results = await self.harness.execute_batch(exec_items)

            er = exec_results[0] if exec_results else {}
            success = er.get("success", False)
            metrics = er.get("metrics") or {}
            has_geom = success and (er.get("mesh_stats", {}).get("faces", 0) >= 4)

            results.append(EvalResult(
                object_id=item["object_id"],
                condition=condition,
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
    ) -> AggregatedMetrics:
        """Aggregate results into summary metrics with CIs."""
        n = len(results)
        if n == 0:
            return AggregatedMetrics(
                condition=condition, test_set=test_set,
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
            dataset_dir = Path(self.cfg.sft.train_path).parent
            eval_id_path = dataset_dir / "eval_id.jsonl"
            eval_ood_path = dataset_dir / "eval_ood.jsonl"

            test_sets = {}
            if eval_id_path.exists():
                test_sets["id"] = str(eval_id_path)
            elif Path(self.cfg.sft.val_path).exists():
                test_sets["id"] = self.cfg.sft.val_path

            if eval_ood_path.exists():
                test_sets["ood"] = str(eval_ood_path)

            if not test_sets:
                raise FileNotFoundError(
                    "No evaluation dataset found. Expected eval_id.jsonl/eval_ood.jsonl "
                    f"in {dataset_dir} or fallback val set at {self.cfg.sft.val_path}."
                )

        all_agg: list[AggregatedMetrics] = []

        for ts_name, ts_path in test_sets.items():
            dataset = Blender3DDataset(ts_path, storage_cfg=self.cfg.storage)

            for cond_name, gen_fn in conditions.items():
                log.info(f"Evaluating {cond_name} on {ts_name}...")
                results = await self.evaluate_condition(cond_name, dataset, gen_fn)
                agg = self.aggregate(results, cond_name, ts_name)
                all_agg.append(agg)

                prefix = f"eval/{cond_name}/{ts_name}"
                self.wb.log({
                    f"{prefix}/execution_rate": agg.execution_rate,
                    f"{prefix}/geometry_rate": agg.geometry_rate,
                    f"{prefix}/f_score_005": agg.mean_f_score_005,
                    f"{prefix}/chamfer": agg.mean_chamfer,
                    f"{prefix}/hausdorff_90": agg.mean_hausdorff_90,
                    f"{prefix}/normal_consistency": agg.mean_normal_consistency,
                    f"{prefix}/reward": agg.mean_reward,
                })

                log.info(
                    f"  {cond_name}/{ts_name}: "
                    f"exec={agg.execution_rate:.2f} "
                    f"f005={agg.mean_f_score_005:.3f} "
                    f"cd={agg.mean_chamfer:.5f} "
                    f"reward={agg.mean_reward:.3f}"
                )

        self.wb.finish()
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


def run_eval(cfg: ProjectConfig | None = None):
    """Entry point for evaluation.

    When called programmatically, pass a ProjectConfig directly or ``None``
    to use Pydantic defaults.  For CLI usage with overrides and sweeps,
    run via ``python -m training.eval_runner`` (Hydra handles config).
    """
    if cfg is None:
        cfg = load_config()
    runner = EvalRunner(cfg)
    results = asyncio.run(runner.run_full_evaluation())
    runner.save_results(results, Path(cfg.output_dir) / "eval_results.json")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from configs.structured import register_configs

    from config import _load_env_file

    register_configs()
    _load_env_file()

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def _hydra_main(hydra_cfg: DictConfig) -> None:
        raw = OmegaConf.to_container(hydra_cfg, resolve=True)
        cfg = ProjectConfig(**raw)
        run_eval(cfg)

    _hydra_main()
