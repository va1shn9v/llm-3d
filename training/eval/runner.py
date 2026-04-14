"""
Evaluation pipeline for Blender code generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config import EvalConditionConfig, EvalConditionsConfig, ProjectConfig, load_config, load_config_from_cli
from environments.blender_3d.dataset import Blender3DDataset
from environments.blender_3d.harness import Blender3DHarness
from training.common.tinker import (
    build_tinker_service_client,
    clean_generated_code,
    render_prompt_to_model_input,
    require_tinker_types,
)
from training.common.tracking import WandbLogger

log = logging.getLogger(__name__)


def _path_exists(path: str) -> bool:
    return path.startswith("hf://") or Path(path).exists()


def _has_geometry(exec_result: dict[str, Any]) -> bool:
    return bool(exec_result.get("success")) and exec_result.get("mesh_stats", {}).get("faces", 0) >= 4


def _selection_key(exec_result: dict[str, Any]) -> tuple[float, float, int, int]:
    metrics = exec_result.get("metrics") or {}
    return (
        float(exec_result.get("reward", 0.0)),
        float(metrics.get("f_score_005", 0.0)),
        int(_has_geometry(exec_result)),
        int(bool(exec_result.get("success", False))),
    )


@dataclass
class EvalResult:
    object_id: str
    caption: str
    condition: str
    test_set: str
    model_ref: str
    success: bool
    has_geometry: bool
    metrics: dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    code: str = ""
    error: str = ""
    selected_sample_index: int = 0
    num_samples: int = 1


@dataclass
class AggregatedMetrics:
    condition: str
    test_set: str
    model_ref: str
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
    """Compute a bootstrap confidence interval."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    samples = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True)
    means = np.sort(samples.mean(axis=1))
    lo = float(means[int(n_bootstrap * alpha / 2)])
    hi = float(means[int(n_bootstrap * (1 - alpha / 2))])
    return (round(lo, 6), round(hi, 6))


class TinkerSamplingGenerator:
    """Thin adapter around a Tinker sampling client."""

    def __init__(
        self,
        name: str,
        condition_cfg: EvalConditionConfig,
        project_cfg: ProjectConfig,
        service_client: Any,
    ):
        require_tinker_types()
        self.name = name
        self.cfg = condition_cfg
        self.project_cfg = project_cfg
        self._sem = asyncio.Semaphore(project_cfg.eval.max_concurrent_tinker)
        self.temperature = (
            condition_cfg.temperature
            if condition_cfg.temperature != 0.0 or project_cfg.eval.temperature == 0.0
            else project_cfg.eval.temperature
        )

        base_model = condition_cfg.base_model.strip() or project_cfg.rl.base_model.strip()
        model_path = condition_cfg.model_path.strip()

        if model_path:
            self.model_ref = model_path
            self.sampling_client = service_client.create_sampling_client(model_path=model_path)
        else:
            self.model_ref = base_model
            self.sampling_client = service_client.create_sampling_client(base_model=base_model)

        self.tokenizer = self.sampling_client.get_tokenizer()

    async def generate(self, messages: list[dict[str, str]]) -> list[str]:
        types = require_tinker_types()
        model_input = render_prompt_to_model_input(messages, self.tokenizer)
        params = types.SamplingParams(
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.temperature,
            stop=list(self.cfg.stop),
        )

        async with self._sem:
            sample_result = await self.sampling_client.sample_async(
                prompt=model_input,
                num_samples=self.cfg.num_samples,
                sampling_params=params,
            )

        codes = [clean_generated_code(self.tokenizer.decode(seq.tokens)) for seq in sample_result.sequences]
        return codes or [""]


class EvalRunner:
    """Run evaluation conditions by sampling on Tinker and scoring on Modal."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        if not self.cfg.modal.endpoint:
            raise ValueError("modal.endpoint must be configured before running eval")

        self.harness = Blender3DHarness(
            modal_endpoint=self.cfg.modal.endpoint,
            auth_token=self.cfg.modal.auth_token,
            reward_cfg=self.cfg.reward,
        )
        self.wb = WandbLogger(
            self.cfg.logging,
            run_name="eval",
            tags=["eval"],
            extra_config=self.cfg.eval.model_dump(),
        )
        self._service_client: Any | None = None

    def _ensure_service_client(self) -> Any:
        if self._service_client is None:
            self._service_client = build_tinker_service_client()
        return self._service_client

    def _iter_condition_configs(self) -> list[tuple[str, EvalConditionConfig]]:
        conditions_cfg: EvalConditionsConfig = self.cfg.eval.conditions
        return [
            ("baseline", conditions_cfg.baseline),
            ("candidate", conditions_cfg.candidate),
            ("reference", conditions_cfg.reference),
        ]

    def build_default_conditions(self) -> dict[str, TinkerSamplingGenerator]:
        enabled = [(name, cfg) for name, cfg in self._iter_condition_configs() if cfg.enabled]
        if not enabled:
            raise ValueError("No eval conditions are enabled under eval.conditions")

        service_client = self._ensure_service_client()
        generators: dict[str, TinkerSamplingGenerator] = {}
        for name, cond_cfg in enabled:
            generator = TinkerSamplingGenerator(name, cond_cfg, self.cfg, service_client)
            generators[name] = generator
            log.info(
                "Eval condition %s -> %s (samples=%d temp=%.3f)",
                name,
                generator.model_ref,
                cond_cfg.num_samples,
                generator.temperature,
            )
        return generators

    async def evaluate_condition(
        self,
        condition: str,
        test_set: str,
        test_dataset: Blender3DDataset,
        generator: TinkerSamplingGenerator,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        total = len(test_dataset)
        if self.cfg.eval.max_cases_per_test_set is not None:
            total = min(total, self.cfg.eval.max_cases_per_test_set)

        batch_size = max(1, self.cfg.eval.batch_size)

        for start in range(0, total, batch_size):
            items = [test_dataset[idx] for idx in range(start, min(start + batch_size, total))]
            prompts = [test_dataset.format_prompt(item) for item in items]
            code_groups = await asyncio.gather(*(generator.generate(prompt) for prompt in prompts))

            exec_items: list[dict[str, Any]] = []
            sample_meta: list[tuple[int, int, str]] = []
            for item_idx, (item, codes) in enumerate(zip(items, code_groups, strict=False)):
                for sample_idx, code in enumerate(codes):
                    exec_items.append(
                        {
                            "object_id": item["object_id"],
                            "code": code,
                            "text_description": item["text"],
                            "seed": self.cfg.seed,
                        }
                    )
                    sample_meta.append((item_idx, sample_idx, code))

            exec_results = await self.harness.execute_batch(exec_items)
            grouped: dict[int, list[tuple[int, str, dict[str, Any]]]] = {}
            for (item_idx, sample_idx, code), exec_result in zip(sample_meta, exec_results, strict=False):
                grouped.setdefault(item_idx, []).append((sample_idx, code, exec_result))

            for item_idx, item in enumerate(items):
                candidates = grouped.get(item_idx, [])
                if not candidates:
                    candidates = [(0, "", {"success": False, "metrics": {}, "reward": 0.0})]

                best_sample_idx, best_code, best_exec = max(
                    candidates,
                    key=lambda candidate: _selection_key(candidate[2]),
                )

                metrics = best_exec.get("metrics") or {}
                results.append(
                    EvalResult(
                        object_id=item["object_id"],
                        caption=item["text"],
                        condition=condition,
                        test_set=test_set,
                        model_ref=generator.model_ref,
                        success=bool(best_exec.get("success", False)),
                        has_geometry=_has_geometry(best_exec),
                        metrics=metrics,
                        reward=float(best_exec.get("reward", 0.0)),
                        code=best_code,
                        error=str(best_exec.get("error", "")),
                        selected_sample_index=best_sample_idx,
                        num_samples=len(candidates),
                    )
                )

        return results

    def aggregate(
        self,
        results: list[EvalResult],
        condition: str,
        test_set: str,
        model_ref: str,
    ) -> AggregatedMetrics:
        n = len(results)
        if n == 0:
            return AggregatedMetrics(
                condition=condition,
                test_set=test_set,
                model_ref=model_ref,
                n_total=0,
                execution_rate=0.0,
                geometry_rate=0.0,
                mean_f_score_005=0.0,
                mean_chamfer=0.0,
                mean_hausdorff_90=0.0,
                mean_normal_consistency=0.0,
                mean_reward=0.0,
            )

        execution_rate = sum(1 for result in results if result.success) / n
        geometry_rate = sum(1 for result in results if result.has_geometry) / n

        f_scores = [result.metrics.get("f_score_005", 0.0) for result in results if result.success]
        chamfers = [result.metrics.get("chamfer", 1.0) for result in results if result.success]
        hausdorffs = [result.metrics.get("hausdorff_90", 1.0) for result in results if result.success]
        normals = [result.metrics.get("normal_consistency", 0.0) for result in results if result.success]
        rewards = [result.reward for result in results]

        bootstrap_samples = self.cfg.eval.bootstrap_samples
        ci_f = bootstrap_ci(f_scores, bootstrap_samples) if f_scores else (0.0, 0.0)
        ci_c = bootstrap_ci(chamfers, bootstrap_samples) if chamfers else (0.0, 0.0)

        return AggregatedMetrics(
            condition=condition,
            test_set=test_set,
            model_ref=model_ref,
            n_total=n,
            execution_rate=round(execution_rate, 4),
            geometry_rate=round(geometry_rate, 4),
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
        conditions: dict[str, TinkerSamplingGenerator] | None = None,
        test_sets: dict[str, str] | None = None,
    ) -> tuple[list[AggregatedMetrics], list[EvalResult]]:
        if conditions is None:
            conditions = self.build_default_conditions()

        if test_sets is None:
            test_sets = {}
            if _path_exists(self.cfg.eval.id_path):
                test_sets["id"] = self.cfg.eval.id_path
            if self.cfg.eval.ood_path and _path_exists(self.cfg.eval.ood_path):
                test_sets["ood"] = self.cfg.eval.ood_path

            if not test_sets:
                raise FileNotFoundError(
                    "No evaluation dataset found. Configure eval.id_path and, optionally, eval.ood_path."
                )

        all_agg: list[AggregatedMetrics] = []
        all_results: list[EvalResult] = []

        for test_set_name, test_set_path in test_sets.items():
            dataset = Blender3DDataset(
                test_set_path,
                storage_cfg=self.cfg.storage,
                system_prompt=self.cfg.dataset.system_prompt,
            )
            for condition_name, generator in conditions.items():
                log.info("Evaluating %s on %s via %s", condition_name, test_set_name, generator.model_ref)
                results = await self.evaluate_condition(condition_name, test_set_name, dataset, generator)
                agg = self.aggregate(results, condition_name, test_set_name, generator.model_ref)
                all_agg.append(agg)
                all_results.extend(results)

                prefix = f"eval/{condition_name}/{test_set_name}"
                self.wb.log(
                    {
                        f"{prefix}/execution_rate": agg.execution_rate,
                        f"{prefix}/geometry_rate": agg.geometry_rate,
                        f"{prefix}/f_score_005": agg.mean_f_score_005,
                        f"{prefix}/chamfer": agg.mean_chamfer,
                        f"{prefix}/hausdorff_90": agg.mean_hausdorff_90,
                        f"{prefix}/normal_consistency": agg.mean_normal_consistency,
                        f"{prefix}/reward": agg.mean_reward,
                    }
                )

                log.info(
                    "%s/%s: exec=%.2f geom=%.2f f005=%.3f cd=%.5f reward=%.3f",
                    condition_name,
                    test_set_name,
                    agg.execution_rate,
                    agg.geometry_rate,
                    agg.mean_f_score_005,
                    agg.mean_chamfer,
                    agg.mean_reward,
                )

        self.wb.finish()
        return all_agg, all_results

    def save_results(self, aggregated: list[AggregatedMetrics], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in aggregated], handle, indent=2)
        log.info("Saved %d aggregated results to %s", len(aggregated), output_path)

    def save_detailed_results(self, results: list[EvalResult], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result)) + "\n")
        log.info("Saved %d detailed results to %s", len(results), output_path)


def run_eval(cfg: ProjectConfig | None = None) -> None:
    runner = EvalRunner(cfg)
    aggregated, detailed = asyncio.run(runner.run_full_evaluation())
    output_dir = Path(runner.cfg.output_dir)
    runner.save_results(aggregated, output_dir / "eval_results.json")
    if runner.cfg.eval.save_details:
        runner.save_detailed_results(detailed, output_dir / "eval_details.jsonl")


def main(argv: Sequence[str] | None = None) -> None:
    cfg = load_config_from_cli(
        description="Run evaluation across the configured test sets and model conditions.",
        argv=argv,
    )
    run_eval(cfg)


if __name__ == "__main__":
    main()
