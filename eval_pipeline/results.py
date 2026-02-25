"""
Result data structures and reporting (table + JSON output).
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

from eval_pipeline.config import Config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: RESULTS REPORTING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectResult:
    """Results for a single object evaluated by a single model."""
    uid: str
    model: str
    code_executed: bool
    response_time: float
    metrics: Optional[Dict[str, float]]
    rlvr_reward: float
    error: Optional[str] = None
    generated_code: Optional[str] = None
    blender_error: Optional[str] = None
    generated_mesh_path: Optional[str] = None
    gt_mesh_path: Optional[str] = None


def print_results_table(results: List[ObjectResult], config: Config):
    """Pretty-print a results summary table."""

    print("\n" + "═" * 90)
    print("  EVALUATION RESULTS")
    print("═" * 90)
    print(f"  Category: {config.category} | Objects: {config.max_objects} "
          f"| Views: {config.num_views}")
    print("─" * 90)

    # ── Per-model aggregation ────────────────────────────────────────────
    model_stats: Dict[str, List[ObjectResult]] = {}
    for r in results:
        model_stats.setdefault(r.model, []).append(r)

    # Header
    print(f"  {'Model':<35} {'Exec%':>6} {'CD↓':>8} {'F@.02↑':>8} "
          f"{'HD90↓':>8} {'NC↑':>6} {'Reward↑':>8} {'Time':>6}")
    print("─" * 90)

    for model, model_results in model_stats.items():
        n = len(model_results)
        exec_rate = sum(1 for r in model_results if r.code_executed) / n * 100

        # Average metrics over successfully executed results
        successful = [r for r in model_results if r.metrics is not None]
        if successful:
            avg = lambda key: np.mean([r.metrics[key] for r in successful])
            avg_cd = avg("chamfer_distance")
            avg_f = avg("f_score@0.02")
            avg_hd = avg("hausdorff_90")
            avg_nc = avg("normal_consistency")
        else:
            avg_cd = avg_f = avg_hd = avg_nc = float('nan')

        avg_reward = np.mean([r.rlvr_reward for r in model_results])
        avg_time = np.mean([r.response_time for r in model_results])

        # Truncate long model names
        short_name = model.split("/")[-1][:33]
        print(
            f"  {short_name:<35} {exec_rate:>5.0f}% {avg_cd:>8.5f} "
            f"{avg_f:>8.3f} {avg_hd:>8.4f} {avg_nc:>6.3f} "
            f"{avg_reward:>8.3f} {avg_time:>5.1f}s"
        )

    print("─" * 90)
    print("  CD = Chamfer Distance (lower is better)")
    print("  F@.02 = F-Score at τ=0.02 (higher is better)")
    print("  HD90 = 90th-percentile Hausdorff Distance (lower is better)")
    print("  NC = Normal Consistency (higher is better)")
    print("  Reward = Composite RLVR reward [0, 1] (higher is better)")
    print("═" * 90 + "\n")


def save_results_json(results: List[ObjectResult], config: Config):
    """Save detailed results to a JSON file for later analysis."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "results": [
            {
                "uid": r.uid,
                "model": r.model,
                "code_executed": r.code_executed,
                "response_time": r.response_time,
                "metrics": r.metrics,
                "rlvr_reward": r.rlvr_reward,
                "error": r.error,
                "generated_code": r.generated_code,
                "blender_error": r.blender_error,
                "generated_mesh_path": r.generated_mesh_path,
                "gt_mesh_path": r.gt_mesh_path,
            }
            for r in results
        ],
    }

    json_path = os.path.join(config.output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"Results saved to {json_path}")
