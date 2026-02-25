"""
Main evaluation pipeline orchestrator.

Parallelization strategy (two-phase per object):
  Phase 1 -- LLM queries:  all models queried concurrently via ThreadPool
             (I/O-bound HTTP calls, no contention).
  Phase 2 -- Blender exec: code execution + view rendering + metrics run
             concurrently with a limited worker pool (CPU/RAM-bound).
"""

import os
import sys
import shutil
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval_pipeline.config import Config
from eval_pipeline.dataset import download_objects
from eval_pipeline.rendering import render_views, render_mesh_views
from eval_pipeline.llm import query_llm
from eval_pipeline.execution import execute_generated_code
from eval_pipeline.metrics import compute_all_metrics, compute_rlvr_reward
from eval_pipeline.results import ObjectResult, print_results_table, save_results_json

log = logging.getLogger(__name__)


def _process_single_model(
    model: str,
    code: str,
    response_time: float,
    uid: str,
    obj_dir: str,
    image_paths: List[str],
    gt_view_paths: List[str],
    gt_path: str,
    config: Config,
) -> ObjectResult:
    """
    Run the Blender execution → view rendering → metric computation pipeline
    for a single model's generated code. Designed to be submitted to a
    ThreadPoolExecutor so multiple models can execute in parallel.
    """
    model_dir = os.path.join(obj_dir, model.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    input_views_dir = os.path.join(model_dir, "input_views")
    os.makedirs(input_views_dir, exist_ok=True)
    for img_path in image_paths:
        shutil.copy2(img_path, input_views_dir)

    gt_copies_dir = os.path.join(model_dir, "gt_views")
    os.makedirs(gt_copies_dir, exist_ok=True)
    for gv_path in gt_view_paths:
        shutil.copy2(gv_path, gt_copies_dir)

    gt_mesh_copy = os.path.join(model_dir, "ground_truth.obj")
    if os.path.exists(gt_path):
        shutil.copy2(gt_path, gt_mesh_copy)

    with open(os.path.join(model_dir, "generated_code.py"), "w") as f:
        f.write(code)

    # ── Execute the generated code ────────────────────────────────────
    log.info(f"  [{model}] Executing generated code...")
    success, gen_mesh_path, exec_log = execute_generated_code(
        code, config, model_dir
    )

    with open(os.path.join(model_dir, "execution_log.txt"), "w") as f:
        f.write(exec_log)

    if not success:
        error_summary = "Code execution failed"
        lines = exec_log.split('\n')
        for i, line in enumerate(lines):
            if 'Error:' in line or 'Traceback' in line:
                error_context = '\n'.join(lines[i:i+5])
                error_summary = error_context[:300]
                break

        log.warning(f"  [{model}] ✗ Execution failed: {error_summary[:100]}")
        return ObjectResult(
            uid=uid, model=model, code_executed=False,
            response_time=response_time, metrics=None,
            rlvr_reward=0.0, error="Code execution failed",
            generated_code=code, blender_error=error_summary,
            gt_mesh_path=gt_mesh_copy if os.path.exists(gt_mesh_copy) else None,
        )

    log.info(f"  [{model}] ✓ Code executed. Rendering + computing metrics...")

    # ── Render views of the generated mesh ────────────────────────────
    gen_views_dir = os.path.join(model_dir, "generated_views")
    gen_view_paths = render_mesh_views(
        gen_mesh_path, gen_views_dir, config, prefix="gen_view"
    )
    log.info(f"  [{model}] Rendered {len(gen_view_paths)} generated views.")

    # ── Compute metrics ───────────────────────────────────────────────
    metrics = compute_all_metrics(gt_path, gen_mesh_path, config)

    if metrics is None:
        log.warning(f"  [{model}] ✗ Metric computation failed")
        return ObjectResult(
            uid=uid, model=model, code_executed=True,
            response_time=response_time, metrics=None,
            rlvr_reward=0.1,
            error="Metric computation failed",
            generated_code=code,
            generated_mesh_path=gen_mesh_path,
            gt_mesh_path=gt_mesh_copy if os.path.exists(gt_mesh_copy) else None,
        )

    reward = compute_rlvr_reward(metrics, True)

    log.info(
        f"  [{model}] 📊 CD={metrics['chamfer_distance']:.5f}  "
        f"F@.02={metrics['f_score@0.02']:.3f}  "
        f"HD90={metrics['hausdorff_90']:.4f}  "
        f"NC={metrics['normal_consistency']:.3f}  "
        f"Reward={reward:.3f}"
    )

    return ObjectResult(
        uid=uid, model=model, code_executed=True,
        response_time=response_time, metrics=metrics,
        rlvr_reward=reward,
        generated_code=code,
        generated_mesh_path=gen_mesh_path,
        gt_mesh_path=gt_mesh_copy if os.path.exists(gt_mesh_copy) else None,
    )


def run_pipeline(config: Config):
    """
    Main evaluation pipeline.

    Parallelization strategy (two-phase per object):
      Phase 1 – LLM queries:  all models queried concurrently via ThreadPool
               (I/O-bound HTTP calls, no contention).
      Phase 2 – Blender exec: code execution + view rendering + metrics run
               concurrently with a limited worker pool (CPU/RAM-bound).
    """

    llm_workers = min(config.max_llm_workers, len(config.models))
    blender_workers = min(config.max_blender_workers, len(config.models))

    log.info("=" * 60)
    log.info("  IMAGE → 3D CODE GENERATION: LLM EVALUATION")
    log.info("=" * 60)
    log.info(f"  Category:   {config.category}")
    log.info(f"  Objects:    {config.max_objects}")
    log.info(f"  Views:      {config.num_views}")
    log.info(f"  Models:     {', '.join(config.models)}")
    log.info(f"  Output:     {config.output_dir}")
    log.info(f"  Parallel:   {llm_workers} LLM workers, "
             f"{blender_workers} Blender workers")
    log.info("=" * 60)

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Step 1: Download objects ─────────────────────────────────────────
    log.info("\n📦 STEP 1: Downloading objects from Objaverse...")
    objects = download_objects(config)

    if not objects:
        log.error("No objects downloaded. Exiting.")
        sys.exit(1)

    all_results: List[ObjectResult] = []

    # ── Step 2-5: Process each object ────────────────────────────────────
    for obj_idx, (uid, glb_path) in enumerate(objects.items()):
        log.info(f"\n{'─'*60}")
        log.info(f"🧊 OBJECT {obj_idx+1}/{len(objects)}: {uid[:16]}...")
        log.info(f"   File: {glb_path}")

        # ── Step 2: Render input views (sequential, once per object) ──
        obj_dir = os.path.join(config.output_dir, f"obj_{uid[:12]}")
        render_dir = os.path.join(obj_dir, "renders")
        image_paths, gt_path = render_views(glb_path, render_dir, config)

        if not image_paths or gt_path is None:
            log.warning(f"  Skipping object {uid[:12]} — rendering failed.")
            for model in config.models:
                all_results.append(ObjectResult(
                    uid=uid, model=model, code_executed=False,
                    response_time=0, metrics=None, rlvr_reward=0.0,
                    error="Rendering failed",
                    generated_code=None,
                ))
            continue

        gt_views_dir = os.path.join(obj_dir, "gt_views")
        gt_view_paths = render_mesh_views(
            gt_path, gt_views_dir, config, prefix="gt_view"
        )

        # ──────────────────────────────────────────────────────────────
        # PHASE 1: Query all LLMs in parallel (I/O-bound)
        # ──────────────────────────────────────────────────────────────
        log.info(f"\n  ⚡ Phase 1: Querying {len(config.models)} models in parallel "
                 f"({llm_workers} workers)...")
        llm_results: Dict[str, Tuple[Optional[str], float]] = {}

        with ThreadPoolExecutor(max_workers=llm_workers) as executor:
            future_to_model = {
                executor.submit(query_llm, model, image_paths, config): model
                for model in config.models
            }
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    _, code, response_time = future.result()
                    llm_results[model] = (code, response_time)
                    status = f"{len(code)} chars" if code else "no code"
                    log.info(f"  ✓ {model} responded ({status}, {response_time:.1f}s)")
                except Exception as exc:
                    log.warning(f"  ✗ {model} raised exception: {exc}")
                    llm_results[model] = (None, 0.0)

        # ──────────────────────────────────────────────────────────────
        # PHASE 2: Execute code + render + metrics in parallel
        #          (CPU-bound, limited workers to avoid OOM)
        # ──────────────────────────────────────────────────────────────
        models_with_code = {
            m: (code, rt) for m, (code, rt) in llm_results.items() if code is not None
        }
        models_without_code = {
            m: rt for m, (code, rt) in llm_results.items() if code is None
        }

        for model, rt in models_without_code.items():
            all_results.append(ObjectResult(
                uid=uid, model=model, code_executed=False,
                response_time=rt, metrics=None,
                rlvr_reward=0.0, error="No code returned",
                generated_code=None,
            ))

        if models_with_code:
            log.info(f"\n  ⚡ Phase 2: Executing {len(models_with_code)} models in parallel "
                     f"({blender_workers} Blender workers)...")

            with ThreadPoolExecutor(max_workers=blender_workers) as executor:
                future_to_model = {
                    executor.submit(
                        _process_single_model,
                        model, code, response_time,
                        uid, obj_dir, image_paths, gt_view_paths,
                        gt_path, config,
                    ): model
                    for model, (code, response_time) in models_with_code.items()
                }
                for future in as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as exc:
                        log.error(f"  ✗ {model} worker crashed: {exc}")
                        all_results.append(ObjectResult(
                            uid=uid, model=model, code_executed=False,
                            response_time=models_with_code[model][1],
                            metrics=None, rlvr_reward=0.0,
                            error=f"Worker exception: {exc}",
                            generated_code=models_with_code[model][0],
                        ))

    # ── Step 6: Report results ───────────────────────────────────────────
    print_results_table(all_results, config)
    save_results_json(all_results, config)

    log.info("✅ Evaluation complete!")
