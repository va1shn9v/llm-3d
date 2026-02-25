"""
=================================================================================
 IMAGE → 3D CODE GENERATION: LLM EVALUATION PIPELINE
=================================================================================

 PURPOSE:
   This script evaluates how well different LLMs can generate Blender Python code
   that reconstructs a 3D object from input images. This is the core evaluation
   loop for an RLVR (Reinforcement Learning with Verifiable Rewards) environment
   where the "verifiable reward" comes from comparing the generated mesh against
   ground truth using geometric metrics.

 PIPELINE:
   1. Download 3D objects from Objaverse by category (e.g., "chair")
   2. Render multi-view input images of each object
   3. Send images to LLMs via OpenRouter, asking them to generate Blender Python
   4. Execute the generated code to produce a mesh
   5. Compare the generated mesh to ground truth using geometric metrics
   6. Produce a results report

 METRICS RATIONALE (why these specific metrics):
 ─────────────────────────────────────────────────
   We chose metrics that would work well as RLVR reward signals. A good reward
   signal for 3D generation needs to be: (a) differentiable in spirit (smooth,
   not just binary), (b) capture different failure modes, (c) computationally
   cheap enough to run thousands of times during RL training.

   1. CHAMFER DISTANCE (CD):
      The workhorse metric for 3D. Measures the average nearest-neighbor distance
      between two point clouds sampled from the meshes.
      CD = (1/|A|) Σ min_b ||a-b||² + (1/|B|) Σ min_a ||b-a||²
      ✓ Smooth and continuous — small improvements in mesh quality → small
        improvements in CD. This makes it ideal as an RL reward signal.
      ✓ Symmetric — penalizes both missing geometry and extra geometry.
      ✗ Can be fooled by "mean-seeking" behavior (a sphere gets decent CD
        against many shapes). That's why we also need F-Score.

   2. F-SCORE @ multiple thresholds (F@τ):
      Precision/recall for 3D. At threshold τ, a predicted point is "correct"
      if it's within τ distance of any ground-truth point (and vice versa).
      F@τ = 2 * (precision * recall) / (precision + recall)
      ✓ More interpretable than CD — "what % of the surface is within τ?"
      ✓ At tight thresholds (τ=0.01), captures fine detail fidelity.
        At loose thresholds (τ=0.05), captures overall shape correctness.
      ✓ Bounded [0,1] — directly usable as a reward signal without normalization.
      This is arguably the BEST single metric for RLVR reward design.

   3. HAUSDORFF DISTANCE (HD):
      The worst-case nearest-neighbor distance. Captures the single biggest
      geometric error between the two meshes.
      HD = max(max_a min_b ||a-b||, max_b min_a ||b-a||)
      ✓ Catches catastrophic failures that CD averages away (e.g., one rogue
        vertex flying off to infinity while the rest of the mesh is perfect).
      ✓ Useful as a "penalty term" in reward design — you want to discourage
        any part of the mesh being wildly wrong.
      ✗ Very sensitive to outliers — a single bad vertex dominates.
        We use the 90th-percentile variant to be more robust.

   4. NORMAL CONSISTENCY (NC):
      Measures alignment of surface normals between corresponding points.
      For each point on mesh A, find the nearest point on mesh B, and compute
      the dot product of their normals. Average over all points.
      ✓ Captures surface *orientation* quality, not just position.
        Two meshes can be positionally close but have inverted/wrong normals,
        which would look terrible when rendered.
      ✓ Important for downstream use — procedural assets need correct normals
        for lighting/shading in game engines.

   5. CODE EXECUTION SUCCESS (binary):
      Did the generated code even run without errors?
      ✓ The most basic "verifiable" signal — essential as a gating reward
        in RLVR (no point computing geometric metrics if code crashed).
      ✓ In early RL training, this will be the primary learning signal.

 USAGE:
   python eval_image_to_3d_code.py

   Or configure via environment variables / editing the CONFIG section below.

 DEPENDENCIES:
   pip install objaverse trimesh numpy scipy Pillow requests tqdm

   For rendering input views, one of:
     - bpy (pip install bpy)  — Blender as Python module
     - Blender CLI             — standalone blender binary

 AUTHOR: Your Name
 DATE:   2025-02
 LICENSE: MIT
=================================================================================
"""

from eval_pipeline.config import Config, parse_args, setup_logging
from eval_pipeline.pipeline import run_pipeline

__all__ = ["Config", "parse_args", "setup_logging", "run_pipeline"]
