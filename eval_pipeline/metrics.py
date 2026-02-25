"""
Geometric metrics for comparing generated meshes against ground truth.

All metrics are designed to work well as RLVR (Reinforcement Learning with
Verifiable Rewards) reward signals. See the package-level docstring in
eval_pipeline/__init__.py for the full rationale behind each metric choice.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from eval_pipeline.config import Config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: COMPUTE GEOMETRIC METRICS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_sample_mesh(obj_path: str, num_points: int) -> Optional[Dict]:
    """
    Load an .obj mesh using trimesh and sample points from its surface.

    Returns a dict with:
      - 'points': (N, 3) array of surface points
      - 'normals': (N, 3) array of surface normals at those points
      - 'mesh': the trimesh object (for visualization/debugging)

    Returns None if the mesh can't be loaded or is degenerate.
    """
    import trimesh

    try:
        mesh = trimesh.load(obj_path, force='mesh', process=True)
    except Exception as e:
        log.warning(f"  Failed to load mesh {obj_path}: {e}")
        return None

    # Handle Scene objects (multiple meshes in one file)
    if isinstance(mesh, trimesh.Scene):
        # Concatenate all geometries
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            log.warning(f"  No valid geometry in {obj_path}")
            return None
        mesh = trimesh.util.concatenate(meshes)

    if not isinstance(mesh, trimesh.Trimesh):
        log.warning(f"  Loaded object is not a Trimesh: {type(mesh)}")
        return None

    if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
        log.warning(f"  Degenerate mesh in {obj_path} (verts={mesh.vertices.shape[0]})")
        return None

    # ── Normalize the mesh to unit bounding box ──────────────────────────
    # Even though we ask the LLM to normalize, we do it again here to be safe.
    # This ensures fair comparison regardless of the LLM's normalization quality.
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    extent = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    if extent > 1e-8:
        mesh.vertices /= extent

    # ── Sample points uniformly from the surface ─────────────────────────
    # trimesh.sample.sample_surface returns points AND face indices
    try:
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        normals = mesh.face_normals[face_indices]
    except Exception as e:
        log.warning(f"  Failed to sample from mesh: {e}")
        return None

    return {
        "points": np.array(points),
        "normals": np.array(normals),
        "mesh": mesh,
    }


def compute_chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    """
    Compute the Chamfer Distance between two point clouds.

    CD = (1/|A|) Σ_a min_b ||a - b||² + (1/|B|) Σ_b min_a ||b - a||²

    This is the L2 Chamfer Distance (using squared distances), which is
    standard in the 3D generation literature (e.g., AtlasNet, OccNet).

    Lower is better. 0 = perfect match.
    """
    # Build KD-trees for fast nearest-neighbor lookup
    # KD-trees give O(N log N) instead of O(N²) brute force
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    # A → B: for each point in A, find nearest in B
    dist_a_to_b, _ = tree_b.query(points_a)
    # B → A: for each point in B, find nearest in A
    dist_b_to_a, _ = tree_a.query(points_b)

    # Squared L2 distances, averaged
    cd = np.mean(dist_a_to_b ** 2) + np.mean(dist_b_to_a ** 2)
    return float(cd)


def compute_f_score(
    points_a: np.ndarray,
    points_b: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    """
    Compute the F-Score at a given distance threshold.

    F-Score is the harmonic mean of precision and recall:
    - Precision: fraction of predicted points within τ of any GT point
    - Recall: fraction of GT points within τ of any predicted point

    This is arguably the best single metric for 3D evaluation because:
    1. It's bounded [0, 1] — directly interpretable as "% correct"
    2. It captures BOTH false positives (extra geometry) via precision
       and false negatives (missing geometry) via recall
    3. Different thresholds τ capture different levels of detail

    Returns:
        (f_score, precision, recall)
    """
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    # Precision: how many predicted (B) points are close to GT (A)?
    dist_b_to_a, _ = tree_a.query(points_b)
    precision = np.mean(dist_b_to_a < threshold)

    # Recall: how many GT (A) points are close to predicted (B)?
    dist_a_to_b, _ = tree_b.query(points_a)
    recall = np.mean(dist_a_to_b < threshold)

    # F-Score (harmonic mean, avoids division by zero)
    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0.0

    return float(f_score), float(precision), float(recall)


def compute_hausdorff_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    percentile: float = 90.0,
) -> float:
    """
    Compute the (robust) Hausdorff Distance between two point clouds.

    Standard Hausdorff is the maximum nearest-neighbor distance, but this
    is extremely sensitive to a single outlier vertex. We use the 90th
    percentile instead, which is standard practice and much more stable.

    HD_90 = max(P90(min_b ||a-b||), P90(min_a ||b-a||))

    Higher = worse. Captures the worst-case error in the reconstruction.
    Useful as a penalty term in RLVR reward to prevent rogue geometry.
    """
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)

    hd = max(
        np.percentile(dist_a_to_b, percentile),
        np.percentile(dist_b_to_a, percentile),
    )
    return float(hd)


def compute_normal_consistency(
    points_a: np.ndarray,
    normals_a: np.ndarray,
    points_b: np.ndarray,
    normals_b: np.ndarray,
) -> float:
    """
    Compute Normal Consistency between two meshes.

    For each point in A, find the nearest point in B, then compute the
    absolute dot product of their normals. Average over all points in
    both directions.

    NC = 0.5 * (mean |n_a · n_b_nearest| + mean |n_b · n_a_nearest|)

    Range: [0, 1]. Higher is better.
    - 1.0 = all normals perfectly aligned
    - 0.0 = all normals perpendicular (very unlikely in practice)

    Why absolute value? Because normal direction can be flipped (inward vs
    outward facing) and both are valid representations. We care about the
    surface orientation, not the convention.

    This metric catches a common failure mode: a mesh that has the right
    shape but wrong normals (e.g., inverted faces), which would look
    broken when rendered in a game engine.
    """
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    # A → B direction
    _, idx_a_to_b = tree_b.query(points_a)
    dots_a = np.abs(np.sum(normals_a * normals_b[idx_a_to_b], axis=1))
    nc_a = np.mean(dots_a)

    # B → A direction
    _, idx_b_to_a = tree_a.query(points_b)
    dots_b = np.abs(np.sum(normals_b * normals_a[idx_b_to_a], axis=1))
    nc_b = np.mean(dots_b)

    return float(0.5 * (nc_a + nc_b))


def compute_all_metrics(
    gt_path: str,
    gen_path: str,
    config: Config,
) -> Optional[Dict[str, float]]:
    """
    Compute all geometric metrics between ground-truth and generated meshes.

    Returns a dict of metric_name -> value, or None if meshes can't be loaded.
    """
    gt_data = load_and_sample_mesh(gt_path, config.num_sample_points)
    gen_data = load_and_sample_mesh(gen_path, config.num_sample_points)

    if gt_data is None or gen_data is None:
        return None

    gt_pts = gt_data["points"]
    gen_pts = gen_data["points"]
    gt_norms = gt_data["normals"]
    gen_norms = gen_data["normals"]

    metrics = {}

    # ── Chamfer Distance ─────────────────────────────────────────────────
    metrics["chamfer_distance"] = compute_chamfer_distance(gt_pts, gen_pts)

    # ── F-Score at multiple thresholds ───────────────────────────────────
    for tau in config.f_score_thresholds:
        f, p, r = compute_f_score(gt_pts, gen_pts, threshold=tau)
        metrics[f"f_score@{tau}"] = f
        metrics[f"precision@{tau}"] = p
        metrics[f"recall@{tau}"] = r

    # ── Hausdorff Distance (90th percentile) ─────────────────────────────
    metrics["hausdorff_90"] = compute_hausdorff_distance(gt_pts, gen_pts)

    # ── Normal Consistency ───────────────────────────────────────────────
    metrics["normal_consistency"] = compute_normal_consistency(
        gt_pts, gt_norms, gen_pts, gen_norms
    )

    return metrics


def compute_rlvr_reward(metrics: Dict[str, float], code_executed: bool) -> float:
    """
    Compute a composite RLVR reward from the individual metrics.

    This is an example reward function you'd use during RL training.
    The design follows several principles:

    1. GATING: If code didn't execute, reward = 0. No partial credit for
       broken code — this gives a clear learning signal to produce valid code.

    2. WEIGHTED COMBINATION: Different metrics capture different qualities.
       F-Score dominates because it's the most informative single metric.

    3. BOUNDED [0, 1]: Important for stable RL training — unbounded rewards
       cause gradient explosion.

    You'd tune these weights based on what matters for your application.
    """
    if not code_executed or metrics is None:
        return 0.0

    # F-Score at medium threshold is the primary signal (0-1)
    f_score_weight = 0.50
    f_score_val = metrics.get("f_score@0.02", 0.0)

    # Normal consistency (0-1)
    nc_weight = 0.15
    nc_val = metrics.get("normal_consistency", 0.0)

    # Chamfer distance penalty (invert and clip to 0-1 range)
    # CD for normalized meshes typically ranges from 0 to ~0.1
    cd_weight = 0.20
    cd_val = max(0, 1.0 - metrics.get("chamfer_distance", 1.0) * 20)

    # Hausdorff penalty (catches worst-case errors)
    hd_weight = 0.15
    hd_val = max(0, 1.0 - metrics.get("hausdorff_90", 1.0) * 5)

    reward = (
        f_score_weight * f_score_val
        + nc_weight * nc_val
        + cd_weight * cd_val
        + hd_weight * hd_val
    )

    return float(np.clip(reward, 0, 1))
