"""
Modal function: compute geometric similarity metrics between two meshes.
"""

from __future__ import annotations

import io
from typing import Any

import modal

from modal_infra.images import metrics_image

app = modal.App("llm3d-metrics-worker")


def _normalize_mesh(mesh):
    """Center mesh at origin, scale so max extent = 1.0."""
    import numpy as np

    center = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2
    mesh.vertices -= center
    extent = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max()
    if extent > 1e-8:
        mesh.vertices /= extent
    return mesh


def _load_mesh(data: bytes):
    """Load mesh from OBJ bytes, normalize."""
    import trimesh

    mesh = trimesh.load(io.BytesIO(data), file_type="obj", force="mesh", process=True)
    if hasattr(mesh, "geometry"):
        parts = [g for g in mesh.geometry.values() if hasattr(g, "vertices")]
        if parts:
            mesh = trimesh.util.concatenate(parts)
        else:
            return None
    if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
        return None
    return _normalize_mesh(mesh)


@app.function(image=metrics_image, cpu=2, memory=2048, timeout=60)
def compute_metrics(gen_mesh_bytes: bytes, gt_mesh_bytes: bytes,
                    num_points: int = 10_000) -> dict[str, Any]:
    """Compute geometric similarity metrics.

    Both meshes are normalized to [-1,1]^3. Returns chamfer, f_score_{001,005},
    hausdorff_90, normal_consistency.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    gen = _load_mesh(gen_mesh_bytes)
    gt = _load_mesh(gt_mesh_bytes)

    if gen is None or gt is None:
        return {
            "chamfer": float("inf"),
            "f_score_001": 0.0,
            "f_score_005": 0.0,
            "hausdorff_90": float("inf"),
            "normal_consistency": 0.0,
            "error": "Could not load one or both meshes",
        }

    import trimesh
    gen_pts, gen_fi = trimesh.sample.sample_surface(gen, num_points)
    gt_pts, gt_fi = trimesh.sample.sample_surface(gt, num_points)
    gen_normals = gen.face_normals[gen_fi]
    gt_normals = gt.face_normals[gt_fi]

    gen_pts = np.asarray(gen_pts)
    gt_pts = np.asarray(gt_pts)

    tree_gen = cKDTree(gen_pts)
    tree_gt = cKDTree(gt_pts)

    d_gen_to_gt, idx_g2gt = tree_gt.query(gen_pts)
    d_gt_to_gen, idx_gt2g = tree_gen.query(gt_pts)

    chamfer = float(np.mean(d_gen_to_gt ** 2) + np.mean(d_gt_to_gen ** 2))

    def f_score(threshold):
        prec = float(np.mean(d_gen_to_gt < threshold))
        rec = float(np.mean(d_gt_to_gen < threshold))
        if prec + rec > 0:
            return 2 * prec * rec / (prec + rec)
        return 0.0

    hausdorff_90 = float(max(
        np.percentile(d_gen_to_gt, 90),
        np.percentile(d_gt_to_gen, 90),
    ))

    dots_fwd = np.abs(np.sum(gen_normals * gt_normals[idx_g2gt], axis=1))
    dots_bwd = np.abs(np.sum(gt_normals * gen_normals[idx_gt2g], axis=1))
    nc = float(0.5 * (np.mean(dots_fwd) + np.mean(dots_bwd)))

    return {
        "chamfer": chamfer,
        "f_score_001": f_score(0.01),
        "f_score_005": f_score(0.05),
        "hausdorff_90": hausdorff_90,
        "normal_consistency": nc,
        "error": "",
    }
