"""
Fit bpy_lib code to extracted part meshes (Section 3.2 of spec).

Template matching approach: for each part mesh, try all bpy_lib part types,
optimize parameters to minimize Chamfer Distance, accept best fit below threshold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    part_type: str
    code: str
    chamfer_distance: float
    params: dict[str, Any]
    accepted: bool


def _chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """L2 Chamfer Distance between two point clouds."""
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a)
    d_ba, _ = tree_a.query(pts_b)
    return float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))


def _sample_mesh_points(mesh, n: int = 2048) -> np.ndarray:
    """Sample n points from a trimesh."""
    import trimesh
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts)


def _normalize_mesh(mesh):
    center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
    mesh.vertices -= center
    extent = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    if extent > 1e-8:
        mesh.vertices /= extent
    return mesh


# ---------------------------------------------------------------------------
# Per-type optimizers
# ---------------------------------------------------------------------------

def _fit_primitive(target_pts: np.ndarray, target_mesh) -> FitResult:
    """Try all primitive types, optimize scale/rotation."""
    import trimesh

    bbox = target_mesh.vertices.max(0) - target_mesh.vertices.min(0)
    best = FitResult("primitive", "", float("inf"), {}, False)

    for ptype in ["cube", "cylinder", "uv_sphere", "cone", "torus"]:
        if ptype == "cube":
            template = trimesh.creation.box(extents=bbox)
        elif ptype == "cylinder":
            r = (bbox[0] + bbox[1]) / 4
            h = bbox[2]
            template = trimesh.creation.cylinder(radius=max(r, 0.01), height=max(h, 0.01))
        elif ptype == "uv_sphere":
            r = bbox.max() / 2
            template = trimesh.creation.icosphere(radius=max(r, 0.01))
        elif ptype == "cone":
            r = (bbox[0] + bbox[1]) / 4
            h = bbox[2]
            template = trimesh.creation.cone(radius=max(r, 0.01), height=max(h, 0.01))
        elif ptype == "torus":
            R = max(bbox[:2].mean() / 2, 0.05)
            r = min(bbox[2] / 2, R * 0.8)
            try:
                template = trimesh.creation.torus(major_radius=R, minor_radius=max(r, 0.01))
            except Exception:
                continue
        else:
            continue

        _normalize_mesh(template)
        tpts = _sample_mesh_points(template)
        cd = _chamfer_distance(target_pts, tpts)

        if cd < best.chamfer_distance:
            scale = bbox.tolist()
            best = FitResult(
                part_type="primitive",
                code=(
                    f"create_primitive(\n"
                    f"    name='{{name}}',\n"
                    f"    primitive_type='{ptype}',\n"
                    f"    location={{location}},\n"
                    f"    scale={[round(s, 4) for s in scale]},\n"
                    f"    rotation={{rotation}},\n"
                    f")"
                ),
                chamfer_distance=cd,
                params={"primitive_type": ptype, "scale": scale},
                accepted=False,
            )

    return best


def _fit_translation(target_pts: np.ndarray, target_mesh) -> FitResult:
    """Approximate with a swept shape."""
    bbox = target_mesh.vertices.max(0) - target_mesh.vertices.min(0)

    z_slices = 5
    z_min, z_max = target_mesh.vertices[:, 2].min(), target_mesh.vertices[:, 2].max()
    z_step = (z_max - z_min) / z_slices if z_slices > 1 else 1.0

    trajectory = []
    for i in range(z_slices):
        z = z_min + z_step * (i + 0.5)
        trajectory.append([0.0, 0.0, round(float(z), 4)])

    r = float(min(bbox[0], bbox[1]) / 2)
    section = [[round(r, 4), 0.0]] * 8

    code = (
        f"create_translation(\n"
        f"    name='{{name}}',\n"
        f"    section_points={section},\n"
        f"    section_type='circle',\n"
        f"    trajectory_points={trajectory},\n"
        f"    trajectory_type='polyline',\n"
        f")"
    )

    return FitResult(
        part_type="translation",
        code=code,
        chamfer_distance=float("inf"),
        params={"trajectory_points": trajectory},
        accepted=False,
    )


def _fit_bridge_loop(target_pts: np.ndarray, target_mesh) -> FitResult:
    """Approximate with connected cross-sections."""
    z_min, z_max = target_mesh.vertices[:, 2].min(), target_mesh.vertices[:, 2].max()
    n_loops = min(4, max(2, int((z_max - z_min) / 0.2)))

    loops = []
    positions = []
    for i in range(n_loops):
        z = z_min + (z_max - z_min) * i / max(n_loops - 1, 1)
        mask = np.abs(target_mesh.vertices[:, 2] - z) < (z_max - z_min) / (2 * n_loops)
        if mask.sum() < 3:
            r = 0.1
        else:
            slice_pts = target_mesh.vertices[mask]
            r = float(np.sqrt(slice_pts[:, 0] ** 2 + slice_pts[:, 1] ** 2).mean())

        n_pts = 8
        shape = [
            [round(r * math.cos(2 * math.pi * j / n_pts), 4),
             round(r * math.sin(2 * math.pi * j / n_pts), 4),
             0.0]
            for j in range(n_pts)
        ]
        loops.append(shape)
        positions.append([0.0, 0.0, round(float(z), 4)])

    code = (
        f"create_bridge_loop(\n"
        f"    name='{{name}}',\n"
        f"    loop_shapes={loops},\n"
        f"    positions={positions},\n"
        f")"
    )

    return FitResult(
        part_type="bridge_loop",
        code=code,
        chamfer_distance=float("inf"),
        params={"n_loops": n_loops},
        accepted=False,
    )


# ---------------------------------------------------------------------------
# Main fitting API
# ---------------------------------------------------------------------------

_FITTERS = {
    "primitive": _fit_primitive,
    "translation": _fit_translation,
    "bridge_loop": _fit_bridge_loop,
}


def fit_part_to_code(
    part_mesh_path: str,
    cd_threshold: float = 5e-3,
    num_sample_points: int = 2048,
) -> FitResult:
    """Find best bpy_lib code for a given part mesh.

    Tries all part types, returns best fit. Marks as accepted if CD < threshold.
    """
    import trimesh

    mesh = trimesh.load(part_mesh_path, force="mesh", process=True)
    if hasattr(mesh, "geometry"):
        parts = list(mesh.geometry.values())
        if parts:
            mesh = trimesh.util.concatenate(parts)

    _normalize_mesh(mesh)
    target_pts = _sample_mesh_points(mesh, num_sample_points)

    best = FitResult("none", "", float("inf"), {}, False)

    for ptype, fitter in _FITTERS.items():
        try:
            result = fitter(target_pts, mesh)
            if result.chamfer_distance < best.chamfer_distance:
                best = result
        except Exception as e:
            log.debug(f"Fitter {ptype} failed: {e}")
            continue

    best.accepted = best.chamfer_distance < cd_threshold
    return best


def fit_all_parts(
    part_paths: list[str],
    cd_threshold: float = 5e-3,
) -> list[FitResult]:
    """Fit bpy_lib code to a list of part mesh files."""
    results = []
    for path in part_paths:
        try:
            r = fit_part_to_code(path, cd_threshold)
            results.append(r)
        except Exception as e:
            log.warning(f"Failed to fit {path}: {e}")
            results.append(FitResult("none", "", float("inf"), {}, False))
    return results
