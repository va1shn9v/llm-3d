"""
Modal function: compute geometric similarity metrics + CLIP text-3D alignment.
"""

from __future__ import annotations

import io
from typing import Any

import modal

app = modal.App("llm3d-metrics-worker")

metrics_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "trimesh>=4.0",
        "numpy>=1.24",
        "scipy>=1.11",
        "torch>=2.1",
        "transformers>=4.36",
        "pillow>=10.0",
        "charset-normalizer>=3.0",
    )
)


def _normalize_mesh(mesh):
    """Center mesh at origin, scale so max extent = 1.0."""
    import numpy as np

    center = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2
    mesh.vertices -= center
    extent = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max()
    if extent > 1e-8:
        mesh.vertices /= extent
    return mesh


def _guess_mesh_format(data: bytes, fallback: str = "obj") -> str:
    """Infer mesh format from bytes when the caller doesn't know it."""
    header = data[:32]
    if header.startswith(b"glTF"):
        return "glb"

    stripped = data.lstrip()
    if stripped.startswith(b"ply"):
        return "ply"
    if stripped.startswith(b"solid "):
        return "stl"
    if stripped.startswith((b"v ", b"o ", b"g ", b"mtllib", b"usemtl", b"#")):
        return "obj"
    if stripped.startswith((b"{", b"[")) and b'"asset"' in data[:512]:
        return "gltf"
    return fallback


def _load_mesh(data: bytes, mesh_format: str | None = None):
    """Load mesh from bytes, normalize, and tolerate multiple mesh formats."""
    import trimesh

    file_type = (mesh_format or _guess_mesh_format(data)).lower().lstrip(".")
    try:
        mesh = trimesh.load(io.BytesIO(data), file_type=file_type, force="mesh", process=True)
    except Exception:
        return None
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
def compute_metrics(
    gen_mesh_bytes: bytes,
    gt_mesh_bytes: bytes,
    num_points: int = 10_000,
    gen_mesh_format: str = "obj",
    gt_mesh_format: str | None = None,
) -> dict[str, Any]:
    """Compute geometric similarity metrics.

    Both meshes are normalized to [-1,1]^3. Returns chamfer, f_score_{001,005},
    hausdorff_90, normal_consistency.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    gen = _load_mesh(gen_mesh_bytes, gen_mesh_format)
    gt = _load_mesh(gt_mesh_bytes, gt_mesh_format)

    if gen is None or gt is None:
        return {
            "chamfer": float("inf"),
            "f_score_001": 0.0,
            "f_score_005": 0.0,
            "hausdorff_90": float("inf"),
            "normal_consistency": 0.0,
            "error": (
                f"Could not load one or both meshes "
                f"(gen_format={gen_mesh_format}, gt_format={gt_mesh_format or 'auto'})"
            ),
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


@app.function(image=metrics_image, cpu=2, memory=4096, timeout=120)
def compute_clip_score(
    rendered_image_paths: list[str],
    text: str,
    model_name: str = "openai/clip-vit-large-patch14",
) -> float:
    """Compute CLIP cosine similarity between rendered mesh views and text.

    Loads rendered images, encodes them with CLIP vision encoder, encodes the
    text with CLIP text encoder, and returns the average cosine similarity.

    Returns a score in [0, 1] (clamped).
    """
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if not rendered_image_paths or not text:
        return 0.0

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    images = []
    for path in rendered_image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception:
            continue

    if not images:
        return 0.0

    with torch.no_grad():
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarities = (image_embeds @ text_embeds.T).squeeze(-1)
        avg_sim = float(similarities.mean())

    return max(0.0, min(1.0, (avg_sim + 1) / 2))
