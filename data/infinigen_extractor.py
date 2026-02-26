"""
Extract parts from Infinigen Indoor procedural objects (Section 3.1 of spec).

Each Infinigen object comes pre-decomposed into semantic parts. This module
exports each part as a separate mesh and records metadata.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ExtractedPart:
    object_id: str
    part_index: int
    label: str
    mesh_path: str
    bbox: list[float] = field(default_factory=list)
    transform: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class ExtractedObject:
    object_id: str
    category: str
    parts: list[ExtractedPart] = field(default_factory=list)
    mesh_path: str = ""


def normalize_mesh_file(mesh_path: str, output_path: str) -> dict:
    """Normalize a mesh to [-1, 1]^3 and save. Returns metadata."""
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    if hasattr(mesh, "geometry"):
        parts = list(mesh.geometry.values())
        if parts:
            mesh = trimesh.util.concatenate(parts)

    center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
    mesh.vertices -= center
    extent = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    if extent > 1e-8:
        mesh.vertices /= extent

    mesh.export(output_path)

    return {
        "num_vertices": int(mesh.vertices.shape[0]),
        "num_faces": int(mesh.faces.shape[0]),
        "bbox_size": (mesh.vertices.max(0) - mesh.vertices.min(0)).tolist(),
    }


def extract_parts_from_blend(
    blend_path: str,
    output_dir: str,
    object_id: str,
    category: str,
) -> ExtractedObject | None:
    """Extract named parts from a .blend file.

    This runs as a Blender subprocess script. Returns an ExtractedObject
    with all parts exported as individual .obj files.
    """
    import subprocess
    import tempfile

    script = f"""\
import bpy, json, os, sys

output_dir = "{output_dir}"
os.makedirs(output_dir, exist_ok=True)

parts = []
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    if obj.data is None or len(obj.data.vertices) < 3:
        continue

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    part_path = os.path.join(output_dir, f"{{obj.name}}.obj")
    bpy.ops.wm.obj_export(
        filepath=part_path,
        export_selected_objects=True,
        export_uv=False,
        export_materials=False,
    )

    bb = [list(obj.bound_box[i]) for i in range(8)]
    parts.append({{
        "name": obj.name,
        "path": part_path,
        "location": list(obj.location),
        "rotation": list(obj.rotation_quaternion) if obj.rotation_mode == 'QUATERNION'
                    else list(obj.rotation_euler.to_quaternion()),
        "scale": list(obj.scale),
        "num_verts": len(obj.data.vertices),
    }})

with open(os.path.join(output_dir, "parts_meta.json"), "w") as f:
    json.dump(parts, f)
"""

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as sf:
        sf.write(script)
        script_path = sf.name

    try:
        proc = subprocess.run(
            ["blender", blend_path, "--background", "--python", script_path],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        log.warning(f"Failed to extract parts from {blend_path}: {e}")
        return None
    finally:
        os.unlink(script_path)

    meta_path = os.path.join(output_dir, "parts_meta.json")
    if not os.path.exists(meta_path):
        log.warning(f"No parts_meta.json produced for {blend_path}")
        return None

    with open(meta_path) as f:
        parts_meta = json.load(f)

    extracted = ExtractedObject(object_id=object_id, category=category)
    for i, pm in enumerate(parts_meta):
        extracted.parts.append(ExtractedPart(
            object_id=object_id,
            part_index=i,
            label=pm["name"],
            mesh_path=pm["path"],
            bbox=[],
            transform={
                "location": pm["location"],
                "rotation": pm["rotation"],
                "scale": pm["scale"],
            },
        ))

    return extracted


def extract_objects_from_directory(
    infinigen_output_dir: str,
    output_dir: str,
    categories: list[str] | None = None,
) -> list[ExtractedObject]:
    """Walk an Infinigen output directory and extract all objects."""
    results = []
    infinigen_path = Path(infinigen_output_dir)

    for cat_dir in sorted(infinigen_path.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        if categories and category not in categories:
            continue

        for obj_dir in sorted(cat_dir.iterdir()):
            if not obj_dir.is_dir():
                continue

            blend_files = list(obj_dir.glob("*.blend"))
            if not blend_files:
                continue

            object_id = f"obj_{category}_{obj_dir.name}"
            part_output = os.path.join(output_dir, object_id)

            extracted = extract_parts_from_blend(
                str(blend_files[0]), part_output, object_id, category,
            )
            if extracted:
                results.append(extracted)

    log.info(f"Extracted {len(results)} objects from {infinigen_output_dir}")
    return results
