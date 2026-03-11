"""
Modal function: execute raw Blender Python code in an isolated container.

Runs raw `import bpy` scripts in an isolated container. Sets EXPORT_PATH env var
so scripts know where to write their output mesh. Auto-exports if the script
doesn't write to EXPORT_PATH itself.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Any

import modal

try:
    from modal_infra.images import blender_image
except ModuleNotFoundError:
    from images import blender_image

app = modal.App("llm3d-blender-worker")
volume = modal.Volume.from_name("llm3d-data", create_if_missing=True)

_WRAPPER_TEMPLATE = """\
import sys, os

os.environ["EXPORT_PATH"] = "{export_path}"

# ===== BEGIN USER CODE =====
{user_code}
# ===== END USER CODE =====

# Auto-export if user code didn't write to EXPORT_PATH
import bpy as _bpy
import os as _os
if not _os.path.exists("{export_path}"):
    try:
        mesh_objects = [o for o in _bpy.data.objects if o.type == 'MESH']
        if mesh_objects:
            _bpy.ops.object.select_all(action='DESELECT')
            for o in mesh_objects:
                o.select_set(True)
            _bpy.context.view_layer.objects.active = mesh_objects[0]
            _bpy.ops.wm.obj_export(
                filepath="{export_path}",
                export_selected_objects=True,
                export_uv=False,
                export_materials=False,
            )
    except Exception:
        pass
"""


@app.function(image=blender_image, cpu=2, memory=4096, timeout=150)  # keep in sync with ModalConfig
def execute_blender_code(code: str, seed: int = 42) -> dict[str, Any]:
    """Execute raw Blender Python code and return the exported mesh.

    Returns dict with keys: success, mesh_bytes, mesh_stats, error, elapsed.
    """
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory() as tmp:
        export_path = os.path.join(tmp, "output.obj")
        script_path = os.path.join(tmp, "script.py")

        wrapped = _WRAPPER_TEMPLATE.format(
            export_path=export_path.replace("\\", "\\\\"),
            user_code=code,
        )
        with open(script_path, "w") as f:
            f.write(wrapped)

        try:
            proc = subprocess.run(
                ["blender", "--background", "--python", script_path],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "mesh_bytes": None,
                "mesh_stats": None,
                "error": "Blender execution timed out (120s)",
                "elapsed": time.monotonic() - t0,
            }
        except Exception as e:
            return {
                "success": False,
                "mesh_bytes": None,
                "mesh_stats": None,
                "error": str(e),
                "elapsed": time.monotonic() - t0,
            }

        stderr_tail = (proc.stderr or "")[-500:]

        if not os.path.exists(export_path):
            return {
                "success": False,
                "mesh_bytes": None,
                "mesh_stats": None,
                "error": f"No output mesh produced. stderr: {stderr_tail}",
                "elapsed": time.monotonic() - t0,
            }

        with open(export_path, "rb") as f:
            mesh_bytes = f.read()

        stats = _get_mesh_stats(mesh_bytes)

        return {
            "success": True,
            "mesh_bytes": mesh_bytes,
            "mesh_stats": stats,
            "error": "",
            "elapsed": time.monotonic() - t0,
        }


@app.function(image=blender_image, volumes={"/data": volume}, timeout=30)
def store_mesh_artifact(uid: str, mesh_bytes: bytes, subdir: str = "synthetic") -> str:
    """Persist a validated mesh to the Modal Volume for downstream reuse."""
    path = f"/data/{subdir}/{uid}.obj"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(mesh_bytes)
    volume.commit()
    return path


_sync_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=1.5.0",
)


@app.function(image=_sync_image, volumes={"/data": volume}, timeout=600, cpu=2, memory=4096)
def sync_from_hf_bucket(
    hf_bucket_path: str,
    volume_subdir: str = "meshes",
) -> int:
    """Pull files from an HF Bucket into the Modal Volume.

    Args:
        hf_bucket_path: Full ``hf://buckets/...`` path to sync from.
        volume_subdir: Subdirectory under ``/data`` on the volume.

    Returns:
        Number of files synced.
    """
    import os
    import shutil

    from huggingface_hub import HfFileSystem

    hffs = HfFileSystem()
    dest_root = f"/data/{volume_subdir}"
    os.makedirs(dest_root, exist_ok=True)

    bucket_path = hf_bucket_path.replace("hf://", "")
    entries = hffs.ls(bucket_path, detail=True)
    count = 0
    for entry in entries:
        if entry["type"] == "file":
            name = entry["name"].split("/")[-1]
            local_dest = os.path.join(dest_root, name)
            with hffs.open(entry["name"], "rb") as src, open(local_dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1

    volume.commit()
    return count


def _get_mesh_stats(mesh_bytes: bytes) -> dict:
    """Quick vertex/face count from raw OBJ bytes."""
    import io

    import trimesh

    try:
        mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type="obj", force="mesh", process=True)
        if hasattr(mesh, "geometry"):
            meshes = list(mesh.geometry.values())
            total_v = sum(m.vertices.shape[0] for m in meshes)
            total_f = sum(m.faces.shape[0] for m in meshes)
            watertight = all(m.is_watertight for m in meshes)
        else:
            total_v = mesh.vertices.shape[0]
            total_f = mesh.faces.shape[0]
            watertight = mesh.is_watertight
        return {"vertices": int(total_v), "faces": int(total_f), "is_watertight": watertight}
    except Exception:
        return {"vertices": 0, "faces": 0, "is_watertight": False}
