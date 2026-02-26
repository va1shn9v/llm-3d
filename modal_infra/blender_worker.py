"""
Modal function: execute Blender Python code in an isolated container.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from typing import Any

import modal

from modal_infra.images import blender_image

app = modal.App("llm3d-blender-worker")

_WRAPPER_TEMPLATE = """\
import sys, os
sys.path.insert(0, "/opt/bpy_lib")
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
os.environ["EXPORT_PATH"] = "{export_path}"

# ===== BEGIN USER CODE =====
{user_code}
# ===== END USER CODE =====

# Auto-export if user code didn't call export_scene()
import os as _os
if not _os.path.exists("{export_path}"):
    try:
        mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
        if mesh_objects:
            bpy.ops.object.select_all(action='DESELECT')
            for o in mesh_objects:
                o.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            bpy.ops.wm.obj_export(
                filepath="{export_path}",
                export_selected_objects=True,
                export_uv=False,
                export_materials=False,
            )
    except Exception:
        pass
"""


@app.function(image=blender_image, cpu=2, memory=4096, timeout=150)
def execute_blender_code(code: str, seed: int = 42) -> dict[str, Any]:
    """Execute Blender Python code and return the exported mesh.

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


def _get_mesh_stats(mesh_bytes: bytes) -> dict:
    """Quick vertex/face count from raw OBJ bytes."""
    import trimesh
    import io

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
