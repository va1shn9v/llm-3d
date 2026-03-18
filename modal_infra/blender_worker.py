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
from pathlib import Path
from typing import Any

import modal

app = modal.App("llm3d-blender-worker")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dev_env() -> None:
    """Load repo-local dev.env for Modal deploy-time configuration."""
    env_path = _PROJECT_ROOT / "dev.env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


_load_dev_env()

_BV = os.environ.get("BLENDER_VERSION", "4.2.0")
blender_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget", "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3",
        "libxrender1", "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
        "libxext6", "libgomp1", "libxkbcommon0",
    )
    .run_commands(
        f"wget -q https://download.blender.org/release/Blender{_BV[:3]}/"
        f"blender-{_BV}-linux-x64.tar.xz -O /tmp/blender.tar.xz",
        "mkdir -p /opt/blender && tar xf /tmp/blender.tar.xz"
        " --strip-components=1 -C /opt/blender",
        "ln -s /opt/blender/blender /usr/local/bin/blender",
        "rm /tmp/blender.tar.xz",
    )
    .pip_install("trimesh>=4.0", "numpy>=1.24", "scipy>=1.11")
)
volume = modal.Volume.from_name(
    os.environ.get("LLM3D_MODAL__VOLUME_NAME", "llm3d-data"),
    create_if_missing=True,
)

_EXPORT_PATH_PLACEHOLDER = "___EXPORT_PATH___"
_USER_CODE_PLACEHOLDER = "___USER_CODE___"

_WRAPPER_TEMPLATE = """\
import sys, os

os.environ["EXPORT_PATH"] = "___EXPORT_PATH___"

# ===== BLENDER 4.x COMPATIBILITY SHIM =====
import bpy as _bpy_shim

# Patch out use_auto_smooth (removed in Blender 4.0+)
_OrigMeshType = type(_bpy_shim.data.meshes.new("__probe"))
_bpy_shim.data.meshes.remove(_bpy_shim.data.meshes["__probe"])

if not hasattr(_OrigMeshType, "use_auto_smooth"):
    _OrigMeshType.use_auto_smooth = property(lambda self: False, lambda self, v: None)
    _OrigMeshType.auto_smooth_angle = property(lambda self: 0.0, lambda self, v: None)

# Patch legacy OBJ export operator if missing
if not hasattr(_bpy_shim.ops.export_scene, "obj"):
    def _legacy_obj_export(**kw):
        filepath = kw.pop("filepath", kw.pop("path", ""))
        use_selection = kw.pop("use_selection", False)
        return _bpy_shim.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=use_selection,
        )
    _bpy_shim.ops.export_scene.obj = _legacy_obj_export

# Ensure mathutils is accessible as bpy.mathutils for broken scripts
import mathutils as _mathutils_mod
if not hasattr(_bpy_shim, "mathutils"):
    _bpy_shim.mathutils = _mathutils_mod

del _OrigMeshType, _bpy_shim, _mathutils_mod
# ===== END COMPATIBILITY SHIM =====

# ===== BEGIN USER CODE =====
___USER_CODE___
# ===== END USER CODE =====

# ===== AUTO-EXPORT FALLBACK =====
import bpy as _bpy
import os as _os

_export_path = "___EXPORT_PATH___"

def _apply_modifiers_for_export():
    for obj in list(_bpy.data.objects):
        if obj.type != 'MESH':
            continue
        try:
            _bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            for mod in list(obj.modifiers):
                try:
                    _bpy.ops.object.modifier_apply(modifier=mod.name)
                except Exception:
                    pass
            obj.select_set(False)
        except Exception:
            pass

def _robust_export(path):
    mesh_objects = [o for o in _bpy.data.objects if o.type == 'MESH']
    curve_objects = [o for o in _bpy.data.objects if o.type == 'CURVE']
    exportable = mesh_objects + curve_objects
    if not exportable:
        return False

    _bpy.ops.object.select_all(action='DESELECT')
    for o in exportable:
        o.select_set(True)
    _bpy.context.view_layer.objects.active = exportable[0]

    try:
        _bpy.ops.wm.obj_export(
            filepath=path,
            export_selected_objects=True,
            export_uv=False,
            export_normals=True,
            export_materials=False,
            apply_modifiers=True,
        )
        return True
    except Exception:
        return False

_needs_export = True
if _os.path.exists(_export_path):
    _fsize = _os.path.getsize(_export_path)
    if _fsize > 100:
        _needs_export = False
    else:
        _os.remove(_export_path)

if _needs_export:
    _apply_modifiers_for_export()
    _robust_export(_export_path)
"""


_CODE_REPLACEMENTS = [
    ("'BLENDER_EEVEE'", "'BLENDER_EEVEE_NEXT'"),
    ('"BLENDER_EEVEE"', '"BLENDER_EEVEE_NEXT"'),
    ("bpy.ops.export_scene.obj(", "bpy.ops.wm.obj_export("),
    ('.inputs["Specular"]', '.inputs["Specular IOR Level"]'),
    (".inputs['Specular']", ".inputs['Specular IOR Level']"),
    ('.inputs["Clearcoat"]', '.inputs["Coat Weight"]'),
    (".inputs['Clearcoat']", ".inputs['Coat Weight']"),
    ('.inputs["Sheen"]', '.inputs["Sheen Weight"]'),
    (".inputs['Sheen']", ".inputs['Sheen Weight']"),
]

_CODE_LINE_REMOVALS = [
    "use_auto_smooth",
    "auto_smooth_angle",
]


def _sanitize_code(code: str) -> str:
    """Apply Blender 4.2 compatibility fixes to user code before execution."""
    for old, new in _CODE_REPLACEMENTS:
        code = code.replace(old, new)

    lines = code.splitlines(keepends=True)
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if any(pat in stripped for pat in _CODE_LINE_REMOVALS):
            if stripped.startswith("#"):
                cleaned.append(line)
            else:
                cleaned.append(line.replace(stripped, f"pass  # removed: {stripped}"))
        else:
            cleaned.append(line)
    return "".join(cleaned)


@app.function(image=blender_image, cpu=2, memory=4096, timeout=150)  # keep in sync with ModalConfig
def execute_blender_code(code: str, seed: int = 42) -> dict[str, Any]:
    """Execute raw Blender Python code and return the exported mesh.

    Returns dict with keys: success, mesh_bytes, mesh_stats, error, elapsed.
    """
    t0 = time.monotonic()

    code = _sanitize_code(code)

    with tempfile.TemporaryDirectory() as tmp:
        export_path = os.path.join(tmp, "output.obj")
        script_path = os.path.join(tmp, "script.py")

        safe_export_path = export_path.replace("\\", "\\\\")
        wrapped = _WRAPPER_TEMPLATE.replace(
            _EXPORT_PATH_PLACEHOLDER, safe_export_path,
        ).replace(
            _USER_CODE_PLACEHOLDER, code,
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


_SUPPORTED_MESH_EXTS = (".obj", ".glb", ".gltf", ".ply", ".stl")


def _load_gt_mesh_from_volume(object_id: str, volume_subdir: str = "meshes") -> tuple[bytes, str] | None:
    for ext in _SUPPORTED_MESH_EXTS:
        path = Path(f"/data/{volume_subdir}/{object_id}{ext}")
        if path.exists():
            return path.read_bytes(), ext.lstrip(".")
    return None


@app.function(image=blender_image, volumes={"/data": volume}, timeout=120, cpu=2, memory=4096)
def compute_metrics_against_volume_mesh(
    object_id: str,
    gen_mesh_bytes: bytes,
    gen_mesh_format: str = "obj",
    num_points: int = 10_000,
    volume_subdir: str = "meshes",
) -> dict[str, Any]:
    """Compare generated mesh bytes against a GT mesh stored in the Modal volume."""
    gt = _load_gt_mesh_from_volume(object_id, volume_subdir)
    if gt is None:
        return {
            "chamfer": float("inf"),
            "f_score_001": 0.0,
            "f_score_005": 0.0,
            "hausdorff_90": float("inf"),
            "normal_consistency": 0.0,
            "error": f"GT mesh not found in Modal volume for {object_id}",
        }

    gt_bytes, gt_format = gt
    compute_metrics = modal.Function.from_name("llm3d-metrics-worker", "compute_metrics")
    return compute_metrics.remote(
        gen_mesh_bytes,
        gt_bytes,
        num_points,
        gen_mesh_format,
        gt_format,
    )


def _get_mesh_stats(mesh_bytes: bytes) -> dict:
    """Quick vertex/face count from raw OBJ bytes.

    Strips mtllib/usemtl references before parsing to avoid resolver
    failures when loading from in-memory bytes (no filesystem context).
    Falls back to manual line counting if trimesh still fails.
    """
    import io
    import re
    import sys

    import trimesh

    if not mesh_bytes or len(mesh_bytes) < 10:
        return {"vertices": 0, "faces": 0, "is_watertight": False}

    cleaned = re.sub(
        rb"^(mtllib|usemtl)\s+.*$", b"", mesh_bytes, flags=re.MULTILINE,
    )

    for attempt_bytes in [cleaned, mesh_bytes]:
        try:
            mesh = trimesh.load(
                io.BytesIO(attempt_bytes),
                file_type="obj",
                force="mesh",
                process=False,
            )
            if hasattr(mesh, "geometry"):
                meshes = list(mesh.geometry.values())
                total_v = sum(m.vertices.shape[0] for m in meshes)
                total_f = sum(m.faces.shape[0] for m in meshes)
                watertight = all(m.is_watertight for m in meshes)
            else:
                total_v = mesh.vertices.shape[0]
                total_f = mesh.faces.shape[0]
                watertight = mesh.is_watertight
            if total_v > 0:
                return {
                    "vertices": int(total_v),
                    "faces": int(total_f),
                    "is_watertight": watertight,
                }
        except Exception as exc:
            print(f"[_get_mesh_stats] trimesh parse attempt failed: {exc}", file=sys.stderr)

    total_v = 0
    total_f = 0
    for line in mesh_bytes.split(b"\n"):
        stripped = line.strip()
        if stripped.startswith(b"v "):
            total_v += 1
        elif stripped.startswith(b"f "):
            total_f += 1

    return {"vertices": total_v, "faces": total_f, "is_watertight": False}
