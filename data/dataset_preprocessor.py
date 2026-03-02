"""
Dataset preprocessor: load 3D objects, render multi-view images, and store them.

Supports two input modes:
  1. Local directory of mesh files (OBJ, GLB, GLTF, PLY, STL)
  2. HuggingFace dataset (e.g. InternRobotics/MeshCoderDataset)

For each object, renders N evenly-spaced azimuth views using Blender in
headless mode, normalizes to unit bounding box, and saves PNGs with a
transparent background.

Usage:
    python -m data.dataset_preprocessor --mesh-dir ./meshes --output-dir ./data/renders
    python -m data.dataset_preprocessor --hf-dataset InternRobotics/MeshCoderDataset --output-dir ./data/renders
    python -m data.dataset_preprocessor --mesh-dir ./meshes --num-views 5 --resolution 512 512
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import ProjectConfig, ViewConfig, load_config

log = logging.getLogger(__name__)

SUPPORTED_MESH_EXTENSIONS = {".obj", ".glb", ".gltf", ".ply", ".stl"}

_RENDER_SCRIPT = """\
import bpy, sys, os, json, math, platform

args = json.loads(sys.argv[sys.argv.index("--") + 1])
mesh_path = args["mesh_path"]
output_dir = args["output_dir"]
num_views = args["num_views"]
resolution = args["resolution"]
elevation_deg = args.get("elevation_deg", 25.0)
engine = args.get("engine", "CYCLES")
camera_distance = args.get("camera_distance", 2.5)
sun_energy = args.get("sun_energy", 3.0)
film_transparent = args.get("film_transparent", True)
cycles_samples = args.get("cycles_samples", 128)

bpy.ops.wm.read_factory_settings(use_empty=True)

os_name = platform.system()
is_headless = os.environ.get("DISPLAY") is None and os_name == "Linux"

# --- GPU configuration ---
def setup_gpu():
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons.get("cycles")
    if not cycles_prefs:
        print("WARNING: Cycles addon not found", file=sys.stderr)
        return False

    cp = cycles_prefs.preferences

    if os_name == "Darwin":
        candidates = ["METAL"]
    else:
        candidates = ["OPTIX", "CUDA", "HIP"]

    activated = False
    for dev_type in candidates:
        try:
            cp.compute_device_type = dev_type
            cp.get_devices()
            gpu_devices = [d for d in cp.devices if d.type != "CPU"]
            if gpu_devices:
                for d in cp.devices:
                    d.use = d.type != "CPU"
                activated = True
                print(f"GPU enabled: {dev_type}, "
                      f"devices: {[d.name for d in gpu_devices]}")
                break
        except (TypeError, RuntimeError):
            continue

    if not activated:
        print("WARNING: No GPU devices found, falling back to CPU", file=sys.stderr)
        return False

    return True

gpu_available = setup_gpu()

# Auto-select engine: on headless Linux without EGL, EEVEE will fail; use Cycles.
if engine == "BLENDER_EEVEE_NEXT" and is_headless:
    print("Headless Linux detected — switching from EEVEE to CYCLES for GPU rendering",
          file=sys.stderr)
    engine = "CYCLES"

scene = bpy.context.scene
scene.render.engine = engine

if engine == "CYCLES":
    scene.cycles.device = "GPU" if gpu_available else "CPU"
    scene.cycles.samples = cycles_samples
    scene.cycles.use_auto_tile = True
    scene.cycles.tile_size = 512 if gpu_available else 64

ext = os.path.splitext(mesh_path)[1].lower()
if ext == ".obj":
    bpy.ops.wm.obj_import(filepath=mesh_path)
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=mesh_path)
elif ext == ".ply":
    bpy.ops.wm.ply_import(filepath=mesh_path)
elif ext == ".stl":
    bpy.ops.wm.stl_import(filepath=mesh_path)
else:
    bpy.ops.wm.obj_import(filepath=mesh_path)

meshes = [o for o in bpy.data.objects if o.type == 'MESH']
if not meshes:
    print("NO_MESHES_FOUND", file=sys.stderr)
    sys.exit(1)

import numpy as np
all_verts = []
for o in meshes:
    mat = o.matrix_world
    for v in o.data.vertices:
        all_verts.append(mat @ v.co)
all_verts = np.array([(v.x, v.y, v.z) for v in all_verts])
center = (all_verts.max(0) + all_verts.min(0)) / 2
extent = (all_verts.max(0) - all_verts.min(0)).max()
scale = 1.0 / extent if extent > 1e-8 else 1.0
for o in meshes:
    o.location.x -= center[0]
    o.location.y -= center[1]
    o.location.z -= center[2]
    o.scale *= scale

scene.render.resolution_x = resolution[0]
scene.render.resolution_y = resolution[1]
scene.render.film_transparent = film_transparent
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

light_data = bpy.data.lights.new("Sun", type="SUN")
light_data.energy = sun_energy
light_obj = bpy.data.objects.new("Sun", light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.rotation_euler = (math.radians(50), math.radians(10), math.radians(30))

cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj

elevation = math.radians(elevation_deg)
os.makedirs(output_dir, exist_ok=True)

for i in range(num_views):
    azimuth = 2 * math.pi * i / num_views
    x = camera_distance * math.cos(elevation) * math.cos(azimuth)
    y = camera_distance * math.cos(elevation) * math.sin(azimuth)
    z = camera_distance * math.sin(elevation)
    cam_obj.location = (x, y, z)

    direction = -cam_obj.location.normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    scene.render.filepath = os.path.join(output_dir, f"view_{i}.png")
    bpy.ops.render.render(write_still=True)

print("RENDER_COMPLETE")
"""


def discover_meshes(mesh_dir: str | Path) -> list[Path]:
    """Find all supported mesh files in a directory tree."""
    mesh_dir = Path(mesh_dir)
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    meshes = []
    for p in sorted(mesh_dir.rglob("*")):
        if p.suffix.lower() in SUPPORTED_MESH_EXTENSIONS and p.is_file():
            meshes.append(p)

    log.info(f"Discovered {len(meshes)} mesh files in {mesh_dir}")
    return meshes


def load_hf_meshes(
    dataset_name: str,
    split: str = "train",
    mesh_key: str = "mesh",
    cache_dir: str | Path | None = None,
) -> list[tuple[Path, str]]:
    """Download meshes from a HuggingFace dataset and write to a cache directory.

    Expects the dataset to contain a column with mesh file bytes or paths.
    Returns list of (mesh_path, object_id) tuples — the object_id is taken
    directly from the dataset so it matches the ground truth code records.
    """
    from datasets import load_dataset

    cache_dir = Path(cache_dir or tempfile.mkdtemp(prefix="llm3d_hf_meshes_"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, split=split)
    log.info(f"Loaded HF dataset '{dataset_name}' split='{split}': {len(ds)} samples")

    results: list[tuple[Path, str]] = []
    for idx, sample in enumerate(ds):
        obj_id = str(sample.get("id", sample.get("object_id", f"obj_{idx:06d}")))
        obj_dir = cache_dir / obj_id
        obj_dir.mkdir(parents=True, exist_ok=True)

        if mesh_key in sample and sample[mesh_key] is not None:
            mesh_data = sample[mesh_key]
            ext = ".obj"
            if isinstance(mesh_data, dict):
                ext = mesh_data.get("format", ".obj")
                if not ext.startswith("."):
                    ext = f".{ext}"
                mesh_data = mesh_data.get("bytes", mesh_data.get("data", b""))

            mesh_path = obj_dir / f"mesh{ext}"
            if isinstance(mesh_data, bytes):
                mesh_path.write_bytes(mesh_data)
            elif isinstance(mesh_data, str) and Path(mesh_data).exists():
                import shutil
                shutil.copy2(mesh_data, mesh_path)
            else:
                log.warning(f"Skipping {obj_id}: unsupported mesh data type")
                continue

            results.append((mesh_path, obj_id))

    log.info(f"Extracted {len(results)} meshes to {cache_dir}")
    return results


def derive_object_id(mesh_path: Path) -> str:
    """Derive a stable object_id from a mesh file path.

    Convention (matches dataset_builder expectations):
      - {dir}/{object_id}/mesh.{ext}  →  object_id  (HF dataset layout)
      - {dir}/{object_id}/anything.{ext}  →  object_id  (parent dir = id)
      - {dir}/{object_id}.{ext}  →  object_id  (flat layout, stem = id)
    """
    parent = mesh_path.parent.name
    stem = mesh_path.stem
    if parent and parent not in (".", "") and stem in ("mesh", "model", "input"):
        return parent
    if parent and parent not in (".", ""):
        return parent
    return stem


def render_object(
    mesh_path: str | Path,
    output_dir: str | Path,
    blender_path: str = "blender",
    view_cfg: ViewConfig | None = None,
    object_id: str | None = None,
) -> dict:
    """Render multi-view images of a single mesh using Blender.

    Returns a result dict with object_id, image paths, and status.
    Pass object_id explicitly to guarantee it matches your dataset's ground truth.
    """
    mesh_path = Path(mesh_path)
    if view_cfg is None:
        view_cfg = ViewConfig()

    if object_id is None:
        object_id = derive_object_id(mesh_path)

    obj_output_dir = Path(output_dir) / object_id
    obj_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = obj_output_dir / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        expected_views = [obj_output_dir / f"view_{i}.png" for i in range(view_cfg.num_views)]
        if all(p.exists() for p in expected_views) and existing.get("num_views") == view_cfg.num_views:
            log.debug(f"Skipping {object_id}: already rendered")
            return existing

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_RENDER_SCRIPT)
        script_path = f.name

    try:
        args_json = json.dumps({
            "mesh_path": str(mesh_path.resolve()),
            "output_dir": str(obj_output_dir.resolve()),
            "num_views": view_cfg.num_views,
            "resolution": list(view_cfg.resolution),
            "elevation_deg": view_cfg.elevation_deg,
            "engine": view_cfg.engine,
            "camera_distance": view_cfg.camera_distance_factor,
            "sun_energy": view_cfg.sun_energy,
            "film_transparent": view_cfg.film_transparent,
            "cycles_samples": view_cfg.cycles_samples,
        })

        import platform as _platform
        cmd = [blender_path, "--background"]
        if view_cfg.engine == "BLENDER_EEVEE_NEXT" and _platform.system() == "Linux":
            cmd += ["--gpu-backend", "egl"]
        cmd += ["--python", script_path, "--", args_json]

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )

        image_paths = []
        for i in range(view_cfg.num_views):
            img = obj_output_dir / f"view_{i}.png"
            if img.exists():
                image_paths.append(str(img))

        success = len(image_paths) == view_cfg.num_views
        if not success:
            log.warning(
                f"{object_id}: rendered {len(image_paths)}/{view_cfg.num_views} views. "
                f"stderr: {proc.stderr[-500:] if proc.stderr else '(empty)'}"
            )

        result = {
            "object_id": object_id,
            "mesh_path": str(mesh_path),
            "image_paths": image_paths,
            "num_views": view_cfg.num_views,
            "resolution": list(view_cfg.resolution),
            "success": success,
        }

        manifest.write_text(json.dumps(result, indent=2))
        return result

    finally:
        os.unlink(script_path)


def preprocess_dataset(
    mesh_entries: list[Path] | list[tuple[Path, str]],
    output_dir: str | Path,
    cfg: ProjectConfig | None = None,
    max_workers: int = 4,
) -> list[dict]:
    """Render multi-view images for all meshes in parallel.

    mesh_entries can be:
      - list[Path]               — object_id derived from path
      - list[tuple[Path, str]]   — explicit (path, object_id) pairs

    Returns list of result dicts, one per object.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[Path, str | None]] = []
    for item in mesh_entries:
        if isinstance(item, tuple):
            entries.append(item)
        else:
            entries.append((item, None))

    log.info(
        f"Preprocessing {len(entries)} objects → {output_dir} "
        f"({cfg.views.num_views} views @ {cfg.views.resolution})"
    )

    results: list[dict] = []

    if max_workers <= 1:
        for i, (mp, oid) in enumerate(entries):
            log.info(f"[{i + 1}/{len(entries)}] Rendering {mp.name}")
            r = render_object(mp, output_dir, cfg.blender_path, cfg.views, object_id=oid)
            results.append(r)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for mp, oid in entries:
                fut = pool.submit(
                    render_object, mp, output_dir, cfg.blender_path, cfg.views, oid,
                )
                futures[fut] = mp

            for i, fut in enumerate(as_completed(futures), 1):
                mp = futures[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    status = "ok" if r["success"] else "partial"
                    log.info(f"[{i}/{len(entries)}] {mp.name}: {status}")
                except Exception as e:
                    log.error(f"[{i}/{len(entries)}] {mp.name}: FAILED — {e}")
                    results.append({
                        "object_id": mp.stem,
                        "mesh_path": str(mp),
                        "image_paths": [],
                        "num_views": 0,
                        "success": False,
                        "error": str(e),
                    })

    succeeded = sum(1 for r in results if r["success"])
    log.info(
        f"Preprocessing complete: {succeeded}/{len(results)} objects rendered "
        f"({100 * succeeded / max(len(results), 1):.1f}% success)"
    )

    summary_path = output_dir / "preprocess_summary.json"
    summary = {
        "total_objects": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "num_views": cfg.views.num_views,
        "resolution": list(cfg.views.resolution),
        "objects": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary written to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess 3D objects: load, render multi-view images, and store."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mesh-dir", type=str, help="Directory containing mesh files")
    source.add_argument("--hf-dataset", type=str, help="HuggingFace dataset name")

    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for renders")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--num-views", type=int, default=None, help="Number of views per object (default: from config)")
    parser.add_argument("--resolution", type=int, nargs=2, default=None, help="Render resolution WxH (default: from config)")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel render workers")
    parser.add_argument("--hf-split", type=str, default="train", help="HuggingFace dataset split")
    parser.add_argument("--hf-mesh-key", type=str, default="mesh", help="Column name for mesh data")
    parser.add_argument("--blender-path", type=str, default=None, help="Path to Blender executable")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config)

    if args.num_views is not None:
        cfg.views.num_views = args.num_views
    if args.resolution is not None:
        cfg.views.resolution = tuple(args.resolution)
    if args.blender_path is not None:
        cfg.blender_path = args.blender_path

    output_dir = args.output_dir or f"{cfg.data_dir}/renders"

    if args.mesh_dir:
        mesh_paths = discover_meshes(args.mesh_dir)
    else:
        mesh_paths = load_hf_meshes(
            args.hf_dataset,
            split=args.hf_split,
            mesh_key=args.hf_mesh_key,
        )

    if not mesh_paths:
        log.error("No mesh files found. Exiting.")
        return

    preprocess_dataset(mesh_paths, output_dir, cfg, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
