"""
Prepare the Objaverse render dataset.

Selects 130 LVIS categories (sorted alphabetically), downloads objects,
renders 5 views per object via Blender, and organises them into train/test splits.

Split logic:
  - First 100 categories: 20 objects -> train, 5 objects -> test
  - Last  30 categories:  5 objects -> test  (unseen categories at test time)

Output layout:
  objaverse_render_data/
    train/<category>/<uid>/view_00.png … view_04.png + ground_truth.obj
    test/<category>/<uid>/view_00.png  … view_04.png + ground_truth.obj
    metadata.json

Usage:
    python prepare_objaverse_data.py                        # defaults
    python prepare_objaverse_data.py --engine CYCLES        # higher quality
    python prepare_objaverse_data.py --resolution 256       # faster renders
    python prepare_objaverse_data.py --fast                 # aggressive speed (prototyping)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import textwrap
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset split constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_CATEGORIES = 130
TRAIN_CATEGORIES = 100
NOVEL_CATEGORIES = 30

TRAIN_OBJECTS_PER_CAT = 20
TEST_OBJECTS_PER_CAT_SEEN = 5
TEST_OBJECTS_PER_CAT_NOVEL = 5

NUM_VIEWS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Blender render script generation
# ─────────────────────────────────────────────────────────────────────────────
def _blender_render_script(
    glb_path: str,
    output_dir: str,
    resolution: int,
    engine: str,
    samples: int,
) -> str:
    """Return a self-contained Blender Python script for 5-view rendering."""

    if engine == "EEVEE":
        engine_block = textwrap.dedent(f"""\
            # Try EEVEE names across Blender versions (5.x / 4.2+ / 4.0-4.1)
            for ename in ['BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT']:
                try:
                    scene.render.engine = ename
                    break
                except Exception:
                    continue
            scene.eevee.taa_render_samples = {samples}

            # Disable post-processing unnecessary for dataset renders
            for attr in ['use_gtao', 'use_bloom', 'use_ssr']:
                if hasattr(scene.eevee, attr):
                    setattr(scene.eevee, attr, False)
            for attr in ['shadow_cube_size', 'shadow_cascade_size']:
                if hasattr(scene.eevee, attr):
                    setattr(scene.eevee, attr, '256')
        """)
    else:
        engine_block = textwrap.dedent(f"""\
            scene.render.engine = 'CYCLES'
            scene.cycles.device = 'GPU'
            scene.cycles.samples = {samples}

            # Enable Metal GPU (Apple Silicon)
            prefs = bpy.context.preferences
            cprefs = prefs.addons['cycles'].preferences
            cprefs.compute_device_type = 'METAL'
            cprefs.get_devices()
            for device in cprefs.devices:
                device.use = True

            scene.cycles.use_denoising = True
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'

            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = 0.05
            scene.cycles.tile_size = 256
            scene.cycles.max_bounces = 4
            scene.cycles.diffuse_bounces = 2
            scene.cycles.glossy_bounces = 2
            scene.cycles.transmission_bounces = 2
        """)

    return textwrap.dedent(f"""\
        import bpy, math, os, mathutils

        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        bpy.ops.import_scene.gltf(filepath=r"{glb_path}")

        mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
        if not mesh_objects:
            raise RuntimeError("No mesh objects in imported file")

        bpy.ops.object.select_all(action='DESELECT')
        for o in mesh_objects:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objects[0]
        if len(mesh_objects) > 1:
            bpy.ops.object.join()

        obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Normalise to unit bounding box centred at origin
        corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        bmin = mathutils.Vector((min(c.x for c in corners),
                                  min(c.y for c in corners),
                                  min(c.z for c in corners)))
        bmax = mathutils.Vector((max(c.x for c in corners),
                                  max(c.y for c in corners),
                                  max(c.z for c in corners)))
        centre = (bmin + bmax) / 2.0
        extent = max(bmax - bmin)
        if extent < 1e-8:
            extent = 1.0
        for v in obj.data.vertices:
            v.co = (mathutils.Vector(v.co) - centre) / extent

        # Export ground-truth mesh
        gt_path = os.path.join(r"{output_dir}", "ground_truth.obj")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.obj_export(filepath=gt_path,
                               export_selected_objects=True,
                               export_uv=False,
                               export_materials=False)

        # Render engine
        scene = bpy.context.scene
        {textwrap.indent(engine_block, "        ").strip()}
        scene.render.resolution_x = {resolution}
        scene.render.resolution_y = {resolution}
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'

        scene.render.use_simplify = True
        scene.render.simplify_subdivision_render = 0

        # 3-point lighting
        bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
        bpy.context.object.data.energy = 100
        bpy.context.object.data.size = 2
        bpy.ops.object.light_add(type='AREA', location=(-2, 2, 2))
        bpy.context.object.data.energy = 40
        bpy.context.object.data.size = 3
        bpy.ops.object.light_add(type='AREA', location=(0, 3, 1))
        bpy.context.object.data.energy = 60
        bpy.context.object.data.size = 1

        # Camera
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        scene.camera = cam
        cam.data.lens = 50
        cam_d = 2.5

        for i in range({NUM_VIEWS}):
            az = (2 * math.pi * i) / {NUM_VIEWS}
            el = math.radians(20 + 10 * math.sin(2 * math.pi * i / {NUM_VIEWS}))
            cam.location = (cam_d * math.cos(el) * math.cos(az),
                            cam_d * math.cos(el) * math.sin(az),
                            cam_d * math.sin(el))
            d = mathutils.Vector((0, 0, 0)) - cam.location
            cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
            scene.render.filepath = os.path.join(r"{output_dir}", f"view_{{i:02d}}.png")
            bpy.ops.render.render(write_still=True)
            print(f"RENDERED view_{{i:02d}}")

        print("RENDER_COMPLETE")
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Single-object rendering
# ─────────────────────────────────────────────────────────────────────────────
def render_object(
    glb_path: str,
    output_dir: str,
    blender_path: str,
    resolution: int,
    engine: str,
    samples: int,
    timeout: int = 300,
) -> bool:
    """Render 5 views of a .glb object. Returns True on success. Skips if already done."""
    os.makedirs(output_dir, exist_ok=True)

    expected = [os.path.join(output_dir, f"view_{i:02d}.png") for i in range(NUM_VIEWS)]
    if all(os.path.exists(p) for p in expected):
        log.info("    already rendered, skipping")
        return True

    script = _blender_render_script(glb_path, output_dir, resolution, engine, samples)
    script_path = os.path.join(output_dir, "_render.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    try:
        result = subprocess.run(
            [blender_path, "--background", "--python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            log.warning("    Blender exit code %d", result.returncode)
            if result.stderr:
                log.debug("    %s", result.stderr[-300:])
            return False
    except FileNotFoundError:
        log.error("Blender not found at '%s'", blender_path)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        log.warning("    Blender timed out (%ds)", timeout)
        return False

    return all(os.path.exists(p) for p in expected)


# ─────────────────────────────────────────────────────────────────────────────
# Category processing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _render_uid_list(
    uids: List[str],
    objects: Dict[str, str],
    dest_root: str,
    category: str,
    blender_path: str,
    resolution: int,
    engine: str,
    samples: int,
) -> List[str]:
    """Render a list of UIDs and return those that succeeded."""
    ok: List[str] = []
    for uid in uids:
        if uid not in objects:
            log.warning("    %s: download failed, skipping", uid)
            continue
        out = os.path.join(dest_root, category, uid)
        log.info("    [%d/%d] %s", len(ok) + 1, len(uids), uid)
        if render_object(objects[uid], out, blender_path, resolution, engine, samples):
            ok.append(uid)
        else:
            log.warning("    %s: render failed", uid)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--blender-path", default="blender",
                        help="Path to Blender binary (default: blender)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", default="./objaverse_render_data",
                        help="Root output directory (default: ./objaverse_render_data)")
    parser.add_argument("--download-processes", type=int, default=4,
                        help="Parallel download workers (default: 4)")
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE",
                        help="Blender render engine (default: EEVEE, much faster)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution in px (default: 512)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Render samples (default: 32 for EEVEE, 24 for CYCLES)")
    parser.add_argument("--fast", action="store_true",
                        help="Aggressive speed settings (resolution=256, low samples)")
    args = parser.parse_args()

    if args.fast:
        if args.resolution == 512:
            args.resolution = 256
        if args.samples is None:
            samples = 16 if args.engine == "EEVEE" else 8
        else:
            samples = args.samples
    else:
        samples = args.samples or (32 if args.engine == "EEVEE" else 24)
    rng = np.random.RandomState(args.seed)
    root = os.path.abspath(args.output_dir)
    train_root = os.path.join(root, "train")
    test_root = os.path.join(root, "test")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    # ── 1. Load LVIS annotations ─────────────────────────────────────────
    import objaverse  # noqa: E402 — deferred so --help is fast

    log.info("Loading Objaverse LVIS annotations ...")
    lvis = objaverse.load_lvis_annotations()
    all_categories = sorted(lvis.keys())
    log.info("Total LVIS categories available: %d", len(all_categories))

    if len(all_categories) < NUM_CATEGORIES:
        log.error("Need %d categories but only %d available", NUM_CATEGORIES, len(all_categories))
        sys.exit(1)

    selected = all_categories[:NUM_CATEGORIES]
    seen_cats = selected[:TRAIN_CATEGORIES]
    novel_cats = selected[TRAIN_CATEGORIES:]

    log.info("Selected %d categories (%d seen + %d novel)",
             NUM_CATEGORIES, TRAIN_CATEGORIES, NOVEL_CATEGORIES)
    log.info("Seen  (first 5): %s", seen_cats[:5])
    log.info("Novel (first 5): %s", novel_cats[:5])
    log.info("Engine: %s | Resolution: %d | Samples: %d", args.engine, args.resolution, samples)

    metadata: Dict[str, Any] = {
        "config": {
            "num_categories": NUM_CATEGORIES,
            "train_categories": TRAIN_CATEGORIES,
            "novel_categories": NOVEL_CATEGORIES,
            "train_objects_per_cat": TRAIN_OBJECTS_PER_CAT,
            "test_objects_per_cat_seen": TEST_OBJECTS_PER_CAT_SEEN,
            "test_objects_per_cat_novel": TEST_OBJECTS_PER_CAT_NOVEL,
            "num_views": NUM_VIEWS,
            "resolution": args.resolution,
            "engine": args.engine,
            "seed": args.seed,
        },
        "seen_categories": seen_cats,
        "novel_categories": novel_cats,
        "train": {},
        "test": {},
    }

    expected_train = TRAIN_CATEGORIES * TRAIN_OBJECTS_PER_CAT
    expected_test = (TRAIN_CATEGORIES * TEST_OBJECTS_PER_CAT_SEEN
                     + NOVEL_CATEGORIES * TEST_OBJECTS_PER_CAT_NOVEL)
    log.info("Target: %d train, %d test objects", expected_train, expected_test)

    # ── 2. Seen categories (first 100): train + test ─────────────────────
    for cat_idx, category in enumerate(seen_cats, 1):
        log.info("[%d/%d] %s", cat_idx, NUM_CATEGORIES, category)
        uids = lvis[category]
        needed = TRAIN_OBJECTS_PER_CAT + TEST_OBJECTS_PER_CAT_SEEN

        if len(uids) < needed:
            log.warning("  %d objects available (need %d) — using all, splitting proportionally",
                        len(uids), needed)
            chosen = list(rng.choice(uids, size=len(uids), replace=False))
            split = max(1, int(len(chosen) * TRAIN_OBJECTS_PER_CAT / needed))
            train_uids = chosen[:split]
            test_uids = chosen[split:]
        else:
            chosen = list(rng.choice(uids, size=needed, replace=False))
            train_uids = chosen[:TRAIN_OBJECTS_PER_CAT]
            test_uids = chosen[TRAIN_OBJECTS_PER_CAT:]

        all_uids = train_uids + test_uids
        log.info("  Downloading %d objects ...", len(all_uids))
        objects = objaverse.load_objects(uids=all_uids, download_processes=args.download_processes)

        log.info("  Rendering %d train objects ...", len(train_uids))
        ok_train = _render_uid_list(
            train_uids, objects, train_root, category,
            args.blender_path, args.resolution, args.engine, samples,
        )

        log.info("  Rendering %d test objects ...", len(test_uids))
        ok_test = _render_uid_list(
            test_uids, objects, test_root, category,
            args.blender_path, args.resolution, args.engine, samples,
        )

        metadata["train"][category] = ok_train
        metadata["test"][category] = ok_test
        log.info("  Done: %d train, %d test", len(ok_train), len(ok_test))

        _save_metadata(metadata, root)

    # ── 3. Novel categories (last 30): test only ─────────────────────────
    for cat_idx, category in enumerate(novel_cats, TRAIN_CATEGORIES + 1):
        log.info("[%d/%d] %s (novel)", cat_idx, NUM_CATEGORIES, category)
        uids = lvis[category]

        if len(uids) < TEST_OBJECTS_PER_CAT_NOVEL:
            log.warning("  %d objects (need %d)", len(uids), TEST_OBJECTS_PER_CAT_NOVEL)
            chosen = list(rng.choice(uids, size=len(uids), replace=False))
        else:
            chosen = list(rng.choice(uids, size=TEST_OBJECTS_PER_CAT_NOVEL, replace=False))

        log.info("  Downloading %d objects ...", len(chosen))
        objects = objaverse.load_objects(uids=chosen, download_processes=args.download_processes)

        log.info("  Rendering %d test objects ...", len(chosen))
        ok_test = _render_uid_list(
            chosen, objects, test_root, category,
            args.blender_path, args.resolution, args.engine, samples,
        )

        metadata["test"][category] = ok_test
        log.info("  Done: %d test", len(ok_test))

        _save_metadata(metadata, root)

    # ── 4. Final summary ─────────────────────────────────────────────────
    n_train = sum(len(v) for v in metadata["train"].values())
    n_test = sum(len(v) for v in metadata["test"].values())
    log.info("Complete! %d train, %d test objects across %d categories",
             n_train, n_test, NUM_CATEGORIES)
    log.info("Output: %s", root)


def _save_metadata(metadata: Dict[str, Any], root: str) -> None:
    """Incrementally persist metadata after each category (crash-safe)."""
    path = os.path.join(root, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
