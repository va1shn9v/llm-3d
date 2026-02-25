"""
Blender script generation and multi-view rendering.

Handles both .glb input rendering (with ground-truth export) and .obj
mesh rendering (for generated/GT view comparison).
"""

import os
import logging
import subprocess
import textwrap
from typing import List, Optional, Tuple

from eval_pipeline.config import Config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: RENDER MULTI-VIEW INPUT IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def create_render_script(
    glb_path: str,
    output_dir: str,
    num_views: int,
    resolution: int,
) -> str:
    """
    Generate a Blender Python script that:
      1. Imports a .glb file
      2. Normalizes it to a unit bounding box (centered at origin)
      3. Sets up camera and lighting
      4. Renders N views evenly spaced around the object
      5. Also exports the ground-truth mesh as .obj for later comparison

    This script is executed via `blender --background --python <script>`.

    The normalization step is CRITICAL for fair metric comparison later —
    all meshes (ground truth and generated) must be in the same coordinate
    system and scale.
    """
    return textwrap.dedent(f"""\
        import bpy
        import math
        import os
        import bmesh
        import mathutils

        # ── Clean the scene ──────────────────────────────────────────────
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Delete everything
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # ── Import the .glb file ─────────────────────────────────────────
        bpy.ops.import_scene.gltf(filepath=r"{glb_path}")

        # ── Merge all mesh objects into one & normalize ──────────────────
        # Some .glb files contain multiple objects; we merge them.
        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        if not mesh_objects:
            raise RuntimeError("No mesh objects found in the imported file!")

        # Select all mesh objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objects[0]

        # Join into single object
        if len(mesh_objects) > 1:
            bpy.ops.object.join()

        obj = bpy.context.active_object

        # Apply all transforms so geometry is in world space
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # ── Normalize to unit bounding box centered at origin ────────────
        # This is essential for fair metric comparison. Without this,
        # a model that outputs a correct shape at the wrong scale would
        # be unfairly penalized.
        bbox_corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        bbox_min = mathutils.Vector((
            min(c.x for c in bbox_corners),
            min(c.y for c in bbox_corners),
            min(c.z for c in bbox_corners),
        ))
        bbox_max = mathutils.Vector((
            max(c.x for c in bbox_corners),
            max(c.y for c in bbox_corners),
            max(c.z for c in bbox_corners),
        ))
        center = (bbox_min + bbox_max) / 2.0
        extent = max(bbox_max - bbox_min)

        # Avoid division by zero for degenerate meshes
        if extent < 1e-8:
            extent = 1.0

        # Center and scale
        for v in obj.data.vertices:
            v.co = (mathutils.Vector(v.co) - center) / extent

        # ── Export ground-truth mesh as .obj ──────────────────────────────
        gt_path = os.path.join(r"{output_dir}", "ground_truth.obj")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.obj_export(
            filepath=gt_path,
            export_selected_objects=True,
            export_uv=False,
            export_materials=False,
        )
        print(f"EXPORTED_GT: {{gt_path}}")

        # ── Set up rendering ─────────────────────────────────────────────
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'  # Use CPU for compatibility
        scene.cycles.samples = 64    # Low samples for speed; enough for LLM input
        scene.render.resolution_x = {resolution}
        scene.render.resolution_y = {resolution}
        scene.render.film_transparent = True  # Transparent background
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'

        # ── Lighting: 3-point setup ──────────────────────────────────────
        # Key light (main)
        bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
        key_light = bpy.context.object
        key_light.data.energy = 100
        key_light.data.size = 2

        # Fill light (softer, opposite side)
        bpy.ops.object.light_add(type='AREA', location=(-2, 2, 2))
        fill_light = bpy.context.object
        fill_light.data.energy = 40
        fill_light.data.size = 3

        # Rim light (behind, for edge definition)
        bpy.ops.object.light_add(type='AREA', location=(0, 3, 1))
        rim_light = bpy.context.object
        rim_light.data.energy = 60
        rim_light.data.size = 1

        # ── Camera setup ─────────────────────────────────────────────────
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        scene.camera = camera
        camera.data.lens = 50  # 50mm focal length

        # Camera distance: must be far enough to frame the full bounding
        # sphere (radius ~0.87 for a unit cube).  With a 50 mm lens the
        # theoretical minimum is ~2.4; we use 2.5 for a small margin.
        cam_distance = 2.5

        # ── Render views evenly spaced around the object ─────────────────
        num_views = {num_views}
        for i in range(num_views):
            # Azimuth: evenly spaced around 360°
            azimuth = (2 * math.pi * i) / num_views
            # Elevation: slight variation to show top/bottom
            # First view is eye-level, others alternate slightly
            elevation = math.radians(20 + 10 * math.sin(2 * math.pi * i / num_views))

            # Spherical to Cartesian
            x = cam_distance * math.cos(elevation) * math.cos(azimuth)
            y = cam_distance * math.cos(elevation) * math.sin(azimuth)
            z = cam_distance * math.sin(elevation)

            camera.location = (x, y, z)

            # Point camera at origin (where the object is centered)
            direction = mathutils.Vector((0, 0, 0)) - camera.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()

            # Render
            filepath = os.path.join(r"{output_dir}", f"view_{{i:02d}}.png")
            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)
            print(f"RENDERED: {{filepath}}")

        print("RENDER_COMPLETE")
    """)


def create_obj_render_script(
    obj_path: str,
    output_dir: str,
    num_views: int,
    resolution: int,
    prefix: str = "gen_view",
) -> str:
    """
    Generate a Blender Python script that renders multi-view images of a .obj mesh.

    Used to render views of the generated mesh (or GT mesh) for visual comparison.
    Uses the same camera setup as the input view renderer so images are directly
    comparable.
    """
    return textwrap.dedent(f"""\
        import bpy
        import math
        import os
        import mathutils

        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Import the .obj mesh
        bpy.ops.wm.obj_import(filepath=r"{obj_path}")

        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        if not mesh_objects:
            raise RuntimeError("No mesh objects found in the imported file!")

        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objects[0]

        if len(mesh_objects) > 1:
            bpy.ops.object.join()

        obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Normalize to unit bounding box centered at origin
        bbox_corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        bbox_min = mathutils.Vector((
            min(c.x for c in bbox_corners),
            min(c.y for c in bbox_corners),
            min(c.z for c in bbox_corners),
        ))
        bbox_max = mathutils.Vector((
            max(c.x for c in bbox_corners),
            max(c.y for c in bbox_corners),
            max(c.z for c in bbox_corners),
        ))
        center = (bbox_min + bbox_max) / 2.0
        extent = max(bbox_max - bbox_min)
        if extent < 1e-8:
            extent = 1.0

        for v in obj.data.vertices:
            v.co = (mathutils.Vector(v.co) - center) / extent

        # Rendering setup
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 64
        scene.render.resolution_x = {resolution}
        scene.render.resolution_y = {resolution}
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'

        # 3-point lighting (same as input renderer)
        bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
        key_light = bpy.context.object
        key_light.data.energy = 100
        key_light.data.size = 2

        bpy.ops.object.light_add(type='AREA', location=(-2, 2, 2))
        fill_light = bpy.context.object
        fill_light.data.energy = 40
        fill_light.data.size = 3

        bpy.ops.object.light_add(type='AREA', location=(0, 3, 1))
        rim_light = bpy.context.object
        rim_light.data.energy = 60
        rim_light.data.size = 1

        # Camera
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        scene.camera = camera
        camera.data.lens = 50
        cam_distance = 2.5

        # Render views from the same angles as the input renderer
        num_views = {num_views}
        for i in range(num_views):
            azimuth = (2 * math.pi * i) / num_views
            elevation = math.radians(20 + 10 * math.sin(2 * math.pi * i / num_views))

            x = cam_distance * math.cos(elevation) * math.cos(azimuth)
            y = cam_distance * math.cos(elevation) * math.sin(azimuth)
            z = cam_distance * math.sin(elevation)

            camera.location = (x, y, z)

            direction = mathutils.Vector((0, 0, 0)) - camera.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()

            filepath = os.path.join(r"{output_dir}", f"{prefix}_{{i:02d}}.png")
            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)
            print(f"RENDERED: {{filepath}}")

        print("RENDER_COMPLETE")
    """)


def render_mesh_views(
    obj_path: str,
    output_dir: str,
    config: Config,
    prefix: str = "gen_view",
) -> List[str]:
    """
    Render multi-view images of a .obj mesh using Blender CLI.

    Returns:
        List of rendered image paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    script_content = create_obj_render_script(
        obj_path=obj_path,
        output_dir=output_dir,
        num_views=config.num_views,
        resolution=config.render_resolution,
        prefix=prefix,
    )
    script_path = os.path.join(output_dir, f"_render_{prefix}_script.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    log.info(f"  Rendering {config.num_views} {prefix} views with Blender...")
    try:
        result = subprocess.run(
            [config.blender_path, "--background", "--python", script_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            log.warning(f"  Blender exited with code {result.returncode} for {prefix}")
            log.debug(f"  STDERR: {result.stderr[-500:]}")
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning(f"  {prefix} rendering failed: {e}")
        return []

    image_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(f"{prefix}_") and f.endswith(".png")
    ])
    log.info(f"  Rendered {len(image_paths)} {prefix} images.")
    return image_paths


def render_views(
    glb_path: str,
    output_dir: str,
    config: Config,
) -> Tuple[List[str], Optional[str]]:
    """
    Render multi-view images of a .glb object using Blender CLI.

    Returns:
        (list_of_image_paths, ground_truth_obj_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Write the Blender script to a temp file
    script_content = create_render_script(
        glb_path=glb_path,
        output_dir=output_dir,
        num_views=config.num_views,
        resolution=config.render_resolution,
    )
    script_path = os.path.join(output_dir, "_render_script.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    # Execute Blender in background mode
    log.info(f"  Rendering {config.num_views} views with Blender...")
    try:
        result = subprocess.run(
            [config.blender_path, "--background", "--python", script_path],
            capture_output=True,
            text=True,
            timeout=120,  # 2 min timeout for rendering
        )

        if result.returncode != 0:
            log.warning(f"  Blender exited with code {result.returncode}")
            log.debug(f"  STDERR: {result.stderr[-500:]}")
            return [], None

    except FileNotFoundError:
        log.error(
            f"Blender not found at '{config.blender_path}'. "
            f"Please install Blender or set the --blender-path argument.\n"
            f"  Download: https://www.blender.org/download/\n"
            f"  Or: pip install bpy  (Python 3.11 required)"
        )
        return [], None
    except subprocess.TimeoutExpired:
        log.warning("  Blender rendering timed out!")
        return [], None

    # Collect rendered images
    image_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("view_") and f.endswith(".png")
    ])

    gt_path = os.path.join(output_dir, "ground_truth.obj")
    gt_path = gt_path if os.path.exists(gt_path) else None

    log.info(f"  Rendered {len(image_paths)} views. GT mesh: {'✓' if gt_path else '✗'}")
    return image_paths, gt_path
