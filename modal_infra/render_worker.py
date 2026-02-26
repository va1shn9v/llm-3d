"""
Modal function: render multi-view images of a mesh using Blender.
"""

from __future__ import annotations

import io
import math
import os
import tempfile
from typing import Any

import modal

from modal_infra.images import blender_image

app = modal.App("llm3d-render-worker")
volume = modal.Volume.from_name("llm3d-data", create_if_missing=True)

_RENDER_SCRIPT = """\
import bpy, sys, os, json, math

args = json.loads(sys.argv[sys.argv.index("--") + 1])
mesh_path = args["mesh_path"]
output_dir = args["output_dir"]
num_views = args["num_views"]
resolution = args["resolution"]
elevation_deg = args.get("elevation_deg", 25.0)
engine = args.get("engine", "BLENDER_EEVEE_NEXT")
camera_distance = args.get("camera_distance", 2.5)

bpy.ops.wm.read_factory_settings(use_empty=True)

# Import mesh
if mesh_path.endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=mesh_path)
elif mesh_path.endswith(".glb") or mesh_path.endswith(".gltf"):
    bpy.ops.import_scene.gltf(filepath=mesh_path)
else:
    bpy.ops.wm.obj_import(filepath=mesh_path)

# Normalize to unit bbox
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
if not meshes:
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
if extent > 1e-8:
    scale = 1.0 / extent
else:
    scale = 1.0
for o in meshes:
    o.location.x -= center[0]
    o.location.y -= center[1]
    o.location.z -= center[2]
    o.scale *= scale

# Setup renderer
scene = bpy.context.scene
scene.render.engine = engine
scene.render.resolution_x = resolution[0]
scene.render.resolution_y = resolution[1]
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

# Lighting
light_data = bpy.data.lights.new("Sun", type="SUN")
light_data.energy = 3.0
light_obj = bpy.data.objects.new("Sun", light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.rotation_euler = (math.radians(50), math.radians(10), math.radians(30))

# Camera
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


@app.function(
    image=blender_image, cpu=2, memory=4096, timeout=300,
    volumes={"/data": volume},
)
def render_mesh_views(
    mesh_bytes: bytes,
    mesh_format: str = "obj",
    num_views: int = 4,
    resolution: tuple[int, int] = (512, 512),
    engine: str = "BLENDER_EEVEE_NEXT",
    elevation_deg: float = 25.0,
    camera_distance: float = 2.5,
) -> list[bytes]:
    """Render multi-view images of a mesh. Returns list of PNG bytes."""
    import json
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        mesh_path = os.path.join(tmp, f"input.{mesh_format}")
        with open(mesh_path, "wb") as f:
            f.write(mesh_bytes)

        output_dir = os.path.join(tmp, "renders")
        script_path = os.path.join(tmp, "render.py")
        with open(script_path, "w") as f:
            f.write(_RENDER_SCRIPT)

        args_json = json.dumps({
            "mesh_path": mesh_path,
            "output_dir": output_dir,
            "num_views": num_views,
            "resolution": list(resolution),
            "elevation_deg": elevation_deg,
            "engine": engine,
            "camera_distance": camera_distance,
        })

        proc = subprocess.run(
            ["blender", "--background", "--python", script_path, "--", args_json],
            capture_output=True, text=True, timeout=240,
        )

        images: list[bytes] = []
        for i in range(num_views):
            img_path = os.path.join(output_dir, f"view_{i}.png")
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    images.append(f.read())

        return images
