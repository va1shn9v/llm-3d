"""
LLM prompt template for image-to-3D code generation.

This file is intentionally kept as a standalone, single-function module so
the prompt is trivially editable without touching any pipeline logic.
"""


_EXAMPLE_CODE = '''\
import bpy
import bmesh
import mathutils

# OBJECT: Simple rectangular table
# PARTS: table top, 4 legs
# PROPORTIONS: top is wide and thin, legs are ~60% of total height
# STRATEGY: cube for top, cylinders for legs, join and normalize

# --- Clear scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# --- Table top ---
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.45))
top = bpy.context.active_object
top.name = "TableTop"
top.scale = (0.5, 0.3, 0.02)
bpy.ops.object.transform_apply(scale=True)

# --- Legs ---
for x, y in [(-0.4, -0.22), (0.4, -0.22), (-0.4, 0.22), (0.4, 0.22)]:
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.44, location=(x, y, 0.22))
    leg = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True)

# --- Join all objects ---
bpy.ops.object.select_all(action='SELECT')
bpy.context.view_layer.objects.active = top
bpy.ops.object.join()

# --- Normalize to unit bounding box ---
obj = bpy.context.active_object
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)
max_dim = max(obj.dimensions)
if max_dim > 0:
    obj.scale = tuple(1.0 / max_dim for _ in range(3))
bpy.ops.object.transform_apply(scale=True, location=True)

# --- Export ---
bpy.ops.object.select_all(action='SELECT')
bpy.ops.wm.obj_export(
    filepath='/tmp/generated_mesh.obj',
    export_selected_objects=True,
    export_uv=False,
    export_materials=False,
)
print("GENERATION_COMPLETE")'''


def build_prompt(num_views: int) -> str:
    """
    Construct the user prompt for the LLM.

    The prompt includes:
    - A planning step (chain-of-thought via comments)
    - A complete working example for the model to follow
    - A common errors section for Blender 4.x pitfalls
    - Strong output format instructions at the end (recency bias)
    """
    return f"""\
Look at these images showing a 3D object from {num_views} viewpoints.
Write a Blender Python script that recreates this object as accurately as possible.

PLAN FIRST — write these as comments at the top of your script:
# OBJECT: [what you see in the images]
# PARTS: [list the 3-5 main structural components]
# PROPORTIONS: [key ratios, e.g., "legs are ~60% of total height"]
# STRATEGY: [which primitives and operations you will use]

SCRIPT REQUIREMENTS:
1. Complete, self-contained Blender Python script using `bpy`.
2. Start by clearing the scene completely.
3. Build the mesh using Blender primitives, modifiers, bmesh, or curves.
   Prefer procedural construction over raw vertex lists.
4. Break the object into logical parts (e.g., seat, legs, backrest for a chair).
5. Join all parts into a single mesh object.
6. Normalize the final mesh:
   - Center at origin (0, 0, 0)
   - Scale so max bounding box extent = 1.0
7. Export to: /tmp/generated_mesh.obj
8. Print "GENERATION_COMPLETE" when done.

CRITICAL — AVOID THESE BLENDER 4.x ERRORS:
- Use `bpy.ops.wm.obj_export()`, NOT `bpy.ops.export_scene.obj()` (removed in Blender 4.x)
- Always call `bpy.ops.object.transform_apply()` BEFORE joining meshes
- Never call `bpy.ops.object.mode_set(mode='EDIT')` without an active selected object
- If using bmesh, always call `bm.to_mesh(mesh)` and `bm.free()` when done
- Make sure at least one object exists and is selected before calling obj_export
- Do NOT use `bpy.data.objects.remove()` inside a loop over `bpy.data.objects`

EXAMPLE — a simple table (adapt the structure for what YOU see):
```python
{_EXAMPLE_CODE}
```

Now write YOUR script for the object shown in the images above.

IMPORTANT: Return ONLY a ```python code block. No explanations before or after.
Start your response with ```python and end with ```."""
