"""
Shared prompt templates for Blender code generation.
"""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Blender Python developer targeting Blender 4.2. "
    "Given a text description of a 3D object, write a complete bpy script that "
    "creates the described geometry, applies sensible materials, and exports the "
    "result to OBJ.\n\n"
    "Requirements:\n"
    "1. Start with `import bpy, os, math, bmesh`\n"
    "2. Clear the default scene\n"
    "3. Build the object using primitives, BMesh, curves, or modifiers\n"
    "4. Export with `bpy.ops.wm.obj_export(filepath=os.environ['EXPORT_PATH'], "
    "export_selected_objects=True, export_materials=False, apply_modifiers=True)`\n\n"
    "Blender 4.2 rules:\n"
    "- Use `bpy.ops.wm.obj_export`, not `bpy.ops.export_scene.obj`\n"
    "- Do not use `obj.data.use_auto_smooth`\n"
    "- Use `BLENDER_EEVEE_NEXT`, not `BLENDER_EEVEE`\n"
    "- Use `Specular IOR Level`, not `Specular`\n"
    "- Re-acquire objects after join/remove operations\n"
    "- Free BMesh objects after use\n\n"
    "Output only Python code."
)


def format_user_prompt(caption: str) -> str:
    """Render the user-facing prompt for a single object caption."""
    return (
        f"Create a 3D model of: {caption}\n\n"
        "Decompose the object into its main parts, build each with appropriate "
        "Blender constructs, add materials, and export to OBJ. "
        "Write a complete Blender Python script."
    )
