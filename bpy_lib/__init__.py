"""
bpy_lib — Constrained Blender Python API for LLM-generated 3D code.

This library provides high-level functions wrapping Blender's low-level API,
making the output code shorter, more structured, and easier for an LLM to learn.
Adapted from MeshCoder's approach (arXiv:2508.14879).

Usage in generated code:
    from bpy_lib import *
"""

from bpy_lib.bpy_lib import (
    create_primitive,
    create_curve,
    fill_grid,
    create_translation,
    create_bridge_loop,
    boolean_op,
    create_array_1d,
    create_array_2d,
    bevel,
    subdivide,
    mirror,
    export_scene,
)

__all__ = [
    "create_primitive",
    "create_curve",
    "fill_grid",
    "create_translation",
    "create_bridge_loop",
    "boolean_op",
    "create_array_1d",
    "create_array_2d",
    "bevel",
    "subdivide",
    "mirror",
    "export_scene",
]
