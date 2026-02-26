"""
Constrained Blender Python API library for LLM-generated 3D code.

Every function is idempotent and error-tolerant. If an object with the given
name already exists, it is overwritten. All coordinates are expected in the
normalized range [-1, 1]^3 but the library will not crash on out-of-range values.

Functions handle Blender's context manager (select/deselect, active object,
mode setting) internally so generated code never needs to worry about it.
"""

from __future__ import annotations

import os
import math
import bpy
import bmesh
from mathutils import Vector, Quaternion


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_object_mode():
    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def _deselect_all():
    _ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")


def _select_only(obj: bpy.types.Object):
    _deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _remove_object(name: str):
    """Remove an existing object by name (mesh data included)."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    _select_only(obj)
    mesh_data = obj.data if obj.type == "MESH" else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh_data and mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)


def _apply_transform(obj: bpy.types.Object, location, scale, rotation):
    """Apply location/scale/rotation to an object.
    rotation is [w, x, y, z] quaternion."""
    obj.location = Vector(location)
    obj.scale = Vector(scale)
    if rotation is not None:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Quaternion(rotation)


def _apply_all_modifiers(obj: bpy.types.Object):
    """Apply all modifiers on an object."""
    _select_only(obj)
    for mod in obj.modifiers:
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. Primitives
# ---------------------------------------------------------------------------

_PRIMITIVE_OPS = {
    "cube": bpy.ops.mesh.primitive_cube_add,
    "cylinder": bpy.ops.mesh.primitive_cylinder_add,
    "uv_sphere": bpy.ops.mesh.primitive_uv_sphere_add,
    "cone": bpy.ops.mesh.primitive_cone_add,
    "torus": bpy.ops.mesh.primitive_torus_add,
}


def create_primitive(
    name: str,
    primitive_type: str,
    location: list[float] = (0, 0, 0),
    scale: list[float] = (1, 1, 1),
    rotation: list[float] | None = None,
) -> bpy.types.Object:
    """Create a basic primitive shape with given transform."""
    _remove_object(name)
    _deselect_all()

    op = _PRIMITIVE_OPS.get(primitive_type.lower())
    if op is None:
        raise ValueError(
            f"Unknown primitive_type '{primitive_type}'. "
            f"Choose from: {list(_PRIMITIVE_OPS.keys())}"
        )
    op()

    obj = bpy.context.active_object
    obj.name = name
    if obj.data:
        obj.data.name = f"{name}_mesh"

    _apply_transform(obj, location, scale, rotation)

    _select_only(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return obj


# ---------------------------------------------------------------------------
# 2. Curves and Surfaces
# ---------------------------------------------------------------------------

_HANDLE_MAP = {0: "AUTO", 1: "VECTOR", 2: "FREE_ALIGN"}


def create_curve(
    name: str,
    control_points: list[list[float]],
    handle_type: list[int] | None = None,
    is_cyclic: bool = True,
) -> bpy.types.Object:
    """Create a Bézier curve from control points."""
    _remove_object(name)
    _deselect_all()

    curve_data = bpy.data.curves.new(f"{name}_curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 12

    spline = curve_data.splines.new("BEZIER")
    n = len(control_points)
    spline.bezier_points.add(n - 1)

    for i, pt in enumerate(control_points):
        bp = spline.bezier_points[i]
        bp.co = Vector(pt[:3])
        ht = _HANDLE_MAP.get(handle_type[i] if handle_type and i < len(handle_type) else 0, "AUTO")
        bp.handle_left_type = ht
        bp.handle_right_type = ht

    spline.use_cyclic_u = is_cyclic

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    _select_only(obj)

    return obj


def fill_grid(
    name: str,
    thickness: float = 0.05,
) -> bpy.types.Object:
    """Fill a closed curve to create a surface, then solidify."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise ValueError(f"Object '{name}' not found for fill_grid")

    _select_only(obj)

    bpy.ops.object.convert(target="MESH")

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.contextual_create(bm, geom=bm.edges[:] + bm.verts[:])
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    mod = obj.modifiers.new("Solidify", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 0
    _select_only(obj)
    bpy.ops.object.modifier_apply(modifier=mod.name)

    return obj


# ---------------------------------------------------------------------------
# 3. Translation (Sweep)
# ---------------------------------------------------------------------------

def _make_section_points_2d(
    section_type: str,
    section_points: list[list[float]],
) -> list[Vector]:
    """Convert section specification to a list of 2D Vectors."""
    if section_type == "circle":
        n = len(section_points) if section_points else 16
        radius = abs(section_points[0][0]) if section_points else 0.1
        return [
            Vector((radius * math.cos(2 * math.pi * i / n),
                    radius * math.sin(2 * math.pi * i / n), 0))
            for i in range(n)
        ]
    return [Vector((p[0], p[1], 0)) for p in section_points]


def create_translation(
    name: str,
    section_points: list[list[float]],
    section_type: str = "polygon",
    trajectory_points: list[list[float]] = ((0, 0, 0), (0, 0, 1)),
    trajectory_type: str = "polyline",
    scale_along: list[float] | None = None,
) -> bpy.types.Object:
    """Sweep a 2D cross-section along a 3D trajectory."""
    _remove_object(name)
    _deselect_all()

    traj_data = bpy.data.curves.new(f"{name}_traj", type="CURVE")
    traj_data.dimensions = "3D"
    spline = traj_data.splines.new("NURBS" if trajectory_type == "bezier" else "POLY")

    n_traj = len(trajectory_points)
    if spline.type == "POLY":
        spline.points.add(n_traj - 1)
        for i, pt in enumerate(trajectory_points):
            spline.points[i].co = Vector((*pt[:3], 1.0))
    else:
        spline.points.add(n_traj - 1)
        for i, pt in enumerate(trajectory_points):
            spline.points[i].co = Vector((*pt[:3], 1.0))

    traj_obj = bpy.data.objects.new(f"{name}_traj_obj", traj_data)
    bpy.context.collection.objects.link(traj_obj)

    sec_data = bpy.data.curves.new(f"{name}_sec", type="CURVE")
    sec_data.dimensions = "3D"
    sec_spline = sec_data.splines.new("POLY")
    sec_pts = _make_section_points_2d(section_type, section_points)
    sec_spline.points.add(len(sec_pts) - 1)
    for i, pt in enumerate(sec_pts):
        sec_spline.points[i].co = Vector((*pt[:3], 1.0))
    sec_spline.use_cyclic_u = True

    sec_obj = bpy.data.objects.new(f"{name}_sec_obj", sec_data)
    bpy.context.collection.objects.link(sec_obj)

    traj_data.bevel_object = sec_obj

    if scale_along and len(scale_along) == n_traj:
        for i, s in enumerate(scale_along):
            if spline.type == "POLY":
                spline.points[i].radius = s
            else:
                spline.points[i].radius = s

    _select_only(traj_obj)
    bpy.ops.object.convert(target="MESH")

    traj_obj.name = name
    if traj_obj.data:
        traj_obj.data.name = f"{name}_mesh"

    sec_cleanup = bpy.data.objects.get(f"{name}_sec_obj")
    if sec_cleanup:
        bpy.data.objects.remove(sec_cleanup, do_unlink=True)

    return traj_obj


# ---------------------------------------------------------------------------
# 4. Bridge Loop
# ---------------------------------------------------------------------------

def create_bridge_loop(
    name: str,
    loop_shapes: list[list[list[float]]],
    positions: list[list[float]],
    rotations: list[list[float]] | None = None,
) -> bpy.types.Object:
    """Connect a sequence of 2D cross-sections to form a 3D surface."""
    _remove_object(name)
    _deselect_all()

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    loops_verts: list[list[bmesh.types.BMVert]] = []

    for loop_idx, (shape, pos) in enumerate(zip(loop_shapes, positions)):
        rot = Quaternion(rotations[loop_idx]) if rotations and loop_idx < len(rotations) else Quaternion()
        offset = Vector(pos[:3])

        loop_verts = []
        for pt in shape:
            co = Vector(pt[:3])
            co.rotate(rot)
            co += offset
            loop_verts.append(bm.verts.new(co))
        loops_verts.append(loop_verts)

    bm.verts.ensure_lookup_table()

    for i in range(len(loops_verts) - 1):
        loop_a = loops_verts[i]
        loop_b = loops_verts[i + 1]
        n = min(len(loop_a), len(loop_b))
        for j in range(n):
            j_next = (j + 1) % n
            try:
                bm.faces.new([loop_a[j], loop_a[j_next], loop_b[j_next], loop_b[j]])
            except ValueError:
                pass

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    return obj


# ---------------------------------------------------------------------------
# 5. Boolean Operations
# ---------------------------------------------------------------------------

def boolean_op(
    name: str,
    object_a: str,
    object_b: str,
    operation: str = "UNION",
) -> bpy.types.Object:
    """Apply boolean operation between two existing objects.
    Result replaces object_a. object_b is removed."""
    obj_a = bpy.data.objects.get(object_a)
    obj_b = bpy.data.objects.get(object_b)
    if obj_a is None:
        raise ValueError(f"Boolean target '{object_a}' not found")
    if obj_b is None:
        raise ValueError(f"Boolean operand '{object_b}' not found")

    _select_only(obj_a)
    mod = obj_a.modifiers.new("Boolean", "BOOLEAN")
    mod.operation = operation.upper()
    mod.object = obj_b
    mod.solver = "FAST"

    bpy.ops.object.modifier_apply(modifier=mod.name)

    _remove_object(object_b)

    obj_a.name = name
    if obj_a.data:
        obj_a.data.name = f"{name}_mesh"

    return obj_a


# ---------------------------------------------------------------------------
# 6. Array (Repetition)
# ---------------------------------------------------------------------------

def create_array_1d(
    name: str,
    source_object: str,
    count: int,
    offset: list[float] = (1, 0, 0),
) -> bpy.types.Object:
    """Repeat an object along a 1D direction."""
    obj = bpy.data.objects.get(source_object)
    if obj is None:
        raise ValueError(f"Source object '{source_object}' not found")

    _select_only(obj)
    mod = obj.modifiers.new("Array1D", "ARRAY")
    mod.count = count
    mod.use_relative_offset = False
    mod.use_constant_offset = True
    mod.constant_offset_displace = Vector(offset[:3])

    bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.name = name
    if obj.data:
        obj.data.name = f"{name}_mesh"

    return obj


def create_array_2d(
    name: str,
    source_object: str,
    count_u: int,
    count_v: int,
    offset_u: list[float] = (1, 0, 0),
    offset_v: list[float] = (0, 1, 0),
) -> bpy.types.Object:
    """Repeat an object in a 2D grid pattern."""
    obj = bpy.data.objects.get(source_object)
    if obj is None:
        raise ValueError(f"Source object '{source_object}' not found")

    _select_only(obj)

    mod_u = obj.modifiers.new("ArrayU", "ARRAY")
    mod_u.count = count_u
    mod_u.use_relative_offset = False
    mod_u.use_constant_offset = True
    mod_u.constant_offset_displace = Vector(offset_u[:3])
    bpy.ops.object.modifier_apply(modifier=mod_u.name)

    mod_v = obj.modifiers.new("ArrayV", "ARRAY")
    mod_v.count = count_v
    mod_v.use_relative_offset = False
    mod_v.use_constant_offset = True
    mod_v.constant_offset_displace = Vector(offset_v[:3])
    bpy.ops.object.modifier_apply(modifier=mod_v.name)

    obj.name = name
    if obj.data:
        obj.data.name = f"{name}_mesh"

    return obj


# ---------------------------------------------------------------------------
# 7. Modifiers
# ---------------------------------------------------------------------------

def bevel(
    name: str,
    width: float = 0.02,
    segments: int = 2,
) -> None:
    """Apply bevel modifier to soften edges."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise ValueError(f"Object '{name}' not found for bevel")

    _select_only(obj)
    mod = obj.modifiers.new("Bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    bpy.ops.object.modifier_apply(modifier=mod.name)


def subdivide(
    name: str,
    levels: int = 1,
) -> None:
    """Apply subdivision surface modifier."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise ValueError(f"Object '{name}' not found for subdivide")

    _select_only(obj)
    mod = obj.modifiers.new("Subsurf", "SUBSURF")
    mod.levels = levels
    mod.render_levels = levels
    bpy.ops.object.modifier_apply(modifier=mod.name)


def mirror(
    name: str,
    axis: str = "X",
) -> None:
    """Apply mirror modifier along axis."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise ValueError(f"Object '{name}' not found for mirror")

    _select_only(obj)
    mod = obj.modifiers.new("Mirror", "MIRROR")
    mod.use_axis[0] = "X" in axis.upper()
    mod.use_axis[1] = "Y" in axis.upper()
    mod.use_axis[2] = "Z" in axis.upper()
    bpy.ops.object.modifier_apply(modifier=mod.name)


# ---------------------------------------------------------------------------
# 8. Mesh Export
# ---------------------------------------------------------------------------

def export_scene(filepath: str | None = None) -> str:
    """Export all mesh objects in the scene to OBJ."""
    _ensure_object_mode()

    if filepath is None:
        filepath = os.environ.get("EXPORT_PATH", "/tmp/generated_mesh.obj")

    _deselect_all()
    mesh_objects = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh objects to export")

    for o in mesh_objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True,
        export_uv=False,
        export_materials=False,
    )

    return filepath
