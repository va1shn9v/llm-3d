"""
Unit tests for bpy_lib functions.

These tests verify that each function produces valid geometry in Blender 4.2.
Run with:  blender --background --python -m pytest bpy_lib/tests/test_bpy_lib.py
Or if bpy is installed as a Python module:  python -m pytest bpy_lib/tests/test_bpy_lib.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

try:
    import bpy
    HAS_BPY = True
except ImportError:
    HAS_BPY = False

pytestmark = pytest.mark.skipif(not HAS_BPY, reason="bpy not available")


@pytest.fixture(autouse=True)
def clean_scene():
    """Start each test with a clean scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    yield
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _mesh_objects():
    return [o for o in bpy.data.objects if o.type == "MESH"]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class TestPrimitives:
    @pytest.mark.parametrize("ptype", ["cube", "cylinder", "uv_sphere", "cone", "torus"])
    def test_create_primitive(self, ptype):
        from bpy_lib import create_primitive
        obj = create_primitive(
            name=f"test_{ptype}",
            primitive_type=ptype,
            location=[0.1, -0.2, 0.3],
            scale=[0.5, 0.5, 0.5],
            rotation=[1, 0, 0, 0],
        )
        assert obj is not None
        assert obj.name == f"test_{ptype}"
        assert len(_mesh_objects()) == 1

    def test_idempotent(self):
        from bpy_lib import create_primitive
        create_primitive(name="box", primitive_type="cube", location=[0, 0, 0], scale=[1, 1, 1])
        create_primitive(name="box", primitive_type="cube", location=[0, 0, 0], scale=[0.5, 0.5, 0.5])
        assert len(_mesh_objects()) == 1

    @pytest.mark.parametrize("seed", range(5))
    def test_random_params(self, seed):
        import numpy as np
        from bpy_lib import create_primitive

        rng = np.random.default_rng(seed)
        ptype = rng.choice(["cube", "cylinder", "uv_sphere", "cone", "torus"])
        loc = rng.uniform(-1, 1, 3).tolist()
        sc = rng.uniform(0.1, 2, 3).tolist()

        obj = create_primitive(name=f"rnd_{seed}", primitive_type=ptype, location=loc, scale=sc)
        assert obj is not None
        assert obj.data.vertices is not None
        assert len(obj.data.vertices) > 0


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

class TestCurves:
    def test_create_curve(self):
        from bpy_lib import create_curve
        pts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        obj = create_curve(name="test_curve", control_points=pts, is_cyclic=True)
        assert obj is not None

    def test_fill_grid(self):
        from bpy_lib import create_curve, fill_grid
        pts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        create_curve(name="plate", control_points=pts, is_cyclic=True)
        obj = fill_grid(name="plate", thickness=0.05)
        assert obj is not None
        assert obj.type == "MESH"


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------

class TestBoolean:
    def test_union(self):
        from bpy_lib import create_primitive, boolean_op
        create_primitive(name="a", primitive_type="cube", location=[0, 0, 0], scale=[1, 1, 1])
        create_primitive(name="b", primitive_type="uv_sphere", location=[0.5, 0, 0], scale=[0.5, 0.5, 0.5])
        obj = boolean_op(name="result", object_a="a", object_b="b", operation="UNION")
        assert obj is not None
        assert bpy.data.objects.get("b") is None


# ---------------------------------------------------------------------------
# Array
# ---------------------------------------------------------------------------

class TestArray:
    def test_array_1d(self):
        from bpy_lib import create_primitive, create_array_1d
        create_primitive(name="unit", primitive_type="cube", location=[0, 0, 0], scale=[0.1, 0.1, 0.1])
        obj = create_array_1d(name="row", source_object="unit", count=5, offset=[0.3, 0, 0])
        assert obj is not None

    def test_array_2d(self):
        from bpy_lib import create_primitive, create_array_2d
        create_primitive(name="tile", primitive_type="cube", location=[0, 0, 0], scale=[0.1, 0.1, 0.1])
        obj = create_array_2d(
            name="grid", source_object="tile",
            count_u=3, count_v=3,
            offset_u=[0.3, 0, 0], offset_v=[0, 0.3, 0],
        )
        assert obj is not None


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------

class TestModifiers:
    def test_bevel(self):
        from bpy_lib import create_primitive, bevel
        create_primitive(name="box", primitive_type="cube", location=[0, 0, 0], scale=[1, 1, 1])
        bevel(name="box", width=0.05, segments=2)
        obj = bpy.data.objects["box"]
        assert len(obj.data.vertices) > 8

    def test_subdivide(self):
        from bpy_lib import create_primitive, subdivide
        create_primitive(name="box", primitive_type="cube", location=[0, 0, 0], scale=[1, 1, 1])
        subdivide(name="box", levels=1)
        obj = bpy.data.objects["box"]
        assert len(obj.data.vertices) > 8

    def test_mirror(self):
        from bpy_lib import create_primitive, mirror
        create_primitive(name="half", primitive_type="cube", location=[0.5, 0, 0], scale=[0.5, 1, 1])
        mirror(name="half", axis="X")
        obj = bpy.data.objects["half"]
        assert obj is not None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_scene(self):
        from bpy_lib import create_primitive, export_scene

        create_primitive(name="exportable", primitive_type="cube", location=[0, 0, 0], scale=[1, 1, 1])

        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            path = f.name

        try:
            result = export_scene(filepath=path)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)
