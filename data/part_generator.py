"""
Synthetic part dataset generator (Section 2 of spec).

Generates random parametric parts using the bpy_lib API, executes them in
Blender (via Modal) to get meshes, renders multi-view images, and stores
(rendered_images, code) pairs for SFT training.

Part types: primitive, translation, bridge_loop, boolean, array.
Total target: ~300K part-code pairs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PartSample:
    id: str
    part_type: str
    code: str
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Random parameter samplers
# ---------------------------------------------------------------------------

def _random_quaternion(rng: np.random.Generator) -> list[float]:
    """Uniform random unit quaternion [w, x, y, z]."""
    u = rng.uniform(0, 1, 3)
    q = [
        math.sqrt(1 - u[0]) * math.sin(2 * math.pi * u[1]),
        math.sqrt(1 - u[0]) * math.cos(2 * math.pi * u[1]),
        math.sqrt(u[0]) * math.sin(2 * math.pi * u[2]),
        math.sqrt(u[0]) * math.cos(2 * math.pi * u[2]),
    ]
    return [q[2], q[3], q[0], q[1]]  # reorder to [w, x, y, z]


def _round(v, n=4):
    if isinstance(v, (list, tuple)):
        return [round(float(x), n) for x in v]
    return round(float(v), n)


# ---------------------------------------------------------------------------
# Primitive parts
# ---------------------------------------------------------------------------

def sample_primitive(rng: np.random.Generator, idx: int) -> PartSample:
    ptype = rng.choice(["cube", "cylinder", "uv_sphere", "cone", "torus"])
    log_scale = rng.uniform(-2, 2, 3)
    scale = (10 ** log_scale).tolist()
    rotation = _random_quaternion(rng)

    max_extent = max(scale)
    target = rng.uniform(1.0, 2.0)
    scale = [s * target / max_extent for s in scale]

    half_extents = [s / 2 for s in scale]
    location = [
        float(rng.uniform(-1 + he, 1 - he)) for he in half_extents
    ]

    code = (
        f"from bpy_lib import *\n\n"
        f"create_primitive(\n"
        f"    name='part',\n"
        f"    primitive_type='{ptype}',\n"
        f"    location={_round(location)},\n"
        f"    scale={_round(scale)},\n"
        f"    rotation={_round(rotation)},\n"
        f")\n\n"
        f"export_scene()\n"
    )

    return PartSample(
        id=f"part_prim_{idx:06d}",
        part_type="primitive",
        code=code,
        params={"primitive_type": ptype, "location": location, "scale": scale, "rotation": rotation},
    )


# ---------------------------------------------------------------------------
# Translation parts
# ---------------------------------------------------------------------------

def _sample_section(rng, stype):
    if stype == "rectangle":
        w, h = rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5)
        return [[w, h], [-w, h], [-w, -h], [w, -h]]
    elif stype == "circle":
        r = rng.uniform(0.05, 0.3)
        return [[r, 0]] * 16
    elif stype == "polygon":
        n = int(rng.integers(3, 9))
        r = rng.uniform(0.05, 0.3)
        return [[r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)] for i in range(n)]
    else:
        n = int(rng.integers(4, 9))
        return rng.uniform(-0.3, 0.3, (n, 2)).tolist()


def _sample_trajectory(rng, ttype):
    if ttype == "line":
        return [[0, 0, 0], [0, 0, rng.uniform(0.3, 1.0)]]
    elif ttype == "polyline":
        n = int(rng.integers(3, 7))
        pts = [[0, 0, 0]]
        for _ in range(n - 1):
            last = pts[-1]
            pts.append([last[0] + rng.uniform(-0.2, 0.2),
                        last[1] + rng.uniform(-0.2, 0.2),
                        last[2] + rng.uniform(0.1, 0.3)])
        return pts
    else:
        n = int(rng.integers(4, 8))
        return rng.uniform(-0.5, 0.5, (n, 3)).tolist()


def sample_translation(rng: np.random.Generator, idx: int) -> PartSample:
    stype = rng.choice(["rectangle", "circle", "polygon", "bezier"])
    ttype = rng.choice(["line", "polyline", "bezier"])
    section = _sample_section(rng, stype)
    trajectory = _sample_trajectory(rng, ttype)

    use_scale = rng.random() > 0.5
    scale_along = None
    if use_scale:
        scale_along = rng.uniform(0.5, 1.5, len(trajectory)).tolist()

    code = (
        f"from bpy_lib import *\n\n"
        f"create_translation(\n"
        f"    name='part',\n"
        f"    section_points={_round(section)},\n"
        f"    section_type='{stype}',\n"
        f"    trajectory_points={_round(trajectory)},\n"
        f"    trajectory_type='{ttype}',\n"
    )
    if scale_along:
        code += f"    scale_along={_round(scale_along)},\n"
    code += ")\n\nexport_scene()\n"

    return PartSample(
        id=f"part_trans_{idx:06d}",
        part_type="translation",
        code=code,
        params={"section_type": stype, "trajectory_type": ttype},
    )


# ---------------------------------------------------------------------------
# Bridge loop parts
# ---------------------------------------------------------------------------

def sample_bridge_loop(rng: np.random.Generator, idx: int) -> PartSample:
    num_loops = int(rng.integers(2, 6))
    loops = []
    positions = []
    z = 0.0
    for _ in range(num_loops):
        n = int(rng.integers(4, 9))
        r = rng.uniform(0.05, 0.3)
        shape = [[r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0] for i in range(n)]
        loops.append(shape)
        positions.append([0.0, 0.0, z])
        z += rng.uniform(0.1, 0.5)

    code = (
        f"from bpy_lib import *\n\n"
        f"create_bridge_loop(\n"
        f"    name='part',\n"
        f"    loop_shapes={_round(loops)},\n"
        f"    positions={_round(positions)},\n"
        f")\n\n"
        f"export_scene()\n"
    )

    return PartSample(
        id=f"part_bridge_{idx:06d}",
        part_type="bridge_loop",
        code=code,
        params={"num_loops": num_loops},
    )


# ---------------------------------------------------------------------------
# Boolean parts
# ---------------------------------------------------------------------------

def sample_boolean(rng: np.random.Generator, idx: int) -> PartSample:
    op = rng.choice(["UNION", "INTERSECT", "DIFFERENCE"])
    ptypes = [rng.choice(["cube", "cylinder", "uv_sphere"]) for _ in range(2)]

    parts_code = []
    for i, pt in enumerate(ptypes):
        loc = rng.uniform(-0.3, 0.3, 3).tolist()
        sc = rng.uniform(0.2, 0.6, 3).tolist()
        rot = _random_quaternion(rng)
        parts_code.append(
            f"create_primitive(\n"
            f"    name='obj_{i}',\n"
            f"    primitive_type='{pt}',\n"
            f"    location={_round(loc)},\n"
            f"    scale={_round(sc)},\n"
            f"    rotation={_round(rot)},\n"
            f")"
        )

    code = (
        f"from bpy_lib import *\n\n"
        f"{parts_code[0]}\n"
        f"{parts_code[1]}\n"
        f"boolean_op(name='part', object_a='obj_0', object_b='obj_1', operation='{op}')\n\n"
        f"export_scene()\n"
    )

    return PartSample(
        id=f"part_bool_{idx:06d}",
        part_type="boolean",
        code=code,
        params={"operation": op},
    )


# ---------------------------------------------------------------------------
# Array parts
# ---------------------------------------------------------------------------

def sample_array(rng: np.random.Generator, idx: int) -> PartSample:
    is_2d = rng.random() > 0.5
    ptype = rng.choice(["cube", "cylinder", "uv_sphere"])
    sc = rng.uniform(0.05, 0.15, 3).tolist()
    rot = _random_quaternion(rng)

    base_code = (
        f"create_primitive(\n"
        f"    name='unit',\n"
        f"    primitive_type='{ptype}',\n"
        f"    location=[0, 0, 0],\n"
        f"    scale={_round(sc)},\n"
        f"    rotation={_round(rot)},\n"
        f")"
    )

    if is_2d:
        cu, cv = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        ou = rng.uniform(0.1, 0.4, 3).tolist()
        ov = rng.uniform(0.1, 0.4, 3).tolist()
        array_code = (
            f"create_array_2d(\n"
            f"    name='part', source_object='unit',\n"
            f"    count_u={cu}, count_v={cv},\n"
            f"    offset_u={_round(ou)}, offset_v={_round(ov)},\n"
            f")"
        )
    else:
        cnt = int(rng.integers(2, 9))
        off = rng.uniform(0.1, 0.4, 3).tolist()
        array_code = (
            f"create_array_1d(\n"
            f"    name='part', source_object='unit',\n"
            f"    count={cnt}, offset={_round(off)},\n"
            f")"
        )

    code = f"from bpy_lib import *\n\n{base_code}\n{array_code}\n\nexport_scene()\n"

    return PartSample(
        id=f"part_arr_{idx:06d}",
        part_type="array",
        code=code,
        params={"is_2d": is_2d},
    )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

_SAMPLERS = {
    "primitive": sample_primitive,
    "translation": sample_translation,
    "bridge_loop": sample_bridge_loop,
    "boolean": sample_boolean,
    "array": sample_array,
}


def generate_part_batch(
    part_type: str,
    count: int,
    seed: int = 42,
    start_idx: int = 0,
) -> list[PartSample]:
    """Generate a batch of part samples of a given type."""
    rng = np.random.default_rng(seed)
    sampler = _SAMPLERS[part_type]
    return [sampler(rng, start_idx + i) for i in range(count)]


def generate_all_parts(
    counts: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, list[PartSample]]:
    """Generate all part types with given counts."""
    from config import load_config
    cfg = load_config()

    if counts is None:
        counts = {
            "primitive": cfg.part_generator.num_primitives,
            "translation": cfg.part_generator.num_translations,
            "bridge_loop": cfg.part_generator.num_bridge_loops,
            "boolean": cfg.part_generator.num_booleans,
            "array": cfg.part_generator.num_arrays,
        }

    all_parts = {}
    for ptype, n in counts.items():
        all_parts[ptype] = generate_part_batch(ptype, n, seed=seed)

    return all_parts


def parts_to_jsonl(parts: list[PartSample], output_path: str | Path):
    """Write parts to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in parts:
            record = {
                "id": p.id,
                "part_type": p.part_type,
                "code": p.code,
                "metadata": p.params,
            }
            f.write(json.dumps(record) + "\n")
