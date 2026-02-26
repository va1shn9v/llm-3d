"""
Assemble fitted parts into complete object code (Section 3.3 of spec).

Takes fitted part codes, transforms them back to their original poses,
sorts parts spatially, and concatenates with semantic comments.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data.part_fitter import FitResult
from data.infinigen_extractor import ExtractedObject, ExtractedPart

log = logging.getLogger(__name__)


@dataclass
class AssembledObject:
    object_id: str
    category: str
    code: str
    num_parts: int
    parts_info: list[dict[str, Any]]
    total_cd: float
    accepted: bool


def _spatial_sort_key(part: ExtractedPart) -> tuple[float, float, float]:
    """Sort key: bottom→top, left→right, front→back (z, x, y)."""
    loc = part.transform.get("location", [0, 0, 0])
    return (loc[2], loc[0], loc[1])


def _round_list(v, n=4):
    return [round(float(x), n) for x in v]


def assemble_object(
    extracted: ExtractedObject,
    fit_results: list[FitResult],
    cd_threshold: float = 5e-3,
) -> AssembledObject | None:
    """Assemble an object from fitted parts.

    Returns None if any part failed to fit (CD > threshold).
    """
    if len(fit_results) != len(extracted.parts):
        log.warning(
            f"{extracted.object_id}: mismatch parts={len(extracted.parts)} "
            f"fits={len(fit_results)}"
        )
        return None

    if not all(fr.accepted for fr in fit_results):
        rejected = [
            (i, fr.chamfer_distance)
            for i, fr in enumerate(fit_results) if not fr.accepted
        ]
        log.debug(f"{extracted.object_id}: {len(rejected)} parts rejected")
        return None

    paired = list(zip(extracted.parts, fit_results))
    paired.sort(key=lambda x: _spatial_sort_key(x[0]))

    lines = [
        "from bpy_lib import *",
        "",
        f"# object name: {extracted.category}",
    ]

    parts_info = []
    for idx, (part, fit) in enumerate(paired, 1):
        loc = _round_list(part.transform.get("location", [0, 0, 0]))
        rot = _round_list(part.transform.get("rotation", [1, 0, 0, 0]))

        code = fit.code.replace("{name}", f"{part.label}_{idx}")
        code = code.replace("{location}", str(loc))
        code = code.replace("{rotation}", str(rot))

        lines.append(f"# part_{idx}: {part.label}")
        lines.append(code)

        parts_info.append({
            "label": part.label,
            "type": fit.part_type,
            "cd": round(fit.chamfer_distance, 6),
        })

    lines.extend(["", "export_scene()"])
    full_code = "\n".join(lines) + "\n"

    total_cd = sum(fr.chamfer_distance for fr in fit_results) / max(len(fit_results), 1)

    return AssembledObject(
        object_id=extracted.object_id,
        category=extracted.category,
        code=full_code,
        num_parts=len(fit_results),
        parts_info=parts_info,
        total_cd=round(total_cd, 6),
        accepted=True,
    )


def assemble_all_objects(
    extracted_objects: list[ExtractedObject],
    all_fits: list[list[FitResult]],
    cd_threshold: float = 5e-3,
) -> list[AssembledObject]:
    """Assemble all objects, filtering those that pass quality gate."""
    assembled = []
    for ext, fits in zip(extracted_objects, all_fits):
        result = assemble_object(ext, fits, cd_threshold)
        if result is not None:
            assembled.append(result)

    log.info(
        f"Assembled {len(assembled)}/{len(extracted_objects)} objects "
        f"({100 * len(assembled) / max(len(extracted_objects), 1):.1f}% acceptance)"
    )
    return assembled
