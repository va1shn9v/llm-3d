"""
Objaverse 1.0 quality filtering (Phase 0 of pipeline).

Uses the objaverse package's LVIS annotations (human-verified category labels)
as a quality signal, then applies lightweight metadata filters (face count,
vertex count, animation exclusion).  Outputs a clean UID list for downstream
caption joining and mesh download.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from config import ProjectConfig, load_config

log = logging.getLogger(__name__)


def filter_objaverse_uids(cfg: ProjectConfig | None = None) -> list[str]:
    """Filter Objaverse 1.0 UIDs via LVIS annotations + metadata checks.

    Returns list of UIDs that pass quality thresholds.
    """
    if cfg is None:
        cfg = load_config()

    import objaverse

    filter_cfg = cfg.objaverse_filter

    # LVIS-annotated objects are human-verified to be recognisable real-world
    # objects — a strong quality proxy for Objaverse 1.0.
    log.info("Loading Objaverse LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()

    all_lvis_uids: set[str] = set()
    for uids in lvis.values():
        all_lvis_uids.update(uids)
    log.info("Found %d LVIS-annotated UIDs across %d categories", len(all_lvis_uids), len(lvis))

    candidate_uids = list(all_lvis_uids)

    needs_metadata = (
        filter_cfg.min_face_count > 0
        or filter_cfg.max_vertex_count > 0
        or filter_cfg.exclude_animated
    )
    if needs_metadata:
        log.info("Loading Objaverse annotations for metadata filtering...")
        annotations = objaverse.load_annotations(candidate_uids)

        filtered: list[str] = []
        for uid in candidate_uids:
            ann = annotations.get(uid, {})

            face_count = ann.get("faceCount", 0) or 0
            vertex_count = ann.get("vertexCount", 0) or 0
            anim_count = ann.get("animationCount", 0) or 0

            if filter_cfg.min_face_count and face_count < filter_cfg.min_face_count:
                continue
            if filter_cfg.max_vertex_count and vertex_count > filter_cfg.max_vertex_count:
                continue
            if filter_cfg.exclude_animated and anim_count > 0:
                continue

            filtered.append(uid)

        log.info(
            "Metadata filtering: %d -> %d UIDs", len(candidate_uids), len(filtered),
        )
        candidate_uids = filtered

    if filter_cfg.max_uids and len(candidate_uids) > filter_cfg.max_uids:
        random.seed(cfg.seed)
        random.shuffle(candidate_uids)
        candidate_uids = candidate_uids[: filter_cfg.max_uids]

    log.info("Final filtered set: %d UIDs", len(candidate_uids))

    output_path = Path(filter_cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidate_uids, f)
    log.info("Saved filtered UIDs to %s", output_path)

    return candidate_uids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filter_objaverse_uids()
