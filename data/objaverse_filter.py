"""
Objaverse++ quality filtering (Phase 0 of pipeline).

Downloads Objaverse++ quality annotations from HuggingFace, filters by quality
tier and trait flags, outputs a clean UID list for downstream processing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import ProjectConfig, load_config

log = logging.getLogger(__name__)


def filter_objaverse_uids(cfg: ProjectConfig | None = None) -> list[str]:
    """Filter Objaverse 1.0 UIDs using Objaverse++ quality scores.

    Returns list of UIDs that pass quality thresholds.
    """
    if cfg is None:
        cfg = load_config()

    from datasets import load_dataset

    log.info("Loading Objaverse++ annotations from HuggingFace...")
    ds = load_dataset("cindyxl/ObjaversePlusPlus", split="train")

    filter_cfg = cfg.objaverse_filter
    allowed_tiers = set(filter_cfg.quality_tiers)
    accepted_uids: list[str] = []

    for row in ds:
        quality = row.get("quality_tier") or row.get("quality", "")
        if quality not in allowed_tiers:
            continue

        if filter_cfg.exclude_scenes and row.get("is_scene", False):
            continue

        if filter_cfg.exclude_transparent and row.get("is_transparent", False):
            continue

        uid = row.get("uid") or row.get("object_uid", "")
        if uid:
            accepted_uids.append(uid)

    if filter_cfg.max_uids and len(accepted_uids) > filter_cfg.max_uids:
        import random
        random.seed(cfg.seed)
        random.shuffle(accepted_uids)
        accepted_uids = accepted_uids[: filter_cfg.max_uids]

    log.info(f"Filtered to {len(accepted_uids)} UIDs from {len(ds)} total")

    output_path = Path(filter_cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(accepted_uids, f)
    log.info(f"Saved filtered UIDs to {output_path}")

    return accepted_uids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filter_objaverse_uids()
