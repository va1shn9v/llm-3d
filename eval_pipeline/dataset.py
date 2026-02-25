"""
Download 3D objects from Objaverse by LVIS category.
"""

import sys
import logging
from typing import Dict

import numpy as np

from eval_pipeline.config import Config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD OBJECTS FROM OBJAVERSE
# ══════════════════════════════════════════════════════════════════════════════

def download_objects(config: Config) -> Dict[str, str]:
    """
    Download a subset of Objaverse objects for a given LVIS category.

    Returns:
        Dict mapping UID -> local file path (.glb)
    """
    import objaverse

    log.info(f"Loading LVIS annotations to find '{config.category}' objects...")
    lvis_annotations = objaverse.load_lvis_annotations()

    # ── Validate category exists ─────────────────────────────────────────
    available_categories = sorted(lvis_annotations.keys())
    if config.category not in lvis_annotations:
        # Try case-insensitive match
        match = [c for c in available_categories if c.lower() == config.category.lower()]
        if match:
            config.category = match[0]
        else:
            log.error(
                f"Category '{config.category}' not found in LVIS annotations.\n"
                f"Available categories (showing first 50):\n"
                f"  {', '.join(available_categories[:50])}\n"
                f"  ... ({len(available_categories)} total)"
            )
            sys.exit(1)

    category_uids = lvis_annotations[config.category]
    log.info(
        f"Found {len(category_uids)} objects in category '{config.category}'. "
        f"Downloading {min(config.max_objects, len(category_uids))}..."
    )

    # ── Select subset ────────────────────────────────────────────────────
    # Use a fixed seed for reproducibility across runs
    rng = np.random.RandomState(42)
    selected_uids = list(rng.choice(
        category_uids,
        size=min(config.max_objects, len(category_uids)),
        replace=False,
    ))

    # ── Download ─────────────────────────────────────────────────────────
    objects = objaverse.load_objects(
        uids=selected_uids,
        download_processes=config.download_processes,
    )

    log.info(f"Downloaded {len(objects)} objects successfully.")
    return objects
