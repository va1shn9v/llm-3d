"""
Build SFT / RL / eval datasets from assembled objects (Section 4 of spec).

Converts assembled objects into Qwen2.5-VL chat format, applies curriculum
ordering, view count augmentation, and train/val/eval splits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import load_config, ProjectConfig
from data.object_assembler import AssembledObject

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

_PART_TYPE_COMPLEXITY = {
    "primitive": 1,
    "translation": 2,
    "bridge_loop": 3,
    "boolean": 3,
    "array": 2,
}


def compute_difficulty(obj: AssembledObject) -> float:
    """Heuristic difficulty: 0.4*num_parts + 0.3*code_tokens + 0.3*max_complexity."""
    num_parts = obj.num_parts
    code_tokens = len(obj.code.split())
    max_complexity = max(
        (_PART_TYPE_COMPLEXITY.get(p["type"], 1) for p in obj.parts_info),
        default=1,
    )

    np_score = min(num_parts / 20.0, 1.0)
    ct_score = min(code_tokens / 500.0, 1.0)
    mc_score = max_complexity / 3.0

    return 0.4 * np_score + 0.3 * ct_score + 0.3 * mc_score


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def format_sft_sample(
    obj: AssembledObject,
    image_paths: list[str],
    system_prompt: str,
) -> dict:
    """Format a single sample as Qwen2.5-VL chat messages."""
    user_content = []
    for img_path in image_paths:
        user_content.append({"type": "image", "image_url": img_path})

    user_content.append({
        "type": "text",
        "text": (
            f"Generate Blender Python code to reconstruct this 3D object. "
            f"The object is a {obj.category}. Use the bpy_lib API functions "
            f"(create_primitive, create_curve, fill_grid, create_translation, "
            f"create_bridge_loop, boolean_op, bevel, create_array_1d, "
            f"create_array_2d). Structure the code with part comments and "
            f"call export_scene() at the end."
        ),
    })

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": obj.code},
        ]
    }


# ---------------------------------------------------------------------------
# View count sampling
# ---------------------------------------------------------------------------

def sample_view_count(
    weights: dict[int, float],
    rng: np.random.Generator,
) -> int:
    """Sample a view count from the weighted distribution."""
    counts = list(weights.keys())
    probs = np.array(list(weights.values()))
    probs /= probs.sum()
    return int(rng.choice(counts, p=probs))


def select_views(
    all_image_paths: dict[str, str],
    num_views: int,
) -> list[str]:
    """Select num_views images with maximum azimuth spread."""
    total = len(all_image_paths)
    if num_views >= total:
        return list(all_image_paths.values())

    indices = sorted(range(total), key=lambda i: i * total // num_views)[:num_views]
    keys = sorted(all_image_paths.keys())
    return [all_image_paths[keys[i]] for i in indices]


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

@dataclass
class DatasetSplit:
    name: str
    samples: list[dict] = field(default_factory=list)


def build_datasets(
    objects: list[AssembledObject],
    images_dir: str,
    cfg: ProjectConfig | None = None,
) -> dict[str, DatasetSplit]:
    """Build all dataset splits from assembled objects.

    Returns dict mapping split name to DatasetSplit.
    """
    if cfg is None:
        cfg = load_config()

    rng = np.random.default_rng(cfg.seed)

    objects_with_difficulty = [(obj, compute_difficulty(obj)) for obj in objects]
    if cfg.dataset.curriculum:
        objects_with_difficulty.sort(key=lambda x: x[1])

    n = len(objects_with_difficulty)
    n_train = int(n * cfg.dataset.sft_train_ratio)
    n_val = int(n * cfg.dataset.sft_val_ratio)

    train_objs = objects_with_difficulty[:n_train]
    val_objs = objects_with_difficulty[n_train:n_train + n_val]
    eval_objs = objects_with_difficulty[n_train + n_val:]

    splits = {}
    for split_name, split_objs in [
        ("sft_train", train_objs),
        ("sft_val", val_objs),
        ("eval_id", eval_objs),
    ]:
        samples = []
        for obj, diff in split_objs:
            img_dir = Path(images_dir) / obj.object_id
            all_images = {}
            for i in range(6):
                p = img_dir / f"view_{i}.png"
                if p.exists():
                    all_images[f"view_{i}"] = str(p)

            if not all_images:
                continue

            nv = sample_view_count(cfg.dataset.view_count_weights, rng)
            selected = select_views(all_images, nv)

            sample = format_sft_sample(obj, selected, cfg.dataset.system_prompt)
            sample["metadata"] = {
                "object_id": obj.object_id,
                "category": obj.category,
                "num_parts": obj.num_parts,
                "total_cd": obj.total_cd,
                "difficulty": round(diff, 4),
                "num_views": nv,
            }
            samples.append(sample)

        splits[split_name] = DatasetSplit(name=split_name, samples=samples)
        log.info(f"  {split_name}: {len(samples)} samples")

    return splits


def save_splits(
    splits: dict[str, DatasetSplit],
    output_dir: str | Path,
):
    """Save dataset splits as JSONL files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split in splits.items():
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for sample in split.samples:
                f.write(json.dumps(sample) + "\n")
        log.info(f"Saved {len(split.samples)} samples to {path}")
