"""
Build SFT / RL / eval datasets from synthetic (caption, raw_bpy_code) pairs.

Converts validated pairs into text-only chat format, applies curriculum
ordering by difficulty, and produces train/val/eval/rl_prompts splits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from config import load_config, ProjectConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

def compute_difficulty(entry: dict) -> float:
    """Heuristic difficulty from code length and mesh complexity.

    Higher = harder. Scores roughly in [0, 1].
    """
    code = entry.get("code", "")
    metrics = entry.get("metrics", {})

    code_tokens = len(code.split())
    ct_score = min(code_tokens / 500.0, 1.0)

    vertex_count = metrics.get("vertex_count", 0)
    face_count = metrics.get("face_count", 0)
    vc_score = min(vertex_count / 50_000.0, 1.0)
    fc_score = min(face_count / 20_000.0, 1.0)

    return 0.4 * ct_score + 0.3 * vc_score + 0.3 * fc_score


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def format_sft_sample(
    caption: str,
    code: str,
    system_prompt: str,
) -> dict:
    """Format a single sample as text-only chat messages for SFT."""
    user_text = (
        f"Create a 3D model of: {caption}. "
        f"Write a complete Blender Python script using the bpy module."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": code},
        ]
    }


def format_rl_prompt(
    caption: str,
    system_prompt: str,
) -> dict:
    """Format a prompt for RL (no assistant turn — model generates that)."""
    user_text = (
        f"Create a 3D model of: {caption}. "
        f"Write a complete Blender Python script using the bpy module."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
    }


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

@dataclass
class DatasetSplit:
    name: str
    samples: list[dict] = field(default_factory=list)


def build_datasets(
    synthetic_jsonl_path: str | Path,
    manifest_jsonl_path: str | Path | None = None,
    cfg: ProjectConfig | None = None,
) -> dict[str, DatasetSplit]:
    """Build all dataset splits from synthetic SFT data.

    Reads the validated (caption, code) pairs, computes difficulty, applies
    curriculum ordering, and splits into sft_train / sft_val / eval_id.

    Also builds an rl_prompts split from the full manifest (including captions
    where the teacher failed — those benefit most from RL exploration).
    """
    if cfg is None:
        cfg = load_config()

    entries = []
    with open(synthetic_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    entries_with_diff = [(e, compute_difficulty(e)) for e in entries]
    if cfg.dataset.curriculum:
        entries_with_diff.sort(key=lambda x: x[1])

    n = len(entries_with_diff)
    n_train = int(n * cfg.dataset.sft_train_ratio)
    n_val = int(n * cfg.dataset.sft_val_ratio)

    train_entries = entries_with_diff[:n_train]
    val_entries = entries_with_diff[n_train:n_train + n_val]
    eval_entries = entries_with_diff[n_train + n_val:]

    system_prompt = cfg.dataset.system_prompt
    splits: dict[str, DatasetSplit] = {}

    for split_name, split_entries in [
        ("sft_train", train_entries),
        ("sft_val", val_entries),
        ("eval_id", eval_entries),
    ]:
        samples = []
        for entry, diff in split_entries:
            sample = format_sft_sample(entry["caption"], entry["code"], system_prompt)
            sample["metadata"] = {
                "object_id": entry.get("uid", ""),
                "caption": entry["caption"],
                "difficulty": round(diff, 4),
                "gt_mesh_path": entry.get("gt_mesh_path", ""),
            }
            samples.append(sample)

        splits[split_name] = DatasetSplit(name=split_name, samples=samples)
        log.info(f"  {split_name}: {len(samples)} samples")

    if manifest_jsonl_path:
        rl_samples = []
        with open(manifest_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prompt = format_rl_prompt(entry["caption"], system_prompt)
                prompt["metadata"] = {
                    "object_id": entry.get("uid", ""),
                    "caption": entry["caption"],
                    "gt_mesh_path": entry.get("mesh_path", ""),
                }
                rl_samples.append(prompt)

        splits["rl_prompts"] = DatasetSplit(name="rl_prompts", samples=rl_samples)
        log.info(f"  rl_prompts: {len(rl_samples)} prompts (full manifest)")

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
