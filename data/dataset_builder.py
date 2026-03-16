"""
Build SFT / RL / eval datasets from synthetic (caption, raw_bpy_code) pairs.

Converts validated pairs into text-only chat format, applies curriculum
ordering by difficulty, and produces train/val/eval/rl splits.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import load_config, ProjectConfig, StorageConfig
from data.storage import open_read

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "as", "at", "base", "for", "from", "in", "is", "it",
    "its", "of", "on", "or", "the", "to", "with",
    "bunch", "group", "pair", "pile", "set",
}
_TAIL_BREAK_TOKENS = {
    "along", "at", "featuring", "from", "in", "mounted", "near", "on",
    "over", "under", "with",
}


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

def _make_user_text(caption: str) -> str:
    return (
        f"Create a 3D model of: {caption}\n\n"
        f"Decompose the object into its main parts, build each with appropriate "
        f"Blender constructs (primitives, BMesh, curves, modifiers), add materials, "
        f"and export to OBJ. Write a complete Blender Python script."
    )


def format_sft_sample(
    caption: str,
    code: str,
    system_prompt: str,
) -> dict:
    """Format a single sample as text-only chat messages for SFT."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _make_user_text(caption)},
            {"role": "assistant", "content": code},
        ]
    }


def format_rl_prompt(
    caption: str,
    system_prompt: str,
) -> dict:
    """Format a prompt for RL (no assistant turn — model generates that)."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _make_user_text(caption)},
        ]
    }


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

@dataclass
class DatasetSplit:
    name: str
    samples: list[dict] = field(default_factory=list)


def _load_jsonl(path: str | Path, storage_cfg: StorageConfig | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        f = open_read(str(path), storage_cfg)
    except FileNotFoundError:
        log.warning("JSONL input not found: %s", path)
        return []

    with f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def _caption_tokens(caption: str) -> list[str]:
    return [_normalize_token(tok) for tok in _TOKEN_RE.findall(caption.lower())]


def _extract_headword(caption: str) -> str:
    tokens = _caption_tokens(caption)
    head_segment = tokens
    for idx, token in enumerate(tokens):
        if idx > 0 and token in _TAIL_BREAK_TOKENS:
            head_segment = tokens[:idx]
            break

    for token in reversed(head_segment):
        if token not in _STOPWORDS:
            return token

    for token in reversed(tokens):
        if token not in _STOPWORDS:
            return token
    return "unknown"


def _format_prompt_sample(
    caption: str,
    system_prompt: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    sample = format_rl_prompt(caption, system_prompt)
    sample["metadata"] = metadata
    return sample


def _ratio_to_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    count = int(round(total * ratio))
    if count == 0:
        return 1
    return min(count, total)


def _choose_ood_uids(
    manifest: list[dict[str, Any]],
    target_count: int,
    cfg: ProjectConfig,
    rng: random.Random,
) -> tuple[set[str], list[str]]:
    if target_count <= 0 or not manifest:
        return set(), []

    configured_terms = {
        _normalize_token(term)
        for term in cfg.eval.unseen_categories
        if term
    }

    ood_uids: set[str] = set()
    selected_categories: list[str] = []

    for entry in manifest:
        caption_terms = set(_caption_tokens(entry["caption"]))
        if configured_terms & caption_terms:
            ood_uids.add(entry["uid"])

    for term in sorted(configured_terms):
        if any(term in set(_caption_tokens(entry["caption"])) for entry in manifest):
            selected_categories.append(term)

    if len(ood_uids) >= target_count:
        return ood_uids, selected_categories

    category_to_uids: dict[str, list[str]] = {}
    for entry in manifest:
        category = _extract_headword(entry["caption"])
        category_to_uids.setdefault(category, []).append(entry["uid"])

    candidates = sorted(
        category_to_uids.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )

    for category, uids in candidates:
        if len(ood_uids) >= target_count:
            break
        if category in selected_categories:
            continue

        remaining = target_count - len(ood_uids)
        if selected_categories and len(uids) > max(remaining * 2, remaining + 25):
            continue

        selected_categories.append(category)
        ood_uids.update(uids)

    if len(ood_uids) < target_count:
        remaining = [entry["uid"] for entry in manifest if entry["uid"] not in ood_uids]
        rng.shuffle(remaining)
        ood_uids.update(remaining[: target_count - len(ood_uids)])

    return ood_uids, selected_categories


def _build_object_splits(
    manifest: list[dict[str, Any]],
    cfg: ProjectConfig,
) -> tuple[dict[str, set[str]], list[str]]:
    rng = random.Random(cfg.seed)
    entries = list(manifest)
    rng.shuffle(entries)

    total = len(entries)
    target_ood = _ratio_to_count(total, cfg.dataset.eval_ood_ratio)
    ood_uids, ood_categories = _choose_ood_uids(entries, target_ood, cfg, rng)

    remaining = [entry for entry in entries if entry["uid"] not in ood_uids]
    target_eval_id = min(
        _ratio_to_count(total, cfg.dataset.eval_id_ratio),
        len(remaining),
    )
    eval_id_uids = {entry["uid"] for entry in remaining[:target_eval_id]}

    remaining = [entry for entry in remaining if entry["uid"] not in eval_id_uids]
    target_val = min(
        _ratio_to_count(total, cfg.dataset.sft_val_ratio),
        len(remaining),
    )
    val_uids = {entry["uid"] for entry in remaining[:target_val]}

    train_uids = {
        entry["uid"]
        for entry in remaining
        if entry["uid"] not in val_uids
    }

    splits = {
        "train": train_uids,
        "val": val_uids,
        "eval_id": eval_id_uids,
        "eval_ood": ood_uids,
    }

    overlap = sum(len(a & b) for i, a in enumerate(splits.values()) for b in list(splits.values())[i + 1 :])
    if overlap:
        raise ValueError("Object splits are not disjoint")

    accounted = sum(len(uids) for uids in splits.values())
    if accounted != total:
        raise ValueError(f"Object split accounting mismatch: {accounted} != {total}")

    return splits, ood_categories


def build_datasets(
    synthetic_jsonl_path: str | Path,
    manifest_jsonl_path: str | Path | None = None,
    cfg: ProjectConfig | None = None,
) -> dict[str, DatasetSplit]:
    """Build all dataset splits from synthetic SFT data.

    Builds object-level splits first from the manifest to avoid leakage across
    SFT, RL, and evaluation. Then:
    - SFT train/val are populated from validated synthetic pairs for train/val objects
    - eval_id/eval_ood are prompt-only held-out sets backed by GT meshes
    - rl_prompts only contains train objects (no eval leakage)
    """
    if cfg is None:
        cfg = load_config()

    synthetic_entries = _load_jsonl(synthetic_jsonl_path, cfg.storage)
    manifest_entries = _load_jsonl(manifest_jsonl_path, cfg.storage) if manifest_jsonl_path else []

    manifest_by_uid = {entry["uid"]: entry for entry in manifest_entries if entry.get("uid")}
    system_prompt = cfg.dataset.system_prompt
    splits: dict[str, DatasetSplit] = {}

    if manifest_entries:
        object_splits, ood_categories = _build_object_splits(manifest_entries, cfg)
        log.info(
            "Object splits: train=%d val=%d eval_id=%d eval_ood=%d",
            len(object_splits["train"]),
            len(object_splits["val"]),
            len(object_splits["eval_id"]),
            len(object_splits["eval_ood"]),
        )
        if ood_categories:
            log.info("OOD holdout categories: %s", ", ".join(ood_categories))
    else:
        all_uids = {
            entry.get("uid", "")
            for entry in synthetic_entries
            if entry.get("uid")
        }
        object_splits = {
            "train": set(all_uids),
            "val": set(),
            "eval_id": set(),
            "eval_ood": set(),
        }
        ood_categories = []
        log.warning("Manifest unavailable; using synthetic-only fallback with no held-out eval sets")

    entries_with_diff = [
        (entry, compute_difficulty(entry))
        for entry in synthetic_entries
        if entry.get("uid") in object_splits["train"] | object_splits["val"]
    ]
    if cfg.dataset.curriculum:
        entries_with_diff.sort(key=lambda x: x[1])

    train_entries = [
        (entry, diff)
        for entry, diff in entries_with_diff
        if entry.get("uid") in object_splits["train"]
    ]
    val_entries = [
        (entry, diff)
        for entry, diff in entries_with_diff
        if entry.get("uid") in object_splits["val"]
    ]

    for split_name, split_entries in [
        ("sft_train", train_entries),
        ("sft_val", val_entries),
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

    def build_prompt_split(name: str, uid_set: set[str], split_type: str) -> None:
        samples = []
        for uid in sorted(uid_set):
            entry = manifest_by_uid.get(uid)
            if entry is None:
                continue
            category = _extract_headword(entry["caption"])
            metadata = {
                "object_id": entry.get("uid", ""),
                "caption": entry["caption"],
                "gt_mesh_path": entry.get("mesh_path", ""),
                "split": split_type,
                "category": category,
                "ood_category": category if split_type == "eval_ood" else "",
            }
            samples.append(_format_prompt_sample(entry["caption"], system_prompt, metadata))

        splits[name] = DatasetSplit(name=name, samples=samples)
        log.info("  %s: %d samples", name, len(samples))

    if manifest_entries:
        build_prompt_split("eval_id", object_splits["eval_id"], "eval_id")
        build_prompt_split("eval_ood", object_splits["eval_ood"], "eval_ood")
        build_prompt_split("rl_prompts", object_splits["train"], "train")
        build_prompt_split("rl_val_prompts", object_splits["val"], "val")

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
