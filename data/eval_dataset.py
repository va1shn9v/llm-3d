"""
Build a curated evaluation dataset from an existing manifest.

The selection logic is intentionally simple and conservative:
- prefer single-object captions over scenes and stylized assets
- spread picks across inferred categories
- cap the number of examples per category for diversity
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import ProjectConfig, load_config
from data.storage import open_read, resolve_manifest_path
from prompts import format_user_prompt

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "as", "at", "for", "from", "in", "is", "it", "its",
    "of", "on", "or", "the", "to", "with",
}
_GENERIC_HEADWORDS = {
    "base", "body", "design", "figure", "item", "object", "part", "scene",
    "shape", "side", "structure", "surface", "thing", "top",
}
_BREAK_TOKENS = {
    "along", "at", "featuring", "from", "in", "mounted", "near", "on",
    "over", "under", "with",
}
_REJECT_PATTERNS = (
    re.compile(r"\b(cartoon|illustration|stylized|figurine|character|mascot)\b", re.I),
    re.compile(r"\b(logo|label|warning|sign|text|multilingual)\b", re.I),
    re.compile(r"\b(collection|group|set|pair|row|several|multiple)\b", re.I),
)
_PENALTY_PATTERNS = (
    re.compile(r"\b(attached|including|mounted|surrounded)\b", re.I),
    re.compile(r"\b(background|scene|street|wall|floor)\b", re.I),
)


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


def extract_category(caption: str) -> str:
    """Infer a rough category token from a caption."""
    tokens = _caption_tokens(caption)
    head_segment = tokens
    for idx, token in enumerate(tokens):
        if idx > 0 and token in _BREAK_TOKENS:
            head_segment = tokens[:idx]
            break

    for token in reversed(head_segment):
        if token not in _STOPWORDS and token not in _GENERIC_HEADWORDS:
            return token

    for token in reversed(tokens):
        if token not in _STOPWORDS and token not in _GENERIC_HEADWORDS:
            return token

    return ""


def _is_candidate_caption(caption: str) -> bool:
    tokens = _caption_tokens(caption)
    if len(tokens) < 2 or len(tokens) > 24:
        return False
    if any(pattern.search(caption) for pattern in _REJECT_PATTERNS):
        return False
    return True


def _caption_penalty(caption: str) -> int:
    penalty = 0
    lowered = caption.lower()
    penalty += caption.count(",") * 2
    penalty += max(len(_caption_tokens(caption)) - 12, 0)
    if " and " in lowered:
        penalty += 3
    for pattern in _PENALTY_PATTERNS:
        if pattern.search(caption):
            penalty += 4
    return penalty


def _load_manifest(path: str | Path, cfg: ProjectConfig) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with open_read(str(path), cfg.storage) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_eval_dataset(
    cfg: ProjectConfig | None = None,
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Select a diverse prompt-only evaluation set from the manifest."""
    if cfg is None:
        cfg = load_config()

    selection = cfg.eval.selection
    manifest_path = manifest_path or selection.manifest_path or resolve_manifest_path(cfg.storage)
    output_path = output_path or selection.output_path

    manifest = _load_manifest(manifest_path, cfg)
    candidates: list[dict[str, Any]] = []
    for entry in manifest:
        uid = str(entry.get("uid", "")).strip()
        caption = str(entry.get("caption", "")).strip()
        mesh_path = str(entry.get("mesh_path", "")).strip()
        if not uid or not caption or not mesh_path:
            continue
        if not _is_candidate_caption(caption):
            continue

        category = extract_category(caption)
        if not category:
            continue

        candidates.append(
            {
                "uid": uid,
                "caption": caption,
                "mesh_path": mesh_path,
                "category": category,
                "penalty": _caption_penalty(caption),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in candidates:
        grouped[entry["category"]].append(entry)

    for category_entries in grouped.values():
        category_entries.sort(key=lambda item: (item["penalty"], len(item["caption"]), item["uid"]))

    selected: list[dict[str, Any]] = []
    used_uids: set[str] = set()
    taken_per_category: dict[str, int] = defaultdict(int)
    offsets: dict[str, int] = defaultdict(int)
    categories = sorted(grouped, key=lambda category: (-len(grouped[category]), category))

    while len(selected) < selection.target_size:
        made_progress = False
        for category in categories:
            if taken_per_category[category] >= selection.max_per_category:
                continue

            bucket = grouped[category]
            while offsets[category] < len(bucket):
                candidate = bucket[offsets[category]]
                offsets[category] += 1
                if candidate["uid"] in used_uids:
                    continue
                selected.append(candidate)
                used_uids.add(candidate["uid"])
                taken_per_category[category] += 1
                made_progress = True
                break

            if len(selected) >= selection.target_size:
                break

        if not made_progress:
            break

    if len(selected) < selection.target_size:
        remaining = sorted(
            [entry for entry in candidates if entry["uid"] not in used_uids],
            key=lambda item: (item["penalty"], taken_per_category[item["category"]], len(item["caption"]), item["uid"]),
        )
        for candidate in remaining:
            selected.append(candidate)
            if len(selected) >= selection.target_size:
                break

    output_records: list[dict[str, Any]] = []
    for item in selected[: selection.target_size]:
        output_records.append(
            {
                "messages": [
                    {"role": "system", "content": cfg.dataset.system_prompt},
                    {"role": "user", "content": format_user_prompt(item["caption"])},
                ],
                "metadata": {
                    "object_id": item["uid"],
                    "caption": item["caption"],
                    "gt_mesh_path": item["mesh_path"],
                    "split": "eval_id",
                    "category": item["category"],
                },
            }
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")

    unique_categories = len({record["metadata"]["category"] for record in output_records})
    log.info(
        "Built eval dataset: %d records across %d inferred categories -> %s",
        len(output_records),
        unique_categories,
        output_file,
    )
    return output_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated evaluation set from a manifest.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    build_eval_dataset(
        cfg=cfg,
        manifest_path=args.manifest_path or None,
        output_path=args.output_path or None,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
