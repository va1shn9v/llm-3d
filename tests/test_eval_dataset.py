from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import EvalConfig, EvalSelectionConfig, ProjectConfig, StorageConfig
from data.eval_dataset import build_eval_dataset


def _write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")


def test_build_eval_dataset_filters_noisy_entries_and_keeps_diversity(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    output_path = tmp_path / "eval_id.jsonl"
    _write_manifest(
        manifest_path,
        [
            {"uid": "u1", "caption": "A wooden chair with a curved backrest.", "mesh_path": "hf://chair"},
            {"uid": "u2", "caption": "A metal stool with four legs.", "mesh_path": "hf://stool"},
            {"uid": "u3", "caption": "A cartoon cactus wearing a cowboy hat.", "mesh_path": "hf://bad1"},
            {"uid": "u4", "caption": "A red warning sign with printed text.", "mesh_path": "hf://bad2"},
            {"uid": "u5", "caption": "A ceramic mug with a wide handle.", "mesh_path": "hf://mug"},
            {"uid": "u6", "caption": "A desk lamp with a rounded shade.", "mesh_path": "hf://lamp"},
            {"uid": "u7", "caption": "A ceramic mug with a narrow handle.", "mesh_path": "hf://mug2"},
        ],
    )

    cfg = ProjectConfig(
        storage=StorageConfig(backend="local", local_manifest_path=str(manifest_path)),
        eval=EvalConfig(
            selection=EvalSelectionConfig(
                output_path=str(output_path),
                manifest_path=str(manifest_path),
                target_size=4,
                max_per_category=1,
            )
        ),
    )

    records = build_eval_dataset(cfg)

    assert len(records) == 4
    categories = {record["metadata"]["category"] for record in records}
    assert "mug" in categories
    assert "chair" in categories or "stool" in categories
    captions = {record["metadata"]["caption"] for record in records}
    assert "A cartoon cactus wearing a cowboy hat." not in captions
    assert "A red warning sign with printed text." not in captions

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(written) == 4
