"""
Cap3D caption joining + Objaverse mesh download (Phase 0 of pipeline).

Joins quality-filtered UIDs with Cap3D text captions, downloads the actual
GLB/OBJ meshes from Objaverse for ground-truth reward computation during RLVR.

Outputs a manifest JSONL: each line has {uid, caption, mesh_path}.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import ProjectConfig, load_config

log = logging.getLogger(__name__)


def load_cap3d_captions() -> dict[str, str]:
    """Load Cap3D captions into a uid -> caption dict."""
    from datasets import load_dataset

    log.info("Loading Cap3D captions from HuggingFace...")
    ds = load_dataset("tiange/Cap3D", split="train")

    captions: dict[str, str] = {}
    for row in ds:
        uid = row.get("uid") or row.get("object_uid", "")
        caption = row.get("caption", "")
        if uid and caption:
            captions[uid] = caption

    log.info(f"Loaded {len(captions)} Cap3D captions")
    return captions


def download_meshes(uids: list[str], output_dir: str | Path) -> dict[str, str]:
    """Download Objaverse GLB meshes for given UIDs.

    Returns dict mapping uid -> local mesh file path.
    """
    import objaverse

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading {len(uids)} meshes from Objaverse...")
    objects = objaverse.load_objects(uids=uids, download_processes=8)

    uid_to_path: dict[str, str] = {}
    for uid, path in objects.items():
        if path and Path(path).exists():
            uid_to_path[uid] = str(path)

    log.info(f"Successfully downloaded {len(uid_to_path)}/{len(uids)} meshes")
    return uid_to_path


def build_manifest(
    cfg: ProjectConfig | None = None,
    filtered_uids_path: str | None = None,
) -> list[dict]:
    """Join filtered UIDs with Cap3D captions and download meshes.

    Returns and saves a manifest of {uid, caption, mesh_path} entries.
    """
    if cfg is None:
        cfg = load_config()

    if filtered_uids_path is None:
        filtered_uids_path = cfg.objaverse_filter.output_path

    with open(filtered_uids_path) as f:
        uids: list[str] = json.load(f)
    log.info(f"Loaded {len(uids)} filtered UIDs")

    captions = load_cap3d_captions()

    uids_with_captions = [uid for uid in uids if uid in captions]
    log.info(f"{len(uids_with_captions)}/{len(uids)} UIDs have Cap3D captions")

    mesh_dir = Path(cfg.data_dir) / "meshes"
    uid_to_mesh = download_meshes(uids_with_captions, mesh_dir)

    manifest: list[dict] = []
    for uid in uids_with_captions:
        if uid not in uid_to_mesh:
            continue
        manifest.append({
            "uid": uid,
            "caption": captions[uid],
            "mesh_path": uid_to_mesh[uid],
        })

    manifest_path = Path(cfg.data_dir) / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    log.info(f"Saved manifest with {len(manifest)} entries to {manifest_path}")

    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_manifest()
