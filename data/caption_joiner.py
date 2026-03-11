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

import pandas as pd
from huggingface_hub import hf_hub_download

from config import ProjectConfig, load_config

log = logging.getLogger(__name__)

CAP3D_CSV = "Cap3D_automated_Objaverse_full.csv"


def load_cap3d_captions() -> dict[str, str]:
    """Load Cap3D captions into a uid -> caption dict.

    Downloads the CSV directly from HuggingFace Hub to avoid ClassLabel
    schema issues with `load_dataset`.
    """
    log.info("Loading Cap3D captions from HuggingFace...")
    csv_path = hf_hub_download(
        repo_id="tiange/Cap3D",
        filename=CAP3D_CSV,
        repo_type="dataset",
    )
    df = pd.read_csv(csv_path, header=None, names=["uid", "caption"])
    df = df.dropna(subset=["uid", "caption"])

    captions: dict[str, str] = dict(zip(df["uid"].astype(str), df["caption"].astype(str)))
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
