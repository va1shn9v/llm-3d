"""
Cap3D caption joining + remote Objaverse mesh ingest.

Builds a manifest of {uid, caption, mesh_path} entries where ``mesh_path``
points at an external HF bucket location instead of a local mesh copy.
"""

from __future__ import annotations

import gzip
import json
import logging

import pandas as pd
from huggingface_hub import hf_hub_download

from config import ProjectConfig, load_config
from data.storage import bucket_uri, ensure_bucket_exists, resolve_manifest_path, write_text

log = logging.getLogger(__name__)

CAP3D_CSV = "Cap3D_automated_Objaverse_full.csv"
OBJAVERSE_OBJECT_PATHS = "object-paths.json.gz"


def load_cap3d_captions() -> dict[str, str]:
    """Load Cap3D captions into a uid -> caption dict."""
    log.info("Loading Cap3D captions from HuggingFace...")
    csv_path = hf_hub_download(
        repo_id="tiange/Cap3D",
        filename=CAP3D_CSV,
        repo_type="dataset",
    )
    df = pd.read_csv(csv_path, header=None, names=["uid", "caption"])
    df = df.dropna(subset=["uid", "caption"])

    captions: dict[str, str] = dict(zip(df["uid"].astype(str), df["caption"].astype(str)))
    log.info("Loaded %d Cap3D captions", len(captions))
    return captions


def load_objaverse_object_paths() -> dict[str, str]:
    """Load Objaverse UID -> object path mapping."""
    path = hf_hub_download(
        repo_id="allenai/objaverse",
        filename=OBJAVERSE_OBJECT_PATHS,
        repo_type="dataset",
    )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_manifest(
    cfg: ProjectConfig | None = None,
    filtered_uids_path: str | None = None,
) -> list[dict]:
    """Build a manifest by ingesting meshes remotely into the configured HF bucket."""
    if cfg is None:
        cfg = load_config()

    if cfg.storage.backend != "hf":
        raise ValueError(
            "Remote-first manifest build requires storage.backend='hf'. "
            "Switch the config or implement a separate local-only ingest path."
        )

    if filtered_uids_path is None:
        filtered_uids_path = cfg.objaverse_filter.output_path

    with open(filtered_uids_path, encoding="utf-8") as f:
        uids: list[str] = json.load(f)
    log.info("Loaded %d filtered UIDs", len(uids))

    import modal

    ensure_bucket_exists(cfg.storage)
    bucket_root = bucket_uri("", cfg.storage)
    build_remote_manifest = modal.Function.from_name(
        "llm3d-data-worker",
        "build_remote_mesh_manifest",
    )
    manifest: list[dict] = build_remote_manifest.remote(
        uids,
        bucket_root,
        cfg.storage.mesh_prefix,
    )

    manifest_text = "".join(json.dumps(entry) + "\n" for entry in manifest)
    local_manifest_path = cfg.storage.local_manifest_path
    write_text(local_manifest_path, manifest_text, cfg.storage)

    manifest_path = resolve_manifest_path(cfg.storage)
    if manifest_path != local_manifest_path:
        write_text(manifest_path, manifest_text, cfg.storage)

    log.info(
        "Saved manifest with %d entries to %s (local mirror: %s)",
        len(manifest),
        manifest_path,
        local_manifest_path,
    )

    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_manifest()
