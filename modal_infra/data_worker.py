"""
Modal worker: remote-first Objaverse ingest into HF bucket storage.

Streams meshes directly from Objaverse's Hugging Face dataset into the
configured HF bucket and returns manifest records with ``hf://`` mesh paths.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import modal

from data.caption_joiner import load_cap3d_captions, load_objaverse_object_paths

app = modal.App("llm3d-data-worker")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dev_env() -> None:
    """Load repo-local dev.env for Modal deploy-time configuration."""
    env_path = _PROJECT_ROOT / "dev.env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


_load_dev_env()


def _hf_secrets() -> list[modal.Secret]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return []
    return [
        modal.Secret.from_dict({
            "HF_TOKEN": token,
            "HUGGINGFACE_HUB_TOKEN": token,
        })
    ]


data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=1.0",
        "pandas>=2.0",
        "pydantic>=2.5",
        "pydantic-settings>=2.1",
        "pyyaml>=6.0",
    )
    .add_local_python_source("config", "data")
)


def _objaverse_resolve_url(object_path: str) -> str:
    return f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_path}"


@app.function(
    image=data_image,
    timeout=60 * 60,
    cpu=2,
    memory=4096,
    secrets=_hf_secrets(),
)
def build_remote_mesh_manifest(
    uids: list[str],
    hf_bucket_root: str,
    remote_prefix: str = "meshes",
) -> list[dict[str, Any]]:
    """Stream meshes into HF bucket storage and return manifest entries."""
    from huggingface_hub import HfFileSystem

    if not hf_bucket_root.startswith("hf://"):
        raise ValueError(f"Expected hf:// bucket root, got {hf_bucket_root!r}")

    captions = load_cap3d_captions()
    object_paths = load_objaverse_object_paths()
    hffs = HfFileSystem()

    prefix = remote_prefix.strip("/")
    manifest: list[dict[str, Any]] = []
    uploaded = 0

    for uid in uids:
        caption = captions.get(uid)
        object_path = object_paths.get(uid)
        if not caption or not object_path:
            continue

        suffix = Path(object_path).suffix.lower() or ".glb"
        remote_key = f"{prefix}/{uid}{suffix}" if prefix else f"{uid}{suffix}"
        remote_uri = f"{hf_bucket_root.rstrip('/')}/{remote_key}"

        if not hffs.exists(remote_uri):
            with urllib.request.urlopen(_objaverse_resolve_url(object_path)) as src, hffs.open(remote_uri, "wb") as dst:
                shutil.copyfileobj(src, dst)
            uploaded += 1

        manifest.append({
            "uid": uid,
            "caption": caption,
            "mesh_path": remote_uri,
        })

    print(f"Prepared {len(manifest)} manifest entries ({uploaded} uploaded, {len(manifest) - uploaded} reused)")
    return manifest
