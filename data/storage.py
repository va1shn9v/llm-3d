"""
Storage abstraction for transparent local / HuggingFace Bucket access.

All file reads go through ``open_read`` which resolves ``hf://`` URIs via the
HF Buckets API, caching downloads locally so repeated training restarts don't
re-fetch.  Local paths pass through unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import IO, Any

from config import StorageConfig

log = logging.getLogger(__name__)


def _cache_key(uri: str) -> str:
    """Deterministic filename-safe hash for an HF URI."""
    return hashlib.sha256(uri.encode()).hexdigest()[:24]


def _hf_bucket_url(cfg: StorageConfig) -> str:
    ns = cfg.hf_bucket_namespace
    return f"hf://buckets/{ns}/{cfg.hf_bucket}" if ns else f"hf://buckets/{cfg.hf_bucket}"


def bucket_uri(remote_key: str, cfg: StorageConfig) -> str:
    """Return a fully qualified HF bucket URI for a bucket-relative key."""
    clean_key = remote_key.strip("/")
    if not clean_key:
        return _hf_bucket_url(cfg)
    return f"{_hf_bucket_url(cfg)}/{clean_key}"


def resolve_manifest_path(cfg: StorageConfig) -> str:
    """Return the manifest path for the configured storage backend."""
    if cfg.backend == "hf":
        return bucket_uri(cfg.manifest_key, cfg)
    return cfg.local_manifest_path


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def open_read(
    path_or_uri: str,
    cfg: StorageConfig | None = None,
    *,
    encoding: str | None = "utf-8",
) -> IO[Any]:
    """Return a file-like object for *path_or_uri*.

    * Plain local paths → ``open(path)``
    * ``hf://`` URIs → download to local cache (if missing), then open the
      cached copy.  Subsequent calls return the cache hit instantly.
    """
    if not path_or_uri.startswith("hf://"):
        mode = "r" if encoding else "rb"
        return open(path_or_uri, mode, encoding=encoding)

    if cfg is None:
        from config import load_config
        cfg = load_config().storage

    cache_dir = Path(cfg.cache_dir)
    cache_path = cache_dir / _cache_key(path_or_uri)

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _download_from_hf(path_or_uri, cache_path)

    mode = "r" if encoding else "rb"
    return open(cache_path, mode, encoding=encoding)


def upload(local_path: str | Path, remote_key: str, cfg: StorageConfig) -> None:
    """Upload a single file to the HF Bucket."""
    from huggingface_hub import HfFileSystem

    hffs = HfFileSystem()
    dest = f"{_hf_bucket_url(cfg)}/{remote_key}"
    log.info("Uploading %s -> %s", local_path, dest)
    with open(local_path, "rb") as src, hffs.open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)


def write_text(path_or_uri: str, text: str, cfg: StorageConfig) -> None:
    """Write text either locally or to an HF bucket URI."""
    if not path_or_uri.startswith("hf://"):
        path = Path(path_or_uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        remote_key = path_or_uri.removeprefix(_hf_bucket_url(cfg)).lstrip("/")
        upload(tmp_path, remote_key, cfg)
    finally:
        tmp_path.unlink(missing_ok=True)


def invalidate_cache(path_or_uri: str, cfg: StorageConfig) -> None:
    """Remove a cached copy so the next ``open_read`` re-downloads."""
    cache_path = Path(cfg.cache_dir) / _cache_key(path_or_uri)
    if cache_path.exists():
        cache_path.unlink()
        log.info("Cache invalidated: %s", path_or_uri)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_from_hf(uri: str, dest: Path) -> None:
    """Download a single ``hf://buckets/...`` URI to *dest*."""
    from huggingface_hub import HfFileSystem

    hffs = HfFileSystem()
    log.info("Downloading %s -> %s", uri, dest)
    with hffs.open(uri, "rb") as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)
