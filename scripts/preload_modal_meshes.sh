#!/usr/bin/env bash
# Sync meshes from HF bucket storage into the Modal volume used for reward/metrics.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "=== Preloading meshes into Modal volume ==="
echo "Config: $CONFIG"
echo ""

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

import modal

from config import load_config
from data.storage import bucket_uri

cfg = load_config('$CONFIG')
if cfg.storage.backend != 'hf':
    raise ValueError('Modal preload expects storage.backend=hf')

sync_from_hf_bucket = modal.Function.from_name('llm3d-blender-worker', 'sync_from_hf_bucket')
count = sync_from_hf_bucket.remote(
    bucket_uri(cfg.storage.mesh_prefix, cfg.storage),
    cfg.storage.modal_volume_mesh_subdir,
)
print(f'Synced {count} meshes into Modal volume subdir {cfg.storage.modal_volume_mesh_subdir!r}')
"

echo ""
echo "=== Modal preload complete ==="
