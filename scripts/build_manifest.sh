#!/usr/bin/env bash
# Phase 0b: Join Cap3D captions and ingest meshes remotely into HF bucket storage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load repo-local env vars so Modal deploy sees secrets without manual export.
# shellcheck disable=SC1091
. "$SCRIPT_DIR/load_dev_env.sh"
load_project_env "$PROJECT_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "=== Building manifest (remote mesh ingest -> HF bucket) ==="
echo "Config: $CONFIG"
echo ""

if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]; then
  echo "HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set so Modal can upload meshes to your HF bucket."
  exit 1
fi

echo "Deploying Modal data worker..."
modal deploy modal_infra/data_worker.py
echo ""

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from config import load_config
from data.caption_joiner import build_manifest

cfg = load_config('$CONFIG')
manifest = build_manifest(cfg)
print(f'Manifest built: {len(manifest)} entries')
"

echo ""
echo "=== Manifest build complete ==="
