#!/usr/bin/env bash
# Phase 0b: Join Cap3D captions with filtered UIDs and download meshes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "=== Building manifest (Cap3D join + mesh download) ==="
echo "Config: $CONFIG"
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
