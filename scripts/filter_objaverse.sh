#!/usr/bin/env bash
# Phase 0a: Filter Objaverse++ UIDs by quality.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "=== Filtering Objaverse++ UIDs ==="
echo "Config: $CONFIG"
echo ""

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from config import load_config
from data.objaverse_filter import filter_objaverse_uids

cfg = load_config('$CONFIG')
uids = filter_objaverse_uids(cfg)
print(f'Filtered {len(uids)} UIDs')
"

echo ""
echo "=== Objaverse++ filtering complete ==="
