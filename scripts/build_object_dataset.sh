#!/usr/bin/env bash
# Phase 1b: Build SFT/RL/eval dataset splits from synthetic data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "=== Building dataset splits ==="
echo "Config: $CONFIG"
echo ""

python -c "
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from config import load_config
from data.dataset_builder import build_datasets, save_splits
from data.storage import resolve_manifest_path

cfg = load_config('$CONFIG')
manifest_path = resolve_manifest_path(cfg.storage)

splits = build_datasets(
    synthetic_jsonl_path=cfg.synthetic_gen.output_path,
    manifest_jsonl_path=manifest_path,
    cfg=cfg,
)
dataset_dir = Path(cfg.sft.train_path).parent
save_splits(splits, dataset_dir)

for name, split in splits.items():
    print(f'  {name}: {len(split.samples)} samples')
print(f'Dataset splits saved to {dataset_dir}.')
"

echo ""
echo "=== Dataset build complete ==="
