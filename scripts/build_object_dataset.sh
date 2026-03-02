#!/usr/bin/env bash
# Build full object dataset: extract parts, fit code, assemble, render.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "=== Building object dataset ==="
echo "Config: $CONFIG"
echo ""

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from config import load_config

cfg = load_config('$CONFIG')
print(f'CD threshold: {cfg.quality_gate.cd_threshold}')

# Step 1: Load MeshCoderDataset
from datasets import load_dataset
ds = load_dataset('InternRobotics/MeshCoderDataset')
print(f'Train: {len(ds[\"train\"])} | Val: {len(ds[\"validation\"])} | Test: {len(ds[\"test\"])}')

# Step 2: Build dataset splits
from data.dataset_builder import build_datasets, save_splits
splits = build_datasets(ds, f'{cfg.data_dir}/renders', cfg)
save_splits(splits, f'{cfg.output_dir}/datasets')
print('Dataset splits saved.')
"

echo ""
echo "=== Object dataset build complete ==="
