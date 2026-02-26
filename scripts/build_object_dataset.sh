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
print(f'Categories: {cfg.infinigen.categories}')
print(f'Objects per category: {cfg.infinigen.objects_per_category}')
print(f'CD threshold: {cfg.quality_gate.cd_threshold}')

# Step 1: Extract parts from Infinigen objects
from data.infinigen_extractor import extract_objects_from_directory
extracted = extract_objects_from_directory(
    cfg.infinigen.output_dir,
    f'{cfg.data_dir}/extracted_parts',
    cfg.infinigen.categories,
)
print(f'Extracted {len(extracted)} objects')

# Step 2: Fit parts to bpy_lib code
from data.part_fitter import fit_all_parts
all_fits = []
for obj in extracted:
    part_paths = [p.mesh_path for p in obj.parts]
    fits = fit_all_parts(part_paths, cfg.quality_gate.cd_threshold)
    all_fits.append(fits)

# Step 3: Assemble into full objects
from data.object_assembler import assemble_all_objects
assembled = assemble_all_objects(extracted, all_fits, cfg.quality_gate.cd_threshold)
print(f'Assembled {len(assembled)} objects (from {len(extracted)} extracted)')

# Step 4: Build dataset splits
from data.dataset_builder import build_datasets, save_splits
splits = build_datasets(assembled, f'{cfg.data_dir}/renders', cfg)
save_splits(splits, f'{cfg.output_dir}/datasets')
print('Dataset splits saved.')
"

echo ""
echo "=== Object dataset build complete ==="
