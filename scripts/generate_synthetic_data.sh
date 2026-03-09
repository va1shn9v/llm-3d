#!/usr/bin/env bash
# Phase 1: Generate synthetic SFT data via teacher LLM + Blender validation.
# Also produces hard_prompts.csv for RLVR hard mining.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "=== Generating synthetic SFT data ==="
echo "Config: $CONFIG"
echo ""
echo "This will:"
echo "  1. Call teacher LLM to generate Blender Python code for each caption"
echo "  2. Execute code in Blender via Modal"
echo "  3. Validate against ground-truth meshes (CD, F-Score)"
echo "  4. Track hard prompts (high failure rate) in CSV for RLVR hard mining"
echo ""

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from config import load_config
from data.synthetic_generator import SyntheticGenerator

cfg = load_config('$CONFIG')
gen = SyntheticGenerator(cfg)
validated = gen.generate()
print(f'Generated {len(validated)} validated (caption, code) pairs')
print(f'Hard prompts CSV: {cfg.synthetic_gen.hard_prompts_path}')
"

echo ""
echo "=== Synthetic data generation complete ==="
