#!/usr/bin/env bash
# Generate synthetic part dataset (~300K part-code pairs) on Modal.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/default.yaml}"

echo "=== Generating synthetic parts ==="
echo "Config: $CONFIG"
echo ""

python -c "
from data.part_generator import generate_all_parts, parts_to_jsonl
from config import load_config

cfg = load_config('$CONFIG')
print(f'Generating parts with seed={cfg.seed}')

all_parts = generate_all_parts(seed=cfg.seed)
for ptype, parts in all_parts.items():
    output = f'{cfg.data_dir}/parts/{ptype}.jsonl'
    parts_to_jsonl(parts, output)
    print(f'  {ptype}: {len(parts)} parts → {output}')

total = sum(len(p) for p in all_parts.values())
print(f'Total: {total} parts generated')
"

echo ""
echo "=== Part generation complete ==="
