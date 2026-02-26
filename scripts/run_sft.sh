#!/usr/bin/env bash
# Run SFT training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-training/configs/sft_config.yaml}"

echo "=== Starting SFT Training ==="
echo "Config: $CONFIG"

python -c "
from training.sft_trainer import run_sft
run_sft('$CONFIG')
"

echo "=== SFT Training Complete ==="
