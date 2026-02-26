#!/usr/bin/env bash
# Run full evaluation across all conditions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-training/configs/eval_config.yaml}"

echo "=== Running Evaluation ==="
echo "Config: $CONFIG"

python -c "
from training.eval_runner import run_eval
run_eval('$CONFIG')
"

echo "=== Evaluation Complete ==="
