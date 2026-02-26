#!/usr/bin/env bash
# Run GRPO RL training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-training/configs/rl_config.yaml}"

echo "=== Starting GRPO RL Training ==="
echo "Config: $CONFIG"
echo "Modal endpoint: ${MODAL_ENDPOINT:-<not set>}"

python -c "
from training.rl_trainer import run_rl
run_rl('$CONFIG')
"

echo "=== RL Training Complete ==="
