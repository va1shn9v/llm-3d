#!/usr/bin/env bash
# Phase 2: Run SFT training on Qwen2.5-Coder-7B-Instruct.
#
# Uses Hydra for config management. Override any parameter on the CLI:
#   ./scripts/run_sft.sh sft.learning_rate=5e-5 sft.epochs=5
#
# Run a sweep:
#   ./scripts/run_sft.sh --multirun sft.lora_rank=16,32,64
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Starting SFT Training ==="
echo "Hydra overrides: $*"
echo "Model: Qwen/Qwen2.5-Coder-7B-Instruct"

python -m training.sft_trainer "$@"

echo "=== SFT Training Complete ==="
