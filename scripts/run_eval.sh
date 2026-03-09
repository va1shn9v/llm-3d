#!/usr/bin/env bash
# Run full evaluation across all conditions.
#
# Uses Hydra for config management. Override any parameter on the CLI:
#   ./scripts/run_eval.sh eval.temperature=0.3 output_dir=./output/eval_warm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Running Evaluation ==="
echo "Hydra overrides: $*"

python -m training.eval_runner "$@"

echo "=== Evaluation Complete ==="
