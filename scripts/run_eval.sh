#!/usr/bin/env bash
# Run full evaluation across all conditions.
#
# Override any parameter on the CLI:
#   ./scripts/run_eval.sh eval.temperature=0.3 output_dir=./output/eval_warm
#   ./scripts/run_eval.sh eval.conditions.candidate.enabled=true eval.conditions.candidate.model_path=ckpts/rl-step-500
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
. "$SCRIPT_DIR/load_dev_env.sh"
load_project_env "$PROJECT_ROOT"

echo "=== Running Evaluation ==="
echo "Overrides: $*"

python -m training.eval.runner "$@"

echo "=== Evaluation Complete ==="
