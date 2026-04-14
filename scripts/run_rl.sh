#!/usr/bin/env bash
# Run direct GRPO RL training against the prompt dataset.
#
# Override any parameter on the CLI:
#   ./scripts/run_rl.sh rl.learning_rate=1e-5 rl.steps=200
#   ./scripts/run_rl.sh reward.geometry.resemblance.threshold=0.08
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
. "$SCRIPT_DIR/load_dev_env.sh"
load_project_env "$PROJECT_ROOT"

echo "=== Starting Direct GRPO RL Training ==="
echo "Overrides: $*"
echo ""

python -m training.rl.trainer "$@"

echo "=== RL Training Complete ==="
