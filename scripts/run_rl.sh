#!/usr/bin/env bash
# Phase 3-4: Run GRPO RL training with hard prompt mining.
#
# Uses Hydra for config management. Override any parameter on the CLI:
#   ./scripts/run_rl.sh reward=geometry_heavy rl.learning_rate=1e-5
#
# Run a sweep over reward settings:
#   ./scripts/run_rl.sh --multirun reward.geometry.resemblance.threshold=0.04,0.05,0.06
#
# Use an experiment preset:
#   ./scripts/run_rl.sh +experiment=reward_sweep --multirun
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Starting GRPO RL Training (with hard mining) ==="
echo "Hydra overrides: $*"
echo ""
echo "Hard mining will oversample prompts where teacher LLM failed"
echo "during synthetic data generation (tracked in hard_prompts.csv)."
echo ""

python -m training.rl_trainer "$@"

echo "=== RL Training Complete ==="
