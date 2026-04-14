#!/usr/bin/env bash
# Build a curated eval set from the existing manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
. "$SCRIPT_DIR/load_dev_env.sh"
load_project_env "$PROJECT_ROOT"

CONFIG="${1:-configs/config.yaml}"

echo "=== Building Eval Dataset ==="
echo "Config: $CONFIG"
echo ""

python -m data.eval_dataset --config "$CONFIG"

echo ""
echo "=== Eval dataset build complete ==="
