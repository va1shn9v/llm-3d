#!/usr/bin/env bash
# Deploy the Modal reward API server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Deploying Modal reward API ==="

# Ensure Modal is authenticated
modal token check 2>/dev/null || {
    echo "Modal not authenticated. Run: modal token new"
    exit 1
}

# Deploy
modal deploy modal_infra/reward_server.py

echo ""
echo "=== Deployment complete ==="
echo "Test with: curl <endpoint>/health"
