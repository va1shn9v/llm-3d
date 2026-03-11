#!/usr/bin/env bash
# Sync local data pipeline outputs to HuggingFace Storage Bucket.
#
# Prerequisites:
#   1. Install hf CLI: curl -LsSf https://hf.co/cli/install.sh | bash
#   2. Authenticate:  hf auth login
#   3. Create bucket:  hf buckets create llm3d-data --private
#
# Usage:
#   ./scripts/sync_to_hf.sh                  # sync all (datasets + meshes + renders)
#   ./scripts/sync_to_hf.sh datasets          # sync only datasets
#   ./scripts/sync_to_hf.sh meshes            # sync only meshes
#   ./scripts/sync_to_hf.sh --dry-run         # preview what would be synced
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

HF_NAMESPACE="${HF_NAMESPACE:?Set HF_NAMESPACE to your HuggingFace username or org}"
BUCKET="llm3d-data"
BUCKET_URL="hf://buckets/${HF_NAMESPACE}/${BUCKET}"

DRY_RUN=""
TARGET="all"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        datasets|meshes|renders) TARGET="$arg" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

sync_dir() {
    local src="$1" dest="$2"
    if [ ! -d "$src" ]; then
        echo "Skipping $src (directory not found)"
        return
    fi
    echo "Syncing $src -> $dest"
    hf buckets sync "$src" "$dest" $DRY_RUN
}

if [ "$TARGET" = "all" ] || [ "$TARGET" = "datasets" ]; then
    sync_dir "./output/datasets" "${BUCKET_URL}/datasets"
    sync_dir "./datasets"        "${BUCKET_URL}/datasets"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "meshes" ]; then
    sync_dir "./data/meshes" "${BUCKET_URL}/meshes"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "renders" ]; then
    sync_dir "./data/renders" "${BUCKET_URL}/renders"
fi

echo ""
echo "=== Sync complete ==="
