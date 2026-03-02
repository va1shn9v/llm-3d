#!/usr/bin/env bash
# Preprocess dataset: load 3D objects, render multi-view images, store to disk.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/default.yaml}"
MESH_DIR="${2:-./meshes}"
OUTPUT_DIR="${3:-./data/renders}"
NUM_WORKERS="${4:-4}"

echo "=== Dataset Preprocessing ==="
echo "Config:     $CONFIG"
echo "Mesh dir:   $MESH_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Workers:    $NUM_WORKERS"
echo ""

python -m data.dataset_preprocessor \
    --mesh-dir "$MESH_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --max-workers "$NUM_WORKERS"

echo ""
echo "=== Preprocessing complete ==="
