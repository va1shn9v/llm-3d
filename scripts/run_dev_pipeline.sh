#!/usr/bin/env bash
# Run the full data collection pipeline on 5 examples for local inspection.
#
# Phases:
#   0a  Filter Objaverse 1.0 UIDs via LVIS annotations (capped at 5)
#   0b  Join Cap3D captions + ingest meshes into HF bucket → manifest
#   1   Preload meshes into Modal Volume
#   2   Teacher LLM code gen → Blender validation via Modal → synthetic SFT pairs
#
# Outputs land in datasets/dev/ so they don't conflict with production data.
#
# Prerequisites:
#   - dev.env filled in (OPENAI_API_KEY, MODAL_TOKEN_ID/SECRET, HF_TOKEN)
#   - `pip install -e ".[data,modal]"` (or uv sync)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
. "$SCRIPT_DIR/load_dev_env.sh"
load_project_env "$PROJECT_ROOT"

CONFIG="configs/dev_5.yaml"
DEV_DIR="datasets/dev"

mkdir -p "$DEV_DIR"

# ── Helpers ───────────────────────────────────────────────────────────
banner() { printf '\n\033[1;36m=== %s ===\033[0m\n\n' "$1"; }
sep()    { printf '\n\033[0;90m%s\033[0m\n' "────────────────────────────────────────"; }

# ── Phase 0a: Filter Objaverse 1.0 UIDs ──────────────────────────────
banner "Phase 0a: Filtering Objaverse 1.0 UIDs via LVIS (max 5)"

python -c "
import logging, json
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from config import load_config
from data.objaverse_filter import filter_objaverse_uids

cfg = load_config('$CONFIG')
uids = filter_objaverse_uids(cfg)
print(f'\nFiltered {len(uids)} UIDs')
print('UIDs:', json.dumps(uids, indent=2))
"

sep
echo "Output: $DEV_DIR/filtered_uids.json"
echo "Contents:"
python -m json.tool "$DEV_DIR/filtered_uids.json"

# ── Phase 0b: Caption join + remote ingest ───────────────────────────
banner "Phase 0b: Building manifest (Cap3D captions + remote mesh ingest)"

modal deploy modal_infra/data_worker.py

python -c "
import logging, json
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from config import load_config
from data.caption_joiner import build_manifest
from data.storage import resolve_manifest_path

cfg = load_config('$CONFIG')
manifest = build_manifest(cfg, filtered_uids_path='$DEV_DIR/filtered_uids.json')
manifest_path = resolve_manifest_path(cfg.storage)
print(f'\nManifest: {len(manifest)} entries')
print(f'Manifest path: {manifest_path}')
for i, entry in enumerate(manifest):
    print(f'  [{i}] uid={entry[\"uid\"]}')
    print(f'      caption: {entry[\"caption\"]}')
    print(f'      mesh:    {entry[\"mesh_path\"]}')
"

sep
echo "Output: data/manifest.jsonl"
if [ -s "data/manifest.jsonl" ]; then
  echo "Contents:"
  python -c "
import json
with open('data/manifest.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        print(json.dumps(obj, indent=2))
        print()
"
else
  echo "(empty or missing — no manifest entries produced)"
fi

# ── Deploy Modal apps ─────────────────────────────────────────────────
banner "Deploying Modal apps (blender-worker + metrics-worker)"
modal deploy modal_infra/blender_worker.py
modal deploy modal_infra/metrics_worker.py

banner "Phase 1: Preloading meshes into Modal Volume"
bash ./scripts/preload_modal_meshes.sh "$CONFIG"

# ── Phase 2: Synthetic data generation ───────────────────────────────
banner "Phase 2: Synthetic generation (teacher LLM → Blender → validation)"
echo "This calls OpenAI and Modal — make sure dev.env has the right keys."
echo ""

python -c "
import asyncio, logging, json
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

from config import load_config
from data.synthetic_generator import SyntheticGenerator

cfg = load_config('$CONFIG')
gen = SyntheticGenerator(cfg)
validated = asyncio.run(gen.generate())

print(f'\nValidated {len(validated)} / 5 pairs')
for i, pair in enumerate(validated):
    print(f'\n  [{i}] uid={pair[\"uid\"]}')
    print(f'      caption:  {pair[\"caption\"]}')
    print(f'      obj:      {pair.get(\"obj_path\", \"(not saved)\")}')
    print(f'      metrics:  {json.dumps(pair.get(\"metrics\", {}), indent=2)}')
    code_preview = pair[\"code\"][:200].replace(chr(10), chr(10) + '        ')
    print(f'      code:     {code_preview}...')
"

sep
echo "Output: $DEV_DIR/synthetic_sft.jsonl"
if [ -s "$DEV_DIR/synthetic_sft.jsonl" ]; then
  echo "Contents:"
  python -c "
import json
with open('$DEV_DIR/synthetic_sft.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        print(json.dumps(obj, indent=2))
        print()
"
else
  echo "(empty or missing — no synthetic pairs produced)"
fi

# ── Summary ──────────────────────────────────────────────────────────
banner "Done — dev outputs"
echo "  Filtered UIDs:   $DEV_DIR/filtered_uids.json"
echo "  Manifest:        data/manifest.jsonl"
echo "  Synthetic pairs: $DEV_DIR/synthetic_sft.jsonl"
echo "  Hard prompts:    $DEV_DIR/hard_prompts.csv"
echo "  Checkpoint:      $DEV_DIR/synthetic_checkpoint.json"
echo "  OBJ meshes:      $DEV_DIR/objs/"
echo ""
if [ -d "$DEV_DIR/objs" ] && ls "$DEV_DIR/objs/"*.obj 1>/dev/null 2>&1; then
  echo "Generated OBJ files:"
  ls -lh "$DEV_DIR/objs/"*.obj
  echo ""
  echo "Inspect with:  bun tools/obj-viewer/server.ts $DEV_DIR/objs/"
fi
echo ""
echo "Inspect any of these files to review the pipeline output."
