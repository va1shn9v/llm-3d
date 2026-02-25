# Image → 3D Code Generation: LLM Evaluation Pipeline

Evaluates how well different LLMs can generate Blender Python code that
reconstructs a 3D object from multi-view input images. Designed as the
foundation for an RLVR (Reinforcement Learning with Verifiable Rewards)
environment for procedural 3D generation.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Blender (pick one):
#    Option A: Standalone (recommended)
sudo apt install blender
#    OR download from https://www.blender.org/download/

#    Option B: Python module (requires Python 3.11)
pip install bpy

# 3. Run evaluation
export OPENROUTER_API_KEY="sk-or-your-key-here"

# Quick test: 1 object, 1 model
python eval_image_to_3d_code.py \
  --category chair \
  --max-objects 1 \
  --num-views 4 \
  --models openai/gpt-4o \
  --api-key $OPENROUTER_API_KEY

# Full evaluation: multiple objects and models
python eval_image_to_3d_code.py \
  --category chair \
  --max-objects 5 \
  --num-views 6 \
  --models anthropic/claude-sonnet-4 openai/gpt-4o google/gemini-2.5-flash \
  --api-key $OPENROUTER_API_KEY
```

## Configuration Cheat Sheet

| Flag                | Default   | Description                              |
|---------------------|-----------|------------------------------------------|
| `--category`        | chair     | LVIS category (chair, table, lamp, etc.) |
| `--max-objects`     | 5         | Number of objects to evaluate            |
| `--num-views`       | 4         | Input views per object (4-8 recommended) |
| `--render-resolution`| 512      | Image resolution in pixels               |
| `--models`          | claude+gpt| Space-separated OpenRouter model IDs     |
| `--max-tokens`      | 4096      | Max response tokens                      |
| `--temperature`     | 0.2       | LLM sampling temperature                 |
| `--blender-path`    | blender   | Path to Blender binary                   |
| `--output-dir`      | ./eval_results | Where to save everything            |
| `--num-sample-points`| 10000    | Points sampled per mesh for metrics      |
| `--code-timeout`    | 60        | Seconds before killing bad code          |

## Output Structure

```
eval_results/
├── eval_results.json          # Full results (metrics, timing, errors)
├── obj_abc123def456/          # Per-object directory
│   ├── renders/
│   │   ├── view_00.png        # Input views sent to LLMs
│   │   ├── view_01.png
│   │   ├── view_02.png
│   │   ├── view_03.png
│   │   ├── ground_truth.obj   # GT mesh (normalized)
│   │   └── _render_script.py  # Blender render script
│   ├── anthropic_claude-sonnet-4/
│   │   ├── generated_code.py  # What the LLM produced
│   │   ├── generated_mesh.obj # The resulting mesh
│   │   └── execution_log.txt  # Blender stdout/stderr
│   └── openai_gpt-4o/
│       └── ...
```

## Metrics Explained

| Metric              | Range    | Direction | What it measures                     |
|---------------------|----------|-----------|--------------------------------------|
| Chamfer Distance    | [0, ∞)   | ↓ lower   | Average surface-to-surface distance  |
| F-Score @ τ         | [0, 1]   | ↑ higher  | % of surface within τ of ground truth|
| Hausdorff 90th      | [0, ∞)   | ↓ lower   | Worst-case geometric error (robust)  |
| Normal Consistency  | [0, 1]   | ↑ higher  | Surface orientation correctness      |
| RLVR Reward         | [0, 1]   | ↑ higher  | Composite reward (weighted combo)    |

## Adapting for RLVR Training

The `compute_rlvr_reward()` function is the reward signal you'd plug into
your Tinker API training loop. The key insight: this reward is **verifiable**
(computed programmatically from the generated mesh vs ground truth) making
it suitable for RLVR-style training. See the docstring in the script for
the reward weight rationale.