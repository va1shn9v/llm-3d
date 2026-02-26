# llm-3d: Image-to-3D Code Generation via RLVR

Train a Vision-Language Model (Qwen2.5-VL-7B-Instruct) to generate executable Blender Python code that reconstructs 3D objects from multi-view images, using Reinforcement Learning with Verifiable Rewards.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    TINKER (Training)                 │
│  SFT (LoRA) → GRPO/RL → Eval → Checkpoints         │
│  Model: Qwen2.5-VL-7B-Instruct                     │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS (reward API)
┌──────────────────────▼──────────────────────────────┐
│            MODAL (Sandbox Execution)                 │
│  Blender code execution → Mesh comparison metrics   │
│  Environment wrapped via prime-rl-env (verifiers)   │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
bpy_lib/                   # Constrained Blender Python API library
├── bpy_lib.py             # 8 function families (primitives, curves, booleans, etc.)
└── tests/                 # Unit tests (run in Blender)

data/                      # Data pipeline
├── part_generator.py      # Synthetic part generation (~300K samples)
├── infinigen_extractor.py # Extract parts from Infinigen Indoor objects
├── part_fitter.py         # Fit bpy_lib code to part meshes (template matching)
├── object_assembler.py    # Assemble parts into full object code
├── dataset_builder.py     # Build SFT/RL/eval datasets in Qwen2.5-VL chat format
└── configs/categories.yaml

modal_infra/               # Modal serverless functions
├── images.py              # Docker image definitions (Blender 4.2)
├── blender_worker.py      # Code execution in isolated containers
├── metrics_worker.py      # Chamfer Distance, F-Score, Hausdorff, Normal Consistency
├── render_worker.py       # Multi-view rendering
└── reward_server.py       # FastAPI reward API (POST /reward/batch)

environments/blender_3d/   # Verifiers environment wrapper
├── dataset.py             # Blender3DDataset — prompts
├── harness.py             # Blender3DHarness — Modal execution client
├── rubric.py              # Blender3DRubric — gated sparse+dense reward
└── blender_3d.py          # Blender3DEnvironment — combined

training/                  # Training pipelines
├── sft_trainer.py         # SFT with LoRA on Tinker
├── rl_trainer.py          # GRPO RL on Tinker + Modal
├── eval_runner.py         # Full evaluation framework
└── configs/               # YAML configs for SFT, RL, eval

config.py                  # Central Pydantic config system
configs/default.yaml       # Default configuration
scripts/                   # Shell scripts for each pipeline stage
```

## Setup

```bash
# Install dependencies
pip install -e ".[all]"

# Or with specific extras
pip install -e ".[modal]"     # Modal infra only
pip install -e ".[training]"  # Training deps only
pip install -e ".[data]"      # Data pipeline deps only
```

## Pipeline

### Phase 1: Generate synthetic parts

```bash
./scripts/generate_parts.sh
```

### Phase 2: Build object dataset

```bash
./scripts/build_object_dataset.sh
```

### Phase 3: Deploy reward API

```bash
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
./scripts/deploy_reward_api.sh
```

### Phase 4: SFT training

```bash
./scripts/run_sft.sh
```

### Phase 5: RL training

```bash
export MODAL_ENDPOINT=https://your-workspace--reward-api-web.modal.run
export MODAL_TOKEN=...
./scripts/run_rl.sh
```

### Phase 6: Evaluation

```bash
./scripts/run_eval.sh
```

## Configuration

All settings are managed through `config.py` with Pydantic models. Override via:
- YAML file: `configs/default.yaml`
- Environment variables: `LLM3D_` prefix (e.g., `LLM3D_SEED=123`)
- Programmatic: `load_config(yaml_path, seed=123)`

## Key Design Decisions

- **bpy_lib API**: Constrained set of high-level Blender functions the model learns to use
- **Gated reward**: Discrete gates (0.0→0.05→0.10→0.15→0.20) + continuous quality (0.3→1.0)
- **Modal execution**: Isolated Blender containers with 120s timeout per sample
- **Curriculum SFT**: Easy→Medium→Hard ordering by part count and type complexity
- **GRPO**: Group-relative advantages with 8 completions per prompt

## Reward Structure

| Gate | Condition | Reward |
|------|-----------|--------|
| 0 | Empty / no import | 0.00 |
| 1 | Execution failed | 0.05 |
| 2 | No geometry (<4 faces) | 0.10 |
| 3 | Degenerate (>100K verts) | 0.15 |
| 4 | No resemblance (F@0.05 < 0.05) | 0.20 |
| 5+ | Quality: 0.3 + 0.7 × min(1, F@0.05/0.6) | 0.30–1.00 |

Plus 10% weight on format reward (import, comments, export_scene).

## Metrics

- **Execution Rate**: % of code that runs without error
- **F-Score@0.05**: Primary quality metric
- **Chamfer Distance (L2)**: 10K points (training) / 100K points (eval)
- **Hausdorff 90th**: Worst-case error
- **Normal Consistency**: Surface orientation accuracy
