# llm-3d

Train a code model to generate executable Blender 4.2 Python from text descriptions, then improve it with RL using server-side execution and mesh-based rewards on Modal.

## Architecture

```text
Objaverse filter
  -> caption join + remote mesh ingest to HF bucket
  -> manifest.jsonl with hf:// mesh references
  -> preload GT meshes into Modal volume
  -> synthetic teacher generation + Blender validation
  -> SFT / RL / eval dataset splits
  -> SFT on Tinker
  -> RL (GRPO) on Tinker
  -> reward execution via Modal /reward/batch
  -> eval via the same reward harness
```

Current runtime split:

- Local scripts orchestrate jobs, config loading, and dataset preparation.
- Tinker handles SFT and RL optimization.
- Modal executes Blender code, computes mesh metrics, serves the reward API, and stores GT meshes in a shared volume.
- Hugging Face bucket storage is the durable mesh/manifest backend.

The current training stack is built around `Qwen/Qwen2.5-Coder-7B-Instruct`, Hydra config groups layered over the typed config schema in `config.py`, and a server-side reward path on Modal backed by Hugging Face bucket storage plus a shared Modal volume for GT meshes.

## Repo Layout

```text
config.py                     Typed root config schema + env loading
configs/config.yaml           Hydra root config for training/eval
configs/default.yaml          Single-file config used by non-Hydra data scripts
configs/rl/                   RL presets
configs/sft/                  SFT presets
configs/reward/               Reward presets
configs/experiment/           Hydra sweep presets

data/                         Objaverse filtering, manifest build, synthetic data, dataset splits
environments/blender_3d/      Dataset, Modal harness, rubric, environment wrapper
modal_infra/reward_server.py  Modal FastAPI reward API
training/sft_trainer.py       SFT entrypoint
training/rl_trainer.py        GRPO RL entrypoint
training/eval_runner.py       Evaluation entrypoint
scripts/                      Shell launchers for each phase
```

## Setup

```bash
pip install -e ".[all]"
```

Useful extras:

- `pip install -e ".[training]"`
- `pip install -e ".[modal]"`
- `pip install -e ".[data]"`
- `pip install -e ".[dev]"`

## Environment

Create `dev.env` from the example and fill in credentials:

```bash
cp dev.env.example dev.env
```

Important variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Teacher model for synthetic generation |
| `HF_TOKEN` | Hugging Face bucket access |
| `LLM3D_STORAGE__HF_BUCKET` | Bucket name |
| `LLM3D_STORAGE__HF_BUCKET_NAMESPACE` | HF namespace / org |
| `MODAL_TOKEN_ID` | Modal auth |
| `MODAL_TOKEN_SECRET` | Modal auth |
| `LLM3D_MODAL__ENDPOINT` | Deployed reward API base URL |
| `LLM3D_MODAL__AUTH_TOKEN` | Client token sent to the reward API |
| `REWARD_API_TOKEN` | Server-side token checked by the Modal API |
| `TINKER_API_KEY` | Tinker training access |
| `WANDB_API_KEY` | Optional experiment logging |

`config.load_config()` auto-loads `dev.env`. Variables prefixed with `LLM3D_` map into nested config fields using `__`, for example `LLM3D_MODAL__ENDPOINT -> modal.endpoint`.

## Config System

There are two config entry modes in the repo today:

1. Hydra entrypoints for training and eval.
   Files: `training/sft_trainer.py`, `training/rl_trainer.py`, `training/eval_runner.py`
   Root config: [`configs/config.yaml`](configs/config.yaml)

2. Direct `load_config()` calls for the data pipeline and helper scripts.
   Default YAML: [`configs/default.yaml`](configs/default.yaml)

The typed schema lives in [`config.py`](config.py). Hydra config groups are registered from [`configs/structured.py`](configs/structured.py).

### Main Config Sections

- `views`: render settings such as `num_views`, resolution, engine, lighting
- `objaverse_filter`: UID filtering thresholds and output path
- `storage`: HF bucket settings, cache, manifest key, Modal volume mesh subdir
- `synthetic_gen`: teacher model/provider/API, concurrency, thresholds, output paths
- `hard_mining`: RL hard prompt oversampling settings
- `dataset`: split ratios, curriculum, system prompt
- `modal`: CPU / memory / timeout / concurrency settings and API endpoint auth
- `metrics`: point counts, F-score thresholds, CLIP model settings
- `reward`: reward weights and per-check thresholds
- `sft`: LoRA and SFT hyperparameters
- `rl`: GRPO hyperparameters
- `eval`: test sizes and bootstrap settings
- `logging`: logging level and W&B config

### How To Set Configs

For reusable secrets or machine-local settings:

- edit `dev.env`
- or export env vars such as `LLM3D_MODAL__ENDPOINT=...`

For data jobs that accept a YAML path:

```bash
./scripts/filter_objaverse.sh configs/default.yaml
./scripts/generate_synthetic_data.sh configs/dev_5.yaml
```

For training and eval jobs, pass Hydra overrides through the shell wrappers:

```bash
./scripts/run_sft.sh sft.learning_rate=5e-5 sft.epochs=5
./scripts/run_rl.sh rl=fast_iter reward=geometry_heavy
./scripts/run_eval.sh eval.temperature=0.2 output_dir=./output/eval_debug
```

For multirun sweeps:

```bash
./scripts/run_rl.sh --multirun reward.geometry.resemblance.threshold=0.04,0.05,0.06
./scripts/run_rl.sh +experiment=reward_sweep --multirun
```

Available preset groups:

- RL presets: [`configs/rl/default.yaml`](configs/rl/default.yaml), [`configs/rl/fast_iter.yaml`](configs/rl/fast_iter.yaml), [`configs/rl/long_run.yaml`](configs/rl/long_run.yaml)
- Reward presets: [`configs/reward/default.yaml`](configs/reward/default.yaml), [`configs/reward/geometry_heavy.yaml`](configs/reward/geometry_heavy.yaml), [`configs/reward/clip_heavy.yaml`](configs/reward/clip_heavy.yaml), [`configs/reward/aggressive_gates.yaml`](configs/reward/aggressive_gates.yaml)
- Experiment sweeps: [`configs/experiment/reward_sweep.yaml`](configs/experiment/reward_sweep.yaml), [`configs/experiment/gate_ablation.yaml`](configs/experiment/gate_ablation.yaml)

## Current Defaults

Selected defaults from the live config:

- SFT base model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- SFT: `epochs=3`, `batch_size=8`, `grad_accum_steps=4`, `learning_rate=1e-4`
- RL: `algorithm=grpo`, `steps=1000`, `batch_size=16`, `num_completions=8`, `learning_rate=5e-6`
- Modal reward API: `reward_cpu=4`, `reward_memory_mb=8192`, `reward_timeout_s=600`, `reward_concurrency=50`
- Storage backend: `hf`
- Views: `num_views=4`, `resolution=[512, 512]`

## Reward

The current rubric is defined in [`environments/blender_3d/rubric.py`](environments/blender_3d/rubric.py) and configured by [`configs/reward/default.yaml`](configs/reward/default.yaml).

Reward is:

```text
reward =
  geometric_weight * geometry_score +
  text_alignment_weight * text_alignment_score +
  format_reward_weight * format_score
```

Default top-level weights:

- `geometric_weight = 0.7`
- `text_alignment_weight = 0.2`
- `format_reward_weight = 0.1`

### Geometry Score

`geometry_score` is the weighted average of binary checks:

- `non_empty`
- `import_bpy`
- `exec_success`
- `min_faces >= 4`
- `max_vertices <= 100000`
- `metrics_available`
- `resemblance`, defined as `f_score_005 >= 0.05`

### Text Alignment Score

Configured as a binary threshold:

- `clip_score >= 0.25`
- gated by `requires_resemblance=true` by default

Important: the current Modal `/reward/batch` and `/reward/single` flow does not pass a `clip_score` into the rubric, so `text_alignment_reward` is effectively `0` in the live reward API path unless that execution path is extended.

### Format Score

`format_score` is the weighted average of binary checks:

- `import_first`
- `has_comments`
- `clears_scene`
- `has_export`

### Reward Presets

- `reward=default`: balanced geometry / text-alignment / format weights
- `reward=geometry_heavy`: 0.85 / 0.10 / 0.05
- `reward=clip_heavy`: 0.50 / 0.40 / 0.10
- `reward=aggressive_gates`: stricter geometry thresholds and heavier resemblance weighting

## Launching Jobs

### Full Data Pipeline

```bash
./scripts/filter_objaverse.sh
./scripts/build_manifest.sh
./scripts/generate_synthetic_data.sh
./scripts/build_object_dataset.sh
```

What each stage does:

- [`scripts/filter_objaverse.sh`](scripts/filter_objaverse.sh): filters candidate Objaverse UIDs
- [`scripts/build_manifest.sh`](scripts/build_manifest.sh): joins captions and remotely ingests meshes into your HF bucket
- [`scripts/preload_modal_meshes.sh`](scripts/preload_modal_meshes.sh): syncs bucket meshes into the Modal volume used by reward and synthetic validation
- [`scripts/generate_synthetic_data.sh`](scripts/generate_synthetic_data.sh): teacher generation plus Blender validation, also writes `hard_prompts.csv`
- [`scripts/build_object_dataset.sh`](scripts/build_object_dataset.sh): produces `sft_train.jsonl`, `sft_val.jsonl`, `rl_prompts.jsonl`, and eval splits

### Modal Reward API

Deploy the reward API:

```bash
./scripts/deploy_reward_api.sh
```

This deploys [`modal_infra/reward_server.py`](modal_infra/reward_server.py), which exposes:

- `POST /reward/batch`
- `POST /reward/single`
- `POST /render`
- `POST /execute`
- `GET /health`

### Training

Run SFT:

```bash
./scripts/run_sft.sh
```

Run RL:

```bash
./scripts/run_rl.sh
```

Run eval:

```bash
./scripts/run_eval.sh
```

The shell wrappers only forward arguments. Equivalent direct entrypoints are:

```bash
python -m training.sft_trainer
python -m training.rl_trainer
python -m training.eval_runner
```

### Example Launches

Quick RL debug run:

```bash
./scripts/run_rl.sh rl=fast_iter reward=geometry_heavy logging.wandb_enabled=true
```

Longer RL run:

```bash
./scripts/run_rl.sh rl=long_run reward=aggressive_gates
```

Reward sweep:

```bash
./scripts/run_rl.sh +experiment=reward_sweep --multirun
```

Small end-to-end smoke test:

```bash
./scripts/run_dev_pipeline.sh
```

That uses [`configs/dev_5.yaml`](configs/dev_5.yaml) and runs the data flow on five objects.

## Training / Eval Notes

- RL hard mining is enabled by default and uses `datasets/hard_prompts.csv` when present.
- The RL trainer sends the current reward config with each reward batch request, so Hydra reward overrides apply server-side without redeploying the API.
- Eval reuses the same `Blender3DHarness` and reward API path as RL.
- If `wandb_enabled=false`, training still runs without W&B.

## Metrics

Primary metrics returned by the reward/eval path:

- `f_score_005`
- `chamfer`
- `hausdorff_90`
- `normal_consistency`
- execution rate
- geometry rate
- mean reward

Configured point counts:

- `metrics.num_sample_points_fast = 10000`
- `metrics.num_sample_points_eval = 100000`

Current implementation note: the live Modal reward batch path currently calls metrics with `10000` sample points.
