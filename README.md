# llm-3d

Direct RL and evaluation for Blender Python code generation.

This repo is intentionally scoped to two workflows:

1. Build a curated evaluation set from an existing manifest of objects already stored in your HF bucket.
2. Run direct RL on an instruct model, then evaluate generated Blender code through the same Modal reward path.

## What Remains

- `training/rl/`: RL sampler + direct GRPO trainer
- `training/eval/`: Tinker sampling + Modal-backed evaluation
- `training/common/`: shared Tinker and experiment-tracking helpers
- `data/eval_dataset.py`: builds a diverse eval set from an existing manifest
- `environments/blender_3d/`: prompt dataset, reward harness, and rubric
- `modal_infra/`: Blender execution worker, metrics worker, and reward API
- `scripts/build_eval_dataset.sh`: curate the eval set
- `scripts/preload_modal_meshes.sh`: sync GT meshes from the HF bucket into the Modal volume
- `scripts/deploy_reward_api.sh`: deploy the reward API
- `scripts/run_rl.sh`: direct RL
- `scripts/run_eval.sh`: evaluation

Removed from the repo:

- synthetic teacher generation
- SFT training
- Objaverse filtering / manifest ingestion
- image-conditioned preprocessing and render workers
- stale SFT/image-era configs

## Setup

```bash
pip install -e ".[all]"
```

Create `dev.env` from the example:

```bash
cp dev.env.example dev.env
```

Important variables:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face bucket access |
| `LLM3D_STORAGE__HF_BUCKET` | Bucket name |
| `LLM3D_STORAGE__HF_BUCKET_NAMESPACE` | HF namespace / org |
| `MODAL_TOKEN_ID` | Modal auth |
| `MODAL_TOKEN_SECRET` | Modal auth |
| `LLM3D_MODAL__ENDPOINT` | Deployed reward API base URL |
| `LLM3D_MODAL__AUTH_TOKEN` | Client token sent to the reward API |
| `LLM3D_MODAL__VOLUME_NAME` | Modal volume name |
| `REWARD_API_TOKEN` | Server-side token checked by the reward API |
| `TINKER_API_KEY` | Tinker access |
| `WANDB_API_KEY` | Optional experiment logging |

`config.load_config()` reads `configs/config.yaml` and applies a small explicit env overlay for the Modal/HF fields above.

## Config

The repo now uses a single default config file: `configs/config.yaml`.

Override any value directly on the CLI with dotted `key=value` assignments:

```bash
./scripts/run_rl.sh rl.learning_rate=1e-5 rl.steps=200
./scripts/run_eval.sh eval.max_cases_per_test_set=100
./scripts/run_eval.sh eval.conditions.candidate.enabled=true eval.conditions.candidate.model_path=ckpts/rl-step-500
```

The active config surface is:

- `dataset.system_prompt`
- `storage.*` for the manifest and HF bucket
- `modal.endpoint`, `modal.auth_token`, `modal.volume_name`
- `reward.*` for geometry/format scoring
- `rl.*` for direct RL hyperparameters and prompt dataset path
- `eval.*` for eval dataset paths, selection settings, and comparison conditions
- `logging.*`

## Eval Set

The repo assumes you already have a manifest of `{uid, caption, mesh_path}` entries, either locally or in the configured HF bucket.

Build a curated eval set:

```bash
./scripts/build_eval_dataset.sh
```

By default this writes `datasets/eval_id.jsonl` using:

- `eval.selection.output_path`
- `eval.selection.manifest_path` if set, otherwise `storage.manifest_key`
- `eval.selection.target_size`
- `eval.selection.max_per_category`

The selector is conservative: it filters obviously noisy captions and round-robins across inferred categories to keep the set diverse.

## Reward Path

Reward is intentionally simple and live:

```text
reward =
  geometric_weight * geometry_score +
  format_reward_weight * format_score
```

Geometry checks:

- non-empty code
- `import bpy`
- execution success
- minimum face count
- maximum vertex count
- metrics available
- resemblance via `f_score_005`

Format checks:

- import first
- has comments
- clears scene
- has export

There is no dead CLIP/text-alignment path in the active repo.

## Modal

Sync meshes into the Modal volume used by reward/eval:

```bash
./scripts/preload_modal_meshes.sh
```

Deploy the reward API:

```bash
./scripts/deploy_reward_api.sh
```

Active endpoints:

- `POST /reward/batch`
- `POST /reward/single`
- `POST /execute`
- `GET /health`
- `GET /artifacts/pair/{uid}`
- `GET /artifacts/gt/{uid}`

## Training

Run direct RL on the instruct base model configured in `rl.base_model`:

```bash
./scripts/run_rl.sh
```

The RL path lives under `training/rl/` and expects a real Tinker client plus a real prompt dataset at `rl.prompt_path`. There is no dummy fallback.

## Evaluation

Run evaluation against `eval.id_path` and optional `eval.ood_path`:

```bash
./scripts/run_eval.sh
```

Each enabled eval condition can point either at:

- a raw base model via `base_model`
- a trained adapter/checkpoint via `model_path`

The default intended comparison is:

- `baseline`: base instruct model
- `candidate`: current RL run
- `reference`: optional extra checkpoint
