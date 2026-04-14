# llm-3d

Train and evaluate instruct models that generate Blender Python for 3D object creation.

The project is built around one loop:

1. Build a curated evaluation set from an existing object manifest.
2. Sync ground-truth meshes into Modal and deploy the reward API.
3. Run direct RL on an instruct model.
4. Evaluate checkpoints against the same reward path.

## Setup

Install dependencies:

```bash
pip install -e ".[all]"
```

Create a local env file:

```bash
cp dev.env.example dev.env
```

Important environment variables:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face bucket access |
| `LLM3D_STORAGE__HF_BUCKET` | HF bucket name |
| `LLM3D_STORAGE__HF_BUCKET_NAMESPACE` | HF namespace / org |
| `MODAL_TOKEN_ID` | Modal auth |
| `MODAL_TOKEN_SECRET` | Modal auth |
| `LLM3D_MODAL__ENDPOINT` | Reward API base URL |
| `LLM3D_MODAL__AUTH_TOKEN` | Client token sent to the reward API |
| `LLM3D_MODAL__VOLUME_NAME` | Modal volume name |
| `REWARD_API_TOKEN` | Server-side token checked by the reward API |
| `TINKER_API_KEY` | Tinker access |
| `WANDB_API_KEY` | Optional W&B logging |

## Config

The default config lives in [configs/config.yaml](/Users/vaishnavp/Desktop/llm-3d/configs/config.yaml).

Use dotted CLI overrides for experiments:

```bash
./scripts/run_rl.sh rl.learning_rate=1e-5 rl.steps=200
./scripts/run_eval.sh eval.max_cases_per_test_set=100
./scripts/run_eval.sh eval.conditions.candidate.enabled=true eval.conditions.candidate.model_path=ckpts/rl-step-500
```

Main config sections:

- `storage.*`: HF manifest and mesh storage
- `modal.*`: reward API endpoint and auth
- `reward.*`: geometry and format reward settings
- `rl.*`: direct RL hyperparameters and prompt dataset path
- `eval.*`: eval dataset paths, selection settings, and model conditions
- `logging.*`: logging and W&B

## Data

The eval-set builder expects a manifest of JSONL records with:

```json
{"uid": "...", "caption": "...", "mesh_path": "..."}
```

By default the manifest is read from `storage.manifest_key` in the configured HF bucket, or from `eval.selection.manifest_path` if you set one explicitly.

The curated eval output is written to `eval.selection.output_path`, which defaults to `datasets/eval_id.jsonl`.

## Commands

Build the curated eval set:

```bash
./scripts/build_eval_dataset.sh
```

Sync meshes into the Modal volume:

```bash
./scripts/preload_modal_meshes.sh
```

Deploy the reward stack:

```bash
./scripts/deploy_reward_api.sh
```

Run direct RL:

```bash
./scripts/run_rl.sh
```

Run evaluation:

```bash
./scripts/run_eval.sh
```

## Reward

Reward is computed as:

```text
reward =
  geometric_weight * geometry_score +
  format_reward_weight * format_score
```

Geometry checks include execution success, face/vertex limits, metric availability, and resemblance via `f_score_005`.

Format checks include `import bpy` structure, scene clearing, comments, and export behavior.

## Code Layout

- [training/rl](/Users/vaishnavp/Desktop/llm-3d/training/rl): RL prompt sampling and training loop
- [training/eval](/Users/vaishnavp/Desktop/llm-3d/training/eval): evaluation runner and condition handling
- [training/common](/Users/vaishnavp/Desktop/llm-3d/training/common): shared Tinker and tracking helpers
- [data/eval_dataset.py](/Users/vaishnavp/Desktop/llm-3d/data/eval_dataset.py): eval-set builder
- [environments/blender_3d](/Users/vaishnavp/Desktop/llm-3d/environments/blender_3d): prompt dataset, reward harness, rubric
- [modal_infra](/Users/vaishnavp/Desktop/llm-3d/modal_infra): Blender execution, metrics, and reward API

## Typical Flow

```bash
./scripts/build_eval_dataset.sh
./scripts/preload_modal_meshes.sh
./scripts/deploy_reward_api.sh
./scripts/run_rl.sh
./scripts/run_eval.sh
```
