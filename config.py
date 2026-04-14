"""
Central configuration system for the retained RL + eval workflow.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

import yaml
from pydantic import BaseModel, Field

from prompts import DEFAULT_SYSTEM_PROMPT

_ENV_FILE = "dev.env"
_DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def _load_env_file(path: str | Path = _ENV_FILE) -> None:
    """Load variables from an env file into ``os.environ`` (won't overwrite).

    Keeps third-party SDKs (Tinker, Modal, W&B) working via their own env vars.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class DatasetConfig(BaseModel):
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class ModalConfig(BaseModel):
    endpoint: str = ""
    auth_token: str = ""
    volume_name: str = "llm3d-data"


class BinaryRewardConfig(BaseModel):
    enabled: bool = True
    weight: float = 1.0


class NumericBinaryRewardConfig(BinaryRewardConfig):
    threshold: float


class GeometryRewardConfig(BaseModel):
    non_empty: BinaryRewardConfig = Field(default_factory=BinaryRewardConfig)
    import_bpy: BinaryRewardConfig = Field(default_factory=BinaryRewardConfig)
    exec_success: BinaryRewardConfig = Field(default_factory=BinaryRewardConfig)
    min_faces: NumericBinaryRewardConfig = Field(
        default_factory=lambda: NumericBinaryRewardConfig(threshold=4)
    )
    max_vertices: NumericBinaryRewardConfig = Field(
        default_factory=lambda: NumericBinaryRewardConfig(threshold=100_000)
    )
    metrics_available: BinaryRewardConfig = Field(default_factory=BinaryRewardConfig)
    resemblance: NumericBinaryRewardConfig = Field(
        default_factory=lambda: NumericBinaryRewardConfig(threshold=0.05)
    )


class FormatRewardConfig(BaseModel):
    import_first: BinaryRewardConfig = Field(default_factory=BinaryRewardConfig)
    has_comments: BinaryRewardConfig = Field(default_factory=BinaryRewardConfig)
    clears_scene: BinaryRewardConfig = Field(default_factory=BinaryRewardConfig)
    has_export: BinaryRewardConfig = Field(default_factory=BinaryRewardConfig)


class RewardConfig(BaseModel):
    geometric_weight: float = 0.9
    format_reward_weight: float = 0.1
    geometry: GeometryRewardConfig = Field(default_factory=GeometryRewardConfig)
    format: FormatRewardConfig = Field(default_factory=FormatRewardConfig)


class RLConfig(BaseModel):
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    lora_rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    init_state_path: str = ""
    steps: int = 1000
    batch_size: int = 16
    num_completions: int = 8
    learning_rate: float = 5e-6
    temperature: float = 0.7
    max_new_tokens: int = 4096
    stop: list[str] = Field(default_factory=lambda: ["<|im_end|>"])
    checkpoint_every: int = 100
    log_every: int = 10
    prompt_path: str = "datasets/rl_prompts.jsonl"


class EvalConditionConfig(BaseModel):
    enabled: bool = True
    base_model: str = ""
    model_path: str = ""
    temperature: float = 0.0
    max_new_tokens: int = 4096
    num_samples: int = 1
    stop: list[str] = Field(default_factory=lambda: ["<|im_end|>"])


class EvalConditionsConfig(BaseModel):
    baseline: EvalConditionConfig = Field(
        default_factory=lambda: EvalConditionConfig(enabled=True)
    )
    candidate: EvalConditionConfig = Field(
        default_factory=lambda: EvalConditionConfig(enabled=False)
    )
    reference: EvalConditionConfig = Field(
        default_factory=lambda: EvalConditionConfig(enabled=False)
    )


class EvalConfig(BaseModel):
    id_path: str = "datasets/eval_id.jsonl"
    ood_path: str = ""
    temperature: float = 0.0
    bootstrap_samples: int = 10_000
    batch_size: int = 16
    max_concurrent_tinker: int = 8
    save_details: bool = True
    max_cases_per_test_set: int | None = None
    conditions: EvalConditionsConfig = Field(default_factory=EvalConditionsConfig)
    selection: "EvalSelectionConfig" = Field(default_factory=lambda: EvalSelectionConfig())


class EvalSelectionConfig(BaseModel):
    output_path: str = "datasets/eval_id.jsonl"
    manifest_path: str = ""
    target_size: int = 500
    max_per_category: int = 2


EvalConfig.model_rebuild()


class StorageConfig(BaseModel):
    """Remote storage via HuggingFace buckets."""
    backend: str = "hf"
    hf_bucket: str = "llm3d-data"
    hf_bucket_namespace: str = ""
    cache_dir: str = ".cache/hf_data"
    local_manifest_path: str = "data/manifest.jsonl"
    manifest_key: str = "datasets/manifest.jsonl"
    mesh_prefix: str = "meshes"
    modal_volume_mesh_subdir: str = "meshes"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "./logs"
    wandb_project: str = "text-to-3d-rlvr"
    wandb_enabled: bool = False


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Root configuration — aggregates all sub-configs."""

    model_config = {"extra": "ignore"}

    project_name: str = "llm-3d"
    seed: int = 42
    output_dir: str = "./output"

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _parse_override_value(raw_value: str) -> Any:
    if raw_value == "":
        return ""
    return yaml.safe_load(raw_value)


def _set_nested_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = data
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        raise ValueError("Override key cannot be empty")

    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def apply_cli_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    updated = dict(data)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override {override!r}. Expected dotted assignments like rl.learning_rate=1e-5."
            )
        key, raw_value = override.split("=", 1)
        _set_nested_value(updated, key, _parse_override_value(raw_value))
    return updated


def load_config(
    yaml_path: str | Path | None = None,
    cli_overrides: Sequence[str] | None = None,
    **overrides: Any,
) -> ProjectConfig:
    """Build a ``ProjectConfig`` from YAML + env overlay + optional CLI overrides."""
    _load_env_file()

    path = Path(yaml_path) if yaml_path is not None else _DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}

    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    modal = data.setdefault("modal", {})
    storage = data.setdefault("storage", {})
    logging_cfg = data.setdefault("logging", {})

    modal["endpoint"] = os.environ.get("LLM3D_MODAL__ENDPOINT", modal.get("endpoint", ""))
    modal["auth_token"] = os.environ.get("LLM3D_MODAL__AUTH_TOKEN", modal.get("auth_token", ""))
    modal["volume_name"] = os.environ.get("LLM3D_MODAL__VOLUME_NAME", modal.get("volume_name", "llm3d-data"))

    storage["hf_bucket"] = os.environ.get("LLM3D_STORAGE__HF_BUCKET", storage.get("hf_bucket", "llm3d-data"))
    storage["hf_bucket_namespace"] = os.environ.get(
        "LLM3D_STORAGE__HF_BUCKET_NAMESPACE",
        storage.get("hf_bucket_namespace", ""),
    )
    storage["cache_dir"] = os.environ.get("LLM3D_STORAGE__CACHE_DIR", storage.get("cache_dir", ".cache/hf_data"))

    if "WANDB_PROJECT" in os.environ:
        logging_cfg["wandb_project"] = os.environ["WANDB_PROJECT"]

    if cli_overrides:
        data = apply_cli_overrides(data, cli_overrides)

    if overrides:
        data = _deep_merge(data, overrides)

    return ProjectConfig(**data)


def build_config_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted config overrides like rl.learning_rate=1e-5",
    )
    return parser


def load_config_from_cli(
    *,
    description: str,
    argv: Sequence[str] | None = None,
) -> ProjectConfig:
    args = build_config_arg_parser(description).parse_args(list(argv) if argv is not None else None)
    return load_config(args.config, cli_overrides=args.overrides)
