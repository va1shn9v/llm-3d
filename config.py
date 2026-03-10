"""
Central configuration system using Pydantic models.

Provides typed, validated configuration for every component of the pipeline:
data curation, synthetic generation, Modal infrastructure, training, and evaluation.

CLI entry points use Hydra for config groups, overrides, and multirun sweeps::

    python -m training.rl_trainer reward=geometry_heavy rl.learning_rate=1e-5
    python -m training.rl_trainer --multirun reward.f_score_target=0.4,0.6,0.8

For programmatic use::

    from config import load_config, ProjectConfig
    cfg = load_config()                          # Pydantic defaults + env vars
    cfg = load_config("configs/default.yaml")    # from YAML
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

_ENV_FILE = "dev.env"


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

class ViewConfig(BaseModel):
    """Rendering config — used for CLIP eval rendering."""
    num_views: int = 4
    resolution: tuple[int, int] = (512, 512)
    engine: str = "CYCLES"
    elevation_deg: float = 25.0
    film_transparent: bool = True
    sun_energy: float = 3.0
    camera_distance_factor: float = 2.5
    cycles_samples: int = 128


class QualityGateConfig(BaseModel):
    min_faces: int = 4
    max_vertices: int = 100_000
    cd_threshold: float = 0.05
    min_f_score_005: float = 0.05


class ObjaverseFilterConfig(BaseModel):
    """Objaverse++ quality filtering."""
    quality_tiers: list[str] = Field(default_factory=lambda: ["High", "Superior"])
    exclude_scenes: bool = True
    exclude_transparent: bool = True
    max_uids: int | None = None
    output_path: str = "datasets/filtered_uids.json"


class SyntheticGenConfig(BaseModel):
    """Teacher LLM synthetic data generation."""
    teacher_model: str = "gpt-4o-mini"
    teacher_provider: str = "openai"
    temperature: float = 0.7
    max_attempts_per_caption: int = 3
    cd_threshold: float = 0.05
    f_score_threshold: float = 0.1
    min_faces: int = 4
    max_vertices: int = 100_000
    batch_size: int = 50
    max_concurrent_blender: int = 32
    output_path: str = "datasets/synthetic_sft.jsonl"
    hard_prompts_path: str = "datasets/hard_prompts.csv"
    checkpoint_path: str = "datasets/synthetic_checkpoint.json"


class HardMiningConfig(BaseModel):
    """Controls hard prompt sampling during RLVR."""
    enabled: bool = True
    hard_prompts_csv: str = "datasets/hard_prompts.csv"
    hard_prompt_ratio: float = 0.4
    min_failure_rate: float = 0.5
    min_attempts: int = 2


class DatasetConfig(BaseModel):
    sft_train_ratio: float = 0.90
    sft_val_ratio: float = 0.05
    eval_id_ratio: float = 0.03
    eval_ood_ratio: float = 0.02
    curriculum: bool = True
    difficulty_weights: dict[str, float] = Field(default_factory=lambda: {
        "code_length_tokens": 0.4,
        "vertex_count": 0.3,
        "face_count": 0.3,
    })
    system_prompt: str = (
        "You are an expert Blender Python developer. Given a text description of a 3D object, "
        "write a complete bpy script that creates the described geometry. Use standard bpy "
        "operations (primitives, BMesh, modifiers, booleans, curves). The script must: "
        "1. Start with `import bpy` and clear the default scene. "
        "2. Create the geometry described. "
        "3. Export the result to OBJ at the path from os.environ['EXPORT_PATH']. "
        "Output only the Python code, no explanations."
    )


class ModalConfig(BaseModel):
    blender_version: str = "4.2.0"
    blender_cpu: int = 2
    blender_memory_mb: int = 4096
    blender_timeout_s: int = 150
    exec_timeout_s: int = 120
    metrics_cpu: int = 2
    metrics_memory_mb: int = 2048
    metrics_timeout_s: int = 60
    render_cpu: int = 2
    render_memory_mb: int = 4096
    render_timeout_s: int = 300
    reward_cpu: int = 4
    reward_memory_mb: int = 8192
    reward_timeout_s: int = 600
    reward_concurrency: int = 50
    reward_keep_warm: int = 1
    max_parallel_workers: int = 64
    endpoint: str = ""
    auth_token: str = ""
    volume_name: str = "llm3d-data"


class MetricsConfig(BaseModel):
    num_sample_points_fast: int = 10_000
    num_sample_points_eval: int = 100_000
    f_score_thresholds: list[float] = Field(default_factory=lambda: [0.01, 0.05])
    hausdorff_percentile: float = 90.0
    clip_model: str = "openai/clip-vit-large-patch14"
    clip_num_views: int = 4


class RewardConfig(BaseModel):
    gate_empty: float = 0.0
    gate_no_import: float = 0.0
    gate_exec_fail: float = 0.05
    gate_no_geometry: float = 0.10
    gate_degenerate: float = 0.15
    gate_metrics_fail: float = 0.15
    gate_no_resemblance: float = 0.20
    quality_floor: float = 0.30
    quality_ceil: float = 1.0
    f_score_target: float = 0.6
    geometric_weight: float = 0.7
    text_alignment_weight: float = 0.2
    format_reward_weight: float = 0.1


class TinkerConfig(BaseModel):
    """Tinker platform settings.  TINKER_API_KEY is read from env by the SDK."""
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"


class SFTConfig(BaseModel):
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 64
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_seq_length: int = 32768
    eval_every_n_steps: int = 500
    eval_num_samples: int = 200
    eval_temperature: float = 0.0
    train_path: str = "datasets/sft_train.jsonl"
    val_path: str = "datasets/sft_val.jsonl"


class RLConfig(BaseModel):
    algorithm: str = "grpo"
    sft_checkpoint: str = "sft-epoch-2"
    steps: int = 1000
    batch_size: int = 16
    num_completions: int = 8
    learning_rate: float = 5e-6
    kl_coeff: float = 0.05
    clip_ratio: float = 0.2
    temperature: float = 0.7
    max_new_tokens: int = 4096
    checkpoint_every: int = 100
    log_every: int = 10
    prompt_path: str = "datasets/rl_prompts.jsonl"


class EvalConfig(BaseModel):
    id_test_size: int = 1500
    ood_objaverse: int = 500
    ood_gso: int = 300
    ood_unseen_categories: int = 200
    unseen_categories: list[str] = Field(default_factory=lambda: [
        "fork", "spoon", "tv", "window", "door",
    ])
    temperature: float = 0.0
    bootstrap_samples: int = 10_000


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "./logs"
    wandb_project: str = "text-to-3d-rlvr"
    wandb_enabled: bool = False


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class ProjectConfig(BaseSettings):
    """Root configuration — aggregates all sub-configs."""

    project_name: str = "llm-3d"
    seed: int = 42
    output_dir: str = "./output"
    data_dir: str = "./data"
    blender_path: str = "blender"

    views: ViewConfig = Field(default_factory=ViewConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    objaverse_filter: ObjaverseFilterConfig = Field(default_factory=ObjaverseFilterConfig)
    synthetic_gen: SyntheticGenConfig = Field(default_factory=SyntheticGenConfig)
    hard_mining: HardMiningConfig = Field(default_factory=HardMiningConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    tinker: TinkerConfig = Field(default_factory=TinkerConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {
        "env_prefix": "LLM3D_",
        "env_nested_delimiter": "__",
        "extra": "ignore",
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_config(
    yaml_path: str | Path | None = None,
    **overrides: Any,
) -> ProjectConfig:
    """Build a ``ProjectConfig`` from optional YAML + keyword overrides.

    For CLI usage with config groups, overrides, and multirun sweeps, prefer
    the Hydra entry points (``python -m training.rl_trainer ...``).

    This function remains available for programmatic and data-pipeline use.
    """
    _load_env_file()

    data: dict = {}

    if yaml_path is not None:
        path = Path(yaml_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

    if overrides:
        data.update(overrides)

    return ProjectConfig(**data)
