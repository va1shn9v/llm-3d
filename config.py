"""
Central configuration system using Pydantic models with YAML file loading.

Provides typed, validated configuration for every component of the pipeline:
bpy_lib, data generation, Modal infrastructure, training, and evaluation.

Usage:
    from config import load_config, ProjectConfig
    cfg = load_config("configs/default.yaml")
    # or with overrides:
    cfg = load_config("configs/default.yaml", modal_endpoint="https://...")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class BpyLibConfig(BaseModel):
    coordinate_range: float = 1.0
    max_vertices_warn: int = 100_000
    export_format: str = "obj"
    default_export_path: str = "/tmp/generated_mesh.obj"


class ViewConfig(BaseModel):
    num_views: int = 4
    resolution: tuple[int, int] = (512, 512)
    engine: str = "BLENDER_EEVEE_NEXT"
    elevation_deg: float = 25.0
    film_transparent: bool = True
    sun_energy: float = 3.0
    camera_distance_factor: float = 2.5


class PartGeneratorConfig(BaseModel):
    num_primitives: int = 50_000
    num_translations: int = 100_000
    num_bridge_loops: int = 50_000
    num_booleans: int = 50_000
    num_arrays: int = 50_000
    min_faces: int = 4
    cd_accept_threshold: float = 5e-3
    modal_batch_size: int = 256
    seed: int = 42


class InfinigenConfig(BaseModel):
    categories: list[str] = Field(default_factory=lambda: [
        "chair", "table_dining", "sofa", "lamp", "bottle", "cup", "bowl",
        "vase", "toilet", "shelf", "tv_stand", "desk", "bathtub", "jar", "plate",
    ])
    objects_per_category: int = 5_000
    cd_accept_threshold: float = 5e-3
    infinigen_path: str = "./third_party/infinigen"
    output_dir: str = "./data/infinigen_objects"


class QualityGateConfig(BaseModel):
    min_faces: int = 4
    max_vertices: int = 100_000
    cd_threshold: float = 5e-3
    min_f_score_005: float = 0.05


class DatasetConfig(BaseModel):
    sft_train_ratio: float = 0.90
    sft_val_ratio: float = 0.05
    eval_id_ratio: float = 0.03
    eval_ood_ratio: float = 0.02
    curriculum: bool = True
    difficulty_weights: dict[str, float] = Field(default_factory=lambda: {
        "num_parts": 0.4,
        "code_length_tokens": 0.3,
        "max_part_complexity": 0.3,
    })
    view_count_weights: dict[int, float] = Field(default_factory=lambda: {
        1: 0.10, 2: 0.20, 3: 0.20, 4: 0.40, 6: 0.10,
    })
    system_prompt: str = (
        "You are a 3D modeling assistant. Given images of a 3D object, "
        "generate Blender Python code using the bpy_lib API that reconstructs "
        "the object. Output executable code only, no explanations."
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
    voxel_grid_size: int = 32


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
    format_reward_weight: float = 0.1


class SFTConfig(BaseModel):
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 64
    target_modules: list[str] = Field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
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


class EvalConfig(BaseModel):
    id_test_size: int = 1500
    ood_objaverse: int = 500
    ood_gso: int = 300
    ood_unseen_categories: int = 200
    unseen_categories: list[str] = Field(default_factory=lambda: [
        "fork", "spoon", "tv", "window", "door",
    ])
    view_counts: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 6])
    temperature: float = 0.0
    bootstrap_samples: int = 10_000


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "./logs"
    wandb_project: str = "image-to-3d-rlvr"
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

    bpy_lib: BpyLibConfig = Field(default_factory=BpyLibConfig)
    views: ViewConfig = Field(default_factory=ViewConfig)
    part_generator: PartGeneratorConfig = Field(default_factory=PartGeneratorConfig)
    infinigen: InfinigenConfig = Field(default_factory=InfinigenConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {"env_prefix": "LLM3D_", "extra": "ignore"}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(
    yaml_path: str | Path | None = None,
    **overrides: Any,
) -> ProjectConfig:
    """Load configuration from optional YAML file with programmatic overrides.

    Precedence (highest wins): overrides > YAML > env vars > defaults.
    """
    data: dict = {}

    if yaml_path is not None:
        path = Path(yaml_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

    if overrides:
        data = _deep_merge(data, overrides)

    return ProjectConfig(**data)
