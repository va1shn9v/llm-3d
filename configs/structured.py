"""
Register Pydantic config models as Hydra structured configs via hydra-zen.

Importing this module populates the Hydra ConfigStore so that CLI overrides
are type-checked against the Pydantic schemas at instantiation time.
"""

from __future__ import annotations

from hydra_zen import ZenStore, builds

from config import (
    DatasetConfig,
    EvalConfig,
    HardMiningConfig,
    LoggingConfig,
    MetricsConfig,
    ModalConfig,
    ObjaverseFilterConfig,
    ProjectConfig,
    QualityGateConfig,
    RewardConfig,
    RLConfig,
    SFTConfig,
    SyntheticGenConfig,
    TinkerConfig,
    ViewConfig,
)

store = ZenStore(name="llm3d")

_SUB_CONFIGS = {
    "reward": RewardConfig,
    "rl": RLConfig,
    "sft": SFTConfig,
}

_INTERNAL_CONFIGS = {
    "views": ViewConfig,
    "quality_gate": QualityGateConfig,
    "objaverse_filter": ObjaverseFilterConfig,
    "synthetic_gen": SyntheticGenConfig,
    "hard_mining": HardMiningConfig,
    "dataset": DatasetConfig,
    "modal": ModalConfig,
    "metrics": MetricsConfig,
    "tinker": TinkerConfig,
    "eval": EvalConfig,
    "logging": LoggingConfig,
}

for group_name, model_cls in _SUB_CONFIGS.items():
    cfg_node = builds(model_cls, populate_full_signature=True)
    store(cfg_node, group=group_name, name="default")

for name, model_cls in _INTERNAL_CONFIGS.items():
    builds(model_cls, populate_full_signature=True)

ProjectCfg = builds(ProjectConfig, populate_full_signature=True)
store(ProjectCfg, name="config")


def register_configs() -> None:
    """Call once at app startup to push all structured configs into Hydra's store."""
    store.add_to_hydra_store(overwrite_ok=True)
