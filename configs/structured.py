"""
Register Pydantic config models as Hydra structured configs via hydra-zen.

Importing this module populates the Hydra ConfigStore so that CLI overrides
are type-checked against the Pydantic schemas at instantiation time.
"""

from __future__ import annotations

from hydra_zen import ZenStore, builds

from config import (
    ProjectConfig,
    RewardConfig,
    RLConfig,
    SFTConfig,
)

store = ZenStore(name="llm3d")

_SUB_CONFIGS = {
    "reward": RewardConfig,
    "rl": RLConfig,
    "sft": SFTConfig,
}

for group_name, model_cls in _SUB_CONFIGS.items():
    cfg_node = builds(model_cls, populate_full_signature=True)
    store(cfg_node, group=group_name, name="default")

ProjectCfg = builds(ProjectConfig, populate_full_signature=True)
store(ProjectCfg, name="config")


def register_configs() -> None:
    """Call once at app startup to push all structured configs into Hydra's store."""
    store.add_to_hydra_store(overwrite_ok=True)
