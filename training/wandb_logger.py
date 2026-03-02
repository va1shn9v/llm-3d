"""
Weights & Biases logging wrapper.

Provides a thin abstraction so the rest of the training code only calls
`WandbLogger` and never touches `wandb` directly.  When wandb is disabled
(or not installed) every method is a silent no-op.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import LoggingConfig

log = logging.getLogger(__name__)

_HAS_WANDB = True
try:
    import wandb
except ImportError:
    _HAS_WANDB = False


class WandbLogger:
    """Lazy, config-driven wandb logger."""

    def __init__(
        self,
        cfg: LoggingConfig,
        *,
        run_name: str | None = None,
        tags: list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
    ):
        self._enabled = cfg.wandb_enabled and _HAS_WANDB
        self._run: Any = None

        if cfg.wandb_enabled and not _HAS_WANDB:
            log.warning("wandb logging requested but `wandb` is not installed — skipping")
            return

        if not self._enabled:
            return

        self._run = wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            tags=tags,
            config=extra_config or {},
            reinit=True,
        )
        log.info("wandb run started: %s", self._run.url)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._run is not None

    def log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        if not self.enabled:
            return
        wandb.log(payload, step=step)

    def log_summary(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for k, v in payload.items():
            wandb.run.summary[k] = v

    def define_metric(self, name: str, *, step_metric: str = "step") -> None:
        if not self.enabled:
            return
        wandb.define_metric(name, step_metric=step_metric)

    def finish(self) -> None:
        if not self.enabled:
            return
        wandb.finish()
        self._run = None
