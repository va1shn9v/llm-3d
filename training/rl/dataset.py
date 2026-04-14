"""Prompt sampling for RL training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import ProjectConfig
from data.storage import open_read

log = logging.getLogger(__name__)


class RLPromptSampler:
    """Uniform prompt sampler for RL training."""

    def __init__(self, jsonl_path: str | Path, cfg: ProjectConfig):
        self.rng = np.random.default_rng(cfg.seed)
        self.items: list[dict[str, Any]] = []

        with open_read(str(jsonl_path), cfg.storage) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                meta = record.get("metadata", {})
                self.items.append(
                    {
                        "object_id": meta.get("object_id", ""),
                        "caption": meta.get("caption", ""),
                        "gt_mesh_path": meta.get("gt_mesh_path", ""),
                        "messages": record.get("messages", []),
                    }
                )

        if not self.items:
            raise ValueError(f"No RL prompts found in {jsonl_path}")

        log.info("Loaded %d RL prompts from %s", len(self.items), jsonl_path)

    def sample(self, n: int) -> list[dict[str, Any]]:
        indices = self.rng.choice(len(self.items), size=min(n, len(self.items)), replace=False)
        return [self.items[i] for i in indices]
