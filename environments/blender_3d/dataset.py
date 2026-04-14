"""
Prompt dataset for Blender code generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import StorageConfig
from data.storage import open_read
from prompts import DEFAULT_SYSTEM_PROMPT, format_user_prompt

log = logging.getLogger(__name__)

class Blender3DDataset:
    """Dataset of prompt records for Blender Python generation.

    Each item provides:
    - text: caption text
    - object_id: UID for reward lookup
    - gt_code: optional reference code
    - gt_mesh_path: path/URI to the ground-truth mesh
    - prompt: rendered user prompt text
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        storage_cfg: StorageConfig | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.jsonl_path = str(jsonl_path)
        self.storage_cfg = storage_cfg
        self.system_prompt = system_prompt
        self._items: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        with open_read(self.jsonl_path, self.storage_cfg) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._items.append(record)
        log.info(f"Loaded {len(self._items)} items from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._items[idx]
        meta = record.get("metadata", {})

        gt_code = ""
        messages = record.get("messages", [])
        for msg in messages:
            if msg["role"] == "assistant":
                gt_code = msg["content"]
                break

        caption = meta.get("caption", "")

        return {
            "text": caption,
            "object_id": meta.get("object_id", f"item_{idx}"),
            "gt_code": gt_code,
            "gt_mesh_path": meta.get("gt_mesh_path", ""),
            "prompt": self.format_prompt_text(caption),
        }

    def format_prompt(self, item: dict[str, Any]) -> list[dict]:
        """Format as text-only chat messages."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.format_prompt_text(item["text"])},
        ]
        return messages

    @staticmethod
    def format_prompt_text(caption: str) -> str:
        return format_user_prompt(caption)
