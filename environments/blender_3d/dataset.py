"""
Blender3DDataset — provides text prompts for text-to-3D code generation training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert Blender Python developer. Given a text description of a 3D object, "
    "write a complete bpy script that creates the described geometry. Use standard bpy "
    "operations (primitives, BMesh, modifiers, booleans, curves). The script must: "
    "1. Start with `import bpy` and clear the default scene. "
    "2. Create the geometry described. "
    "3. Export the result to OBJ at the path from os.environ['EXPORT_PATH']. "
    "Output only the Python code, no explanations."
)


class Blender3DDataset:
    """Dataset of text descriptions → Blender Python code problems.

    Each item provides:
    - text:         str (the caption / text description)
    - object_id:    str (UID for looking up GT mesh)
    - gt_code:      str (for SFT reference, not used in RL)
    - gt_mesh_path: str (path to ground-truth mesh for reward computation)
    - prompt:       str (formatted user prompt text)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
    ):
        self.jsonl_path = Path(jsonl_path)
        self._items: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        with open(self.jsonl_path) as f:
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.format_prompt_text(item["text"])},
        ]
        return messages

    @staticmethod
    def format_prompt_text(caption: str) -> str:
        return (
            f"Create a 3D model of: {caption}. "
            f"Write a complete Blender Python script using the bpy module."
        )
