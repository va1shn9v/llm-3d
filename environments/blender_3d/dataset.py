"""
Blender3DDataset — provides (multi-view images, category) prompts for training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a 3D modeling assistant. Given images of a 3D object, "
    "generate Blender Python code using the bpy_lib API that reconstructs "
    "the object. Output executable code only, no explanations."
)


class Blender3DDataset:
    """Dataset of (multi-view images, category) → code problems.

    Each item provides:
    - images:    list of PIL Images (the views)
    - category:  str (object category name)
    - object_id: str (for looking up GT mesh)
    - gt_code:   str (for SFT reference, not used in RL)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        images_base_dir: str | Path = ".",
        num_views: int = 4,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.images_base = Path(images_base_dir)
        self.num_views = num_views
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

        images = []
        messages = record.get("messages", [])
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image":
                            img_path = self.images_base / item["image_url"]
                            try:
                                images.append(Image.open(img_path).convert("RGBA"))
                            except Exception:
                                pass

        images = images[:self.num_views]

        gt_code = ""
        for msg in messages:
            if msg["role"] == "assistant":
                gt_code = msg["content"]
                break

        return {
            "images": images,
            "category": meta.get("category", "unknown"),
            "object_id": meta.get("object_id", f"item_{idx}"),
            "gt_code": gt_code,
            "prompt": self.format_prompt_text(meta.get("category", "object")),
        }

    def format_prompt(self, item: dict[str, Any]) -> list[dict]:
        """Format as Qwen2.5-VL chat messages."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                *[{"type": "image", "image": img} for img in item["images"]],
                {"type": "text", "text": self.format_prompt_text(item["category"])},
            ]},
        ]
        return messages

    @staticmethod
    def format_prompt_text(category: str) -> str:
        return (
            f"Generate Blender Python code to reconstruct this {category}. "
            f"Use the bpy_lib API functions (create_primitive, create_curve, "
            f"fill_grid, create_translation, create_bridge_loop, boolean_op, "
            f"bevel, create_array_1d, create_array_2d). Structure the code "
            f"with part comments and call export_scene() at the end."
        )
