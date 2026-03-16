"""
Blender3DDataset — provides text prompts for text-to-3D code generation training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import StorageConfig
from data.storage import open_read

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert Blender Python developer targeting Blender 4.2. "
    "Given a text description of a 3D object, write a complete bpy script that creates "
    "the described geometry with appropriate materials.\n\n"
    "APPROACH: Decompose the object into logical parts, build each with the best Blender "
    "construct (primitives, BMesh, curves, modifiers), add materials via Principled BSDF, "
    "then export.\n\n"
    "REQUIRED STRUCTURE:\n"
    "1. `import bpy, os, math, bmesh` and clear the scene\n"
    "2. Create geometry for each part\n"
    "3. Add materials (Principled BSDF)\n"
    "4. Apply smooth shading via bpy.ops.object.shade_smooth()\n"
    "5. Select all and export: bpy.ops.wm.obj_export(filepath=os.environ['EXPORT_PATH'], "
    "export_selected_objects=True, export_materials=False, apply_modifiers=True)\n\n"
    "BLENDER 4.2 RULES:\n"
    "- NEVER use obj.data.use_auto_smooth (removed)\n"
    "- NEVER use bpy.ops.export_scene.obj() (removed)\n"
    "- Use 'BLENDER_EEVEE_NEXT' not 'BLENDER_EEVEE'\n"
    "- Use 'Specular IOR Level' not 'Specular' in Principled BSDF\n"
    "- mathutils is top-level: from mathutils import Vector\n"
    "- After join()/remove(), re-acquire object references\n"
    "- BMesh: always ensure_lookup_table() and bm.free()\n\n"
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
        storage_cfg: StorageConfig | None = None,
    ):
        self.jsonl_path = str(jsonl_path)
        self.storage_cfg = storage_cfg
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.format_prompt_text(item["text"])},
        ]
        return messages

    @staticmethod
    def format_prompt_text(caption: str) -> str:
        return (
            f"Create a 3D model of: {caption}\n\n"
            f"Decompose the object into its main parts, build each with appropriate "
            f"Blender constructs (primitives, BMesh, curves, modifiers), add materials, "
            f"and export to OBJ. Write a complete Blender Python script."
        )
