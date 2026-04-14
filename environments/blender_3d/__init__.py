"""
Core environment primitives for Blender code generation.

Components:
  - Blender3DDataset: prompt dataset wrapper
  - Blender3DHarness: executes code in the Modal Blender sandbox
  - Blender3DRubric: computes binary-threshold rewards from execution results
"""

from .dataset import Blender3DDataset
from .harness import Blender3DHarness
from .rubric import Blender3DRubric

__all__ = [
    "Blender3DDataset",
    "Blender3DHarness",
    "Blender3DRubric",
]
