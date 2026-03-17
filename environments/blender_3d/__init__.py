"""
Blender 3D verifiers environment — wraps the Blender sandbox as a
verifiable environment for RLVR training via prime-rl-env (verifiers).

Components:
  - Blender3DDataset:     Provides text prompts for code generation
  - Blender3DHarness:     Executes code in Modal Blender sandbox
  - Blender3DRubric:      Computes configurable binary reward from execution results
  - Blender3DEnvironment: Combines Dataset + Harness with server-side rewards
"""

from environments.blender_3d.dataset import Blender3DDataset
from environments.blender_3d.harness import Blender3DHarness
from environments.blender_3d.rubric import Blender3DRubric
from environments.blender_3d.blender_3d import Blender3DEnvironment

__all__ = [
    "Blender3DDataset",
    "Blender3DHarness",
    "Blender3DRubric",
    "Blender3DEnvironment",
]
