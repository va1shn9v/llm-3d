"""
Synthetic Self-Play Data Generator

This script generates paired (image, code) training data WITHOUT requiring
pre-existing paired datasets like MeshCoder.

Strategy: "Write-Execute-Render-Verify" loop
============================================

Instead of:  image → LLM → code → compare to ground truth
We do:       LLM → code → execute → mesh → render → (images, code) pair

The key insight: we don't NEED a ground truth mesh. We can ask the LLM
to generate objects from TEXT descriptions, execute the code, verify it
produces reasonable geometry, then render the resulting mesh to create
the input images. The (rendered_images, code) pair is self-consistent
by construction.

This is analogous to "expert iteration" or "self-play" in RL:
  1. LLM generates code for "a wooden chair with 4 legs and a curved back"
  2. We execute the code in Blender → get a mesh
  3. We validate the mesh (is it reasonable? non-degenerate? right scale?)
  4. We render the mesh from multiple viewpoints → input images
  5. Store (images, code) as an SFT training sample

The model learns: "given images that look like THIS, produce code like THAT"

Advantages over ground-truth comparison:
  - No dependency on paired datasets (MeshCoder, etc.)
  - Unlimited scale (can generate millions of samples)
  - Code is guaranteed to execute (we verified it)
  - Images perfectly match the code output (self-consistent)
  - Can target specific categories / complexity levels

Disadvantages:
  - No guarantee the code is OPTIMAL (it's whatever the teacher LLM produced)
  - Quality ceiling is limited by the teacher model
  - Need to filter aggressively for geometric plausibility

Usage:
    python generate_selfplay.py --config selfplay_config.yaml
    python generate_selfplay.py --categories chair table lamp --num-per-category 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ============================================================================
# Text prompts for object generation (no images needed!)
# ============================================================================
OBJECT_PROMPTS = {
    "chair": [
        "a simple wooden dining chair with 4 straight legs, a flat seat, and a straight backrest",
        "an office swivel chair with a curved seat, 5 wheeled legs, and adjustable armrests",
        "a minimalist stool with 3 angled legs and a round seat",
        "a rocking chair with curved runners, armrests, and vertical slats in the backrest",
        "an adirondack chair with wide armrests and a reclined seat",
        "a bar stool with a round seat, a single pedestal leg, and a circular footrest",
        "a folding chair with an X-shaped leg structure and a flat seat",
        "a throne-like chair with a tall ornate backrest and thick armrests",
    ],
    "table": [
        "a rectangular dining table with 4 straight legs",
        "a round coffee table with 3 angled legs",
        "a desk with a flat top, 4 legs, and a drawer compartment underneath",
        "an oval dining table with a central pedestal base",
        "a side table with a square top and shelf underneath",
        "a folding table with X-shaped collapsible legs",
        "a console table: long, narrow, wall-height with 4 legs",
        "a workbench with a thick top, 4 sturdy legs, and a lower shelf",
    ],
    "lamp": [
        "a table lamp with a cylindrical base, thin stem, and conical lampshade",
        "a floor lamp with a heavy round base, tall thin pole, and dome shade",
        "a desk lamp with an articulated arm and small cone shade",
        "a pendant lamp: just a sphere hanging from a thin wire",
        "a mushroom lamp with a dome shade sitting directly on a rounded base",
        "a tripod floor lamp with 3 splayed legs and a drum shade",
        "a lantern-style lamp: hexagonal frame with panels, handle on top",
        "a minimalist cube lamp: a hollow cube that emits light",
    ],
    "sofa": [
        "a 3-seat sofa with rectangular cushions, armrests, and short legs",
        "a 2-seat loveseat with rounded armrests and no visible legs",
        "an L-shaped sectional sofa with a chaise extension",
        "a futon: a flat rectangular frame that folds into a sofa shape",
        "a single-seat armchair with wide cushioned armrests",
    ],
    "bookshelf": [
        "a tall bookshelf with 5 evenly spaced horizontal shelves and 2 vertical sides",
        "a cube storage unit: 3x3 grid of open cubby holes",
        "a ladder bookshelf: shelves get narrower toward the top, leaning shape",
        "a low wide bookshelf with 3 shelves, suitable under a window",
        "a corner bookshelf: triangular shelves that fit in a corner",
    ],
    "cabinet": [
        "a kitchen cabinet: rectangular box with 2 doors and a shelf inside",
        "a filing cabinet: tall narrow box with 3 pull-out drawers",
        "a TV stand: low wide cabinet with 2 doors and an open shelf on top",
        "a wardrobe: tall cabinet with 2 doors and a hanging rod inside",
        "a bathroom vanity: cabinet with a flat top, 2 doors, single sink cutout on top",
    ],
    "vase": [
        "a tall cylindrical vase with a slightly wider opening at the top",
        "a round bulbous vase with a narrow neck",
        "a geometric vase: hexagonal cross-section, straight sides",
        "a wavy vase with a sinusoidal profile curve",
        "an amphora: two handles on the sides, narrow base, wide middle, narrow neck",
    ],
}

# Master prompt template for text-to-code generation
CODE_GEN_PROMPT = """Generate a Blender Python script that creates a 3D model of:

{description}

Requirements:
1. Start with `import bpy` and clear the default scene
2. Build the geometry using bpy.ops.mesh primitives, transforms, and modifiers
3. Use meaningful variable names that describe each part (e.g., seat, leg, backrest)
4. Center the object at the origin
5. Normalize so the object fits within a 1x1x1 bounding box
6. Export as OBJ to the path stored in EXPORT_PATH environment variable

# PLANNING (write as comments):
# PARTS: List the main geometric parts
# PROPORTIONS: Approximate relative sizes
# STRATEGY: Which primitives and operations to use

Important:
- Use bpy.ops.mesh.primitive_cube_add(), primitive_cylinder_add(), etc.
- Apply transforms with bpy.ops.object.transform_apply()
- For Blender 4.x: use bpy.ops.wm.obj_export() not export_scene.obj()
- Set EXPORT_PATH = os.environ.get("EXPORT_PATH", "/tmp/generated.obj")

Return ONLY a ```python code block."""


@dataclass
class SelfPlaySample:
    id: str
    category: str
    description: str
    code: str
    mesh_path: str
    render_paths: list[str]
    mesh_stats: dict          # vertices, faces, bounding box, etc.
    execution_success: bool
    validation_passed: bool
    generation_model: str
    generation_time: float
    # Pseudo-metrics (no ground truth, so we use geometric heuristics)
    reward: float = 0.0
    f_score_005: float = 0.0
    chamfer_distance: float = 0.0


class SelfPlayGenerator:
    """Generates self-play training data via write-execute-render-verify."""
    
    def __init__(
        self,
        output_dir: str = "./data/synthetic",
        blender_path: str = "blender",
        api_key: str | None = None,
        model: str = "grok-4.1",
        blender_timeout: int = 120,
    ):
        self.output_dir = Path(output_dir)
        self.blender_path = blender_path
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.blender_timeout = blender_timeout
        
        # Create output structure
        (self.output_dir / "code").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "meshes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "renders").mkdir(parents=True, exist_ok=True)
    
    def generate_batch(
        self,
        categories: list[str],
        num_per_category: int = 100,
        models: list[str] | None = None,
    ) -> list[SelfPlaySample]:
        """Generate a batch of self-play samples."""
        models = models or [self.model]
        all_samples = []
        
        for category in categories:
            prompts = OBJECT_PROMPTS.get(category, [])
            if not prompts:
                logger.warning(f"No prompts defined for category: {category}")
                continue
            
            for i in range(num_per_category):
                # Pick a random description (with variation)
                base_prompt = random.choice(prompts)
                description = self._add_variation(base_prompt)
                model = random.choice(models)
                
                sample_id = f"selfplay_{category}_{i:04d}"
                
                logger.info(f"Generating {sample_id}: {description[:60]}...")
                
                try:
                    sample = self._generate_one(
                        sample_id, category, description, model
                    )
                    if sample and sample.validation_passed:
                        all_samples.append(sample)
                        logger.info(
                            f"  ✓ {sample_id}: {sample.mesh_stats.get('faces', 0)} faces, "
                            f"reward={sample.reward:.3f}"
                        )
                    else:
                        logger.info(f"  ✗ {sample_id}: validation failed")
                
                except Exception as e:
                    logger.error(f"  ✗ {sample_id}: {e}")
        
        # Write results
        results_path = self.output_dir / "selfplay_results.json"
        with open(results_path, "w") as f:
            json.dump([asdict(s) for s in all_samples], f, indent=2)
        
        logger.info(
            f"Generated {len(all_samples)} valid self-play samples "
            f"(saved to {results_path})"
        )
        return all_samples
    
    def _generate_one(
        self,
        sample_id: str,
        category: str,
        description: str,
        model: str,
    ) -> SelfPlaySample | None:
        """Generate a single self-play sample: code → execute → render → verify."""
        
        # Step 1: Generate code via LLM
        t0 = time.time()
        code = self._call_llm(description, model)
        gen_time = time.time() - t0
        
        if not code:
            return None
        
        # Save code
        code_path = self.output_dir / "code" / f"{sample_id}.py"
        code_path.write_text(code)
        
        # Step 2: Execute in Blender
        mesh_path = self.output_dir / "meshes" / f"{sample_id}.obj"
        exec_success = self._execute_code(code, str(mesh_path))
        
        if not exec_success or not mesh_path.exists():
            return SelfPlaySample(
                id=sample_id, category=category, description=description,
                code=code, mesh_path="", render_paths=[],
                mesh_stats={}, execution_success=False,
                validation_passed=False, generation_model=model,
                generation_time=gen_time,
            )
        
        # Step 3: Validate mesh geometry
        mesh_stats = self._validate_mesh(str(mesh_path))
        valid = self._is_plausible(mesh_stats, category)
        
        if not valid:
            return SelfPlaySample(
                id=sample_id, category=category, description=description,
                code=code, mesh_path=str(mesh_path), render_paths=[],
                mesh_stats=mesh_stats, execution_success=True,
                validation_passed=False, generation_model=model,
                generation_time=gen_time,
            )
        
        # Step 4: Render views from the generated mesh
        render_dir = self.output_dir / "renders" / sample_id
        render_dir.mkdir(parents=True, exist_ok=True)
        
        num_views = random.randint(4, 8)  # Render extras, subsample later
        render_paths = self._render_views(str(mesh_path), str(render_dir), num_views)
        
        # Step 5: Compute pseudo-reward (no ground truth, use heuristics)
        reward = self._compute_pseudo_reward(mesh_stats, code, category)
        
        return SelfPlaySample(
            id=sample_id, category=category, description=description,
            code=code, mesh_path=str(mesh_path), render_paths=render_paths,
            mesh_stats=mesh_stats, execution_success=True,
            validation_passed=True, generation_model=model,
            generation_time=gen_time, reward=reward,
            # These are pseudo-metrics for compatibility with the pipeline
            f_score_005=min(reward * 1.2, 1.0),
            chamfer_distance=max(0.001, 0.05 * (1.0 - reward)),
        )
    
    def _call_llm(self, description: str, model: str) -> str | None:
        """Call OpenRouter API to generate Blender code."""
        prompt = CODE_GEN_PROMPT.format(description=description)
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8192,
                    "temperature": 0.3,
                },
                timeout=300,
            )
            response.raise_for_status()
            
            content = response.json()["choices"][0]["message"]["content"]
            return self._extract_code(content)
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def _extract_code(self, response: str) -> str | None:
        """Extract Python code block from LLM response."""
        import re
        
        # Try ```python blocks
        patterns = [
            r'```python\s*\n(.*?)```',
            r'```Python\s*\n(.*?)```',
            r'```py\s*\n(.*?)```',
            r'```\s*\n(.*?import bpy.*?)```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: if response starts with import bpy
        if response.strip().startswith("import bpy"):
            return response.strip()
        
        return None
    
    def _execute_code(self, code: str, export_path: str) -> bool:
        """Execute Blender Python code in a subprocess."""
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            # Prepend safety wrapper
            wrapper = f"""
import os
os.environ["EXPORT_PATH"] = "{export_path}"

# Clear scene
import bpy
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

"""
            f.write(wrapper + code)
            script_path = f.name
        
        try:
            result = subprocess.run(
                [self.blender_path, "--background", "--python", script_path],
                capture_output=True, text=True,
                timeout=self.blender_timeout,
            )
            return result.returncode == 0 and os.path.exists(export_path)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Blender execution failed: {e}")
            return False
        finally:
            os.unlink(script_path)
    
    def _validate_mesh(self, mesh_path: str) -> dict:
        """Load mesh and compute basic geometric statistics."""
        try:
            import trimesh
            mesh = trimesh.load(mesh_path, force="mesh")
            
            bounds = mesh.bounds
            extents = mesh.extents
            
            return {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_watertight": bool(mesh.is_watertight),
                "extents": extents.tolist(),
                "volume": float(mesh.volume) if mesh.is_watertight else 0.0,
                "surface_area": float(mesh.area),
                "bounds_min": bounds[0].tolist(),
                "bounds_max": bounds[1].tolist(),
                "center_of_mass": mesh.center_mass.tolist(),
                "is_degenerate": len(mesh.faces) < 4,
            }
        except Exception as e:
            logger.warning(f"Mesh validation failed: {e}")
            return {"vertices": 0, "faces": 0, "is_degenerate": True}
    
    def _is_plausible(self, stats: dict, category: str) -> bool:
        """Check if mesh geometry is plausible for the given category."""
        if stats.get("is_degenerate", True):
            return False
        if stats.get("vertices", 0) < 8:
            return False
        if stats.get("faces", 0) < 6:
            return False
        
        # Check extents are reasonable (not a flat plane, not a point)
        extents = stats.get("extents", [0, 0, 0])
        if min(extents) < 0.01:  # Nearly flat in one dimension
            return False
        if max(extents) / (min(extents) + 1e-8) > 100:  # Extremely elongated
            return False
        
        return True
    
    def _render_views(
        self, mesh_path: str, output_dir: str, num_views: int
    ) -> list[str]:
        """Render the mesh from multiple viewpoints using Blender."""
        render_script = f"""
import bpy
import math
import os

# Clear scene
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

# Import mesh
bpy.ops.wm.obj_import(filepath="{mesh_path}")

# Center and normalize
obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
if obj is None:
    for o in bpy.data.objects:
        if o.type == 'MESH':
            obj = o
            break

if obj:
    # Normalize to unit bounding box
    dims = obj.dimensions
    max_dim = max(dims) if max(dims) > 0 else 1.0
    obj.scale = (1.0/max_dim, 1.0/max_dim, 1.0/max_dim)
    bpy.ops.object.transform_apply(scale=True)
    obj.location = (0, 0, 0)

# Setup camera and lighting
bpy.ops.object.camera_add(location=(0, -2.5, 1.5))
camera = bpy.context.object
camera.rotation_euler = (math.radians(60), 0, 0)
bpy.context.scene.camera = camera

bpy.ops.object.light_add(type='SUN', location=(2, -2, 4))
light = bpy.context.object
light.data.energy = 3.0

# Render settings
scene = bpy.context.scene
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = True
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32

# Render from multiple viewpoints
num_views = {num_views}
render_paths = []
for i in range(num_views):
    angle = (2 * math.pi / num_views) * i
    radius = 2.5
    elevation = math.radians(30)
    
    x = radius * math.cos(angle) * math.cos(elevation)
    y = radius * math.sin(angle) * math.cos(elevation)
    z = radius * math.sin(elevation)
    
    camera.location = (x, y, z)
    
    # Point camera at origin
    direction = mathutils.Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    output_path = os.path.join("{output_dir}", f"view_{{i:02d}}.png")
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    render_paths.append(output_path)

import mathutils  # Import at top level for the script
print("RENDER_COMPLETE")
"""
        
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            # Fix: import mathutils at top
            fixed_script = "import mathutils\n" + render_script
            f.write(fixed_script)
            script_path = f.name
        
        try:
            subprocess.run(
                [self.blender_path, "--background", "--python", script_path],
                capture_output=True, text=True,
                timeout=self.blender_timeout * 2,  # Rendering takes longer
            )
        except Exception as e:
            logger.warning(f"Rendering failed: {e}")
        finally:
            os.unlink(script_path)
        
        # Collect rendered images
        render_dir = Path(output_dir)
        return sorted(str(p) for p in render_dir.glob("view_*.png"))
    
    def _compute_pseudo_reward(
        self, stats: dict, code: str, category: str
    ) -> float:
        """
        Compute a pseudo-reward based on mesh quality heuristics.
        
        Since we have no ground truth, we use geometric plausibility signals:
        - Mesh complexity (not too simple, not too complex)
        - Watertightness
        - Reasonable proportions
        - Code quality (uses multiple parts, not just one primitive)
        """
        reward = 0.0
        
        # Mesh complexity score (0-0.3)
        faces = stats.get("faces", 0)
        if 12 <= faces <= 500:
            reward += 0.15
        elif 500 < faces <= 5000:
            reward += 0.30
        elif faces > 5000:
            reward += 0.20  # Very complex, might be good
        
        # Watertight bonus (0-0.15)
        if stats.get("is_watertight", False):
            reward += 0.15
        
        # Proportion reasonableness (0-0.15)
        extents = stats.get("extents", [1, 1, 1])
        aspect_ratio = max(extents) / (min(extents) + 1e-8)
        if aspect_ratio < 5:
            reward += 0.15
        elif aspect_ratio < 10:
            reward += 0.08
        
        # Code quality score (0-0.25)
        code_lower = code.lower()
        num_primitives = sum(1 for kw in [
            "primitive_cube_add", "primitive_cylinder_add",
            "primitive_uv_sphere_add", "primitive_cone_add",
        ] if kw in code_lower)
        
        if num_primitives >= 3:
            reward += 0.15
        elif num_primitives >= 2:
            reward += 0.10
        
        if "for " in code:
            reward += 0.05
        if any(kw in code_lower for kw in ["boolean", "bevel", "solidify"]):
            reward += 0.05
        
        # Code structure bonus (0-0.15)
        if code.count("def ") >= 1:
            reward += 0.05
        if "# " in code:  # Has comments
            reward += 0.05
        if len(code) > 1000:
            reward += 0.05
        
        return min(reward, 1.0)
    
    def _add_variation(self, base_prompt: str) -> str:
        """Add random variation to a base prompt for diversity."""
        variations = [
            "",  # No change
            " with rounded edges",
            " with slightly tapered legs",
            " in a modern minimalist style",
            " with proportions similar to IKEA furniture",
            " with thick solid construction",
            " with a slightly asymmetric design",
        ]
        return base_prompt + random.choice(variations)


def main():
    parser = argparse.ArgumentParser(description="Generate self-play SFT data")
    parser.add_argument(
        "--categories", nargs="+",
        default=["chair", "table", "lamp", "bookshelf", "cabinet", "vase"],
    )
    parser.add_argument("--num-per-category", type=int, default=100)
    parser.add_argument("--output-dir", default="./data/synthetic")
    parser.add_argument("--blender-path", default="blender")
    parser.add_argument(
        "--models", nargs="+",
        default=["grok-4.1", "claude-sonnet-4-20250514"],
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    generator = SelfPlayGenerator(
        output_dir=args.output_dir,
        blender_path=args.blender_path,
        model=args.models[0],
    )
    
    samples = generator.generate_batch(
        categories=args.categories,
        num_per_category=args.num_per_category,
        models=args.models,
    )
    
    print(f"\nGenerated {len(samples)} valid self-play samples")
    print(f"Average pseudo-reward: {sum(s.reward for s in samples) / max(len(samples), 1):.3f}")


if __name__ == "__main__":
    main()
