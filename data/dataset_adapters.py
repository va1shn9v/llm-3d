"""
Concrete dataset adapter implementations.

Each adapter reads from its native data format and yields SFTSample records.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

from core.adapters import BaseAdapter, AdapterRegistry
from core.models import SFTSample, GeometricMetrics

logger = logging.getLogger(__name__)


# ============================================================================
# MeshCoder Adapter
# ============================================================================
@AdapterRegistry.register
class MeshCoderAdapter(BaseAdapter):
    """
    Loads paired object-code data from MeshCoder's released dataset.
    
    Expected directory structure:
        data_dir/
            train.json          # [{object_id, category, code_path, mesh_path, metrics}, ...]
            dataset_train/      # Contains point clouds and code files
            dataset_val/
    
    MeshCoder samples already have verified code + metrics, so we mainly
    need to re-render views from the meshes and package as SFTSample.
    """
    
    @property
    def name(self) -> str:
        return "meshcoder"
    
    def validate_config(self):
        data_dir = Path(self.config.get("data_dir", ""))
        if not data_dir.exists():
            logger.warning(f"MeshCoder data_dir not found: {data_dir}")
    
    def estimate_count(self) -> int | None:
        data_dir = Path(self.config["data_dir"])
        train_json = data_dir / self.config.get("train_json", "train.json")
        if train_json.exists():
            with open(train_json) as f:
                return len(json.load(f))
        return None
    
    def load(self) -> Iterator[SFTSample]:
        data_dir = Path(self.config["data_dir"])
        train_json = data_dir / self.config.get("train_json", "train.json")
        
        if not train_json.exists():
            logger.error(f"MeshCoder train.json not found at {train_json}")
            return
        
        with open(train_json) as f:
            entries = json.load(f)
        
        allowed_categories = self.config.get("categories")  # None = all
        max_samples = self.config.get("max_source_samples")
        
        count = 0
        for entry in entries:
            if max_samples and count >= max_samples:
                break
            
            category = entry.get("category", "unknown")
            if allowed_categories and category not in allowed_categories:
                continue
            
            # Read the code file
            code_path = data_dir / entry.get("code_path", "")
            if not code_path.exists():
                continue
            
            code = code_path.read_text(encoding="utf-8", errors="replace")
            
            # Build metrics from MeshCoder's stored evaluation
            mc_metrics = entry.get("metrics", {})
            metrics = GeometricMetrics(
                chamfer_distance=mc_metrics.get("chamfer_distance", 0.0),
                f_score_005=mc_metrics.get("f_score", 0.0),
                normal_consistency=mc_metrics.get("normal_consistency", 0.5),
                execution_success=True,  # MeshCoder only stores successful samples
                rlvr_reward=mc_metrics.get("reward", 0.0),
            )
            
            # If MeshCoder doesn't have a pre-computed reward, estimate one
            if metrics.rlvr_reward == 0.0:
                metrics.rlvr_reward = self._estimate_reward(metrics)
            
            sample = SFTSample(
                source_dataset=self.name,
                object_id=entry.get("object_id", entry.get("id", "")),
                category=category,
                code=code,
                code_source_model="meshcoder_ground_truth",
                gt_mesh_path=str(data_dir / entry.get("mesh_path", "")),
                metrics=metrics,
            )
            
            count += 1
            yield sample
        
        logger.info(f"MeshCoder adapter yielded {count} samples")
    
    def _estimate_reward(self, m: GeometricMetrics) -> float:
        """Fallback reward estimation from available metrics."""
        cd_penalty = max(0, 1.0 - m.chamfer_distance * 100)
        return 0.5 * m.f_score_005 + 0.15 * m.normal_consistency + 0.2 * cd_penalty


# ============================================================================
# Infinigen Adapter
# ============================================================================
@AdapterRegistry.register
class InfinigenAdapter(BaseAdapter):
    """
    Loads procedural generation scripts extracted from Infinigen.
    
    Infinigen's source code contains Python scripts that procedurally generate
    objects. We extract these, execute them to produce meshes, then treat the
    (mesh, code) pair as ground-truth training data.
    
    Expected directory structure:
        code_dir/
            chair_001.py
            table_002.py
            ...
        mesh_dir/
            chair_001.obj
            table_002.obj
            ...
    """
    
    @property
    def name(self) -> str:
        return "infinigen"
    
    def load(self) -> Iterator[SFTSample]:
        code_dir = Path(self.config["code_dir"])
        mesh_dir = Path(self.config["mesh_dir"])
        
        if not code_dir.exists():
            logger.warning(f"Infinigen code_dir not found: {code_dir}")
            return
        
        allowed_categories = self.config.get("categories")
        max_samples = self.config.get("max_source_samples")
        
        count = 0
        for code_path in sorted(code_dir.glob("*.py")):
            if max_samples and count >= max_samples:
                break
            
            stem = code_path.stem
            # Infer category from filename prefix (e.g., "chair_001" -> "chair")
            parts = stem.split("_")
            category = parts[0] if len(parts) > 1 else "unknown"
            
            if allowed_categories and category not in allowed_categories:
                continue
            
            code = code_path.read_text(encoding="utf-8", errors="replace")
            
            # Find corresponding mesh
            mesh_path = None
            for ext in [".obj", ".glb", ".ply", ".stl"]:
                candidate = mesh_dir / f"{stem}{ext}"
                if candidate.exists():
                    mesh_path = str(candidate)
                    break
            
            if mesh_path is None:
                logger.debug(f"No mesh found for {stem}, skipping")
                continue
            
            # Infinigen code is ground truth — we know it executes
            metrics = GeometricMetrics(
                execution_success=True,
                # Actual metrics will be computed by the pipeline's metric stage
                # For now, set a reasonable default for ground-truth code
                rlvr_reward=0.5,  # Placeholder — recomputed in pipeline
            )
            
            sample = SFTSample(
                source_dataset=self.name,
                object_id=stem,
                category=category,
                code=code,
                code_source_model="infinigen_procedural",
                gt_mesh_path=mesh_path,
                metrics=metrics,
            )
            
            count += 1
            yield sample
        
        logger.info(f"Infinigen adapter yielded {count} samples")


# ============================================================================
# Objaverse LLM-Generated Adapter
# ============================================================================
@AdapterRegistry.register
class ObjaverseLLMAdapter(BaseAdapter):
    """
    Loads LLM-generated code for Objaverse objects.
    
    This adapter reads from the output of the eval_image_to_3d_code.py pipeline.
    Each result file contains: object_id, category, model, code, metrics.
    
    Expected directory structure:
        data_dir/
            results/
                results_grok-4.1.json
                results_claude-sonnet.json
                ...
            objects/
                {uid}.glb
                ...
            renders/
                {uid}/
                    view_00.png
                    view_01.png
                    ...
    """
    
    @property
    def name(self) -> str:
        return "objaverse_llm"
    
    def load(self) -> Iterator[SFTSample]:
        data_dir = Path(self.config["data_dir"])
        results_dir = data_dir / "results"
        renders_dir = data_dir / "renders"
        objects_dir = data_dir / "objects"
        
        if not results_dir.exists():
            logger.warning(f"Objaverse results dir not found: {results_dir}")
            return
        
        max_samples = self.config.get("max_source_samples")
        count = 0
        
        for results_file in sorted(results_dir.glob("results_*.json")):
            with open(results_file) as f:
                results = json.load(f)
            
            for result in results:
                if max_samples and count >= max_samples:
                    return
                
                obj_id = result.get("object_id", "")
                code = result.get("generated_code", result.get("code", ""))
                
                if not code or not result.get("execution_success", False):
                    continue
                
                # Collect pre-rendered views if they exist
                obj_render_dir = renders_dir / obj_id
                image_paths = []
                if obj_render_dir.exists():
                    image_paths = sorted(
                        str(p) for p in obj_render_dir.glob("view_*.png")
                    )
                
                metrics = GeometricMetrics(
                    chamfer_distance=result.get("chamfer_distance", 0.0),
                    f_score_001=result.get("f_score_001", 0.0),
                    f_score_002=result.get("f_score_002", 0.0),
                    f_score_005=result.get("f_score_005", 0.0),
                    hausdorff_90=result.get("hausdorff_90", 0.0),
                    normal_consistency=result.get("normal_consistency", 0.0),
                    rlvr_reward=result.get("rlvr_reward", 0.0),
                    execution_success=True,
                )
                
                gt_mesh = objects_dir / f"{obj_id}.glb"
                
                sample = SFTSample(
                    source_dataset=self.name,
                    object_id=obj_id,
                    category=result.get("category", "unknown"),
                    image_paths=image_paths,
                    num_views=len(image_paths),
                    code=code,
                    code_source_model=result.get("model", "unknown"),
                    gt_mesh_path=str(gt_mesh) if gt_mesh.exists() else "",
                    metrics=metrics,
                )
                
                count += 1
                yield sample
        
        logger.info(f"Objaverse LLM adapter yielded {count} samples")


# ============================================================================
# ShapeNet Adapter
# ============================================================================
@AdapterRegistry.register
class ShapeNetAdapter(BaseAdapter):
    """
    Loads ShapeNet/PartNet objects with LLM-generated code.
    
    Similar to Objaverse adapter but reads ShapeNet's directory structure.
    Code must be pre-generated (by running the eval pipeline on ShapeNet objects).
    """
    
    @property
    def name(self) -> str:
        return "shapenet"
    
    def load(self) -> Iterator[SFTSample]:
        data_dir = Path(self.config["data_dir"])
        
        # Look for a unified results file
        results_file = data_dir / "results.json"
        if not results_file.exists():
            logger.warning(f"ShapeNet results not found: {results_file}")
            return
        
        with open(results_file) as f:
            results = json.load(f)
        
        allowed_categories = self.config.get("categories")
        count = 0
        
        for result in results:
            category = result.get("category", "unknown")
            if allowed_categories and category not in allowed_categories:
                continue
            
            code = result.get("generated_code", "")
            if not code or not result.get("execution_success", False):
                continue
            
            metrics = GeometricMetrics(
                chamfer_distance=result.get("chamfer_distance", 0.0),
                f_score_005=result.get("f_score_005", 0.0),
                normal_consistency=result.get("normal_consistency", 0.0),
                rlvr_reward=result.get("rlvr_reward", 0.0),
                execution_success=True,
            )
            
            sample = SFTSample(
                source_dataset=self.name,
                object_id=result.get("object_id", ""),
                category=category,
                code=code,
                code_source_model=result.get("model", "unknown"),
                gt_mesh_path=result.get("gt_mesh_path", ""),
                metrics=metrics,
            )
            
            count += 1
            yield sample
        
        logger.info(f"ShapeNet adapter yielded {count} samples")


# ============================================================================
# Generic JSON Adapter (for custom / user-provided datasets)
# ============================================================================
@AdapterRegistry.register
class GenericJSONAdapter(BaseAdapter):
    """
    Loads any dataset stored as a JSON/JSONL file with a known schema.
    
    Config must include:
        data_file: path to JSON or JSONL file
        field_mapping: dict mapping SFTSample fields to JSON keys
    
    Example field_mapping:
        object_id: "id"
        category: "label"
        code: "blender_script"
        gt_mesh_path: "mesh_file"
    """
    
    @property
    def name(self) -> str:
        return self.config.get("name", "generic_json")
    
    def validate_config(self):
        if "data_file" not in self.config:
            raise ValueError("GenericJSONAdapter requires 'data_file' in config")
        if "field_mapping" not in self.config:
            raise ValueError("GenericJSONAdapter requires 'field_mapping' in config")
    
    def load(self) -> Iterator[SFTSample]:
        data_file = Path(self.config["data_file"])
        mapping = self.config["field_mapping"]
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return
        
        # Support both JSON (list) and JSONL (one object per line)
        if data_file.suffix == ".jsonl":
            entries = []
            with open(data_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        else:
            with open(data_file) as f:
                entries = json.load(f)
        
        count = 0
        for entry in entries:
            code = entry.get(mapping.get("code", "code"), "")
            if not code:
                continue
            
            sample = SFTSample(
                source_dataset=self.name,
                object_id=str(entry.get(mapping.get("object_id", "id"), count)),
                category=entry.get(mapping.get("category", "category"), "unknown"),
                code=code,
                code_source_model=entry.get(mapping.get("model", "model"), "unknown"),
                gt_mesh_path=entry.get(mapping.get("gt_mesh_path", "mesh_path"), ""),
                metrics=GeometricMetrics(
                    rlvr_reward=entry.get(mapping.get("reward", "reward"), 0.0),
                    execution_success=entry.get(mapping.get("success", "success"), True),
                    chamfer_distance=entry.get(mapping.get("chamfer", "chamfer_distance"), 0.0),
                    f_score_005=entry.get(mapping.get("f_score", "f_score_005"), 0.0),
                ),
            )
            count += 1
            yield sample
        
        logger.info(f"GenericJSON adapter yielded {count} samples")
