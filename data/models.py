"""
Core data models for the SFT dataset pipeline.

Every sample flowing through the pipeline is a SFTSample dataclass.
Adapters produce them, quality gates filter them, samplers select them,
and formatters serialize them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class DifficultyBucket(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXCELLENT = "excellent"


class CodeComplexityBucket(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class GeometricMetrics:
    """Metrics from comparing generated mesh to ground truth."""
    chamfer_distance: float = 0.0
    f_score_001: float = 0.0
    f_score_002: float = 0.0
    f_score_005: float = 0.0
    hausdorff_90: float = 0.0
    normal_consistency: float = 0.0
    rlvr_reward: float = 0.0
    execution_success: bool = False


@dataclass
class CodeFeatures:
    """Structural features extracted from generated Blender Python code."""
    num_primitives: int = 0
    uses_boolean: bool = False
    uses_array: bool = False
    uses_curves: bool = False
    uses_extrude: bool = False
    uses_bevel: bool = False
    uses_math: bool = False
    num_parts: int = 0          # Approximate count of separate objects created
    code_length: int = 0
    num_functions: int = 0      # Number of `def ` occurrences
    num_loops: int = 0          # for/while loops
    complexity_bucket: CodeComplexityBucket = CodeComplexityBucket.SIMPLE


@dataclass
class SFTSample:
    """
    A single sample in the SFT dataset pipeline.
    
    This is the universal record type that flows through every stage:
      Adapter → QualityGate → FeatureExtractor → Sampler → Formatter
    """
    # --- Identity ---
    sample_id: str = ""                     # Unique ID (hash of source + object_id)
    source_dataset: str = ""                # e.g., "meshcoder", "infinigen", "objaverse_llm"
    object_id: str = ""                     # Original object identifier
    category: str = "unknown"               # Semantic category (chair, table, ...)
    
    # --- Inputs (for the VLM) ---
    image_paths: list[str] = field(default_factory=list)  # Paths to rendered view images
    num_views: int = 0                      # How many views (1-6)
    
    # --- Output (the target code) ---
    code: str = ""                          # Blender Python script
    code_source_model: str = ""             # Which LLM generated it (or "ground_truth")
    
    # --- Ground truth ---
    gt_mesh_path: str = ""                  # Path to ground-truth mesh
    gen_mesh_path: str = ""                 # Path to generated mesh (from executing code)
    
    # --- Metrics ---
    metrics: GeometricMetrics = field(default_factory=GeometricMetrics)
    
    # --- Computed features ---
    code_features: CodeFeatures = field(default_factory=CodeFeatures)
    difficulty_bucket: DifficultyBucket = DifficultyBucket.EASY
    
    # --- Pipeline metadata ---
    passed_quality_gate: bool = False
    selected_for_sft: bool = False
    
    def __post_init__(self):
        if not self.sample_id:
            raw = f"{self.source_dataset}:{self.object_id}:{self.code_source_model}"
            self.sample_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DPOPair:
    """A preference pair for DPO training."""
    prompt_images: list[str]
    prompt_text: str
    chosen_code: str
    chosen_reward: float
    rejected_code: str
    rejected_reward: float
    object_id: str
    category: str
    source_dataset: str


@dataclass
class PipelineStats:
    """Aggregate statistics for a pipeline run."""
    total_source_samples: int = 0
    passed_quality_gate: int = 0
    selected_for_sft: int = 0
    dpo_pairs_generated: int = 0
    
    per_dataset: dict = field(default_factory=dict)
    per_category: dict = field(default_factory=dict)
    per_difficulty: dict = field(default_factory=dict)
    per_complexity: dict = field(default_factory=dict)
    per_view_count: dict = field(default_factory=dict)
    
    reward_stats: dict = field(default_factory=dict)  # mean, std, min, max, median
