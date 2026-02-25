"""
Processing stages that transform and filter SFTSamples.

Pipeline flow:
    Adapter → QualityGate → FeatureExtractor → ViewSampler → Sampler → Formatter
    
Each stage is a callable that takes an iterable of SFTSample and yields
filtered/transformed SFTSample records.
"""
from __future__ import annotations

import logging
import math
import random
import re
from typing import Iterator, Sequence

from core.models import (
    SFTSample,
    CodeFeatures,
    CodeComplexityBucket,
    DifficultyBucket,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Quality Gate
# ============================================================================
class QualityGate:
    """
    Filters samples based on configurable quality thresholds.
    
    Samples that fail any criterion are marked passed_quality_gate=False
    and excluded from downstream selection.
    """
    
    def __init__(self, config: dict):
        self.min_reward = config.get("min_reward", 0.10)
        self.min_f_score = config.get("min_f_score_005", 0.08)
        self.max_chamfer = config.get("max_chamfer", 0.05)
        self.min_code_len = config.get("min_code_length", 200)
        self.max_code_len = config.get("max_code_length", 15000)
        self.require_geo_ops = config.get("require_geometry_ops", True)
        self.geo_keywords = config.get("geometry_op_keywords", [])
        
        self._counts = {"total": 0, "passed": 0, "reasons": {}}
    
    def __call__(self, samples: Iterator[SFTSample]) -> Iterator[SFTSample]:
        for sample in samples:
            self._counts["total"] += 1
            passed, reason = self._check(sample)
            sample.passed_quality_gate = passed
            
            if passed:
                self._counts["passed"] += 1
                yield sample
            else:
                self._counts["reasons"][reason] = (
                    self._counts["reasons"].get(reason, 0) + 1
                )
    
    def _check(self, s: SFTSample) -> tuple[bool, str]:
        if not s.metrics.execution_success:
            return False, "execution_failed"
        if s.metrics.rlvr_reward < self.min_reward:
            return False, "low_reward"
        if s.metrics.f_score_005 < self.min_f_score:
            return False, "low_f_score"
        if self.max_chamfer > 0 and s.metrics.chamfer_distance > self.max_chamfer:
            return False, "high_chamfer"
        if len(s.code) < self.min_code_len:
            return False, "code_too_short"
        if len(s.code) > self.max_code_len:
            return False, "code_too_long"
        if self.require_geo_ops and self.geo_keywords:
            code_lower = s.code.lower()
            if not any(kw.lower() in code_lower for kw in self.geo_keywords):
                return False, "no_geometry_ops"
        return True, ""
    
    @property
    def stats(self) -> dict:
        return dict(self._counts)


# ============================================================================
# Feature Extractor
# ============================================================================
class FeatureExtractor:
    """
    Extracts structural code features and assigns difficulty/complexity buckets.
    """
    
    def __init__(self, config: dict, difficulty_thresholds: dict | None = None):
        self.primitives_kw = config.get("primitives_keywords", [])
        self.boolean_kw = config.get("boolean_keywords", ["boolean"])
        self.array_kw = config.get("array_keywords", ["array", "for "])
        self.curve_kw = config.get("curve_keywords", ["curve", "bezier"])
        self.extrude_kw = config.get("extrude_keywords", ["extrude"])
        self.math_kw = config.get("math_keywords", ["import math", "sin(", "cos("])
        
        self.difficulty_thresholds = difficulty_thresholds or {
            "easy": [0.10, 0.25],
            "medium": [0.25, 0.45],
            "hard": [0.45, 0.65],
            "excellent": [0.65, 1.00],
        }
    
    def __call__(self, samples: Iterator[SFTSample]) -> Iterator[SFTSample]:
        for sample in samples:
            sample.code_features = self._extract_features(sample.code)
            sample.difficulty_bucket = self._assign_difficulty(
                sample.metrics.rlvr_reward
            )
            yield sample
    
    def _extract_features(self, code: str) -> CodeFeatures:
        code_lower = code.lower()
        
        num_primitives = sum(
            1 for kw in self.primitives_kw if kw.lower() in code_lower
        )
        
        # Count approximate number of separate objects
        # Heuristic: count bpy.ops.mesh.primitive_* and bpy.ops.object.select_all
        num_parts = max(1, len(re.findall(
            r'bpy\.ops\.mesh\.primitive_\w+_add', code
        )))
        
        code_length = len(code)
        
        # Determine complexity bucket based on multiple signals
        complexity_score = 0
        if num_primitives >= 3:
            complexity_score += 1
        if any(kw.lower() in code_lower for kw in self.boolean_kw):
            complexity_score += 2
        if any(kw.lower() in code_lower for kw in self.curve_kw):
            complexity_score += 2
        if any(kw.lower() in code_lower for kw in self.math_kw):
            complexity_score += 1
        if code.count("def ") >= 2:
            complexity_score += 1
        if code.count("for ") >= 3:
            complexity_score += 1
        if code_length > 5000:
            complexity_score += 1
        
        if complexity_score <= 2:
            bucket = CodeComplexityBucket.SIMPLE
        elif complexity_score <= 5:
            bucket = CodeComplexityBucket.MODERATE
        else:
            bucket = CodeComplexityBucket.COMPLEX
        
        return CodeFeatures(
            num_primitives=num_primitives,
            uses_boolean=any(kw.lower() in code_lower for kw in self.boolean_kw),
            uses_array=any(kw.lower() in code_lower for kw in self.array_kw),
            uses_curves=any(kw.lower() in code_lower for kw in self.curve_kw),
            uses_extrude=any(kw.lower() in code_lower for kw in self.extrude_kw),
            uses_bevel="bevel" in code_lower,
            uses_math=any(kw.lower() in code_lower for kw in self.math_kw),
            num_parts=num_parts,
            code_length=code_length,
            num_functions=code.count("def "),
            num_loops=code.count("for ") + code.count("while "),
            complexity_bucket=bucket,
        )
    
    def _assign_difficulty(self, reward: float) -> DifficultyBucket:
        for bucket_name, (lo, hi) in self.difficulty_thresholds.items():
            if lo <= reward < hi:
                return DifficultyBucket(bucket_name)
        # Default: if reward >= highest threshold, assign excellent
        return DifficultyBucket.EXCELLENT


# ============================================================================
# View Sampler
# ============================================================================
class ViewSampler:
    """
    Assigns how many views each sample will have, and selects which views.
    
    If the sample already has pre-rendered views (from the adapter),
    this stage subsamples them. If not, it flags them for rendering.
    """
    
    def __init__(self, config: dict):
        self.min_views = config.get("min_views", 1)
        self.max_views = config.get("max_views", 6)
        self.count_weights = config.get("count_weights", {
            1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20, 5: 0.10, 6: 0.10
        })
        # Normalize to integer keys (YAML may parse as strings)
        self.count_weights = {int(k): v for k, v in self.count_weights.items()}
        
        self.selection_strategy = config.get("selection_strategy", "azimuth_spread")
        self.elevation_jitter = config.get("elevation_jitter_deg", 10)
        self.render_resolution = config.get("render_resolution", [512, 512])
    
    def __call__(self, samples: Iterator[SFTSample]) -> Iterator[SFTSample]:
        # Pre-compute normalized weights
        counts = sorted(self.count_weights.keys())
        weights = [self.count_weights[c] for c in counts]
        
        for sample in samples:
            # Sample number of views
            num_views = random.choices(counts, weights=weights, k=1)[0]
            num_views = max(self.min_views, min(self.max_views, num_views))
            
            # If we have pre-rendered views, select a subset
            if sample.image_paths and len(sample.image_paths) >= num_views:
                sample.image_paths = self._select_views(
                    sample.image_paths, num_views
                )
            else:
                # Flag: needs rendering. Store the desired count so the
                # rendering stage knows how many views to produce.
                sample.image_paths = []
            
            sample.num_views = num_views
            yield sample
    
    def _select_views(
        self, all_views: list[str], n: int
    ) -> list[str]:
        """Select n views from available views using the configured strategy."""
        total = len(all_views)
        
        if n >= total:
            return all_views[:n]
        
        if self.selection_strategy == "azimuth_spread":
            # Assume views are evenly spaced in azimuth (view_00, view_01, ...)
            # Pick maximally separated indices
            step = total / n
            indices = [int(i * step) % total for i in range(n)]
            return [all_views[i] for i in indices]
        
        elif self.selection_strategy == "fixed_canonical":
            # 1 view: front 3/4 (index 1 if 8 views)
            # 2 views: front + back
            # 3+ views: evenly spaced starting from front
            if n == 1:
                return [all_views[min(1, total - 1)]]
            elif n == 2:
                return [all_views[0], all_views[total // 2]]
            else:
                step = total / n
                indices = [int(i * step) % total for i in range(n)]
                return [all_views[i] for i in indices]
        
        else:  # random
            return sorted(random.sample(all_views, n))
    
    def get_render_params(self, num_views: int) -> list[dict]:
        """
        Generate camera parameters for rendering num_views views.
        
        Returns list of {azimuth, elevation, distance} dicts for Blender.
        """
        params = []
        for i in range(num_views):
            azimuth = (360.0 / num_views) * i
            elevation = 30.0 + random.uniform(
                -self.elevation_jitter, self.elevation_jitter
            )
            params.append({
                "azimuth": azimuth,
                "elevation": elevation,
                "distance": 2.0,
                "resolution": self.render_resolution,
            })
        return params
