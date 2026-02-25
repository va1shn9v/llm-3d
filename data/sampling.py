"""
Sample selection strategies.

Given a pool of quality-gated, feature-extracted SFTSamples and a target
count K, select the best K samples for SFT training.

Three strategies:
  1. StratifiedSampler  — cross-product stratification by category × difficulty × complexity
  2. FacilityLocationSampler — k-means clustering in feature space, pick best per cluster
  3. RewardTopKSampler  — simple top-K by reward with category diversity floor
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Sequence

import numpy as np

from core.models import SFTSample, DifficultyBucket, CodeComplexityBucket

logger = logging.getLogger(__name__)


# ============================================================================
# Base Sampler
# ============================================================================
class BaseSampler:
    def __init__(self, config: dict, target_k: int):
        self.config = config
        self.target_k = target_k
    
    def select(self, samples: list[SFTSample]) -> list[SFTSample]:
        raise NotImplementedError


# ============================================================================
# 1. Stratified Sampler
# ============================================================================
class StratifiedSampler(BaseSampler):
    """
    Stratifies samples across multiple axes (category, difficulty, complexity)
    and selects proportionally from each stratum.
    
    This ensures coverage: you won't get 5000 samples all from the "chair"
    category at "easy" difficulty with "simple" code.
    """
    
    def select(self, samples: list[SFTSample]) -> list[SFTSample]:
        cfg = self.config.get("stratified", {})
        min_per_stratum = cfg.get("min_per_stratum", 1)
        overflow_strategy = cfg.get("overflow_strategy", "drop_lowest_reward")
        
        # Build strata
        strata: dict[tuple, list[SFTSample]] = defaultdict(list)
        for s in samples:
            key = self._stratum_key(s)
            strata[key].append(s)
        
        logger.info(
            f"StratifiedSampler: {len(samples)} samples across "
            f"{len(strata)} strata, target {self.target_k}"
        )
        
        # Sort each stratum by reward (descending)
        for key in strata:
            strata[key].sort(key=lambda s: -s.metrics.rlvr_reward)
        
        # Allocate budget across strata
        non_empty = {k: v for k, v in strata.items() if len(v) > 0}
        base_allocation = max(min_per_stratum, self.target_k // len(non_empty))
        
        selected: list[SFTSample] = []
        
        # First pass: take base_allocation from each stratum
        for key, stratum in non_empty.items():
            take = min(base_allocation, len(stratum))
            selected.extend(stratum[:take])
        
        # If under budget, greedily add highest-reward remaining samples
        if len(selected) < self.target_k:
            selected_ids = {s.sample_id for s in selected}
            remaining = [
                s for s in samples if s.sample_id not in selected_ids
            ]
            remaining.sort(key=lambda s: -s.metrics.rlvr_reward)
            
            needed = self.target_k - len(selected)
            selected.extend(remaining[:needed])
        
        # If over budget, trim
        if len(selected) > self.target_k:
            if overflow_strategy == "drop_lowest_reward":
                selected.sort(key=lambda s: -s.metrics.rlvr_reward)
                selected = selected[:self.target_k]
            elif overflow_strategy == "drop_random":
                random.shuffle(selected)
                selected = selected[:self.target_k]
            else:
                selected = selected[:self.target_k]
        
        for s in selected:
            s.selected_for_sft = True
        
        logger.info(f"StratifiedSampler selected {len(selected)} samples")
        self._log_distribution(selected)
        return selected
    
    def _stratum_key(self, s: SFTSample) -> tuple:
        return (
            s.category,
            s.difficulty_bucket.value,
            s.code_features.complexity_bucket.value,
        )
    
    def _log_distribution(self, selected: list[SFTSample]):
        cats = defaultdict(int)
        diffs = defaultdict(int)
        comps = defaultdict(int)
        for s in selected:
            cats[s.category] += 1
            diffs[s.difficulty_bucket.value] += 1
            comps[s.code_features.complexity_bucket.value] += 1
        
        logger.info(f"  Categories: {dict(cats)}")
        logger.info(f"  Difficulties: {dict(diffs)}")
        logger.info(f"  Complexities: {dict(comps)}")


# ============================================================================
# 2. Facility Location Sampler (k-means diversity selection)
# ============================================================================
class FacilityLocationSampler(BaseSampler):
    """
    Clusters samples in feature space, then picks the best sample from
    each cluster. Maximizes diversity while maintaining quality.
    
    Uses MiniBatchKMeans for scalability (handles 100K+ samples easily).
    """
    
    def select(self, samples: list[SFTSample]) -> list[SFTSample]:
        cfg = self.config.get("facility_location", {})
        selection_strategy = cfg.get("selection_within_cluster", "max_reward")
        
        # Extract feature vectors
        X, valid_samples = self._build_feature_matrix(samples)
        
        if len(valid_samples) <= self.target_k:
            logger.warning(
                f"FacilityLocation: only {len(valid_samples)} valid samples "
                f"(<= target {self.target_k}), returning all"
            )
            for s in valid_samples:
                s.selected_for_sft = True
            return valid_samples
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(self.target_k, len(valid_samples))
        
        logger.info(
            f"FacilityLocation: clustering {len(valid_samples)} samples "
            f"into {n_clusters} clusters"
        )
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(1024, len(valid_samples)),
            random_state=42,
            n_init=3,
        )
        labels = kmeans.fit_predict(X_scaled)
        
        # Select best from each cluster
        selected: list[SFTSample] = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            if selection_strategy == "max_reward":
                best_idx = max(
                    cluster_indices,
                    key=lambda i: valid_samples[i].metrics.rlvr_reward,
                )
            elif selection_strategy == "random":
                best_idx = random.choice(cluster_indices)
            elif selection_strategy == "closest_to_centroid":
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(
                    X_scaled[cluster_indices] - centroid, axis=1
                )
                best_idx = cluster_indices[np.argmin(distances)]
            else:
                best_idx = cluster_indices[0]
            
            selected.append(valid_samples[best_idx])
        
        for s in selected:
            s.selected_for_sft = True
        
        logger.info(f"FacilityLocation selected {len(selected)} samples")
        return selected
    
    def _build_feature_matrix(
        self, samples: list[SFTSample]
    ) -> tuple[np.ndarray, list[SFTSample]]:
        """Build a numeric feature matrix from code features + metrics."""
        features = []
        valid = []
        
        for s in samples:
            cf = s.code_features
            m = s.metrics
            
            vec = [
                cf.num_primitives,
                float(cf.uses_boolean),
                float(cf.uses_array),
                float(cf.uses_curves),
                float(cf.uses_extrude),
                float(cf.uses_bevel),
                float(cf.uses_math),
                cf.num_parts,
                cf.code_length / 1000.0,  # Normalize
                cf.num_functions,
                cf.num_loops,
                m.rlvr_reward,
                m.chamfer_distance * 100,  # Scale up
                m.f_score_005,
                m.normal_consistency,
            ]
            
            features.append(vec)
            valid.append(s)
        
        return np.array(features, dtype=np.float32), valid


# ============================================================================
# 3. Reward Top-K Sampler
# ============================================================================
class RewardTopKSampler(BaseSampler):
    """
    Simplest strategy: sort by reward, take top K.
    Optionally enforces a minimum number of samples per category
    to prevent category collapse.
    """
    
    def select(self, samples: list[SFTSample]) -> list[SFTSample]:
        cfg = self.config.get("reward_top_k", {})
        min_per_category = cfg.get("ensure_min_per_category", 20)
        
        # Group by category
        by_category: dict[str, list[SFTSample]] = defaultdict(list)
        for s in samples:
            by_category[s.category].append(s)
        
        # Sort each category by reward
        for cat in by_category:
            by_category[cat].sort(key=lambda s: -s.metrics.rlvr_reward)
        
        selected: list[SFTSample] = []
        selected_ids: set[str] = set()
        
        # First: ensure minimum per category
        for cat, cat_samples in by_category.items():
            take = min(min_per_category, len(cat_samples))
            for s in cat_samples[:take]:
                if s.sample_id not in selected_ids:
                    selected.append(s)
                    selected_ids.add(s.sample_id)
        
        # Then: fill remaining budget with global top-reward
        if len(selected) < self.target_k:
            all_remaining = [
                s for s in samples if s.sample_id not in selected_ids
            ]
            all_remaining.sort(key=lambda s: -s.metrics.rlvr_reward)
            
            needed = self.target_k - len(selected)
            for s in all_remaining[:needed]:
                selected.append(s)
                selected_ids.add(s.sample_id)
        
        # Trim if over
        if len(selected) > self.target_k:
            selected.sort(key=lambda s: -s.metrics.rlvr_reward)
            selected = selected[:self.target_k]
        
        for s in selected:
            s.selected_for_sft = True
        
        logger.info(f"RewardTopK selected {len(selected)} samples")
        return selected


# ============================================================================
# Factory
# ============================================================================
SAMPLERS = {
    "stratified": StratifiedSampler,
    "facility_location": FacilityLocationSampler,
    "reward_top_k": RewardTopKSampler,
}


def create_sampler(method: str, config: dict, target_k: int) -> BaseSampler:
    if method not in SAMPLERS:
        raise ValueError(
            f"Unknown sampling method '{method}'. "
            f"Available: {list(SAMPLERS.keys())}"
        )
    return SAMPLERS[method](config, target_k)
