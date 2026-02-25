"""
Main pipeline orchestrator.

Reads config, instantiates adapters, runs all stages, produces output.

Usage:
    python -m pipeline.run --config config/default.yaml
    python -m pipeline.run --config config/default.yaml --override global.total_samples=3000
    python -m pipeline.run --config config/no_meshcoder.yaml  # Alternative without MeshCoder
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from core.models import SFTSample, PipelineStats
from core.adapters import AdapterRegistry

# Import adapters to trigger registration
import adapters.dataset_adapters  # noqa: F401

from stages.processing import QualityGate, FeatureExtractor, ViewSampler
from stages.sampling import create_sampler
from stages.formatting import SFTFormatter, DPOFormatter

logger = logging.getLogger(__name__)


def load_config(config_path: str, overrides: list[str] | None = None) -> dict:
    """Load YAML config with optional dot-notation overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            parts = key.split(".")
            
            # Navigate to the right nested dict
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            
            # Try to parse value as int, float, bool, or keep as string
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
            
            d[parts[-1]] = value
            logger.info(f"Config override: {key} = {value}")
    
    return config


def run_pipeline(config: dict) -> PipelineStats:
    """Execute the full SFT dataset pipeline."""
    stats = PipelineStats()
    
    global_cfg = config["global"]
    seed = global_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    
    target_k = global_cfg["total_samples"]
    output_dir = Path(global_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Stage 1: Load samples from all enabled dataset sources
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading dataset sources")
    logger.info("=" * 60)
    
    all_samples: list[SFTSample] = []
    dataset_configs = config.get("datasets", {})
    
    # Compute normalized weights for enabled datasets
    enabled = {k: v for k, v in dataset_configs.items() if v.get("enabled", True)}
    total_weight = sum(v["weight"] for v in enabled.values())
    
    for ds_name, ds_cfg in enabled.items():
        adapter_name = ds_cfg["adapter"]
        weight = ds_cfg["weight"] / total_weight  # Normalize
        
        logger.info(f"  Loading {ds_name} (adapter={adapter_name}, weight={weight:.2%})")
        
        try:
            adapter_cls = AdapterRegistry.get(adapter_name)
            adapter = adapter_cls(ds_cfg.get("config", {}))
            
            source_samples = list(adapter.load())
            
            # Tag samples with their difficulty thresholds
            for s in source_samples:
                s._dataset_weight = weight
                s._difficulty_thresholds = ds_cfg.get("difficulty_thresholds", {})
                s._difficulty_distribution = ds_cfg.get("difficulty_distribution", {})
            
            all_samples.extend(source_samples)
            stats.per_dataset[ds_name] = {
                "loaded": len(source_samples),
                "weight": weight,
            }
            
            logger.info(f"    → Loaded {len(source_samples)} samples")
        
        except Exception as e:
            logger.error(f"    → Failed to load {ds_name}: {e}")
            stats.per_dataset[ds_name] = {"loaded": 0, "error": str(e)}
    
    stats.total_source_samples = len(all_samples)
    logger.info(f"Total source samples: {stats.total_source_samples}")
    
    if not all_samples:
        logger.error("No samples loaded from any source. Aborting.")
        return stats
    
    # ================================================================
    # Stage 2: Quality gate
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: Quality gate")
    logger.info("=" * 60)
    
    quality_gate = QualityGate(config.get("quality_gate", {}))
    gated_samples = list(quality_gate(iter(all_samples)))
    
    stats.passed_quality_gate = len(gated_samples)
    logger.info(f"Passed quality gate: {stats.passed_quality_gate}/{stats.total_source_samples}")
    logger.info(f"Rejection reasons: {quality_gate.stats.get('reasons', {})}")
    
    # ================================================================
    # Stage 3: Feature extraction
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 3: Feature extraction")
    logger.info("=" * 60)
    
    feature_extractor = FeatureExtractor(
        config.get("code_features", {}),
        # Use per-dataset thresholds if available, else global
        difficulty_thresholds=config.get("datasets", {})
            .get(list(enabled.keys())[0] if enabled else "", {})
            .get("difficulty_thresholds"),
    )
    featured_samples = list(feature_extractor(iter(gated_samples)))
    
    # Log feature distribution
    complexity_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    for s in featured_samples:
        complexity_counts[s.code_features.complexity_bucket.value] += 1
        difficulty_counts[s.difficulty_bucket.value] += 1
    
    logger.info(f"Complexity distribution: {dict(complexity_counts)}")
    logger.info(f"Difficulty distribution: {dict(difficulty_counts)}")
    
    # ================================================================
    # Stage 4: View sampling
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 4: View sampling")
    logger.info("=" * 60)
    
    view_sampler = ViewSampler(config.get("views", {}))
    viewed_samples = list(view_sampler(iter(featured_samples)))
    
    view_counts = defaultdict(int)
    for s in viewed_samples:
        view_counts[s.num_views] += 1
    logger.info(f"View count distribution: {dict(sorted(view_counts.items()))}")
    stats.per_view_count = dict(view_counts)
    
    # ================================================================
    # Stage 5: Sample selection
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 5: Sample selection")
    logger.info("=" * 60)
    
    sampling_cfg = config.get("sampling", {})
    method = sampling_cfg.get("method", "stratified")
    
    sampler = create_sampler(method, sampling_cfg, target_k)
    selected = sampler.select(viewed_samples)
    
    stats.selected_for_sft = len(selected)
    
    # Compute reward statistics
    rewards = [s.metrics.rlvr_reward for s in selected]
    stats.reward_stats = {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
    }
    logger.info(f"Selected {len(selected)} samples. Reward stats: {stats.reward_stats}")
    
    # ================================================================
    # Stage 6: Format and write output
    # ================================================================
    logger.info("=" * 60)
    logger.info("STAGE 6: Formatting output")
    logger.info("=" * 60)
    
    formatter = SFTFormatter(config.get("output", {}))
    sft_path = formatter.format_samples(selected, output_dir)
    logger.info(f"SFT dataset written to: {sft_path}")
    
    # ================================================================
    # Stage 7 (optional): DPO pairs
    # ================================================================
    output_cfg = config.get("output", {})
    if output_cfg.get("generate_dpo_pairs", False):
        logger.info("=" * 60)
        logger.info("STAGE 7: Generating DPO pairs")
        logger.info("=" * 60)
        
        # Use ALL samples (including failed ones) for DPO pair mining
        dpo_formatter = DPOFormatter(output_cfg)
        dpo_pairs = dpo_formatter.generate_pairs(all_samples)
        
        if dpo_pairs:
            dpo_path = dpo_formatter.write_pairs(dpo_pairs)
            stats.dpo_pairs_generated = len(dpo_pairs)
            logger.info(f"DPO pairs written to: {dpo_path}")
    
    # ================================================================
    # Write pipeline report
    # ================================================================
    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "total_source_samples": stats.total_source_samples,
            "passed_quality_gate": stats.passed_quality_gate,
            "selected_for_sft": stats.selected_for_sft,
            "dpo_pairs_generated": stats.dpo_pairs_generated,
            "per_dataset": stats.per_dataset,
            "per_view_count": stats.per_view_count,
            "reward_stats": stats.reward_stats,
            "quality_gate_stats": quality_gate.stats,
            "config_summary": {
                "target_k": target_k,
                "sampling_method": method,
                "enabled_datasets": list(enabled.keys()),
            },
        }, f, indent=2)
    
    logger.info(f"Pipeline report written to: {report_path}")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="SFT Dataset Mining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m pipeline.run --config config/default.yaml

  # Override total samples and sampling method
  python -m pipeline.run --config config/default.yaml \\
      --override global.total_samples=3000 \\
      --override sampling.method=facility_location

  # Run without MeshCoder (alternative strategy)
  python -m pipeline.run --config config/no_meshcoder.yaml

  # Quick test with small dataset
  python -m pipeline.run --config config/default.yaml \\
      --override global.total_samples=100
        """,
    )
    
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override config values (dot notation): key=value"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and validate config without running pipeline"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    config = load_config(args.config, args.override)
    
    # Add file handler if configured
    log_cfg = config.get("logging", {})
    if log_cfg.get("log_file"):
        log_path = Path(log_cfg["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(fh)
    
    if args.dry_run:
        logger.info("Dry run — config loaded successfully:")
        logger.info(json.dumps(config, indent=2, default=str))
        return
    
    start = time.time()
    stats = run_pipeline(config)
    elapsed = time.time() - start
    
    logger.info(f"Total pipeline time: {elapsed:.1f}s")
    logger.info(f"Final: {stats.selected_for_sft} SFT samples, "
                f"{stats.dpo_pairs_generated} DPO pairs")


if __name__ == "__main__":
    main()
