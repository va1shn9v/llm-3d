"""
Output formatters: serialize selected SFTSamples into training-ready formats.

Supports:
  - JSONL (one JSON object per line, standard for most training frameworks)
  - Parquet (columnar, efficient for large datasets)
  - HuggingFace datasets format
  
Also handles DPO pair generation from the full sample pool.
"""
from __future__ import annotations

import base64
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from core.models import SFTSample, DPOPair, PipelineStats

logger = logging.getLogger(__name__)


class SFTFormatter:
    """
    Converts selected SFTSamples into chat-format training records.
    
    Output schema (JSONL, one per line):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ...
                {"type": "text", "text": "Given these N view(s) of a 3D object, ..."}
            ]},
            {"role": "assistant", "content": "import bpy\n..."}
        ],
        "metadata": { ... }  // optional
    }
    """
    
    def __init__(self, config: dict):
        self.format = config.get("format", "jsonl")
        self.chat_template = config.get("chat_template", "chatml")
        self.include_system = config.get("include_system_prompt", True)
        self.system_prompt = config.get("system_prompt", "").strip()
        self.include_metadata = config.get("include_metadata", True)
        self.metadata_fields = config.get("metadata_fields", [])
        self.embed_images = config.get("embed_images_base64", False)
    
    def format_samples(
        self,
        samples: list[SFTSample],
        output_dir: Path,
    ) -> Path:
        """Write all samples to the configured format. Returns output path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.format == "jsonl":
            return self._write_jsonl(samples, output_dir)
        elif self.format == "parquet":
            return self._write_parquet(samples, output_dir)
        elif self.format == "huggingface":
            return self._write_huggingface(samples, output_dir)
        else:
            raise ValueError(f"Unknown output format: {self.format}")
    
    def _write_jsonl(self, samples: list[SFTSample], output_dir: Path) -> Path:
        output_path = output_dir / "sft_train.jsonl"
        
        with open(output_path, "w") as f:
            for sample in samples:
                record = self._sample_to_chat(sample)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"Wrote {len(samples)} samples to {output_path}")
        return output_path
    
    def _write_parquet(self, samples: list[SFTSample], output_dir: Path) -> Path:
        import pandas as pd
        
        records = [self._sample_to_chat(s) for s in samples]
        df = pd.DataFrame(records)
        
        output_path = output_dir / "sft_train.parquet"
        df.to_parquet(output_path, index=False)
        
        logger.info(f"Wrote {len(samples)} samples to {output_path}")
        return output_path
    
    def _write_huggingface(
        self, samples: list[SFTSample], output_dir: Path
    ) -> Path:
        """Write in HuggingFace datasets format (arrow + metadata)."""
        # Fall back to JSONL which HF datasets can load directly
        return self._write_jsonl(samples, output_dir)
    
    def _sample_to_chat(self, sample: SFTSample) -> dict:
        """Convert a single SFTSample to a chat-format training record."""
        messages = []
        
        # System message
        if self.include_system and self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        
        # User message with images + text prompt
        user_content = []
        
        # Add images
        for img_path in sample.image_paths:
            if self.embed_images:
                # Read and base64-encode the image
                img_data = self._encode_image(img_path)
                if img_data:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    })
            else:
                # Reference by path (for local training)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": img_path},
                })
        
        # Text prompt
        prompt = self._build_prompt(sample.num_views)
        user_content.append({"type": "text", "text": prompt})
        
        messages.append({"role": "user", "content": user_content})
        
        # Assistant response (the target code)
        messages.append({
            "role": "assistant",
            "content": sample.code,
        })
        
        record = {"messages": messages}
        
        # Metadata
        if self.include_metadata:
            meta = {}
            field_map = {
                "source_dataset": sample.source_dataset,
                "category": sample.category,
                "object_id": sample.object_id,
                "num_views": sample.num_views,
                "reward": sample.metrics.rlvr_reward,
                "f_score_005": sample.metrics.f_score_005,
                "chamfer_distance": sample.metrics.chamfer_distance,
                "code_complexity_bucket": sample.code_features.complexity_bucket.value,
                "difficulty_bucket": sample.difficulty_bucket.value,
                "code_source_model": sample.code_source_model,
            }
            for field in self.metadata_fields:
                if field in field_map:
                    meta[field] = field_map[field]
            record["metadata"] = meta
        
        return record
    
    def _build_prompt(self, num_views: int) -> str:
        view_word = "view" if num_views == 1 else "views"
        return (
            f"You are given {num_views} {view_word} of a 3D object. "
            f"Generate a Blender Python script that recreates this object's geometry.\n\n"
            f"Requirements:\n"
            f"1. Clear the default scene\n"
            f"2. Build the geometry using bpy operations\n"
            f"3. Normalize the result to fit within a unit bounding box centered at origin\n"
            f"4. Export as OBJ to the path specified by the EXPORT_PATH variable\n\n"
            f"Return ONLY a ```python code block."
        )
    
    def _encode_image(self, path: str) -> str | None:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            logger.warning(f"Failed to encode image {path}: {e}")
            return None


class DPOFormatter:
    """
    Generates DPO preference pairs from the full sample pool.
    
    For each object that has multiple code generations with different rewards,
    creates (chosen, rejected) pairs where the reward gap exceeds a threshold.
    """
    
    def __init__(self, config: dict):
        self.min_reward_gap = config.get("dpo_min_reward_gap", 0.15)
        self.output_dir = Path(config.get("dpo_output_dir", "./output/dpo_pairs"))
    
    def generate_pairs(
        self, all_samples: list[SFTSample]
    ) -> list[DPOPair]:
        """
        Find all objects with multiple attempts, create preference pairs.
        """
        # Group by (source_dataset, object_id) to find multiple attempts
        by_object: dict[str, list[SFTSample]] = defaultdict(list)
        for s in all_samples:
            key = f"{s.source_dataset}:{s.object_id}"
            by_object[key].append(s)
        
        pairs: list[DPOPair] = []
        
        for key, attempts in by_object.items():
            if len(attempts) < 2:
                continue
            
            # Sort by reward descending
            attempts.sort(key=lambda s: -s.metrics.rlvr_reward)
            
            # Create pairs: best vs each worse attempt
            best = attempts[0]
            for worse in attempts[1:]:
                gap = best.metrics.rlvr_reward - worse.metrics.rlvr_reward
                if gap >= self.min_reward_gap:
                    pair = DPOPair(
                        prompt_images=best.image_paths,  # Same images for both
                        prompt_text=f"Given {best.num_views} view(s) of a 3D object...",
                        chosen_code=best.code,
                        chosen_reward=best.metrics.rlvr_reward,
                        rejected_code=worse.code,
                        rejected_reward=worse.metrics.rlvr_reward,
                        object_id=best.object_id,
                        category=best.category,
                        source_dataset=best.source_dataset,
                    )
                    pairs.append(pair)
            
            # Also pair successful vs failed executions
            executed = [s for s in attempts if s.metrics.execution_success]
            failed = [s for s in attempts if not s.metrics.execution_success]
            
            if executed and failed:
                best_exec = executed[0]
                for fail in failed:
                    pair = DPOPair(
                        prompt_images=best_exec.image_paths,
                        prompt_text=f"Given {best_exec.num_views} view(s)...",
                        chosen_code=best_exec.code,
                        chosen_reward=best_exec.metrics.rlvr_reward,
                        rejected_code=fail.code,
                        rejected_reward=0.0,
                        object_id=best_exec.object_id,
                        category=best_exec.category,
                        source_dataset=best_exec.source_dataset,
                    )
                    pairs.append(pair)
        
        logger.info(f"Generated {len(pairs)} DPO preference pairs")
        return pairs
    
    def write_pairs(self, pairs: list[DPOPair]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "dpo_pairs.jsonl"
        
        with open(output_path, "w") as f:
            for pair in pairs:
                record = {
                    "prompt_images": pair.prompt_images,
                    "prompt_text": pair.prompt_text,
                    "chosen": pair.chosen_code,
                    "chosen_reward": pair.chosen_reward,
                    "rejected": pair.rejected_code,
                    "rejected_reward": pair.rejected_reward,
                    "object_id": pair.object_id,
                    "category": pair.category,
                    "source_dataset": pair.source_dataset,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"Wrote {len(pairs)} DPO pairs to {output_path}")
        return output_path
