"""
Configuration and CLI argument parsing for the evaluation pipeline.
"""

import os
import sys
import logging
import argparse
import textwrap
from typing import List
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging():
    """Configure the shared logging format. Call once at startup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these or override via CLI args
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """All configurable parameters in one place."""

    # ── Dataset ──────────────────────────────────────────────────────────────
    category: str = "chair"
    """LVIS category to download from Objaverse. Common options:
       'chair', 'table', 'car', 'airplane', 'lamp', 'sofa', 'bottle',
       'bowl', 'mug', 'vase', 'guitar', 'piano', 'bed', 'bookshelf'"""

    max_objects: int = 5
    """Number of objects to evaluate on. Start small (3-5) for testing."""

    # ── Rendering ────────────────────────────────────────────────────────────
    num_views: int = 4
    """Number of input views to render for each object.
       More views = more information for the LLM, but longer prompts.
       4-6 is a good sweet spot. Views are evenly spaced around the object."""

    render_resolution: int = 512
    """Resolution of rendered input images (square). 512 is good for
       balancing detail vs. API token cost."""

    # ── LLM Evaluation ───────────────────────────────────────────────────────
    openrouter_api_key: str = "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"
    """Your OpenRouter API key. Set via OPENROUTER_API_KEY env var or here."""

    models: List[str] = field(default_factory=lambda: [
        "anthropic/claude-sonnet-4.6",
        "x-ai/grok-4.1-fast",
        "google/gemini-3.1-pro-preview",
        "openai/gpt-5.2-codex",
        "qwen/qwen3.5-plus-02-15",
        "qwen/qwen3-vl-8b-thinking",
        "qwen/qwen3-vl-8b-instruct",
        "qwen/qwen3-vl-30b-a3b-thinking",
    ])
    """List of OpenRouter model identifiers to evaluate.
       See https://openrouter.ai/models for available models.
       These should be vision-capable models (can process images)."""

    max_tokens: int = 8192
    """Default max tokens for LLM response. Per-model overrides are
       applied by get_model_params() in llm.py at runtime."""

    temperature: float = 0.1
    """Low temperature for more reliable code generation.
       For RLVR training, you'd use higher temperature for exploration."""

    # ── Metrics ──────────────────────────────────────────────────────────────
    num_sample_points: int = 10000
    """Number of points to sample from each mesh surface for metric
       computation. 10K is standard in the literature. More = more precise
       but slower. For fast RL reward, you could drop to 2048."""

    f_score_thresholds: List[float] = field(default_factory=lambda: [
        0.01, 0.02, 0.05
    ])
    """Thresholds (in normalized coordinates) for F-Score computation.
       0.01 = very tight (fine detail), 0.05 = loose (overall shape).
       Objects are normalized to fit in a unit sphere before comparison."""

    # ── Paths ────────────────────────────────────────────────────────────────
    output_dir: str = "./eval_results"
    """Where to save results, renders, and generated meshes."""

    blender_path: str = "blender"
    """Path to Blender binary. Used for rendering input views and
       executing generated code. Set to full path if not on $PATH.
       If using bpy module, this is only needed as fallback."""

    # ── Execution ────────────────────────────────────────────────────────────
    code_timeout: int = 60
    """Timeout in seconds for executing generated Blender code.
       Prevents infinite loops from hanging the pipeline."""

    download_processes: int = 4
    """Number of parallel processes for downloading Objaverse objects."""

    max_llm_workers: int = 5
    """Max concurrent LLM API queries per object. These are I/O-bound so
       we can safely fire many in parallel. Capped to len(models) at runtime."""

    max_blender_workers: int = 2
    """Max concurrent Blender subprocesses for code execution + rendering.
       Each Blender instance uses ~500MB-1GB RAM. 2 is conservative for
       M2 Macs with 8GB. Bump to 3-4 on 16GB+ machines."""


def parse_args() -> Config:
    """Parse CLI arguments into a Config object, using Config class defaults for all unset args."""
    _defaults = Config()

    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on image-to-3D code generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            EXAMPLES:
              # Basic run with defaults (all settings taken from Config class)
              python eval_image_to_3d_code.py

              # Custom category, more objects, different models
              python eval_image_to_3d_code.py \\
                --category lamp \\
                --max-objects 10 \\
                --num-views 6 \\
                --models anthropic/claude-sonnet-4 openai/gpt-4o \\
                --api-key sk-or-...

              # Quick test with 1 object
              python eval_image_to_3d_code.py \\
                --category chair --max-objects 1 --num-views 4 \\
                --models openai/gpt-4o --api-key sk-or-...
        """),
    )

    parser.add_argument("--category", type=str, default=_defaults.category,
                        help=f"LVIS category to evaluate on (default: {_defaults.category})")
    parser.add_argument("--max-objects", type=int, default=_defaults.max_objects,
                        help=f"Number of objects to evaluate (default: {_defaults.max_objects})")
    parser.add_argument("--num-views", type=int, default=_defaults.num_views,
                        help=f"Number of input views per object (default: {_defaults.num_views})")
    parser.add_argument("--render-resolution", type=int, default=_defaults.render_resolution,
                        help=f"Render resolution in pixels (default: {_defaults.render_resolution})")
    parser.add_argument("--models", nargs="+", default=_defaults.models,
                        help="OpenRouter model IDs to evaluate")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--max-tokens", type=int, default=_defaults.max_tokens,
                        help=f"Max tokens for LLM response (default: {_defaults.max_tokens})")
    parser.add_argument("--temperature", type=float, default=_defaults.temperature,
                        help=f"LLM temperature (default: {_defaults.temperature})")
    parser.add_argument("--blender-path", type=str, default=_defaults.blender_path,
                        help=f"Path to Blender binary (default: '{_defaults.blender_path}')")
    parser.add_argument("--output-dir", type=str, default=_defaults.output_dir,
                        help=f"Output directory (default: {_defaults.output_dir})")
    parser.add_argument("--num-sample-points", type=int, default=_defaults.num_sample_points,
                        help=f"Points to sample per mesh for metrics (default: {_defaults.num_sample_points})")
    parser.add_argument("--code-timeout", type=int, default=_defaults.code_timeout,
                        help=f"Timeout for code execution in seconds (default: {_defaults.code_timeout})")
    parser.add_argument("--max-llm-workers", type=int, default=_defaults.max_llm_workers,
                        help=f"Max concurrent LLM API queries (default: {_defaults.max_llm_workers})")
    parser.add_argument("--max-blender-workers", type=int, default=_defaults.max_blender_workers,
                        help=f"Max concurrent Blender processes (default: {_defaults.max_blender_workers})")

    args = parser.parse_args()

    # Resolve API key: CLI arg → env var → Config class default
    resolved_api_key = (
        args.api_key
        or os.environ.get("OPENROUTER_API_KEY", "")
        or _defaults.openrouter_api_key
    )

    config = Config(
        category=args.category,
        max_objects=args.max_objects,
        num_views=args.num_views,
        render_resolution=args.render_resolution,
        models=args.models,
        openrouter_api_key=resolved_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        blender_path=args.blender_path,
        output_dir=args.output_dir,
        num_sample_points=args.num_sample_points,
        code_timeout=args.code_timeout,
        max_llm_workers=args.max_llm_workers,
        max_blender_workers=args.max_blender_workers,
    )

    if not config.openrouter_api_key:
        parser.error(
            "OpenRouter API key required. Set via --api-key or OPENROUTER_API_KEY env var.\n"
            "  Get one at: https://openrouter.ai/keys"
        )

    return config
