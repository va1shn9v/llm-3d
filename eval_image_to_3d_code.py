#!/usr/bin/env python3
"""Entry point for the LLM image-to-3D evaluation pipeline. See eval_pipeline/ for details."""

from eval_pipeline.config import setup_logging, parse_args
from eval_pipeline.pipeline import run_pipeline

if __name__ == "__main__":
    setup_logging()
    config = parse_args()
    run_pipeline(config)
