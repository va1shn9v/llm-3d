"""
Modal Docker image definitions for Blender execution, metrics, and CLIP scoring.
"""

from __future__ import annotations

import modal

from config import load_config

_cfg = load_config()
_bv = _cfg.modal.blender_version


def make_blender_image() -> modal.Image:
    """Blender 4.x image for raw bpy script execution."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "wget", "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3",
            "libxrender1", "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
            "libxext6", "libgomp1",
        )
        .run_commands(
            f"wget -q https://download.blender.org/release/Blender{_bv[:3]}/"
            f"blender-{_bv}-linux-x64.tar.xz -O /tmp/blender.tar.xz",
            "mkdir -p /opt/blender && tar xf /tmp/blender.tar.xz"
            " --strip-components=1 -C /opt/blender",
            "ln -s /opt/blender/blender /usr/local/bin/blender",
            "rm /tmp/blender.tar.xz",
        )
        .pip_install("trimesh>=4.0", "numpy>=1.24", "scipy>=1.11")
    )


def make_metrics_image() -> modal.Image:
    """Image for mesh metrics + CLIP scoring."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "trimesh>=4.0",
            "numpy>=1.24",
            "scipy>=1.11",
            "torch>=2.1",
            "transformers>=4.36",
            "pillow>=10.0",
        )
    )


def make_blender_gpu_image() -> modal.Image:
    """Blender 4.x image with CUDA support for GPU-accelerated rendering."""
    return (
        modal.Image.from_registry("nvidia/cuda:12.2.0-runtime-ubuntu22.04", add_python="3.11")
        .apt_install(
            "wget", "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3",
            "libxrender1", "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
            "libxext6", "libgomp1",
        )
        .run_commands(
            f"wget -q https://download.blender.org/release/Blender{_bv[:3]}/"
            f"blender-{_bv}-linux-x64.tar.xz -O /tmp/blender.tar.xz",
            "mkdir -p /opt/blender && tar xf /tmp/blender.tar.xz"
            " --strip-components=1 -C /opt/blender",
            "ln -s /opt/blender/blender /usr/local/bin/blender",
            "rm /tmp/blender.tar.xz",
        )
        .pip_install("trimesh>=4.0", "numpy>=1.24", "scipy>=1.11")
    )


blender_image = make_blender_image()
blender_gpu_image = make_blender_gpu_image()
metrics_image = make_metrics_image()
