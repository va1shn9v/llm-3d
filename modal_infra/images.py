"""
Modal Docker image definitions for Blender execution and metrics computation.
"""

from __future__ import annotations

import modal

from config import load_config

_cfg = load_config()
_bv = _cfg.modal.blender_version


def make_blender_image() -> modal.Image:
    """Blender 4.x image with bpy_lib baked in."""
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
        .copy_local_file("bpy_lib/bpy_lib.py", "/opt/bpy_lib/bpy_lib.py")
        .copy_local_file("bpy_lib/__init__.py", "/opt/bpy_lib/__init__.py")
        .run_commands(
            "PYVER=$(python3 -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')"
            " && echo 'import sys; sys.path.insert(0, \"/opt/bpy_lib\")'"
            f" >> /opt/blender/{_bv[:3]}/python/lib/python$PYVER/site-packages/usercustomize.py"
        )
    )


def make_metrics_image() -> modal.Image:
    """Lightweight image for mesh metrics computation (no Blender needed)."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("trimesh>=4.0", "numpy>=1.24", "scipy>=1.11")
    )


blender_image = make_blender_image()
metrics_image = make_metrics_image()
