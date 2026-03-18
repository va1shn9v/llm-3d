"""
Modal reward API server — primary endpoint for RL training.

Exposes a FastAPI server that:
  POST /reward/batch  — Parallel Blender exec + metrics for a batch of code samples
  POST /reward/single — Single sample (debugging)
  POST /render        — Render GT mesh views
  POST /execute       — Execute code only, no reward
  GET  /health        — Health check

Deploy:  modal deploy modal_infra/reward_server.py
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import modal
from modal import asgi_app

from config import RewardConfig
from environments.blender_3d.rubric import Blender3DRubric

app = modal.App("llm3d-reward-api")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dev_env() -> None:
    """Load repo-local dev.env for Modal deploy-time configuration."""
    env_path = _PROJECT_ROOT / "dev.env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _runtime_secrets() -> list[modal.Secret]:
    secret_env = {
        key: value
        for key in (
            "REWARD_API_TOKEN",
            "GT_MESH_VOLUME_SUBDIR",
            "LLM3D_STORAGE__MODAL_VOLUME_MESH_SUBDIR",
        )
        if (value := os.environ.get(key))
    }
    if not secret_env:
        return []
    return [modal.Secret.from_dict(secret_env)]


_load_dev_env()

_BV = os.environ.get("BLENDER_VERSION", "4.2.0")
blender_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget", "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3",
        "libxrender1", "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
        "libxext6", "libgomp1",
    )
    .run_commands(
        f"wget -q https://download.blender.org/release/Blender{_BV[:3]}/"
        f"blender-{_BV}-linux-x64.tar.xz -O /tmp/blender.tar.xz",
        "mkdir -p /opt/blender && tar xf /tmp/blender.tar.xz"
        " --strip-components=1 -C /opt/blender",
        "ln -s /opt/blender/blender /usr/local/bin/blender",
        "rm /tmp/blender.tar.xz",
    )
    .pip_install(
        "trimesh>=4.0",
        "numpy>=1.24",
        "scipy>=1.11",
        "pydantic>=2.5",
        "pydantic-settings>=2.1",
        "pyyaml>=6.0",
    )
    .add_local_python_source("config", "environments")
)

execute_blender_code = modal.Function.from_name("llm3d-blender-worker", "execute_blender_code")
compute_metrics = modal.Function.from_name("llm3d-metrics-worker", "compute_metrics")
render_mesh_views = modal.Function.from_name("llm3d-render-worker", "render_mesh_views")
volume = modal.Volume.from_name(
    os.environ.get("LLM3D_MODAL__VOLUME_NAME", "llm3d-data"),
    create_if_missing=True,
)

_START_TIME = time.time()
_GT_MESH_VOLUME_SUBDIR = (
    os.environ.get("GT_MESH_VOLUME_SUBDIR")
    or os.environ.get("LLM3D_STORAGE__MODAL_VOLUME_MESH_SUBDIR")
    or "meshes"
).strip("/") or "meshes"


def _verify_token(token: str | None):
    expected = os.environ.get("REWARD_API_TOKEN", "")
    if expected and token != expected:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Invalid token")


_SUPPORTED_MESH_EXTS = (".obj", ".glb", ".gltf", ".ply", ".stl")


def _load_gt_mesh(object_id: str) -> tuple[bytes, str] | None:
    """Load a ground-truth mesh from the volume, preserving its true format."""
    for ext in _SUPPORTED_MESH_EXTS:
        path = Path(f"/data/{_GT_MESH_VOLUME_SUBDIR}/{object_id}{ext}")
        if path.exists():
            return path.read_bytes(), ext.lstrip(".")
    return None


def _list_synthetic_artifacts(limit: int = 200) -> list[dict[str, Any]]:
    root = Path("/data/synthetic")
    if not root.exists():
        return []

    items: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.obj"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        items.append({
            "uid": path.stem,
            "filename": path.name,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        })
        if len(items) >= limit:
            break
    return items


def _load_synthetic_artifact(uid: str) -> bytes | None:
    path = Path(f"/data/synthetic/{uid}.obj")
    if not path.exists():
        return None
    return path.read_bytes()


def _pair_artifact_status(uid: str) -> dict[str, Any]:
    gt = _load_gt_mesh(uid)
    gen = _load_synthetic_artifact(uid)
    return {
        "uid": uid,
        "generated_available": gen is not None,
        "generated_format": "obj" if gen is not None else None,
        "gt_available": gt is not None,
        "gt_format": gt[1] if gt is not None else None,
    }


def _build_rubric(cfg_data: dict[str, Any] | None) -> Blender3DRubric:
    """Build a shared rubric from serialized config payload."""
    cfg = RewardConfig(**cfg_data) if cfg_data else RewardConfig()
    return Blender3DRubric(cfg)


@app.function(
    image=blender_image,
    cpu=4, memory=8192, timeout=600,
    volumes={"/data": volume},
    secrets=_runtime_secrets(),
    keep_warm=1,
    allow_concurrent_inputs=50,
)
@asgi_app()
def reward_api():
    from fastapi import FastAPI, Query
    from fastapi.responses import Response
    from pydantic import BaseModel

    api = FastAPI(title="LLM-3D Reward API")

    class RewardItem(BaseModel):
        object_id: str
        code: str
        text_description: str = ""
        seed: int = 42

    class BatchRequest(BaseModel):
        items: list[RewardItem]
        reward_config: dict[str, Any] | None = None

    class SingleRequest(BaseModel):
        object_id: str
        code: str
        text_description: str = ""
        seed: int = 42
        reward_config: dict[str, Any] | None = None

    class RenderRequest(BaseModel):
        object_id: str
        num_views: int = 4
        resolution: list[int] = [512, 512]

    class ExecuteRequest(BaseModel):
        code: str
        seed: int = 42
        return_mesh: bool = False

    @api.get("/health")
    async def health():
        return {"status": "ok", "uptime": time.time() - _START_TIME}

    @api.get("/artifacts")
    async def list_artifacts(token: str = Query(None), limit: int = Query(200, ge=1, le=1000)):
        _verify_token(token)
        return {"artifacts": _list_synthetic_artifacts(limit)}

    @api.get("/artifacts/pair/{uid}")
    async def get_pair_status(uid: str, token: str = Query(None)):
        _verify_token(token)
        return _pair_artifact_status(uid)

    @api.get("/artifacts/generated/{uid}")
    async def get_generated_artifact(uid: str, token: str = Query(None)):
        _verify_token(token)
        data = _load_synthetic_artifact(uid)
        if data is None:
            return Response(content="Artifact not found", status_code=404)
        return Response(
            content=data,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{uid}.obj"'},
        )

    @api.get("/artifacts/gt/{uid}")
    async def get_gt_artifact(uid: str, token: str = Query(None)):
        _verify_token(token)
        gt = _load_gt_mesh(uid)
        if gt is None:
            return Response(content="Artifact not found", status_code=404)
        data, mesh_format = gt
        media_type = "model/gltf-binary" if mesh_format == "glb" else "application/octet-stream"
        return Response(
            content=data,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{uid}.{mesh_format}"'},
        )

    @api.get("/artifacts/{uid}")
    async def get_artifact(uid: str, token: str = Query(None)):
        return await get_generated_artifact(uid, token)

    @api.post("/reward/batch")
    async def reward_batch(req: BatchRequest, token: str = Query(None)):
        _verify_token(token)
        rubric = _build_rubric(req.reward_config)

        exec_futures = [
            execute_blender_code.spawn(item.code, item.seed)
            for item in req.items
        ]
        exec_results = [f.get() for f in exec_futures]

        metrics_futures = []
        for item, er in zip(req.items, exec_results):
            if er["success"] and er.get("mesh_bytes"):
                gt = _load_gt_mesh(item.object_id)
                if gt:
                    gt_bytes, gt_format = gt
                    metrics_futures.append(
                        compute_metrics.spawn(er["mesh_bytes"], gt_bytes, 10_000, "obj", gt_format)
                    )
                else:
                    metrics_futures.append(None)
            else:
                metrics_futures.append(None)

        metrics_results = []
        for mf in metrics_futures:
            if mf is not None:
                metrics_results.append(mf.get())
            else:
                metrics_results.append(None)

        rewards = []
        for item, er, mr in zip(req.items, exec_results, metrics_results):
            evaluation = rubric.evaluate(
                item.code,
                {**er, "metrics": mr},
                text_description=item.text_description,
            )
            rewards.append({
                "object_id": item.object_id,
                "reward": evaluation["reward"],
                "base_reward": evaluation["base_reward"],
                "text_alignment_reward": evaluation["text_alignment_reward"],
                "format_reward": evaluation["format_reward"],
                "sub_rewards": evaluation["sub_rewards"],
                "success": er["success"],
                "metrics": mr,
                "elapsed": er.get("elapsed", 0),
                "error": er.get("error", ""),
            })

        valid = [r for r in rewards if r["success"]]
        return {
            "rewards": rewards,
            "execution_rate": len(valid) / max(len(rewards), 1),
            "mean_reward": sum(r["reward"] for r in rewards) / max(len(rewards), 1),
        }

    @api.post("/reward/single")
    async def reward_single(req: SingleRequest, token: str = Query(None)):
        _verify_token(token)
        rubric = _build_rubric(req.reward_config)
        er = execute_blender_code.remote(req.code, req.seed)
        mr = None
        if er["success"] and er.get("mesh_bytes"):
            gt = _load_gt_mesh(req.object_id)
            if gt:
                gt_bytes, gt_format = gt
                mr = compute_metrics.remote(er["mesh_bytes"], gt_bytes, 10_000, "obj", gt_format)
        evaluation = rubric.evaluate(
            req.code,
            {**er, "metrics": mr},
            text_description=req.text_description,
        )
        return {
            "reward": evaluation["reward"],
            "base_reward": evaluation["base_reward"],
            "text_alignment_reward": evaluation["text_alignment_reward"],
            "format_reward": evaluation["format_reward"],
            "sub_rewards": evaluation["sub_rewards"],
            "success": er["success"],
            "metrics": mr,
            "exec_result": {k: v for k, v in er.items() if k != "mesh_bytes"},
        }

    @api.post("/render")
    async def render(req: RenderRequest, token: str = Query(None)):
        _verify_token(token)
        gt = _load_gt_mesh(req.object_id)
        if gt is None:
            return {"error": f"GT mesh not found for {req.object_id}"}
        gt_bytes, gt_format = gt
        imgs = render_mesh_views.remote(
            gt_bytes, gt_format, req.num_views, tuple(req.resolution),
        )
        return {
            "images_b64": [base64.b64encode(img).decode() for img in imgs],
        }

    @api.post("/execute")
    async def execute(req: ExecuteRequest, token: str = Query(None)):
        _verify_token(token)
        er = execute_blender_code.remote(req.code, req.seed)
        result = {k: v for k, v in er.items() if k != "mesh_bytes"}
        if req.return_mesh and er.get("mesh_bytes"):
            result["mesh_b64"] = base64.b64encode(er["mesh_bytes"]).decode()
        return result

    return api
