"""
Modal reward API for Blender code execution and geometry-based scoring.
"""

from __future__ import annotations

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
    return [modal.Secret.from_dict(secret_env)] if secret_env else []


_load_dev_env()

_BLENDER_VERSION = os.environ.get("BLENDER_VERSION", "4.2.0")
blender_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget", "xz-utils", "libxi6", "libxxf86vm1", "libxfixes3",
        "libxrender1", "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
        "libxext6", "libgomp1",
    )
    .run_commands(
        f"wget -q https://download.blender.org/release/Blender{_BLENDER_VERSION[:3]}/"
        f"blender-{_BLENDER_VERSION}-linux-x64.tar.xz -O /tmp/blender.tar.xz",
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
        "pyyaml>=6.0",
    )
    .add_local_python_source("config", "environments", "prompts")
)

execute_blender_code = modal.Function.from_name("llm3d-blender-worker", "execute_blender_code")
compute_metrics = modal.Function.from_name("llm3d-metrics-worker", "compute_metrics")
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
_SUPPORTED_MESH_EXTS = (".obj", ".glb", ".gltf", ".ply", ".stl")


def _verify_token(token: str | None) -> None:
    expected = os.environ.get("REWARD_API_TOKEN", "")
    if expected and token != expected:
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Invalid token")


def _load_gt_mesh(object_id: str) -> tuple[bytes, str] | None:
    for ext in _SUPPORTED_MESH_EXTS:
        path = Path(f"/data/{_GT_MESH_VOLUME_SUBDIR}/{object_id}{ext}")
        if path.exists():
            return path.read_bytes(), ext.lstrip(".")
    return None


def _pair_artifact_status(uid: str) -> dict[str, Any]:
    gt = _load_gt_mesh(uid)
    return {
        "uid": uid,
        "generated_available": False,
        "generated_format": None,
        "gt_available": gt is not None,
        "gt_format": gt[1] if gt is not None else None,
    }


def _build_rubric(cfg_data: dict[str, Any] | None) -> Blender3DRubric:
    return Blender3DRubric(RewardConfig(**cfg_data) if cfg_data else RewardConfig())


@app.function(
    image=blender_image,
    cpu=4,
    memory=8192,
    timeout=600,
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

    class ExecuteRequest(BaseModel):
        code: str
        seed: int = 42

    @api.get("/health")
    async def health():
        return {"status": "ok", "uptime": time.time() - _START_TIME}

    @api.get("/artifacts/pair/{uid}")
    async def get_pair_status(uid: str, token: str = Query(None)):
        _verify_token(token)
        return _pair_artifact_status(uid)

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

    @api.post("/reward/batch")
    async def reward_batch(req: BatchRequest, token: str = Query(None)):
        _verify_token(token)
        rubric = _build_rubric(req.reward_config)

        exec_futures = [execute_blender_code.spawn(item.code, item.seed) for item in req.items]
        exec_results = [future.get() for future in exec_futures]

        metrics_futures = []
        for item, exec_result in zip(req.items, exec_results, strict=False):
            if exec_result["success"] and exec_result.get("mesh_bytes"):
                gt = _load_gt_mesh(item.object_id)
                if gt is not None:
                    gt_bytes, gt_format = gt
                    metrics_futures.append(
                        compute_metrics.spawn(exec_result["mesh_bytes"], gt_bytes, 10_000, "obj", gt_format)
                    )
                    continue
            metrics_futures.append(None)

        metrics_results = [future.get() if future is not None else None for future in metrics_futures]

        rewards = []
        for item, exec_result, metrics_result in zip(req.items, exec_results, metrics_results, strict=False):
            evaluation = rubric.evaluate(
                item.code,
                {**exec_result, "metrics": metrics_result},
                text_description=item.text_description,
            )
            rewards.append(
                {
                    "object_id": item.object_id,
                    "reward": evaluation["reward"],
                    "base_reward": evaluation["base_reward"],
                    "format_reward": evaluation["format_reward"],
                    "sub_rewards": evaluation["sub_rewards"],
                    "success": exec_result["success"],
                    "metrics": metrics_result,
                    "mesh_stats": exec_result.get("mesh_stats"),
                    "elapsed": exec_result.get("elapsed", 0),
                    "error": exec_result.get("error", ""),
                }
            )

        valid = [reward for reward in rewards if reward["success"]]
        return {
            "rewards": rewards,
            "execution_rate": len(valid) / max(len(rewards), 1),
            "mean_reward": sum(reward["reward"] for reward in rewards) / max(len(rewards), 1),
        }

    @api.post("/reward/single")
    async def reward_single(req: SingleRequest, token: str = Query(None)):
        _verify_token(token)
        rubric = _build_rubric(req.reward_config)
        exec_result = execute_blender_code.remote(req.code, req.seed)
        metrics_result = None
        if exec_result["success"] and exec_result.get("mesh_bytes"):
            gt = _load_gt_mesh(req.object_id)
            if gt is not None:
                gt_bytes, gt_format = gt
                metrics_result = compute_metrics.remote(exec_result["mesh_bytes"], gt_bytes, 10_000, "obj", gt_format)
        evaluation = rubric.evaluate(
            req.code,
            {**exec_result, "metrics": metrics_result},
            text_description=req.text_description,
        )
        return {
            "reward": evaluation["reward"],
            "base_reward": evaluation["base_reward"],
            "format_reward": evaluation["format_reward"],
            "sub_rewards": evaluation["sub_rewards"],
            "success": exec_result["success"],
            "metrics": metrics_result,
            "exec_result": {key: value for key, value in exec_result.items() if key != "mesh_bytes"},
        }

    @api.post("/execute")
    async def execute(req: ExecuteRequest, token: str = Query(None)):
        _verify_token(token)
        exec_result = execute_blender_code.remote(req.code, req.seed)
        return {key: value for key, value in exec_result.items() if key != "mesh_bytes"}

    return api
