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

import modal
from modal import asgi_app

from modal_infra.images import blender_image
from modal_infra.blender_worker import execute_blender_code
from modal_infra.metrics_worker import compute_metrics
from modal_infra.render_worker import render_mesh_views

app = modal.App("llm3d-reward-api")
volume = modal.Volume.from_name("llm3d-data", create_if_missing=True)

_START_TIME = time.time()


def _verify_token(token: str | None):
    expected = os.environ.get("REWARD_API_TOKEN", "")
    if expected and token != expected:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Invalid token")


def _load_gt_mesh(object_id: str) -> bytes | None:
    """Load ground-truth mesh from volume."""
    path = f"/data/meshes/{object_id}.obj"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def _compute_reward(
    code: str,
    exec_result: dict,
    metrics_result: dict | None,
) -> float:
    """Gated sparse + dense reward (Section 6 of spec)."""
    if not code or not code.strip():
        return 0.0

    if "from bpy_lib import" not in code and "import bpy" not in code:
        return 0.0

    if not exec_result.get("success", False):
        return 0.05

    stats = exec_result.get("mesh_stats")
    if not stats or stats.get("faces", 0) < 4:
        return 0.10

    if stats.get("vertices", 0) > 100_000:
        return 0.15

    if metrics_result is None or metrics_result.get("error"):
        return 0.15

    f_score = metrics_result.get("f_score_005", 0.0)
    if f_score < 0.05:
        return 0.20

    quality = 0.3 + 0.7 * min(1.0, f_score / 0.6)
    return quality


def _format_reward(code: str) -> float:
    """Bonus for following expected code structure."""
    score = 0.0
    if code.startswith("from bpy_lib import"):
        score += 0.25
    if "# object name:" in code:
        score += 0.25
    if "# part_" in code:
        score += 0.25
    if "export_scene()" in code:
        score += 0.25
    return score


@app.function(
    image=blender_image,
    cpu=4, memory=8192, timeout=600,
    volumes={"/data": volume},
    keep_warm=1,
    allow_concurrent_inputs=50,
)
@asgi_app()
def reward_api():
    from fastapi import FastAPI, Query
    from pydantic import BaseModel

    api = FastAPI(title="LLM-3D Reward API")

    class RewardItem(BaseModel):
        object_id: str
        code: str
        seed: int = 42

    class BatchRequest(BaseModel):
        items: list[RewardItem]

    class SingleRequest(BaseModel):
        object_id: str
        code: str
        seed: int = 42

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

    @api.post("/reward/batch")
    async def reward_batch(req: BatchRequest, token: str = Query(None)):
        _verify_token(token)

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
                    metrics_futures.append(
                        compute_metrics.spawn(er["mesh_bytes"], gt)
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
            base = _compute_reward(item.code, er, mr)
            fmt = _format_reward(item.code)
            total = 0.9 * base + 0.1 * fmt
            rewards.append({
                "object_id": item.object_id,
                "reward": total,
                "base_reward": base,
                "format_reward": fmt,
                "success": er["success"],
                "metrics": mr,
                "elapsed": er.get("elapsed", 0),
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
        er = execute_blender_code.remote(req.code, req.seed)
        mr = None
        if er["success"] and er.get("mesh_bytes"):
            gt = _load_gt_mesh(req.object_id)
            if gt:
                mr = compute_metrics.remote(er["mesh_bytes"], gt)
        base = _compute_reward(req.code, er, mr)
        fmt = _format_reward(req.code)
        return {
            "reward": 0.9 * base + 0.1 * fmt,
            "base_reward": base,
            "format_reward": fmt,
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
        imgs = render_mesh_views.remote(
            gt, "obj", req.num_views, tuple(req.resolution),
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
