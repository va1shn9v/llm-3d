"""
Async synthetic SFT data generation via teacher LLM distillation.

Uses an async producer-consumer architecture:
  - Producer coroutines call AsyncOpenAI (n=samples_per_caption) concurrently
  - Codes are pushed into an asyncio.Queue with backpressure
  - Consumer coroutines pull from the queue, execute in Modal Blender,
    validate against GT meshes, and write validated (caption, code) pairs
  - Validated mesh artifacts are stored in a Modal Volume for downstream use

The hard_prompts.csv tracks every caption's attempt history so we can identify
objects where code generation is hardest -- these get oversampled during RLVR.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from pathlib import Path
from typing import Any

from config import ProjectConfig, load_config
from data.hard_prompts import HardPromptTracker

log = logging.getLogger(__name__)

_SENTINEL = None

_TEACHER_SYSTEM_PROMPT = """\
You are an expert Blender Python developer. Write a complete bpy script \
that creates the described 3D object. Use standard bpy operations (primitives, \
BMesh, modifiers, booleans, curves). The script must:
1. Start with `import bpy` and clear the default scene
2. Create the geometry described
3. Export the result to OBJ at the path from os.environ["EXPORT_PATH"]
Output only the Python code, no explanations."""

_TEACHER_USER_TEMPLATE = "Create a 3D model of: {caption}"


def _classify_error(exec_result: dict[str, Any]) -> str:
    """Classify the failure type for hard prompt tracking."""
    if not exec_result.get("success", False):
        error = exec_result.get("error", "")
        if "timed out" in error.lower():
            return "timeout"
        if "SyntaxError" in error:
            return "syntax"
        return "runtime"

    stats = exec_result.get("mesh_stats")
    if not stats or stats.get("faces", 0) < 4:
        return "no_geometry"
    if stats.get("vertices", 0) > 100_000:
        return "degenerate"

    metrics = exec_result.get("metrics")
    if metrics and metrics.get("f_score_005", 0) < 0.05:
        return "no_resemblance"

    return ""


def _extract_code(text: str) -> str | None:
    """Extract Python code from LLM response, handling markdown fences."""
    text = text.strip()
    if not text:
        return None

    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        if text[start:].startswith("\n"):
            start += 1
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()

    if not text or "import bpy" not in text:
        return None

    return text


class SyntheticGenerator:
    """Generates validated (caption, raw_bpy_code) SFT pairs via async pipeline."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        self.gen_cfg = self.cfg.synthetic_gen
        self.tracker = HardPromptTracker(self.gen_cfg.hard_prompts_path)
        self._checkpoint: dict[str, Any] = {}
        self._load_checkpoint()

        self._tracker_lock = asyncio.Lock()
        self._output_lock = asyncio.Lock()
        self._checkpoint_lock = asyncio.Lock()
        self._best_lock = asyncio.Lock()

        self._shutting_down = False

    def _load_checkpoint(self):
        cp_path = Path(self.gen_cfg.checkpoint_path)
        if cp_path.exists():
            with open(cp_path, encoding="utf-8") as f:
                self._checkpoint = json.load(f)
            log.info(
                "Resumed from checkpoint: %d done",
                len(self._checkpoint.get("completed_uids", [])),
            )

    def _save_checkpoint(self, completed_uids: list[str]):
        cp_path = Path(self.gen_cfg.checkpoint_path)
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint["completed_uids"] = completed_uids
        with open(cp_path, "w", encoding="utf-8") as f:
            json.dump(self._checkpoint, f)

    def _get_teacher_client(self):
        """Initialize async teacher LLM API client."""
        provider = self.gen_cfg.teacher_provider
        if provider == "openai":
            from openai import AsyncOpenAI
            return AsyncOpenAI()
        raise ValueError(f"Unknown teacher provider: {provider}")

    async def _call_teacher_batch(
        self, client: Any, caption: str,
    ) -> list[str]:
        """Generate multiple code completions in a single API call via n param."""
        resp = await client.chat.completions.create(
            model=self.gen_cfg.teacher_model,
            messages=[
                {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": _TEACHER_USER_TEMPLATE.format(caption=caption)},
            ],
            n=self.gen_cfg.samples_per_caption,
            temperature=self.gen_cfg.temperature,
            max_tokens=4096,
        )
        codes: list[str] = []
        for choice in resp.choices:
            code = _extract_code(choice.message.content or "")
            if code:
                codes.append(code)
        return codes

    async def _execute_and_validate_async(
        self, code: str, gt_mesh_path: str,
    ) -> dict[str, Any]:
        """Execute code in Modal Blender and compute metrics, non-blocking."""
        import modal

        execute_blender_code = modal.Function.from_name(
            "llm3d-blender-worker", "execute_blender_code",
        )
        compute_metrics = modal.Function.from_name(
            "llm3d-metrics-worker", "compute_metrics",
        )

        exec_result: dict[str, Any] = await execute_blender_code.remote.aio(code)

        if not exec_result.get("success") or not exec_result.get("mesh_bytes"):
            return exec_result

        stats = exec_result.get("mesh_stats", {})
        if stats.get("faces", 0) < self.gen_cfg.min_faces:
            return exec_result
        if stats.get("vertices", 0) > self.gen_cfg.max_vertices:
            return exec_result

        gt_bytes = await asyncio.to_thread(Path(gt_mesh_path).read_bytes)
        metrics: dict[str, Any] = await compute_metrics.remote.aio(
            exec_result["mesh_bytes"], gt_bytes,
        )
        exec_result["metrics"] = metrics

        return exec_result

    async def _store_artifact_async(self, uid: str, mesh_bytes: bytes) -> str | None:
        """Persist validated mesh to Modal Volume."""
        try:
            import modal

            store_mesh_artifact = modal.Function.from_name(
                "llm3d-blender-worker", "store_mesh_artifact",
            )
            path: str = await store_mesh_artifact.remote.aio(uid, mesh_bytes)
            return path
        except Exception:
            log.warning("Failed to store artifact for %s", uid, exc_info=True)
            return None

    async def _produce(
        self,
        entries: list[dict],
        queue: asyncio.Queue,
        client: Any,
        llm_sem: asyncio.Semaphore,
        completed_uids: set[str],
    ):
        """Producer: call teacher LLM for each entry, push codes into queue."""
        for entry in entries:
            if self._shutting_down:
                break
            uid = entry["uid"]
            caption = entry["caption"]
            gt_mesh_path = entry["mesh_path"]

            if uid in completed_uids:
                continue

            try:
                async with llm_sem:
                    codes = await self._call_teacher_batch(client, caption)
            except Exception:
                log.warning("Teacher call failed for %s", uid, exc_info=True)
                async with self._tracker_lock:
                    self.tracker.record_attempt(
                        uid, caption, success=False,
                        error_type="llm_error", gt_mesh_path=gt_mesh_path,
                    )
                continue

            if not codes:
                async with self._tracker_lock:
                    self.tracker.record_attempt(
                        uid, caption, success=False,
                        error_type="empty_response", gt_mesh_path=gt_mesh_path,
                    )
                continue

            for code in codes:
                await queue.put((uid, caption, code, gt_mesh_path))

    async def _consume(
        self,
        queue: asyncio.Queue,
        blender_sem: asyncio.Semaphore,
        best_per_uid: dict[str, dict],
        completed_uids: set[str],
        progress: dict[str, int],
    ):
        """Consumer: pull codes, execute in Modal, validate, track results."""
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                await queue.put(_SENTINEL)
                break

            uid, caption, code, gt_mesh_path = item

            try:
                async with blender_sem:
                    result = await self._execute_and_validate_async(code, gt_mesh_path)
            except Exception:
                log.warning("Execution failed for %s", uid, exc_info=True)
                async with self._tracker_lock:
                    self.tracker.record_attempt(
                        uid, caption, success=False,
                        error_type="infra_error", gt_mesh_path=gt_mesh_path,
                    )
                queue.task_done()
                continue

            metrics = result.get("metrics", {})
            cd = metrics.get("chamfer", float("inf"))
            f_score = metrics.get("f_score_005", 0.0)
            error_type = _classify_error(result)
            passed = (
                result.get("success", False)
                and cd < self.gen_cfg.cd_threshold
                and f_score > self.gen_cfg.f_score_threshold
            )

            async with self._tracker_lock:
                self.tracker.record_attempt(
                    uid, caption, success=passed,
                    cd=cd, f_score=f_score,
                    error_type=error_type, gt_mesh_path=gt_mesh_path,
                )

            if passed:
                mesh_bytes = result.get("mesh_bytes")

                async with self._best_lock:
                    prev = best_per_uid.get(uid)
                    if prev is None or cd < prev["cd"]:
                        best_per_uid[uid] = {
                            "code": code,
                            "metrics": metrics,
                            "cd": cd,
                            "caption": caption,
                            "gt_mesh_path": gt_mesh_path,
                        }

                if mesh_bytes:
                    asyncio.create_task(self._store_artifact_async(uid, mesh_bytes))

            progress["processed"] += 1
            if progress["processed"] % self.gen_cfg.batch_size == 0:
                await self._periodic_save(completed_uids, best_per_uid)

            queue.task_done()

    async def _periodic_save(
        self, completed_uids: set[str], best_per_uid: dict[str, dict],
    ):
        """Checkpoint + tracker save, async-safe."""
        async with self._checkpoint_lock:
            all_done = list(completed_uids | set(best_per_uid.keys()))
            await asyncio.to_thread(self._save_checkpoint, all_done)
            async with self._tracker_lock:
                await asyncio.to_thread(self.tracker.save)
        log.info("Checkpoint saved: %d UIDs processed", len(all_done))

    async def _flush_best_results(
        self, best_per_uid: dict[str, dict], out_f,
    ) -> list[dict]:
        """Write the best result per UID to the output JSONL."""
        validated: list[dict] = []
        for uid, best in best_per_uid.items():
            pair = {
                "uid": uid,
                "caption": best["caption"],
                "code": best["code"],
                "metrics": best["metrics"],
                "gt_mesh_path": best["gt_mesh_path"],
            }
            validated.append(pair)
            async with self._output_lock:
                out_f.write(json.dumps(pair) + "\n")
                out_f.flush()
        return validated

    async def generate(self, manifest_path: str | Path | None = None) -> list[dict]:
        """Run async synthetic data generation on the full manifest.

        Launches producer coroutines that call the teacher LLM and consumer
        coroutines that execute + validate in Modal Blender. Best-of-N
        selection keeps the lowest-CD passing result per UID.
        """
        if manifest_path is None:
            manifest_path = Path(self.cfg.data_dir) / "manifest.jsonl"

        manifest: list[dict] = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    manifest.append(json.loads(line))

        completed_uids = set(self._checkpoint.get("completed_uids", []))
        pending = [m for m in manifest if m["uid"] not in completed_uids]
        log.info(
            "Processing %d entries (%d already done)", len(pending), len(completed_uids),
        )

        if not pending:
            log.info("Nothing to process")
            output_path = Path(self.gen_cfg.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch(exist_ok=True)
            return []

        client = self._get_teacher_client()
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.gen_cfg.queue_size)
        llm_sem = asyncio.Semaphore(self.gen_cfg.max_concurrent_llm)
        blender_sem = asyncio.Semaphore(self.gen_cfg.max_concurrent_blender)
        best_per_uid: dict[str, dict] = {}
        progress: dict[str, int] = {"processed": 0}

        output_path = Path(self.gen_cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(output_path, "a", encoding="utf-8")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        num_producers = self.gen_cfg.num_producers
        chunk_size = max(1, len(pending) // num_producers)
        producer_chunks = [
            pending[i: i + chunk_size]
            for i in range(0, len(pending), chunk_size)
        ]

        try:
            producers = [
                asyncio.create_task(
                    self._produce(chunk, queue, client, llm_sem, completed_uids),
                    name=f"producer-{i}",
                )
                for i, chunk in enumerate(producer_chunks)
            ]

            consumers = [
                asyncio.create_task(
                    self._consume(
                        queue, blender_sem, best_per_uid,
                        completed_uids, progress,
                    ),
                    name=f"consumer-{i}",
                )
                for i in range(self.gen_cfg.num_consumers)
            ]

            await asyncio.gather(*producers, return_exceptions=True)
            await queue.put(_SENTINEL)
            await asyncio.gather(*consumers, return_exceptions=True)

            validated = await self._flush_best_results(best_per_uid, out_f)

        finally:
            out_f.close()
            all_done = list(completed_uids | set(best_per_uid.keys()))
            self._save_checkpoint(all_done)
            self.tracker.save()

        total = len(manifest)
        rate = len(validated) / max(total, 1) * 100
        log.info(
            "Generation complete: %d/%d validated pairs (%.1f%% acceptance rate)",
            len(validated), total, rate,
        )
        self._log_hard_prompt_stats()

        return validated

    def _handle_shutdown(self):
        log.info("Shutdown signal received, finishing in-flight work...")
        self._shutting_down = True

    def _log_hard_prompt_stats(self):
        """Log summary statistics about hard prompts."""
        records = self.tracker.records
        if not records:
            return

        failure_rates = [r.failure_rate for r in records.values()]
        hard_count = len(self.tracker.get_hard_uids())

        mean_fr = sum(failure_rates) / len(failure_rates)
        log.info(
            "Hard prompt stats: %d total tracked, %d classified as hard "
            "(>50%% failure rate), mean failure rate: %.2f",
            len(records), hard_count, mean_fr,
        )

        error_types: dict[str, int] = {}
        for rec in records.values():
            if rec.last_error_type:
                error_types[rec.last_error_type] = (
                    error_types.get(rec.last_error_type, 0) + 1
                )
        if error_types:
            log.info("Error type distribution: %s", error_types)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    gen = SyntheticGenerator()
    asyncio.run(gen.generate())
