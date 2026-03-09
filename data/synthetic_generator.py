"""
Synthetic SFT data generation via teacher LLM distillation (Phase 1 of pipeline).

For each (uid, caption) pair from the manifest:
1. Prompt a teacher LLM (GPT-4o-mini) to write raw Blender Python code
2. Execute the code in Blender via Modal
3. Compare the output mesh to the ground-truth Objaverse mesh (CD, F-Score)
4. Keep validated (text, code) pairs; track failures in hard_prompts.csv

The hard_prompts.csv tracks every caption's attempt history so we can identify
objects where code generation is hardest — these get oversampled during RLVR.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import ProjectConfig, load_config
from data.hard_prompts import HardPromptTracker

log = logging.getLogger(__name__)

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


class SyntheticGenerator:
    """Generates validated (caption, raw_bpy_code) SFT pairs via teacher LLM."""

    def __init__(self, cfg: ProjectConfig | None = None):
        self.cfg = cfg or load_config()
        self.gen_cfg = self.cfg.synthetic_gen
        self.tracker = HardPromptTracker(self.gen_cfg.hard_prompts_path)
        self._checkpoint: dict[str, Any] = {}
        self._load_checkpoint()

    def _load_checkpoint(self):
        cp_path = Path(self.gen_cfg.checkpoint_path)
        if cp_path.exists():
            with open(cp_path) as f:
                self._checkpoint = json.load(f)
            log.info(f"Resumed from checkpoint: {len(self._checkpoint.get('completed_uids', []))} done")

    def _save_checkpoint(self, completed_uids: list[str]):
        cp_path = Path(self.gen_cfg.checkpoint_path)
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint["completed_uids"] = completed_uids
        with open(cp_path, "w") as f:
            json.dump(self._checkpoint, f)

    def _get_teacher_client(self):
        """Initialize teacher LLM API client."""
        provider = self.gen_cfg.teacher_provider

        if provider == "openai":
            from openai import OpenAI
            return OpenAI()
        elif provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic()
        else:
            raise ValueError(f"Unknown teacher provider: {provider}")

    def _call_teacher(self, client: Any, caption: str) -> str | None:
        """Generate one code completion from the teacher LLM."""
        provider = self.gen_cfg.teacher_provider

        if provider == "openai":
            resp = client.chat.completions.create(
                model=self.gen_cfg.teacher_model,
                messages=[
                    {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": _TEACHER_USER_TEMPLATE.format(caption=caption)},
                ],
                temperature=self.gen_cfg.temperature,
                max_tokens=4096,
            )
            text = resp.choices[0].message.content or ""
            return _extract_code(text)

        elif provider == "anthropic":
            resp = client.messages.create(
                model=self.gen_cfg.teacher_model,
                system=_TEACHER_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": _TEACHER_USER_TEMPLATE.format(caption=caption)},
                ],
                temperature=self.gen_cfg.temperature,
                max_tokens=4096,
            )
            text = resp.content[0].text if resp.content else ""
            return _extract_code(text)

        return None

    def _execute_and_validate(
        self,
        code: str,
        gt_mesh_path: str,
    ) -> dict[str, Any]:
        """Execute code in Blender and compute metrics against GT mesh."""
        from modal_infra.blender_worker import execute_blender_code
        from modal_infra.metrics_worker import compute_metrics

        exec_result = execute_blender_code.remote(code)

        if not exec_result.get("success") or not exec_result.get("mesh_bytes"):
            return exec_result

        stats = exec_result.get("mesh_stats", {})
        if stats.get("faces", 0) < self.gen_cfg.min_faces:
            return exec_result
        if stats.get("vertices", 0) > self.gen_cfg.max_vertices:
            return exec_result

        gt_bytes = Path(gt_mesh_path).read_bytes()
        metrics = compute_metrics.remote(exec_result["mesh_bytes"], gt_bytes)
        exec_result["metrics"] = metrics

        return exec_result

    def generate(self, manifest_path: str | Path | None = None) -> list[dict]:
        """Run synthetic data generation on the full manifest.

        For each caption, tries up to max_attempts. Tracks every attempt in the
        hard prompt CSV regardless of success. Keeps the best completion per
        object (lowest Chamfer Distance).
        """
        if manifest_path is None:
            manifest_path = Path(self.cfg.data_dir) / "manifest.jsonl"

        manifest = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    manifest.append(json.loads(line))

        completed_uids = set(self._checkpoint.get("completed_uids", []))
        pending = [m for m in manifest if m["uid"] not in completed_uids]
        log.info(f"Processing {len(pending)} entries ({len(completed_uids)} already done)")

        client = self._get_teacher_client()
        validated: list[dict] = []

        output_path = Path(self.gen_cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(output_path, "a")

        try:
            for entry in tqdm(pending, desc="Generating synthetic data"):
                uid = entry["uid"]
                caption = entry["caption"]
                gt_mesh_path = entry["mesh_path"]

                best_result: dict | None = None
                best_cd = float("inf")

                for _attempt in range(self.gen_cfg.max_attempts_per_caption):
                    code = self._call_teacher(client, caption)
                    if not code:
                        self.tracker.record_attempt(
                            uid, caption, success=False,
                            error_type="empty_response", gt_mesh_path=gt_mesh_path,
                        )
                        continue

                    try:
                        result = self._execute_and_validate(code, gt_mesh_path)
                    except Exception as e:
                        log.warning(f"Execution failed for {uid}: {e}")
                        self.tracker.record_attempt(
                            uid, caption, success=False,
                            error_type="infra_error", gt_mesh_path=gt_mesh_path,
                        )
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

                    self.tracker.record_attempt(
                        uid, caption, success=passed,
                        cd=cd, f_score=f_score,
                        error_type=error_type, gt_mesh_path=gt_mesh_path,
                    )

                    if passed and cd < best_cd:
                        best_cd = cd
                        best_result = {"code": code, "metrics": metrics}

                if best_result:
                    pair = {
                        "uid": uid,
                        "caption": caption,
                        "code": best_result["code"],
                        "metrics": best_result["metrics"],
                        "gt_mesh_path": gt_mesh_path,
                    }
                    validated.append(pair)
                    out_f.write(json.dumps(pair) + "\n")
                    out_f.flush()

                completed_uids.add(uid)

                if len(completed_uids) % self.gen_cfg.batch_size == 0:
                    self._save_checkpoint(list(completed_uids))
                    self.tracker.save()

        finally:
            out_f.close()
            self._save_checkpoint(list(completed_uids))
            self.tracker.save()

        log.info(
            f"Generation complete: {len(validated)}/{len(manifest)} validated pairs "
            f"({len(validated) / max(len(manifest), 1) * 100:.1f}% acceptance rate)"
        )
        self._log_hard_prompt_stats()

        return validated

    def _log_hard_prompt_stats(self):
        """Log summary statistics about hard prompts."""
        records = self.tracker.records
        if not records:
            return

        failure_rates = [r.failure_rate for r in records.values()]
        hard_count = len(self.tracker.get_hard_uids())

        log.info(
            f"Hard prompt stats: {len(records)} total tracked, "
            f"{hard_count} classified as hard (>50% failure rate), "
            f"mean failure rate: {sum(failure_rates) / len(failure_rates):.2f}"
        )

        error_types: dict[str, int] = {}
        for rec in records.values():
            if rec.last_error_type:
                error_types[rec.last_error_type] = error_types.get(rec.last_error_type, 0) + 1
        if error_types:
            log.info(f"Error type distribution: {error_types}")


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = SyntheticGenerator()
    gen.generate()
