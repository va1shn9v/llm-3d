"""
Hard prompt tracking and loading for RLVR hard mining.

During synthetic data generation, captions where code generation fails are
tracked in a CSV with failure statistics. During RLVR, this CSV is loaded to
oversample hard prompts — the ones the teacher model struggled with are the
ones where RL exploration has the most room to improve.

CSV schema:
    uid, caption, attempts, successes, failure_rate, best_cd, best_f_score,
    last_error_type, gt_mesh_path
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import HardMiningConfig

log = logging.getLogger(__name__)


@dataclass
class HardPromptRecord:
    uid: str
    caption: str
    attempts: int = 0
    successes: int = 0
    failure_rate: float = 1.0
    best_cd: float = float("inf")
    best_f_score: float = 0.0
    last_error_type: str = ""
    gt_mesh_path: str = ""

    def update(self, success: bool, cd: float | None, f_score: float | None, error_type: str = ""):
        self.attempts += 1
        if success:
            self.successes += 1
        if cd is not None and cd < self.best_cd:
            self.best_cd = cd
        if f_score is not None and f_score > self.best_f_score:
            self.best_f_score = f_score
        if error_type:
            self.last_error_type = error_type
        self.failure_rate = 1.0 - (self.successes / self.attempts) if self.attempts > 0 else 1.0


class HardPromptTracker:
    """Accumulates failure statistics during synthetic data generation.

    Writes a CSV that the RL trainer can later load for hard mining.
    """

    _CSV_FIELDS = [
        "uid", "caption", "attempts", "successes", "failure_rate",
        "best_cd", "best_f_score", "last_error_type", "gt_mesh_path",
    ]

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self._records: dict[str, HardPromptRecord] = {}
        if self.csv_path.exists():
            self._load_existing()

    def _load_existing(self):
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = HardPromptRecord(
                    uid=row["uid"],
                    caption=row["caption"],
                    attempts=int(row.get("attempts", 0)),
                    successes=int(row.get("successes", 0)),
                    failure_rate=float(row.get("failure_rate", 1.0)),
                    best_cd=float(row.get("best_cd", "inf")),
                    best_f_score=float(row.get("best_f_score", 0.0)),
                    last_error_type=row.get("last_error_type", ""),
                    gt_mesh_path=row.get("gt_mesh_path", ""),
                )
                self._records[rec.uid] = rec
        log.info(f"Loaded {len(self._records)} existing hard prompt records from {self.csv_path}")

    def record_attempt(
        self,
        uid: str,
        caption: str,
        success: bool,
        cd: float | None = None,
        f_score: float | None = None,
        error_type: str = "",
        gt_mesh_path: str = "",
    ):
        if uid not in self._records:
            self._records[uid] = HardPromptRecord(
                uid=uid, caption=caption, gt_mesh_path=gt_mesh_path,
            )
        self._records[uid].update(success, cd, f_score, error_type)

    def save(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
            writer.writeheader()
            for rec in sorted(self._records.values(), key=lambda r: -r.failure_rate):
                writer.writerow({
                    "uid": rec.uid,
                    "caption": rec.caption,
                    "attempts": rec.attempts,
                    "successes": rec.successes,
                    "failure_rate": round(rec.failure_rate, 4),
                    "best_cd": round(rec.best_cd, 6) if rec.best_cd != float("inf") else "inf",
                    "best_f_score": round(rec.best_f_score, 6),
                    "last_error_type": rec.last_error_type,
                    "gt_mesh_path": rec.gt_mesh_path,
                })
        log.info(f"Saved {len(self._records)} hard prompt records to {self.csv_path}")

    @property
    def records(self) -> dict[str, HardPromptRecord]:
        return self._records

    def get_hard_uids(self, min_failure_rate: float = 0.5, min_attempts: int = 2) -> list[str]:
        """Return UIDs where the teacher model failed above threshold."""
        return [
            uid
            for uid, rec in self._records.items()
            if rec.failure_rate >= min_failure_rate and rec.attempts >= min_attempts
        ]


class HardPromptSampler:
    """Loads hard prompt CSV and provides weighted sampling for RLVR.

    During RL training, a fraction of each batch is drawn from hard prompts
    (high failure rate during synthetic generation). This focuses RL exploration
    on the objects where the teacher model struggled, giving the policy more
    room to discover novel solutions via reward signal.
    """

    def __init__(
        self,
        hard_csv_path: str | Path,
        all_prompts: list[dict],
        cfg: HardMiningConfig | None = None,
        seed: int = 42,
    ):
        if cfg is None:
            cfg = HardMiningConfig()
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self._hard_records: dict[str, HardPromptRecord] = {}
        csv_path = Path(hard_csv_path)
        if csv_path.exists():
            tracker = HardPromptTracker(csv_path)
            self._hard_records = tracker.records

        self._hard_uids = set()
        for uid, rec in self._hard_records.items():
            if rec.failure_rate >= cfg.min_failure_rate and rec.attempts >= cfg.min_attempts:
                self._hard_uids.add(uid)

        self._all_prompts = all_prompts
        prompt_by_uid = {p.get("object_id", ""): p for p in all_prompts}
        self._hard_prompts = [prompt_by_uid[uid] for uid in self._hard_uids if uid in prompt_by_uid]
        self._normal_prompts = [p for p in all_prompts if p.get("object_id", "") not in self._hard_uids]

        log.info(
            f"HardPromptSampler: {len(self._hard_prompts)} hard prompts, "
            f"{len(self._normal_prompts)} normal prompts "
            f"(ratio={cfg.hard_prompt_ratio})"
        )

    @property
    def num_hard(self) -> int:
        return len(self._hard_prompts)

    @property
    def num_normal(self) -> int:
        return len(self._normal_prompts)

    def sample(self, n: int) -> list[dict]:
        """Sample n prompts with hard mining ratio applied."""
        if not self._hard_prompts or not self.cfg.enabled:
            indices = self.rng.choice(len(self._all_prompts), size=min(n, len(self._all_prompts)), replace=False)
            return [self._all_prompts[i] for i in indices]

        n_hard = min(int(n * self.cfg.hard_prompt_ratio), len(self._hard_prompts))
        n_normal = n - n_hard

        hard_idx = self.rng.choice(len(self._hard_prompts), size=n_hard, replace=n_hard > len(self._hard_prompts))
        normal_idx = self.rng.choice(
            len(self._normal_prompts),
            size=min(n_normal, len(self._normal_prompts)),
            replace=n_normal > len(self._normal_prompts),
        )

        batch = [self._hard_prompts[i] for i in hard_idx] + [self._normal_prompts[i] for i in normal_idx]
        self.rng.shuffle(batch)
        return batch
