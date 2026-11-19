#!/usr/bin/env python3
"""
adaptive_agent.py — Phase 3, Task 2.

Feedback-driven heuristic threshold policy. NOT formal reinforcement learning:
no MDP, no Bellman equation, no Q-learning, no value function. The "policy" is
two scalars (t_review, t_block) with an explicit, documented update rule.

The agent:
  - Reads default thresholds (t_review=0.35, t_block=0.70) at init.
  - Accepts feedback events (correct / false_positive / false_negative) on
    individual predictions.
  - Updates thresholds by a small bounded step, persisted to
    results/heuristic_policy_state.json.
  - Does NOT train, retrain, or mutate the frozen XGBoost model.
  - Does NOT modify any file under data/ or models/.

The default thresholds in the UI are what the ARES calculus panel and the
PredictionPipeline(policy="default") uses. If feedback moves the thresholds
away from default, the app displays the new heuristic thresholds but the
calculus panel continues to show the experimental (0.35/0.70 and 0.60/0.7911)
configurations for the *published* results.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
STATE_PATH = RESULTS / "heuristic_policy_state.json"
FEEDBACK_PATH = RESULTS / "feedback.csv"

# Default thresholds — must match results/calculus_experiment.json
DEFAULT_T_REVIEW = 0.35
DEFAULT_T_BLOCK = 0.70

# Permitted threshold ranges (mirror calculus_threshold.BOUNDS, restricted for the
# heuristic agent — the app never sets thresholds outside the calculus grid)
MIN_T_REVIEW = 0.15
MAX_T_REVIEW = 0.60
MIN_T_BLOCK = 0.45
MAX_T_BLOCK = 0.95
MIN_GAP = 0.15

# Feedback → threshold update. Bounded to prevent drift from a single event.
LR_T_REVIEW = 0.01
LR_T_BLOCK = 0.01

# Reward mapping — used for the audit log in feedback.csv, not for any model update.
REWARD = {"correct": +1.0, "false_positive": -1.5, "false_negative": -3.0}

VALID_VERDICTS = tuple(REWARD.keys())


@dataclass
class PolicyState:
    t_review: float
    t_block: float
    n_correct: int = 0
    n_false_positive: int = 0
    n_false_negative: int = 0
    updated_utc: str = ""

    def __post_init__(self):
        if not (MIN_T_REVIEW <= self.t_review <= MAX_T_REVIEW):
            raise ValueError(
                f"t_review={self.t_review} outside [{MIN_T_REVIEW}, {MAX_T_REVIEW}]"
            )
        if not (MIN_T_BLOCK <= self.t_block <= MAX_T_BLOCK):
            raise ValueError(
                f"t_block={self.t_block} outside [{MIN_T_BLOCK}, {MAX_T_BLOCK}]"
            )
        if self.t_block - self.t_review < MIN_GAP - 1e-9:
            raise ValueError(
                f"t_block - t_review = {self.t_block - self.t_review:.3f} < {MIN_GAP}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyState":
        return cls(
            t_review=float(d["t_review"]),
            t_block=float(d["t_block"]),
            n_correct=int(d.get("n_correct", 0)),
            n_false_positive=int(d.get("n_false_positive", 0)),
            n_false_negative=int(d.get("n_false_negative", 0)),
            updated_utc=d.get("updated_utc", ""),
        )


class HeuristicAdaptiveAgent:
    """Heuristic feedback-driven threshold policy.

    Thread-safe file access via fcntl.flock. Never retrains or mutates the model.
    """

    def __init__(self, state_path: Path | None = None, feedback_path: Path | None = None):
        self.state_path = state_path or STATE_PATH
        self.feedback_path = feedback_path or FEEDBACK_PATH
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_default()

    # ---- persistence ----
    def _load_or_default(self) -> PolicyState:
        if not self.state_path.exists():
            return PolicyState(DEFAULT_T_REVIEW, DEFAULT_T_BLOCK,
                               updated_utc=self._now())
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return PolicyState.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, OSError):
            # Malformed state file — fall back to defaults, do NOT crash.
            return PolicyState(DEFAULT_T_REVIEW, DEFAULT_T_BLOCK,
                               updated_utc=self._now())

    def _persist(self) -> None:
        self.state.updated_utc = self._now()
        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=self.state_path.parent, suffix=".tmp", delete=False
        )
        try:
            json.dump(self.state.to_dict(), tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, self.state_path)
        except Exception:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ---- policy ----
    def thresholds(self) -> tuple[float, float]:
        return self.state.t_review, self.state.t_block

    def reset_to_default(self) -> None:
        self.state = PolicyState(DEFAULT_T_REVIEW, DEFAULT_T_BLOCK,
                                 updated_utc=self._now())
        self._persist()

    # ---- feedback ----
    def submit_feedback(
        self,
        sample_id: str,
        predicted_prob: float,
        predicted_action: str,
        verdict: str,
    ) -> dict:
        """Record a feedback event and adjust thresholds.

        verdict ∈ {"correct", "false_positive", "false_negative"}.
        Returns the post-update state dict.
        """
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got {verdict!r}")
        if not (0.0 <= predicted_prob <= 1.0):
            raise ValueError(f"predicted_prob={predicted_prob} outside [0,1]")

        # Update counters
        if verdict == "correct":
            self.state.n_correct += 1
        elif verdict == "false_positive":
            self.state.n_false_positive += 1
            # A FP means we flagged a benign URL → raise thresholds slightly.
            self.state.t_review = min(self.state.t_review + LR_T_REVIEW, MAX_T_REVIEW)
            self.state.t_block = min(self.state.t_block + LR_T_BLOCK, MAX_T_BLOCK)
        elif verdict == "false_negative":
            self.state.n_false_negative += 1
            # A FN means we missed a malicious URL → lower thresholds slightly.
            self.state.t_review = max(self.state.t_review - LR_T_REVIEW, MIN_T_REVIEW)
            self.state.t_block = max(self.state.t_block - LR_T_BLOCK, MIN_T_BLOCK)

        # Preserve the gap invariant after updates
        if self.state.t_block - self.state.t_review < MIN_GAP:
            self.state.t_block = min(self.state.t_review + MIN_GAP, MAX_T_BLOCK)
            if self.state.t_block - self.state.t_review < MIN_GAP:
                self.state.t_review = max(self.state.t_block - MIN_GAP, MIN_T_REVIEW)

        self._persist()

        # Append to feedback.csv (append-only audit log; never read by the model)
        self._append_feedback(sample_id, predicted_prob, predicted_action, verdict)

        return self.state.to_dict()

    def _append_feedback(
        self, sample_id: str, prob: float, action: str, verdict: str
    ) -> None:
        import csv
        new_file = not self.feedback_path.exists()
        with open(self.feedback_path, "a", newline="", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                w = csv.writer(f)
                if new_file:
                    w.writerow(["utc", "sample_id", "predicted_prob",
                                "predicted_action", "verdict",
                                "reward", "t_review_after", "t_block_after"])
                w.writerow([
                    self._now(), sample_id, f"{prob:.6f}", action, verdict,
                    f"{REWARD[verdict]:+.2f}",
                    f"{self.state.t_review:.4f}", f"{self.state.t_block:.4f}",
                ])
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)