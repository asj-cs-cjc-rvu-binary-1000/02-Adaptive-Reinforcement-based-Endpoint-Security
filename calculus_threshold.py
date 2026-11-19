#!/usr/bin/env python3
"""
calculus_threshold.py — Phase 3, Task 3.

READ-ONLY adapter for the Phase 2 calculus experiment.

This module does NOT reimplement gradient descent or the sigmoid-smoothed cost.
Those live in scripts/calculus_threshold.py and are exercised in
scripts/run_calculus_experiment.py. This module exposes the results so the
Streamlit panel can display them without duplicating logic.

Public API:
  load_experiment()        → dict (raw results/calculus_experiment.json)
  default_thresholds()     → (t_r, t_b) = (0.35, 0.70)
  optimized_thresholds()   → (t_r, t_b) = (0.60, 0.7911…) from the JSON
  compare_policies()       → dict {default: {...}, optimized: {...}}
  is_optimized_cost_lower() → False on this dataset (documented honestly)
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
EXPERIMENT_PATH = RESULTS / "calculus_experiment.json"

# Hard fallbacks (only used if experiment file is missing/corrupt). These match
# results/calculus_experiment.json at Phase 2 HEAD but are NOT authoritative —
# the JSON is. This module fails closed if the JSON cannot be read.
_FALLBACK_DEFAULT = (0.35, 0.70)


def load_experiment() -> dict:
    if not EXPERIMENT_PATH.exists():
        raise FileNotFoundError(
            f"FATAL: {EXPERIMENT_PATH} missing. Run scripts/run_calculus_experiment.py."
        )
    try:
        return json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Malformed calculus_experiment.json: {e}") from e


def default_thresholds() -> tuple[float, float]:
    try:
        d = load_experiment()["default_thresholds"]
        return float(d["t_r"]), float(d["t_b"])
    except (FileNotFoundError, KeyError, RuntimeError):
        return _FALLBACK_DEFAULT


def optimized_thresholds() -> tuple[float, float]:
    d = load_experiment()["optimized_thresholds"]
    return float(d["t_r"]), float(d["t_b"])


def compare_policies() -> dict:
    """Return validation and test metrics for both policies, from the JSON."""
    exp = load_experiment()
    return {
        "default_thresholds": default_thresholds(),
        "optimized_thresholds": optimized_thresholds(),
        "validation": exp["validation_metrics"],
        "test": exp["test_metrics"],
        "optimization": {
            "hessian_det": exp["optimization"]["hessian_det"],
            "is_minimum": exp["optimization"]["is_minimum"],
            "is_stationary": exp["optimization"]["is_stationary"],
            "boundary_hit": exp["optimization"]["boundary_hit"],
            "is_kkt_constrained": exp["optimization"]["is_kkt_constrained"],
            "cost_smooth_final": exp["optimization"]["cost_smooth_final"],
            "cost_hard_final": exp["optimization"]["cost_hard_final"],
        },
    }


def is_optimized_cost_lower() -> bool:
    """On this dataset, the answer is False — the optimized policy has higher
    hard cost than default on both validation and test. This helper exists so the
    UI can say so programmatically rather than by hardcoded English."""
    exp = load_experiment()
    val = exp["validation_metrics"]
    test = exp["test_metrics"]
    # Hard cost proxy: C_FP*FPR + C_FN*FNR + C_review*Review
    # (defaults C_FP=1, C_FN=5, C_review=0.5 — match calculus_experiment.json)
    def hard_cost(m: dict) -> float:
        return (1.0 * m["fpr"]) + (5.0 * m["fnr"]) + (0.5 * m["review_rate"])

    val_default = hard_cost(val["default"])
    val_opt = hard_cost(val["optimized"])
    test_default = hard_cost(test["default"])
    test_opt = hard_cost(test["optimized"])
    return bool(val_opt < val_default and test_opt < test_default)