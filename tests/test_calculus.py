"""
tests/test_calculus.py — Calculus module contract tests.

Covers Task 4 area: threshold constraints and optimization outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_experiment_file_exists():
    p = PROJECT_ROOT / "results" / "calculus_experiment.json"
    assert p.exists()


def test_experiment_fit_on_validation_only():
    exp = json.loads((PROJECT_ROOT / "results" / "calculus_experiment.json").read_text())
    assert exp["fit_on"] == "validation split only"
    assert exp["test_used_for_fit"] is False


def test_default_thresholds_match_documented():
    import calculus_threshold as ct
    assert ct.default_thresholds() == pytest.approx((0.35, 0.70), abs=1e-9)


def test_optimized_thresholds_match_recorded():
    import calculus_threshold as ct
    t_r, t_b = ct.optimized_thresholds()
    assert t_r == pytest.approx(0.60, abs=1e-6)
    assert t_b == pytest.approx(0.7911, abs=1e-3)


def test_is_optimized_cost_lower_is_false():
    """On this dataset, the optimized policy has HIGHER hard cost."""
    import calculus_threshold as ct
    assert ct.is_optimized_cost_lower() is False


def test_optimization_hessian_positive():
    exp = json.loads((PROJECT_ROOT / "results" / "calculus_experiment.json").read_text())
    assert exp["optimization"]["hessian_det"] > 0
    assert exp["optimization"]["is_minimum"] is True


def test_scripts_calculus_threshold_has_bounds():
    """The Phase 2 scripts/calculus_threshold.py exposes BOUNDS with expected keys."""
    import calculus_threshold as scripts_ct  # top-level adapter
    # The GD module is in scripts/ — import that instead
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import importlib
    importlib.invalidate_caches()
    import calculus_threshold as _top
    # The scripts/ version may shadow the top-level; skip if same module
    # Instead, load by path:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scripts_calculus_threshold",
        str(PROJECT_ROOT / "scripts" / "calculus_threshold.py"),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert set(m.BOUNDS.keys()) == {"t_r_min", "t_r_max", "t_b_min", "t_b_max", "gap_min"}
    assert set(m.DEFAULT_COST.keys()) >= {"C_FP", "C_FN", "C_review", "tau", "eta"}
    assert callable(m.optimize)


def test_run_calculus_experiment_importable():
    import importlib
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(
        "scripts_run_calculus_experiment",
        str(PROJECT_ROOT / "scripts" / "run_calculus_experiment.py"),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert hasattr(m, "main")