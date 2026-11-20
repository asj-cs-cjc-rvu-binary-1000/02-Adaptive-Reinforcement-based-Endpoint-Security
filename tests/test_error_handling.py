"""
tests/test_error_handling.py — Error-handling tests.

Covers Task 4 area: invalid inputs do not crash.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_predict_pipeline_handles_missing_file(pipeline_default, tmp_path):
    result = pipeline_default.run(tmp_path / "does_not_exist.png")
    # Pipeline should return Undecodable, not crash
    assert result["action"] in ("Undecodable",)


def test_adaptive_agent_handles_empty_state(tmp_path):
    import adaptive_agent as aa
    agent = aa.HeuristicAdaptiveAgent(
        state_path=tmp_path / "nonexistent_state.json",
        feedback_path=tmp_path / "feedback.csv",
    )
    assert agent.thresholds() == (aa.DEFAULT_T_REVIEW, aa.DEFAULT_T_BLOCK)


def test_calculus_adapter_handles_missing_experiment(monkeypatch, tmp_path):
    """calculus_threshold.load_experiment must raise FileNotFoundError on missing file."""
    import calculus_threshold as ct
    monkeypatch.setattr(ct, "EXPERIMENT_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        ct.load_experiment()


def test_calculus_adapter_default_thresholds_fallback(monkeypatch, tmp_path):
    """If experiment is missing, default_thresholds() returns (0.35, 0.70) fallback."""
    import calculus_threshold as ct
    monkeypatch.setattr(ct, "EXPERIMENT_PATH", tmp_path / "missing.json")
    assert ct.default_thresholds() == (0.35, 0.70)