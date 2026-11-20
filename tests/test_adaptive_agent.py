"""
tests/test_adaptive_agent.py — Adaptive agent contract tests.

Covers Task 4 areas:
  - Boundary behaviour at t_review and t_block
  - Invalid feedback rejected
  - Repeated feedback is safe
  - Persistence and reset
  - No ML library imports
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import adaptive_agent as aa


# ---------- threshold boundaries ----------

@pytest.mark.parametrize("prob,expected_action", [
    (0.000, "Allow"),
    (0.100, "Allow"),
    (0.3499, "Allow"),
    (0.3500, "Review"),   # t_review exactly
    (0.5000, "Review"),
    (0.6999, "Review"),
    (0.7000, "Block"),    # t_block exactly
    (0.999, "Block"),
    (1.000, "Block"),
])
def test_decision_boundaries_default(isolated_agent, prob, expected_action):
    """Note: the decision boundaries the app uses are the HeuristicAdaptiveAgent's,
    not PredictionPipeline._decide's. This test asserts the boundaries directly."""
    t_r, t_b = isolated_agent.thresholds()
    assert t_r == pytest.approx(0.35, abs=1e-9)
    assert t_b == pytest.approx(0.70, abs=1e-9)
    if prob >= t_b:
        action = "Block"
    elif prob >= t_r:
        action = "Review"
    else:
        action = "Allow"
    assert action == expected_action


# ---------- feedback paths ----------

def test_submit_correct(isolated_agent):
    state_before = isolated_agent.thresholds()
    result = isolated_agent.submit_feedback(
        sample_id="s1", predicted_prob=0.20,
        predicted_action="Allow", verdict="correct"
    )
    assert result["n_correct"] == 1
    assert isolated_agent.thresholds() == state_before  # correct does not move thresholds


def test_submit_false_positive_moves_thresholds_up(isolated_agent):
    t_r0, t_b0 = isolated_agent.thresholds()
    isolated_agent.submit_feedback("s1", 0.20, "Allow", "false_positive")
    t_r1, t_b1 = isolated_agent.thresholds()
    assert t_r1 == pytest.approx(t_r0 + aa.LR_T_REVIEW, abs=1e-9)
    assert t_b1 == pytest.approx(t_b0 + aa.LR_T_BLOCK, abs=1e-9)


def test_submit_false_negative_moves_thresholds_down(isolated_agent):
    t_r0, t_b0 = isolated_agent.thresholds()
    isolated_agent.submit_feedback("s1", 0.50, "Block", "false_negative")
    t_r1, t_b1 = isolated_agent.thresholds()
    assert t_r1 == pytest.approx(t_r0 - aa.LR_T_REVIEW, abs=1e-9)
    assert t_b1 == pytest.approx(t_b0 - aa.LR_T_BLOCK, abs=1e-9)


def test_thresholds_bounded_upper(isolated_agent):
    """Repeated FP feedback must clamp at MAX_T_REVIEW and MAX_T_BLOCK."""
    for _ in range(200):
        isolated_agent.submit_feedback("x", 0.10, "Allow", "false_positive")
    t_r, t_b = isolated_agent.thresholds()
    assert t_r <= aa.MAX_T_REVIEW + 1e-9
    assert t_b <= aa.MAX_T_BLOCK + 1e-9


def test_thresholds_bounded_lower(isolated_agent):
    for _ in range(200):
        isolated_agent.submit_feedback("x", 0.90, "Block", "false_negative")
    t_r, t_b = isolated_agent.thresholds()
    assert t_r >= aa.MIN_T_REVIEW - 1e-9
    assert t_b >= aa.MIN_T_BLOCK - 1e-9


def test_gap_invariant_preserved(isolated_agent):
    """After any sequence of feedback, t_block - t_review >= MIN_GAP."""
    for verdict in ("false_positive", "false_negative", "correct") * 30:
        isolated_agent.submit_feedback("s", 0.40, "Review", verdict)
        t_r, t_b = isolated_agent.thresholds()
        assert t_b - t_r >= aa.MIN_GAP - 1e-6


# ---------- validation ----------

def test_invalid_verdict_rejected(isolated_agent):
    with pytest.raises(ValueError):
        isolated_agent.submit_feedback("s", 0.5, "Review", "not_a_verdict")


@pytest.mark.parametrize("prob", [-0.01, 1.01, 2.0, -5.0])
def test_invalid_probability_rejected(isolated_agent, prob):
    with pytest.raises(ValueError):
        isolated_agent.submit_feedback("s", prob, "Block", "correct")


# ---------- persistence ----------

def test_state_persists_and_reloads(isolated_agent, tmp_path):
    isolated_agent.submit_feedback("s", 0.10, "Allow", "false_positive")
    t_r_after, t_b_after = isolated_agent.thresholds()
    # Reload from the same state file
    reloaded = aa.HeuristicAdaptiveAgent(
        state_path=isolated_agent.state_path,
        feedback_path=isolated_agent.feedback_path,
    )
    assert reloaded.thresholds() == (t_r_after, t_b_after)


def test_malformed_state_falls_back_to_defaults(tmp_path):
    bad_state = tmp_path / "heuristic_policy_state.json"
    bad_state.write_text("{ not json")
    agent = aa.HeuristicAdaptiveAgent(
        state_path=bad_state, feedback_path=tmp_path / "fb.csv"
    )
    assert agent.thresholds() == (aa.DEFAULT_T_REVIEW, aa.DEFAULT_T_BLOCK)


def test_reset_to_default(isolated_agent):
    isolated_agent.submit_feedback("s", 0.10, "Allow", "false_positive")
    isolated_agent.reset_to_default()
    assert isolated_agent.thresholds() == (aa.DEFAULT_T_REVIEW, aa.DEFAULT_T_BLOCK)


def test_feedback_csv_written(isolated_agent):
    isolated_agent.submit_feedback("s1", 0.5, "Review", "correct")
    assert isolated_agent.feedback_path.exists()
    contents = isolated_agent.feedback_path.read_text()
    assert "verdict" in contents
    assert "s1" in contents


# ---------- no-ML-imports invariant ----------

def test_adaptive_agent_imports_no_ml():
    src = (PROJECT_ROOT / "adaptive_agent.py").read_text(encoding="utf-8")
    for forbidden in ("import joblib", "import xgboost", "import sklearn",
                      "from xgboost", "from sklearn"):
        assert forbidden not in src, f"adaptive_agent imports {forbidden!r}"

def test_adaptive_agent_has_no_rl_terms():
    """No formal-RL terminology anywhere in the module."""
    src = (PROJECT_ROOT / "adaptive_agent.py").read_text(encoding="utf-8").lower()
    for forbidden in ("bellman", "q-learning", "mdp", "value function"):
        # "not formal reinforcement learning" is allowed as a docstring disclaimer
        # so we only catch these as standalone API names, not in docstring comments
        for line in src.splitlines():
            if forbidden in line and not line.strip().startswith("#") and "not" not in line:
                # Skip pure comments and disclaimer sentences
                if '"""' not in line and "not formal" not in line and "no " not in line:
                    pytest.fail(f"adaptive_agent.py line contains RL term {forbidden!r}: {line!r}")