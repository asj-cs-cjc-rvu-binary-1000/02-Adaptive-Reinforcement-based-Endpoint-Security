"""
tests/test_manifest.py — Model-manifest integrity tests.

Covers Task 4 area: valid manifest, corrupted/mismatched manifest.
Also confirms `PredictionPipeline.__init__` fails closed on drift.
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


def test_manifest_exists_and_parses(manifest_path):
    data = json.loads(manifest_path.read_text())
    assert "artifacts" in data
    assert "packages" in data
    assert "git_commit_hash" in data


def test_manifest_pins_expected_packages(manifest_dict):
    assert manifest_dict["packages"]["scikit-learn"].startswith("1.9.")
    assert manifest_dict["packages"]["xgboost"] == "3.0.2"


def test_manifest_includes_source_hashes(manifest_dict):
    names = set(manifest_dict["artifacts"].keys())
    required = {
        "model_selection", "pca_report", "calculus_experiment", "model_joblib",
        "scripts__predict_pipeline", "scripts__calculus_threshold",
        "scripts__run_calculus_experiment", "scripts__train_models",
        "scripts__prepare_features", "scripts__split_dataset",
        "scripts__pca_analysis", "scripts__evaluate_models",
        "root__qr_utils", "root__feature_extractor",
    }
    missing = required - names
    assert not missing, f"manifest missing artifacts: {missing}"


def test_check_manifest_ok():
    import model_manifest
    ok, mismatches = model_manifest.check_manifest()
    assert ok, f"manifest drift: {mismatches}"


def test_pipeline_fails_closed_on_wrong_xgboost(monkeypatch):
    """Simulate a package-version drift; PredictionPipeline must refuse to construct."""
    import model_manifest
    original = model_manifest._collect
    def fake_collect():
        c = original()
        c["packages"]["xgboost"] = "2.1.4"   # simulated drift
        return c
    monkeypatch.setattr(model_manifest, "_collect", fake_collect)
    with pytest.raises(SystemExit):
        import predict_pipeline as pp
        pp.PredictionPipeline(policy="default")


def test_pipeline_fails_closed_on_artifact_drift(monkeypatch):
    """Simulate a sha256 mismatch in a consumed artifact."""
    import model_manifest
    original = model_manifest._collect
    def fake_collect():
        c = original()
        c["artifacts"]["model_joblib"] = dict(c["artifacts"]["model_joblib"])
        c["artifacts"]["model_joblib"]["sha256"] = "0000" * 16
        return c
    monkeypatch.setattr(model_manifest, "_collect", fake_collect)
    with pytest.raises(SystemExit):
        import predict_pipeline as pp
        pp.PredictionPipeline(policy="default")