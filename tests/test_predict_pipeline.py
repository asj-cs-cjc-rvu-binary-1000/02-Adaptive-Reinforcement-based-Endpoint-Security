"""
tests/test_predict_pipeline.py — PredictionPipeline contract tests.

Covers Task 4 areas:
  - Model loading via manifest
  - Prediction output structure
  - Probability range
  - Non-URL payload → Review (never Allow)
  - Undecodable payload → Undecodable
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


REQUIRED_OUTPUT_KEYS = {
    "payload", "payload_category", "features",
    "prob_phishing", "action", "reason", "policy",
}


def test_output_keys_complete(pipeline_default, trad_chehab_paths):
    result = pipeline_default.run(trad_chehab_paths["benign"])
    assert set(result.keys()) == REQUIRED_OUTPUT_KEYS


def test_url_payload_has_features_and_probability(pipeline_default, trad_chehab_paths):
    result = pipeline_default.run(trad_chehab_paths["benign"])
    assert result["payload_category"] == "url"
    assert isinstance(result["features"], dict)
    assert len(result["features"]) == 20
    assert 0.0 <= result["prob_phishing"] <= 1.0
    assert result["action"] in ("Allow", "Review", "Block")


def test_undecodable_image_returns_undecodable_action(pipeline_default, undecodable_path):
    result = pipeline_default.run(undecodable_path)
    assert result["action"] == "Undecodable"
    assert result["prob_phishing"] is None
    assert result["payload"] is None
    assert result["payload_category"] == "undecodable"


def test_non_url_payload_routes_to_review(pipeline_default, tmp_path):
    """Generate a QR containing plain text (not a URL) and assert the pipeline
    refuses to classify it as a URL."""
    import qr_utils
    from PIL import Image
    arr = qr_utils.generate_qr_image("Hello ARES, this is not a URL", scale=6)
    p = tmp_path / "plain_text.png"
    Image.fromarray(arr).save(p)
    result = pipeline_default.run(p)
    assert result["payload_category"] in ("plain_text", "non_url_scheme")
    assert result["action"] == "Review"
    assert result["prob_phishing"] is None
    assert result["features"] is None


def test_probability_in_unit_interval_on_sample_set(pipeline_default, trad_chehab_paths, mendeley_paths):
    """Multiple samples: probability must always be in [0,1] when present."""
    for p in (trad_chehab_paths["benign"], trad_chehab_paths["malicious"],
              mendeley_paths["benign"], mendeley_paths["malicious"]):
        r = pipeline_default.run(p)
        if r["prob_phishing"] is not None:
            assert 0.0 <= r["prob_phishing"] <= 1.0


def test_policy_default_is_0_35_and_0_70(pipeline_default):
    r = pipeline_default.run.__self__  # the pipeline itself
    assert pipeline_default.t_review == pytest.approx(0.35, abs=1e-9)
    assert pipeline_default.t_block == pytest.approx(0.70, abs=1e-9)


def test_policy_optimized_is_0_60_and_0_7911(pipeline_optimized):
    assert pipeline_optimized.t_review == pytest.approx(0.60, abs=1e-6)
    assert pipeline_optimized.t_block == pytest.approx(0.7911, abs=1e-3)


def test_determinism(pipeline_default, trad_chehab_paths):
    """Same input → identical probability across repeated calls."""
    a = pipeline_default.run(trad_chehab_paths["benign"])
    b = pipeline_default.run(trad_chehab_paths["benign"])
    assert a["prob_phishing"] == b["prob_phishing"]
    assert a["action"] == b["action"]


def test_no_network_in_pipeline_sources():
    """predict_pipeline.py and everything it imports must not fetch URLs."""
    for f in ("scripts/predict_pipeline.py", "scripts/qr_utils.py",
              "scripts/feature_extractor.py"):
        src = (PROJECT_ROOT / f).read_text(encoding="utf-8")
        for forbidden in ("import requests", "import urllib.request",
                          "import http.client", "import socket"):
            assert forbidden not in src, f"{f} imports {forbidden!r}"