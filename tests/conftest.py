"""
tests/conftest.py — shared fixtures for the ARES Phase 3 regression suite.

Every fixture is deterministic and depends only on artifacts already committed:
  - models/raw_xgb.joblib   (frozen; sha256 648438e2…)
  - results/model_manifest.json
  - data/raw/qr_codes_29.pickle + qr_codes_29_labels.pickle
  - data/raw/qr-images/{benign,malicious}/*.png

No network. No re-train. No model mutation.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RAW = PROJECT_ROOT / "data" / "raw"
RESULTS = PROJECT_ROOT / "results"


# ---------- QR image fixtures ----------
def _trad_chehab_png_bytes(idx: int, *, pad: int = 20, scale: int = 4) -> bytes:
    """Render a Trad & Chehab matrix as a PNG byte string (same as Task 3 preprocessing)."""
    with open(RAW / "qr_codes_29.pickle", "rb") as f:
        imgs = pickle.load(f)
    arr = (imgs[idx].astype(np.uint8) * 255)
    pil = Image.fromarray(arr, mode="L")
    n = pil.size[0] + 2 * pad
    canvas = Image.new("L", (n, n), 255)
    canvas.paste(pil, (pad, pad))
    canvas = canvas.resize((n * scale, n * scale), Image.NEAREST)
    import io
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _first_index_of_class(label: int) -> int:
    with open(RAW / "qr_codes_29_labels.pickle", "rb") as f:
        labs = pickle.load(f)
    return int(np.where(labs == label)[0][0])


@pytest.fixture(scope="session")
def trad_chehab_paths():
    """First benign and first malicious Trad & Chehab samples (paths on disk)."""
    if not (RAW / "qr_codes_29.pickle").exists():
        pytest.skip("Trad & Chehab pickle missing")
    tmp = PROJECT_ROOT / "tests" / "_tmp_images"
    tmp.mkdir(exist_ok=True)
    benign_idx = _first_index_of_class(0)
    malicious_idx = _first_index_of_class(1)
    benign_path = tmp / f"tc_benign_{benign_idx}.png"
    malicious_path = tmp / f"tc_malicious_{malicious_idx}.png"
    benign_path.write_bytes(_trad_chehab_png_bytes(benign_idx))
    malicious_path.write_bytes(_trad_chehab_png_bytes(malicious_idx))
    return {"benign": benign_path, "malicious": malicious_path}


@pytest.fixture(scope="session")
def mendeley_paths():
    """First benign and first malicious Mendeley PNGs."""
    benign_dir = RAW / "qr-images" / "benign"
    malicious_dir = RAW / "qr-images" / "malicious"
    if not (benign_dir.exists() and malicious_dir.exists()):
        pytest.skip("Mendeley folder missing")
    return {
        "benign": sorted(benign_dir.glob("*.png"))[0],
        "malicious": sorted(malicious_dir.glob("*.png"))[0],
    }


@pytest.fixture
def undecodable_path(tmp_path):
    """Blank 200×200 image with no QR content."""
    p = tmp_path / "blank.png"
    Image.new("L", (200, 200), 255).save(p)
    return p


# ---------- pipeline fixtures ----------
def _make_pipeline(policy: str):
    """Build a PredictionPipeline, converting fail-closed SystemExit into a
    single, explicit pytest.fail so a manifest drift does not cascade into
    a dozen confusing finalizer AssertionErrors from session-scoped teardown.
    """
    from predict_pipeline import PredictionPipeline
    try:
        return PredictionPipeline(policy=policy)
    except SystemExit as e:
        pytest.fail(
            f"PredictionPipeline(policy={policy!r}) failed closed:\n{e}\n"
            "If predict_pipeline.py was edited deliberately, run "
            "`python scripts/model_manifest.py` (no flag) and re-run pytest. "
            "If not, restore the drifted artifact with git.",
            pytrace=False,
        )


@pytest.fixture(scope="session")
def pipeline_default():
    return _make_pipeline("default")


@pytest.fixture(scope="session")
def pipeline_optimized():
    return _make_pipeline("optimized")


# ---------- adaptive agent fixture (isolated state) ----------
@pytest.fixture
def isolated_agent(tmp_path, monkeypatch):
    """HeuristicAdaptiveAgent with isolated state file paths (not the real results/)."""
    from adaptive_agent import HeuristicAdaptiveAgent
    state = tmp_path / "heuristic_policy_state.json"
    feedback = tmp_path / "feedback.csv"
    return HeuristicAdaptiveAgent(state_path=state, feedback_path=feedback)


# ---------- manifest fixture ----------
@pytest.fixture(scope="session")
def manifest_path():
    p = RESULTS / "model_manifest.json"
    if not p.exists():
        pytest.skip("results/model_manifest.json missing")
    return p


@pytest.fixture(scope="session")
def manifest_dict(manifest_path):
    return json.loads(manifest_path.read_text(encoding="utf-8"))