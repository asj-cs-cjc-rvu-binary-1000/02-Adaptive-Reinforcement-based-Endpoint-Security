"""
tests/test_safety.py — Safety-invariant tests.

Covers Task 4 area: non-URL never Allow, undecodable safe, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_non_url_never_allow(pipeline_default, tmp_path):
    """A QR containing plain text must never resolve to Allow."""
    import qr_utils
    from PIL import Image
    payloads = [
        "Hello ARES",
        "WIFI:S:TestNet;T:WPA;P:example;;",
        "mailto:admin@example.com",
        "tel:+15551234567",
    ]
    for text in payloads:
        arr = qr_utils.generate_qr_image(text, scale=6)
        p = tmp_path / f"{abs(hash(text)) % 10**8}.png"
        Image.fromarray(arr).save(p)
        result = pipeline_default.run(p)
        assert result["action"] != "Allow", (
            f"payload {text!r} → action={result['action']!r} (must not be Allow)"
        )


def test_undecodable_never_crashes(pipeline_default, undecodable_path):
    result = pipeline_default.run(undecodable_path)
    assert result["action"] == "Undecodable"
    assert result["prob_phishing"] is None


def test_no_network_in_inference_chain():
    """Structural check: none of the inference modules import a network library.

    `urllib.parse` is pure string parsing and is explicitly allowed; only
    network-capable urllib submodules (`urllib.request`) and other network
    libraries are forbidden.
    """
    import re
    forbidden_patterns = [
        r"\bimport\s+requests\b",
        r"\bfrom\s+requests\b",
        r"\bimport\s+httpx\b",
        r"\bfrom\s+httpx\b",
        r"\bimport\s+aiohttp\b",
        r"\bfrom\s+aiohttp\b",
        r"\bimport\s+urllib\.request\b",
        r"\bfrom\s+urllib\.request\b",
        r"\bimport\s+urllib3\b",
        r"\bfrom\s+urllib3\b",
        r"\bimport\s+socket\b",
        r"\bfrom\s+socket\b",
        r"\bimport\s+http\.client\b",
        r"\bfrom\s+http\.client\b",
        r"\burlopen\b",
        r"\burlretrieve\b",
    ]
    for f in (
        "scripts/predict_pipeline.py",
        "scripts/qr_utils.py",
        "scripts/feature_extractor.py",
        "scripts/model_manifest.py",
    ):
        src = (PROJECT_ROOT / f).read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert not re.search(pattern, src), (
                f"{f} matches forbidden network pattern {pattern!r}"
            )


def test_corrupted_image_safe(pipeline_default, tmp_path):
    """A 1×1 pixel array must not crash the pipeline."""
    from PIL import Image
    p = tmp_path / "tiny.png"
    Image.new("L", (1, 1), 255).save(p)
    result = pipeline_default.run(p)
    assert result["action"] in ("Undecodable", "Review", "Allow", "Block")


def test_invalid_file_extension_safe(pipeline_default, tmp_path):
    """A text file pretending to be an image must not crash the pipeline."""
    p = tmp_path / "not_an_image.png"
    p.write_text("this is not a PNG")
    result = pipeline_default.run(p)
    assert result["action"] in ("Undecodable", "Review")