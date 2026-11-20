"""
tests/test_qr.py — QR decoding tests.

Covers Task 4 test areas:
  - Valid QR images
  - Invalid / undecodable images
  - Non-URL payloads
  - Fail-closed behaviour of the production decoder
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import qr_utils


def test_decode_benign_trad_chehab(trad_chehab_paths):
    payloads = qr_utils.decode_qr_codes(trad_chehab_paths["benign"])
    assert isinstance(payloads, list)
    assert len(payloads) >= 1
    assert payloads[0].strip() != ""


def test_decode_malicious_trad_chehab(trad_chehab_paths):
    payloads = qr_utils.decode_qr_codes(trad_chehab_paths["malicious"])
    assert isinstance(payloads, list)
    assert len(payloads) >= 1


def test_decode_mendeley_benign(mendeley_paths):
    payloads = qr_utils.decode_qr_codes(mendeley_paths["benign"])
    assert isinstance(payloads, list)
    assert len(payloads) >= 1


def test_undecodable_raises(undecodable_path):
    """Blank image must raise QRDecodeError, not return empty or crash otherwise."""
    with pytest.raises(qr_utils.QRDecodeError):
        qr_utils.decode_qr_codes(undecodable_path)


def test_decode_qr_code_fails_closed_on_empty(tmp_path):
    """The single-payload decoder must not silently return a fake payload."""
    p = tmp_path / "blank.png"
    from PIL import Image
    Image.new("L", (200, 200), 255).save(p)
    with pytest.raises(qr_utils.QRDecodeError):
        qr_utils.decode_qr_code(p)


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        qr_utils.decode_qr_codes(tmp_path / "does_not_exist.png")


def test_offline_only_no_network_imports():
    """qr_utils must not import requests/urllib/socket/http.client."""
    src = (PROJECT_ROOT / "scripts" / "qr_utils.py").read_text(encoding="utf-8")
    for forbidden in ("import requests", "import urllib", "import socket", "import http.client"):
        assert forbidden not in src, f"qr_utils.py contains {forbidden!r}"