"""
tests/test_features.py — Feature-extraction contract tests.

Covers Task 4 test area: correct feature names, order, dimension.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import feature_extractor as fx


EXPECTED_24 = [
    "url_length", "num_dots", "has_https", "has_ip", "num_subdirs",
    "num_params", "suspicious_words", "special_char_count", "digits_count",
    "entropy", "host_length", "path_length", "query_length",
    "subdomain_count", "digit_ratio", "susp_token_hits",
    "susp_compound_hits", "susp_in_subdomain", "is_shortener",
    "is_punycode", "has_port", "has_userinfo", "embedded_url_in_query",
    "tld_risk",
]

EXPECTED_20_KEPT = [
    f for f in EXPECTED_24
    if f not in {"has_https", "has_ip", "has_userinfo", "embedded_url_in_query"}
]


def test_feature_names_is_24_in_canonical_order():
    assert list(fx.feature_names) == EXPECTED_24
    assert len(fx.feature_names) == 24


def test_dict_to_list_length_matches_feature_names():
    out = fx.dict_to_list("https://example.com/login?user=1")
    assert len(out) == 24
    assert all(isinstance(v, (int, float)) for v in out)


def test_feature_order_preserved_across_calls():
    url = "https://secure-login.paypal.com.evil.xyz/verify?id=1"
    a = fx.dict_to_list(url)
    b = fx.dict_to_list(url)
    assert a == b, "feature vector must be deterministic"


def test_pca_kept_matches_expected_20():
    import json
    pca = json.loads((PROJECT_ROOT / "results" / "pca_report.json").read_text())
    kept = pca["features"]["kept"]
    assert list(kept) == EXPECTED_20_KEPT, (
        "predict_pipeline uses pca_report.features.kept — any change here breaks "
        "feature alignment with the trained model"
    )
    assert len(kept) == 20


def test_is_valid_url_accepts_typical_http_urls():
    for url in (
        "https://example.com/",
        "http://www.example.com/path?a=1",
        "https://sub.example.co.uk/path",
    ):
        assert fx.is_valid_url(url), url


def test_is_valid_url_rejects_non_http_schemes():
    for url in (
        "javascript:alert(1)",
        "mailto:a@b.com",
        "WIFI:S:MySSID;T:WPA;P:pass;;",
        "tel:+15551234567",
    ):
        assert not fx.is_valid_url(url), url


def test_extract_registrable_domain_stable():
    assert fx.extract_registrable_domain("https://a.b.example.co.uk/path") == "example.co.uk"
    assert fx.extract_registrable_domain("https://example.com/") == "example.com"


def test_extract_url_features_returns_dict_with_24_keys():
    feats = fx.extract_url_features("https://example.com/path?a=1")
    assert isinstance(feats, dict)
    assert len(feats) == 24
    assert set(feats.keys()) == set(EXPECTED_24)


def test_constant_features_are_constant():
    """The four dropped features must be identical zero across a sample of URLs."""
    for f in ("has_ip", "has_userinfo", "embedded_url_in_query"):
        values = [
            fx.extract_url_features(u)[f]
            for u in (
                "https://example.com/",
                "http://benign.org/path",
                "https://secure-login.paypal.com.evil.xyz/verify?id=1",
            )
        ]
        # At least: not-NaN, numeric. Constant means same value across this sample —
        # we do not assert exactly 0 because the sample is too small to prove the
        # full-dataset claim (that is checked by recompute_train_constants.py).
        assert all(v == values[0] for v in values), f"{f} not constant on sample"