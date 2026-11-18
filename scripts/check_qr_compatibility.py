#!/usr/bin/env python3
"""
scripts/check_qr_compatibility.py — Phase 1 Task 3.

For each candidate dataset, decode QR images with the PRODUCTION decoder
(recovered byte-identical from `origin/main`, stored under scripts/), categorize
the decoded payloads, and check feature-extractor compatibility + duplicate /
domain-overlap risks.

Layout (Option Y, chosen by user):
    scripts/qr_utils.py
    scripts/feature_extractor.py
    scripts/check_qr_compatibility.py       <- this file

Datasets:
  A. Trad & Chehab  — NumPy pickle (9987, 69, 69) int16 {0,1}
                      reconstructed to uint8, padded, upscaled before decode
  B. Mendeley       — 500+500 PNGs, PIL mode '1' (1-bit), mixed sizes

Read-only on data/. Writes results/qr_decode_report.json only. Safe to re-run.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Option Y: production modules under scripts/. Also allow repo-root for
# forward-compat with Phase 2 (train_model.py, app.py) if that layout is used.
for _p in (SCRIPTS_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---- Production modules (must be byte-identical to origin/main) -----------
import qr_utils                       # noqa: E402
import feature_extractor as fx        # noqa: E402

# Hard-fail guard — makes the "git show nonexistent:file > file" zero-byte
# trap impossible to disguise as a successful report again.
_REQUIRED_QR = ("decode_qr_code", "decode_qr_codes", "QRDecodeError")
_REQUIRED_FX = ("feature_names", "is_valid_url", "extract_url_features",
                "extract_registrable_domain")
_missing = [n for n in _REQUIRED_QR if not hasattr(qr_utils, n)]
_missing += [n for n in _REQUIRED_FX if not hasattr(fx, n)]
if _missing:
    raise SystemExit(
        "FATAL: production modules are missing required symbols: " + ", ".join(_missing)
        + "\nThis means the module was imported from an empty or wrong file. "
          "Verify with:\n"
          "  wc -c scripts/qr_utils.py scripts/feature_extractor.py\n"
          "  sha256sum scripts/qr_utils.py scripts/feature_extractor.py  "
          "(must NOT be e3b0c442...)\n"
          "Do NOT trust a decode rate from a report produced under this condition."
    )

RAW = PROJECT_ROOT / "data" / "raw"
RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "qr_decode_report.json"

TRAD_IMAGES = RAW / "qr_codes_29.pickle"
TRAD_LABELS = RAW / "qr_codes_29_labels.pickle"
MENDELEY_DIR = RAW / "qr-images"

# Set to an integer to subsample deterministically (for time-bounded runs).
# None = decode every image. Reproducible via RNG_SEED regardless.
# Bounded defaults for a first successful run. Increase after correctness is
# confirmed. Deterministic via RNG_SEED regardless of sample size.
MAX_TRAD_PER_CLASS: int | None = 200       # ~30-90 s total
MAX_MENDELEY_PER_CLASS: int | None = 200   # ~30-90 s total (was: all 500)

RNG_SEED = 42


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_production_file(name: str) -> Path:
    """Return scripts/<name> if present and non-empty, else <repo root>/<name>."""
    for candidate in (SCRIPTS_DIR / name, PROJECT_ROOT / name):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        f"Production file {name!r} not found as scripts/{name} or {name}, "
        f"or present but zero-byte. Recover with:\n"
        f"  git show origin/main:{name} > scripts/{name}\n"
        f"  wc -c scripts/{name}   # MUST be non-zero"
    )


def categorize_payload(payload: str | None) -> str:
    """'url' | 'non_url_scheme' | 'plain_text' | 'empty' | 'undecodable'."""
    if payload is None:
        return "undecodable"
    s = payload.strip()
    if not s:
        return "empty"
    low = s.lower()
    if any(low.startswith(p + ":") for p in
           ("javascript", "data", "mailto", "tel", "sms", "wifi", "ftp",
            "file", "geo", "bitcoin", "smsto", "skype", "otpauth", "vcard")):
        return "non_url_scheme"
    if fx.is_valid_url(s):
        return "url"
    return "plain_text"


def _matrix_to_bgr(matrix: np.ndarray) -> np.ndarray:
    """(69,69) {0,1} int16 -> (H,W,3) uint8 BGR with quiet border + 4x upscale."""
    arr = (matrix.astype(np.uint8) * 255)
    pil = Image.fromarray(arr, mode="L")
    border = 20
    new_side = pil.size[0] + 2 * border
    canvas = Image.new("L", (new_side, new_side), 255)
    canvas.paste(pil, (border, border))
    canvas = canvas.resize((new_side * 4, new_side * 4), Image.NEAREST)
    return np.ascontiguousarray(
        np.array(canvas.convert("L"))[:, :, None].repeat(3, axis=2)
    )


import tempfile
from pathlib import Path as _Path

_TMP_DIR = RESULTS / ".tmp_decode"
_TMP_DIR.mkdir(parents=True, exist_ok=True)

import time as _time
_t0_global = _time.time()


def _log(msg: str) -> None:
    elapsed = _time.time() - _t0_global
    print(f"[{elapsed:6.1f}s] {msg}", flush=True)

def _decode(array_2d_or_3d: np.ndarray) -> list[str]:
    """Call the production path-based decoder on an in-memory image.

    qr_utils.decode_qr_codes / decode_qr_code take `str | Path`, not arrays.
    We persist a temp PNG and call the path API so the decoder we exercise is
    byte-identical to origin/main's production path. The temp file is unlinked
    immediately after the call succeeds or fails.
    """
    import cv2  # local import — only needed here

    # Ensure single-channel (grayscale) uint8 PNG; qr_utils.load_image_safely
    # handles BGR/RGBA itself but grayscale is the smallest faithful artefact.
    if array_2d_or_3d.ndim == 3:
        gray = cv2.cvtColor(array_2d_or_3d, cv2.COLOR_BGR2GRAY)
    else:
        gray = array_2d_or_3d
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    tmp_path: _Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=_TMP_DIR, suffix=".png", delete=False
        ) as tf:
            tmp_path = _Path(tf.name)
        cv2.imwrite(str(tmp_path), gray)
        if hasattr(qr_utils, "decode_qr_codes"):
            return list(qr_utils.decode_qr_codes(tmp_path))
        p = qr_utils.decode_qr_code(tmp_path)
        return [p] if p is not None else []
    except qr_utils.QRDecodeError:
        return []
    except Exception:                       # noqa: BLE001 — any decoder failure is "undecodable"
        return []
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def decode_trad_chehab() -> dict:
    with open(TRAD_IMAGES, "rb") as f:
        images = pickle.load(f)
    with open(TRAD_LABELS, "rb") as f:
        labels = pickle.load(f)
    if images.shape[0] != labels.shape[0]:
        return {"error": f"length mismatch {images.shape[0]} vs {labels.shape[0]}"}

    per_class = {0: Counter(), 1: Counter()}
    per_class_domain = {0: Counter(), 1: Counter()}
    payload_examples = {0: [], 1: []}
    payload_hash = {0: Counter(), 1: Counter()}
    n_processed = 0
    n_feat_ok = 0
    n_feat_fail = 0

    rng = np.random.default_rng(RNG_SEED)
    for lbl in (0, 1):
        class_idx = np.where(labels == lbl)[0]
        rng.shuffle(class_idx)
        if MAX_TRAD_PER_CLASS is not None:
            class_idx = class_idx[:MAX_TRAD_PER_CLASS]
        for i in class_idx:
            n_processed += 1
            if n_processed % 25 == 0:
                _log(f"TradChehab label={lbl} processed={n_processed} "
                     f"decoded_so_far="
                     f"{sum(sum(c.values()) for c in per_class.values()) - sum(c.get('undecodable', 0) for c in per_class.values())}")
            bgr = _matrix_to_bgr(images[int(i)])
            payloads = _decode(bgr)
            if not payloads:
                per_class[lbl]["undecodable"] += 1
                continue
            payload = payloads[0]
            category = categorize_payload(payload)
            per_class[lbl][category] += 1
            payload_hash[lbl][hashlib.sha256(payload.encode()).hexdigest()[:16]] += 1
            if len(payload_examples[lbl]) < 5 and category == "url":
                payload_examples[lbl].append(payload)
            if category == "url":
                try:
                    feat = fx.extract_url_features(payload)
                    if len(feat) == len(fx.feature_names):
                        n_feat_ok += 1
                    else:
                        n_feat_fail += 1
                    per_class_domain[lbl][fx.extract_registrable_domain(payload)] += 1
                except Exception:           # noqa: BLE001
                    n_feat_fail += 1

    benign_domains = set(per_class_domain[0])
    malicious_domains = set(per_class_domain[1])
    overlap = benign_domains & malicious_domains

    def _rate(c: Counter) -> float:
        total = sum(c.values())
        return round((total - c.get("undecodable", 0)) / total, 4) if total else 0.0

    return {
        "dataset": "Trad & Chehab",
        "n_total": int(len(labels)),
        "n_processed": n_processed,
        "sample_cap_per_class": MAX_TRAD_PER_CLASS,
        "per_class_category_counts": {
            "0_benign": dict(per_class[0]),
            "1_malicious": dict(per_class[1]),
        },
        "decode_rate_benign": _rate(per_class[0]),
        "decode_rate_malicious": _rate(per_class[1]),
        "feature_extraction": {
            "attempted_urls": sum(per_class[0].values()) + sum(per_class[1].values())
                              - per_class[0].get("undecodable", 0)
                              - per_class[1].get("undecodable", 0),
            "succeeded": n_feat_ok,
            "failed": n_feat_fail,
        },
        "duplicate_check": {
            str(lbl): {
                "unique_payloads": len(c),
                "total_decoded": sum(c.values()),
                "duplicates": sum(v for v in c.values() if v > 1),
            } for lbl, c in payload_hash.items()
        },
        "domain_overlap": {
            "n_benign_domains": len(benign_domains),
            "n_malicious_domains": len(malicious_domains),
            "n_overlap": len(overlap),
            "overlap_examples": sorted(overlap)[:20],
        },
        "payload_examples": {
            "0_benign": payload_examples[0],
            "1_malicious": payload_examples[1],
        },
    }


def decode_mendeley() -> dict:
    if not MENDELEY_DIR.exists():
        return {"error": "Mendeley folder missing"}

    per_class = {"benign": Counter(), "malicious": Counter()}
    per_class_domain = {"benign": Counter(), "malicious": Counter()}
    payload_examples = {"benign": [], "malicious": []}
    payload_hash = {"benign": Counter(), "malicious": Counter()}
    n_feat_ok = 0
    n_feat_fail = 0
    n_processed = {"benign": 0, "malicious": 0}

    for cls in ("benign", "malicious"):
        files = sorted((MENDELEY_DIR / cls).glob("*.png"))
        if MAX_MENDELEY_PER_CLASS is not None:
            files = files[:MAX_MENDELEY_PER_CLASS]
        for f in files:
            n_processed[cls] += 1
            if n_processed[cls] % 25 == 0:
                _log(f"Mendeley {cls} processed={n_processed[cls]}")
            try:
                pil = Image.open(f).convert("L")
                gray = np.ascontiguousarray(np.array(pil))
            except Exception:               # noqa: BLE001
                per_class[cls]["undecodable"] += 1
                continue
            payloads = _decode(gray)
            if not payloads:
                per_class[cls]["undecodable"] += 1
                continue
            payload = payloads[0]
            category = categorize_payload(payload)
            per_class[cls][category] += 1
            payload_hash[cls][hashlib.sha256(payload.encode()).hexdigest()[:16]] += 1
            if len(payload_examples[cls]) < 5 and category == "url":
                payload_examples[cls].append(payload)
            if category == "url":
                try:
                    feat = fx.extract_url_features(payload)
                    if len(feat) == len(fx.feature_names):
                        n_feat_ok += 1
                    else:
                        n_feat_fail += 1
                    per_class_domain[cls][fx.extract_registrable_domain(payload)] += 1
                except Exception:           # noqa: BLE001
                    n_feat_fail += 1

    benign_domains = set(per_class_domain["benign"])
    malicious_domains = set(per_class_domain["malicious"])
    overlap = benign_domains & malicious_domains

    def _rate(c: Counter) -> float:
        total = sum(c.values())
        return round((total - c.get("undecodable", 0)) / total, 4) if total else 0.0

    return {
        "dataset": "Mendeley",
        "n_processed": n_processed,
        "sample_cap_per_class": MAX_MENDELEY_PER_CLASS,
        "per_class_category_counts": {k: dict(v) for k, v in per_class.items()},
        "decode_rate_benign": _rate(per_class["benign"]),
        "decode_rate_malicious": _rate(per_class["malicious"]),
        "feature_extraction": {
            "attempted_urls": sum(per_class["benign"].values())
                              + sum(per_class["malicious"].values())
                              - per_class["benign"].get("undecodable", 0)
                              - per_class["malicious"].get("undecodable", 0),
            "succeeded": n_feat_ok,
            "failed": n_feat_fail,
        },
        "duplicate_check": {
            cls: {
                "unique_payloads": len(c),
                "total_decoded": sum(c.values()),
                "duplicates": sum(v for v in c.values() if v > 1),
            } for cls, c in payload_hash.items()
        },
        "domain_overlap": {
            "n_benign_domains": len(benign_domains),
            "n_malicious_domains": len(malicious_domains),
            "n_overlap": len(overlap),
            "overlap_examples": sorted(overlap)[:20],
        },
        "payload_examples": payload_examples,
    }


def main() -> int:
    from datetime import datetime, timezone

    qr_path = locate_production_file("qr_utils.py")
    fx_path = locate_production_file("feature_extractor.py")

    report: dict = {
        "phase": "1",
        "task": "3 — QR decode + ARES compatibility",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decoder_provenance": {
            "qr_utils_path": str(qr_path.relative_to(PROJECT_ROOT)),
            "qr_utils_sha256": file_sha256(qr_path),
            "feature_extractor_path": str(fx_path.relative_to(PROJECT_ROOT)),
            "feature_extractor_sha256": file_sha256(fx_path),
            "source": "byte-identical copies of `origin/main`",
        },
        "sample_cap": {
            "trad_per_class": MAX_TRAD_PER_CLASS,
            "mendeley_per_class": MAX_MENDELEY_PER_CLASS,
        },
    }
    try:
        report["trad_chehab"] = decode_trad_chehab()
    except Exception as e:                  # noqa: BLE001
        report["trad_chehab_error"] = f"{type(e).__name__}: {e}"
    try:
        report["mendeley"] = decode_mendeley()
    except Exception as e:                  # noqa: BLE001
        report["mendeley_error"] = f"{type(e).__name__}: {e}"

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps(report, indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())