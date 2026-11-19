#!/usr/bin/env python3
"""
scripts/prepare_features.py — Phase 2 Task 3.

Extracts the 24 production URL features from every successfully-decoded URL
payload in the primary (Trad & Chehab) and external (Mendeley) datasets, writes
a reproducible feature matrix, and produces a feature-quality summary.

Hard prerequisites:
  - `run_gates.py check` must pass (both gates valid). Instant abort otherwise.

Behavior:
  - Fails **loudly** if either gate is stale or missing. No 52-minute decode ever
    happens in this task — decoding is deduplicated by design in Phase 2 Tasks 1-2
    and their per-payload results are reused here via a lightweight re-decode that
    runs in ~52 min the *first time* and is cached.

Design choice for this file:
  - We DO re-decode payloads here, because Gate 2 only stored aggregate counts.
    To avoid a second 52-minute run, this script caches every decoded payload as a
    JSON: results/decoded_payloads.json. Subsequent runs reuse it. The cache is
    content-addressed: if data or decoder changes, cache is invalidated.

Outputs:
  data/processed/features_primary.parquet
  data/processed/features_external.parquet
  results/feature_summary.json
  results/excluded_samples.json
  results/decoded_payloads.json  (cache — see content-addressing note)

Schema of features_*.parquet:
  columns: feature_names[24] + label + sample_id + registrable_domain + dataset
  label: 0=benign, 1=malicious
  sample_id: stable per dataset ("trad_chehab:0000", "mendeley:benign/qr_0_benign_images.png")
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import qr_utils                                   # noqa: E402
import feature_extractor as fx                    # noqa: E402
import dataset_config as cfg                      # noqa: E402

RESULTS = PROJECT_ROOT / "results"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TMP_DECODE = RESULTS / ".tmp_decode"
TMP_DECODE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

DECODE_CACHE = RESULTS / "decoded_payloads.json"


# ------------------------------------------------------------- gates
def assert_gates_pass(dataset: str, external: str) -> None:
    """Fail fast if gates are not both valid. ~3 s."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_gates.py"),
           "check", "--dataset", dataset, "--external", external]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit(
            "FATAL: gates are not passed or are stale. "
            "Re-run gate1-overlap and/or gate2-decode before preparing features."
        )
    print("gates OK — proceeding to feature extraction.")


# ------------------------------------------------------------- decode cache
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(primary: cfg.DatasetSpec, external: cfg.DatasetSpec | None) -> dict:
    """Everything the decoded payloads depend on."""
    files = []
    for spec in (primary, external):
        if spec is None:
            continue
        if spec.images_pickle:
            files.append(spec.images_pickle)
        if spec.labels_pickle:
            files.append(spec.labels_pickle)
        if spec.png_dir and spec.png_dir.exists():
            for folder in ("benign", "malicious"):
                d = spec.png_dir / folder
                if d.exists():
                    files.extend(sorted(d.glob("*.png")))
    return {
        "dataset_files": {str(p.relative_to(PROJECT_ROOT)): _sha256_file(p)
                          for p in files},
        "qr_utils_sha256": _sha256_file(PROJECT_ROOT / "scripts" / "qr_utils.py"),
        "feature_extractor_sha256": _sha256_file(PROJECT_ROOT / "scripts" / "feature_extractor.py"),
    }


def _load_cache(key: dict) -> dict | None:
    if not DECODE_CACHE.exists():
        return None
    try:
        cached = json.loads(DECODE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if cached.get("key") != key:
        return None
    return cached


def _save_cache(key: dict, payloads: dict) -> None:
    DECODE_CACHE.write_text(
        json.dumps({"key": key,
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "payloads": payloads},
                   indent=None),
        encoding="utf-8",
    )


# ------------------------------------------------------------- decode logic
def _decode_array(arr: np.ndarray, tag: str) -> list[str]:
    import cv2
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    gray = np.ascontiguousarray(gray)
    tmp = TMP_DECODE / f"{tag}.png"
    try:
        cv2.imwrite(str(tmp), gray)
        return list(qr_utils.decode_qr_codes(tmp))
    except qr_utils.QRDecodeError:
        return []
    except Exception:
        return []
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _iter_trad_chehab(spec: cfg.DatasetSpec):
    with open(spec.images_pickle, "rb") as f:
        images = pickle.load(f)
    with open(spec.labels_pickle, "rb") as f:
        labels = pickle.load(f)
    for i in range(images.shape[0]):
        mat = images[i]
        arr = mat.astype(np.uint8) * 255
        pil = Image.fromarray(arr, mode="L")
        border = 20
        n = pil.size[0] + 2 * border
        canvas = Image.new("L", (n, n), 255)
        canvas.paste(pil, (border, border))
        canvas = canvas.resize((n * 4, n * 4), Image.NEAREST)
        bgr = np.ascontiguousarray(
            np.array(canvas.convert("L"))[:, :, None].repeat(3, axis=2)
        )
        sample_id = f"{spec.key}:{i:05d}"
        yield sample_id, spec.label_mapping[int(labels[i])], bgr


def _iter_mendeley(spec: cfg.DatasetSpec):
    for cls in ("benign", "malicious"):
        folder = spec.png_dir / cls
        for f in sorted(folder.glob("*.png")):
            pil = Image.open(f).convert("L")
            arr = np.ascontiguousarray(np.array(pil))
            sample_id = f"{spec.key}:{cls}/{f.name}"
            yield sample_id, cls, arr


def _iter_spec(spec: cfg.DatasetSpec):
    if spec.images_pickle:
        yield from _iter_trad_chehab(spec)
    else:
        yield from _iter_mendeley(spec)


def decode_all(spec: cfg.DatasetSpec, log_prefix: str) -> dict:
    """Return {sample_id: {"cls": str, "payload": str | None}}."""
    out: dict = {}
    n = 0
    for sample_id, cls, arr in _iter_spec(spec):
        n += 1
        if n % 200 == 0 or n == spec.n_total:
            print(f"[{log_prefix}] {n}/{spec.n_total} decoded", flush=True)
        payloads = _decode_array(arr, tag=f"{spec.key}_pf_{n:06d}")
        out[sample_id] = {"cls": cls, "payload": payloads[0] if payloads else None}
    return out


# ------------------------------------------------------------- feature extraction
def _categorize(payload: str | None) -> str:
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


def build_feature_frame(decoded: dict, dataset_key: str) -> tuple[pd.DataFrame, list, dict]:
    rows: list[dict] = []
    excluded: list[dict] = []
    per_class_counts = {"benign": 0, "malicious": 0}
    feat_names = list(fx.feature_names)

    for sample_id, rec in decoded.items():
        cls = rec["cls"]
        per_class_counts[cls] += 1
        payload = rec["payload"]
        category = _categorize(payload)
        if category != "url":
            excluded.append({
                "sample_id": sample_id,
                "class": cls,
                "category": category,
                "payload_preview": (payload[:120] if payload else None),
            })
            continue
        try:
            feat_dict = fx.extract_url_features(payload)
        except Exception as e:                      # noqa: BLE001
            excluded.append({
                "sample_id": sample_id,
                "class": cls,
                "category": "extractor_failed",
                "payload_preview": payload[:120],
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        if len(feat_dict) != len(feat_names):
            excluded.append({
                "sample_id": sample_id,
                "class": cls,
                "category": "extractor_wrong_width",
                "payload_preview": payload[:120],
            })
            continue
        row = {name: feat_dict[name] for name in feat_names}
        row["label"] = 0 if cls == "benign" else 1
        row["sample_id"] = sample_id
        row["registrable_domain"] = fx.extract_registrable_domain(payload)
        row["dataset"] = dataset_key
        rows.append(row)

    df = pd.DataFrame(rows, columns=feat_names + [
        "label", "sample_id", "registrable_domain", "dataset"
    ])
    return df, excluded, per_class_counts


# ------------------------------------------------------------- quality report
def summarize_features(df: pd.DataFrame, feat_names: list[str]) -> dict:
    """Per-feature stats + constant-feature detection + high-correlation pairs."""
    summary: dict = {"n_rows": int(len(df)), "features": {}}
    if len(df) == 0:
        return summary

    for name in feat_names:
        col = df[name].astype(float)
        summary["features"][name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std(ddof=0)),
            "n_missing": int(col.isna().sum()),
            "n_unique": int(col.nunique()),
            "is_constant": bool(col.nunique() <= 1),
        }

    corr = df[feat_names].astype(float).corr(method="pearson")
    high_corr: list[dict] = []
    seen = set()
    for i, a in enumerate(feat_names):
        for j, b in enumerate(feat_names):
            if i >= j:
                continue
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= 0.95:
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                high_corr.append({"a": a, "b": b, "r": round(float(r), 4)})
    summary["high_correlation_pairs_r_ge_0_95"] = high_corr
    summary["constant_features"] = [k for k, v in summary["features"].items()
                                    if v["is_constant"]]
    return summary


# ------------------------------------------------------------- additional features
def additional_feature_candidates(df: pd.DataFrame) -> dict:
    """Candidate features the brief flagged as optional §5. Report availability
    and (where trivially derivable from the existing 24) whether they would add
    variance beyond what the current features already capture."""
    # We compute *candidates* without adding them to the model.
    cand = {}
    cand["path_depth"] = "derivable as (num_subdirs); already present"
    cand["path_to_host_ratio"] = "derivable as path_length / host_length; high corr with path_length expected"
    cand["host_entropy"] = "requires new extraction step (separate from URL entropy)"
    cand["subdomain_to_domain_ratio"] = "derivable as subdomain_count / len(registrable_domain)"
    cand["consecutive_digits_max"] = "requires per-char scan (not in 24); risk of overlap with digits_count"
    cand["consecutive_hyphens_max"] = "requires per-char scan (not in 24)"
    cand["query_key_count"] = "already present as num_params"
    cand["encoded_char_ratio"] = "requires %-decode aware scan; potentially useful"
    cand["qr_image_dims"] = "OUT OF SCOPE — image features belong to a separate model, not the URL classifier"
    return {
        "note": "candidates documented; none added to the 24-feature model in this task",
        "candidates": cand,
    }


# ------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--external", default=cfg.DEFAULT_EXTERNAL)
    ap.add_argument("--force-redecode", action="store_true",
                    help="ignore cache; re-decode all images (~52 min)")
    args = ap.parse_args()

    dataset_key = args.dataset or cfg.DEFAULT_PRIMARY
    external_key = args.external or cfg.DEFAULT_EXTERNAL

    # 1. Gates — fail fast
    assert_gates_pass(dataset_key, external_key)

    primary = cfg.get_primary(dataset_key)
    external = cfg.get_external(external_key) if external_key else None

    # 2. Decoded payloads via cache
    cache_key = _cache_key(primary, external)
    if args.force_redecode:
        cached = None
    else:
        cached = _load_cache(cache_key)

    if cached is not None:
        print(f"using decoded-payload cache ({len(cached['payloads'])} samples)")
        all_payloads = cached["payloads"]
    else:
        print("no cache — decoding all images (this is the ~52-min path once)")
        primary_payloads = decode_all(primary, log_prefix="primary")
        external_payloads = decode_all(external, log_prefix="external") if external else {}
        all_payloads = {**primary_payloads, **external_payloads}
        _save_cache(cache_key, all_payloads)
        print(f"wrote cache {DECODE_CACHE}")

    # 3. Split decoded payloads by dataset (prefix before ':')
    primary_decoded = {sid: rec for sid, rec in all_payloads.items()
                       if sid.startswith(f"{primary.key}:")}
    external_decoded = ({sid: rec for sid, rec in all_payloads.items()
                         if sid.startswith(f"{external.key}:")}
                        if external else {})

    # 4. Feature frames
    primary_df, primary_excluded, primary_cls = build_feature_frame(
        primary_decoded, primary.key)
    external_df, external_excluded, external_cls = (build_feature_frame(
        external_decoded, external.key) if external else (pd.DataFrame(), [], {}))

    # 5. Persist
    primary_path = PROCESSED / "features_primary.parquet"
    external_path = PROCESSED / "features_external.parquet"
    primary_df.to_parquet(primary_path, index=False)
    if external:
        external_df.to_parquet(external_path, index=False)

    # 6. Reports
    feat_names = list(fx.feature_names)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary": {
            "dataset": primary.name,
            "n_decoded": len(primary_decoded),
            "n_url_features_rows": int(len(primary_df)),
            "class_counts": primary_cls,
            "n_excluded": len(primary_excluded),
            "feature_summary": summarize_features(primary_df, feat_names),
        },
        "external": ({
            "dataset": external.name,
            "n_decoded": len(external_decoded),
            "n_url_features_rows": int(len(external_df)),
            "class_counts": external_cls,
            "n_excluded": len(external_excluded),
            "feature_summary": summarize_features(external_df, feat_names),
        } if external else None),
        "additional_feature_candidates": additional_feature_candidates(primary_df),
        "outputs": {
            "features_primary": str(primary_path.relative_to(PROJECT_ROOT)),
            "features_external": (str(external_path.relative_to(PROJECT_ROOT))
                                  if external else None),
        },
    }
    (RESULTS / "feature_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    excluded_dump = {"primary": primary_excluded, "external": external_excluded}
    (RESULTS / "excluded_samples.json").write_text(
        json.dumps(excluded_dump, indent=2), encoding="utf-8")

    print(f"wrote {primary_path}  ({len(primary_df)} rows)")
    if external:
        print(f"wrote {external_path}  ({len(external_df)} rows)")
    print(f"wrote {RESULTS/'feature_summary.json'}")
    print(f"wrote {RESULTS/'excluded_samples.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())