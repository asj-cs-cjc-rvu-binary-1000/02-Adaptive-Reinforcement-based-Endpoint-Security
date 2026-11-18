#!/usr/bin/env python3
"""
scripts/run_gates.py — Phase 2 pre-training gates.

Sub-commands:
  gate1-overlap  Full-dataset duplicate + label-conflict + cross-dataset overlap audit
                 → results/full_overlap_audit.json, results/label_conflicts.json
                 Writes results/.gate1_pass marker on success.

  gate2-decode   (Scheduled as Phase 2 Task 2 — NOT implemented in this file yet)

Design:
  - Decodes every image via scripts/qr_utils.py (byte-identical to origin/main).
  - Categorizes payloads (url / plain_text / non_url_scheme / undecodable).
  - Deduplicates by sha256(payload) within class; drops cross-class conflicts.
  - Checks domain overlap primary ∩ external.
  - Emits one JSON per gate. Never writes into data/ — outputs only under results/.
  - Deterministic (no random sampling).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import qr_utils                                   # noqa: E402
import feature_extractor as fx                    # noqa: E402
import dataset_config as cfg                      # noqa: E402

RESULTS = PROJECT_ROOT / "results"
TMP_DECODE = RESULTS / ".tmp_decode"
TMP_DECODE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Hard-fail guard (same discipline as Phase 1 Task 3)
_REQUIRED_QR = ("decode_qr_codes", "decode_qr_code", "QRDecodeError")
_REQUIRED_FX = ("feature_names", "is_valid_url", "extract_url_features",
                "extract_registrable_domain")
_missing = [n for n in _REQUIRED_QR if not hasattr(qr_utils, n)]
_missing += [n for n in _REQUIRED_FX if not hasattr(fx, n)]
if _missing:
    raise SystemExit(f"FATAL: production modules missing symbols: {_missing}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_payload(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _categorize(payload: str) -> str:
    """Reuse Phase 1 categorizer semantics exactly."""
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


def _decode_array(arr: np.ndarray, tag: str) -> list[str]:
    """Write arr to a temp PNG, call the path-based production decoder."""
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
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"length mismatch {images.shape[0]} vs {labels.shape[0]}")
    for i in range(images.shape[0]):
        mat = images[i]
        arr = (mat.astype(np.uint8) * 255)
        pil = Image.fromarray(arr, mode="L")
        border = 20
        n = pil.size[0] + 2 * border
        canvas = Image.new("L", (n, n), 255)
        canvas.paste(pil, (border, border))
        canvas = canvas.resize((n * 4, n * 4), Image.NEAREST)
        bgr = np.ascontiguousarray(np.array(canvas.convert("L"))[:, :, None].repeat(3, axis=2))
        yield i, spec.label_mapping[int(labels[i])], bgr, _sha256_bytes(mat.tobytes())


def _iter_mendeley(spec: cfg.DatasetSpec):
    for cls in ("benign", "malicious"):
        folder = spec.png_dir / cls
        for f in sorted(folder.glob("*.png")):
            pil = Image.open(f).convert("L")
            arr = np.ascontiguousarray(np.array(pil))
            yield str(f.relative_to(PROJECT_ROOT)), cls, arr, _sha256_file(f)


def _iter_spec(spec: cfg.DatasetSpec):
    if spec.images_pickle:
        yield from _iter_trad_chehab(spec)
    else:
        yield from _iter_mendeley(spec)


def _audit_one_dataset(spec: cfg.DatasetSpec, log_prefix: str) -> dict:
    """Decode all images, dedupe by payload hash within class, record conflicts."""
    per_class_cat = Counter()
    per_class_payload_hash = {"benign": {}, "malicious": {}}   # hash -> (count, first_seen_id)
    per_class_image_hash = {"benign": {}, "malicious": {}}     # image hash -> count (dup image)
    label_conflicts: dict[str, dict] = {}                      # payload_hash -> {benign_ids, malicious_ids, payload}
    class_counts = Counter()
    n = 0
    for sample_id, cls, arr, img_hash in _iter_spec(spec):
        n += 1
        class_counts[cls] += 1
        if n % 200 == 0 or n == spec.n_total:
            print(f"[{log_prefix}] {n}/{spec.n_total} processed", flush=True)
        # duplicate image check
        per_class_image_hash[cls].setdefault(img_hash, 0)
        per_class_image_hash[cls][img_hash] += 1
        # decode
        payloads = _decode_array(arr, tag=f"{spec.key}_{hash(str(sample_id)) & 0xffffffff:x}")
        if not payloads:
            per_class_cat[(cls, "undecodable")] += 1
            continue
        payload = payloads[0]
        category = _categorize(payload)
        per_class_cat[(cls, category)] += 1
        if category != "url":
            continue
        ph = _sha256_payload(payload)
        if ph in per_class_payload_hash[cls]:
            per_class_payload_hash[cls][ph][0] += 1
        else:
            per_class_payload_hash[cls][ph] = [1, sample_id, payload]
        # cross-class conflict detection (same payload hash appears under both classes)
        other = "malicious" if cls == "benign" else "benign"
        if ph in per_class_payload_hash[other]:
            rec = label_conflicts.setdefault(ph, {"payload": payload,
                                                 "benign_ids": [], "malicious_ids": []})
            if cls == "benign":
                rec["benign_ids"].append(sample_id)
            else:
                rec["malicious_ids"].append(sample_id)

    # within-class duplicate count (payload-level)
    within_dup = {c: sum(v[0] - 1 for v in per_class_payload_hash[c].values())
                  for c in per_class_payload_hash}
    within_unique = {c: len(per_class_payload_hash[c]) for c in per_class_payload_hash}
    # image-hash duplicates
    dup_images = {c: sum(v - 1 for v in per_class_image_hash[c].values() if v > 1)
                  for c in per_class_image_hash}

    # domains per class (for overlap)
    domains_by_class = {
        c: set(fx.extract_registrable_domain(rec[2])
               for rec in per_class_payload_hash[c].values())
        for c in per_class_payload_hash
    }
    domains_by_class = {c: {d for d in s if d} for c, s in domains_by_class.items()}

    return {
        "dataset": spec.name,
        "n_total": spec.n_total,
        "n_processed": n,
        "class_counts": dict(class_counts),
        "per_class_category": {
            f"{cls}__{cat}": v for (cls, cat), v in per_class_cat.items()
        },
        "payload_dedup": {
            "within_class_duplicates": within_dup,
            "unique_payloads_after_dedup": within_unique,
        },
        "image_duplicates": dup_images,
        "label_conflicts_count": len(label_conflicts),
        "label_conflict_hashes": sorted(label_conflicts.keys()),
        "domains_by_class": {c: sorted(s) for c, s in domains_by_class.items()},
    }


def gate1_overlap(args) -> int:
    primary = cfg.get_primary(args.dataset)
    external = cfg.get_external(args.external) if args.external else None

    print(f"Gate 1 — primary: {primary.key} ({primary.name}), n={primary.n_total}", flush=True)
    primary_result = _audit_one_dataset(primary, log_prefix="primary")

    external_result = None
    if external is not None and external.key != primary.key:
        print(f"Gate 1 — external: {external.key} ({external.name}), n={external.n_total}", flush=True)
        external_result = _audit_one_dataset(external, log_prefix="external")

    # cross-dataset domain overlap
    cross_ds_overlap = None
    if external_result is not None:
        prim_domains = set()
        for s in primary_result["domains_by_class"].values():
            prim_domains.update(s)
        ext_domains = set()
        for s in external_result["domains_by_class"].values():
            ext_domains.update(s)
        cross_ds_overlap = {
            "n_primary_domains": len(prim_domains),
            "n_external_domains": len(ext_domains),
            "n_overlap": len(prim_domains & ext_domains),
            "overlap_examples": sorted(prim_domains & ext_domains)[:30],
        }

    # gate PASS/FAIL: no unresolved label conflicts; within-class dups accounted for
    conflicts = primary_result["label_conflicts_count"]
    pass_flag = (conflicts == 0)

    report = {
        "gate": "gate1-overlap",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary": primary_result,
        "external": external_result,
        "cross_dataset_domain_overlap": cross_ds_overlap,
        "gate_pass": pass_flag,
        "gate_fail_reason": None if pass_flag
                            else f"{conflicts} cross-class label conflicts remain unresolved",
        "production_hashes": {
            "qr_utils_sha256": _sha256_file(PROJECT_ROOT / "scripts" / "qr_utils.py"),
            "feature_extractor_sha256": _sha256_file(PROJECT_ROOT / "scripts" / "feature_extractor.py"),
        },
    }

    out = RESULTS / "full_overlap_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "label_conflicts.json").write_text(
        json.dumps({ph: None for ph in primary_result["label_conflict_hashes"]},
                   indent=2), encoding="utf-8")

    marker = RESULTS / ".gate1_pass"
    if pass_flag:
        marker.write_text("PASS\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()

    print(f"wrote {out}")
    print(f"gate1_pass = {pass_flag}, conflicts = {conflicts}, "
          f"primary within-class dups = {primary_result['payload_dedup']['within_class_duplicates']}")
    if cross_ds_overlap:
        print(f"cross-dataset domain overlap = {cross_ds_overlap['n_overlap']}")
    return 0 if pass_flag else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g1 = sub.add_parser("gate1-overlap")
    g1.add_argument("--dataset", default=None,
                    help="primary dataset key (default: trad_chehab)")
    g1.add_argument("--external", default=cfg.DEFAULT_EXTERNAL,
                    help="external dataset key (default: mendeley; 'none' to skip)")
    args = ap.parse_args()
    if args.cmd == "gate1-overlap":
        return gate1_overlap(args)
    raise SystemExit(f"unknown sub-command: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())