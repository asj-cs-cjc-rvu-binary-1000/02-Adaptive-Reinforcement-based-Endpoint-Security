#!/usr/bin/env python3
"""
scripts/inspect_datasets.py — Phase 1 Task 1+2 inspection for ARES rebuild.

Reads (READ-ONLY) the two candidate datasets under data/raw/:
  1. Trad & Chehab  — data/raw/qr_codes_29.pickle + qr_codes_29_labels.pickle
  2. Mendeley       — data/raw/qr-images/{benign,malicious}/*.png

Emits results/dataset_inspection.json with:
  - sha256 + byte size of every primary file
  - shape / dtype / value-range / class-balance for Trad & Chehab
  - per-class size & PIL-mode histograms for Mendeley
  - label-semantics note (0=benign, 1=malicious; derived from paper's 5005/4982 split)
  - license status  (UNVERIFIED_IN_REPO unless a license file is present)

No network calls. No writes outside results/ and /tmp/. Safe to re-run.
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
RAW = PROJECT_ROOT / "data" / "raw"
RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "dataset_inspection.json"

TRAD_IMAGES = RAW / "qr_codes_29.pickle"
TRAD_LABELS = RAW / "qr_codes_29_labels.pickle"
MENDELEY_DIR = RAW / "qr-images"

# Paper-reported counts; used to derive the label->class mapping empirically
PAPER_BENIGN = 5005
PAPER_MALICIOUS = 4982


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_license_files() -> list[str]:
    hits: list[str] = []
    for pat in ("LICENSE*", "LICENCE*", "CITATION*", "README*"):
        for p in PROJECT_ROOT.rglob(pat):
            if ".git" in p.parts:
                continue
            hits.append(str(p.relative_to(PROJECT_ROOT)))
    return sorted(set(hits))


def inspect_trad_chehab() -> dict:
    info: dict = {"dataset": "Trad & Chehab (Quishing)", "files": {}}
    for label, path in (("images", TRAD_IMAGES), ("labels", TRAD_LABELS)):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        info["files"][label] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
        }

    with open(TRAD_IMAGES, "rb") as f:
        images = pickle.load(f)
    with open(TRAD_LABELS, "rb") as f:
        labels = pickle.load(f)

    if not isinstance(images, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("Expected np.ndarray for both images and labels")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"length mismatch: {images.shape[0]} != {labels.shape[0]}")

    uniq, counts = np.unique(labels, return_counts=True)
    count_map = {int(u): int(c) for u, c in zip(uniq, counts)}

    # Derive label->class mapping against paper counts
    if count_map.get(0) == PAPER_BENIGN and count_map.get(1) == PAPER_MALICIOUS:
        mapping = {0: "benign", 1: "malicious"}
        mapping_source = "empirical: matches paper (5005 benign / 4982 malicious)"
    elif count_map.get(0) == PAPER_MALICIOUS and count_map.get(1) == PAPER_BENIGN:
        mapping = {0: "malicious", 1: "benign"}
        mapping_source = "empirical: inverted vs paper — REVIEW BEFORE USE"
    else:
        mapping = {int(k): "UNKNOWN" for k in count_map}
        mapping_source = "UNRESOLVED: counts do not match paper — DO NOT TRAIN"

    info.update({
        "images_shape": list(images.shape),
        "images_dtype": str(images.dtype),
        "images_value_range": [int(images.min()), int(images.max())],
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
        "label_counts": count_map,
        "label_to_class": mapping,
        "label_mapping_source": mapping_source,
        "n_total": int(images.shape[0]),
        "classes_present": sorted(count_map.keys()),
    })
    return info


def inspect_mendeley() -> dict:
    info: dict = {"dataset": "Mendeley 1,000 QR Images", "files": {}, "per_class": {}}
    if not MENDELEY_DIR.exists():
        raise FileNotFoundError(f"missing {MENDELEY_DIR}")

    for cls in ("benign", "malicious"):
        folder = MENDELEY_DIR / cls
        files = sorted(folder.glob("*.png"))
        sizes = Counter()
        modes = Counter()
        corrupt: list[str] = []
        for f in files:
            try:
                im = Image.open(f)
                im.verify()           # cheap integrity check
                im = Image.open(f)    # reopen after verify
                sizes[im.size] += 1
                modes[im.mode] += 1
            except Exception as e:    # noqa: BLE001
                corrupt.append(f"{f.name}: {type(e).__name__}: {e}")
        info["per_class"][cls] = {
            "folder": str(folder.relative_to(PROJECT_ROOT)),
            "n_files": len(files),
            "size_histogram": {f"{w}x{h}": c for (w, h), c in sizes.items()},
            "mode_histogram": dict(modes),
            "n_corrupt": len(corrupt),
            "corrupt": corrupt[:10],
        }

    total_bytes = sum(p.stat().st_size for p in MENDELEY_DIR.rglob("*.png"))
    info["files"]["total_pngs"] = {
        "n_files": sum(v["n_files"] for v in info["per_class"].values()),
        "bytes": total_bytes,
    }
    return info


def main() -> int:
    report: dict = {
        "inspection_tool": "scripts/inspect_datasets.py",
        "inspection_date_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
        "license_files_found_in_repo": find_license_files(),
        "license_status_per_dataset": {
            "trad_chehab": "UNVERIFIED_IN_REPO (no license file shipped in archive; confirm against publication data-availability statement)",
            "mendeley": "UNVERIFIED_IN_REPO (no license file shipped in archive; confirm at Mendeley Data DOI landing page)",
        },
    }
    try:
        report["trad_chehab"] = inspect_trad_chehab()
    except Exception as e:  # noqa: BLE001
        report["trad_chehab_error"] = f"{type(e).__name__}: {e}"
    try:
        report["mendeley"] = inspect_mendeley()
    except Exception as e:  # noqa: BLE001
        report["mendeley_error"] = f"{type(e).__name__}: {e}"

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps({k: report[k] for k in report if k not in ("mendeley",)}, indent=2, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())