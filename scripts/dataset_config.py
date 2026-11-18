#!/usr/bin/env python3
"""
scripts/dataset_config.py — Phase 2 dataset configuration.

Config A (confirmed): Primary = Trad & Chehab, External = Mendeley.
Every Phase 2 script imports this module. Override via:
  --dataset {trad_chehab,mendeley}
or environment variable:
  ARES_DATASET_PRIMARY / ARES_DATASET_EXTERNAL

Do NOT hardcode dataset identifiers anywhere else. This module is the single
source of truth so a future Config B is a one-line change, not a refactor.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
RESULTS = PROJECT_ROOT / "results"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# ---- Confirmed defaults (Config A) ----
DEFAULT_PRIMARY = "trad_chehab"
DEFAULT_EXTERNAL = "mendeley"

# ---- Dataset registry ----
_DATASETS = {
    "trad_chehab": {
        "name": "Trad & Chehab",
        "images_pickle": RAW / "qr_codes_29.pickle",
        "labels_pickle": RAW / "qr_codes_29_labels.pickle",
        "png_dir": None,
        "label_mapping": {0: "benign", 1: "malicious"},
        "n_total": 9987,
    },
    "mendeley": {
        "name": "Mendeley",
        "images_pickle": None,
        "labels_pickle": None,
        "png_dir": RAW / "qr-images",
        "label_mapping": {"benign": "benign", "malicious": "malicious"},
        "n_total": 1000,
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    images_pickle: Path | None
    labels_pickle: Path | None
    png_dir: Path | None
    label_mapping: dict
    n_total: int


def _resolve_key(arg: str | None, env_var: str, default: str) -> str:
    key = (arg or os.environ.get(env_var) or default).strip().lower()
    if key not in _DATASETS:
        raise ValueError(
            f"Unknown dataset {key!r}. Choose from {sorted(_DATASETS)}."
        )
    return key


def get_primary(key: str | None = None) -> DatasetSpec:
    k = _resolve_key(key, "ARES_DATASET_PRIMARY", DEFAULT_PRIMARY)
    spec = _DATASETS[k]
    return DatasetSpec(
        key=k,
        name=spec["name"],
        images_pickle=spec["images_pickle"],
        labels_pickle=spec["labels_pickle"],
        png_dir=spec["png_dir"],
        label_mapping=spec["label_mapping"],
        n_total=spec["n_total"],
    )


def get_external(key: str | None = None) -> DatasetSpec | None:
    raw = key if key is not None else os.environ.get("ARES_DATASET_EXTERNAL", DEFAULT_EXTERNAL)
    if raw is None or raw.lower() in ("", "none"):
        return None
    k = _resolve_key(raw, "ARES_DATASET_EXTERNAL", DEFAULT_EXTERNAL)
    spec = _DATASETS[k]
    return DatasetSpec(
        key=k,
        name=spec["name"],
        images_pickle=spec["images_pickle"],
        labels_pickle=spec["labels_pickle"],
        png_dir=spec["png_dir"],
        label_mapping=spec["label_mapping"],
        n_total=spec["n_total"],
    )


def list_datasets() -> list[str]:
    return sorted(_DATASETS)