#!/usr/bin/env python3
"""
scripts/split_dataset.py — Phase 2 Task 4.

Grouped 70:15:15 split by registrable domain (seed 42), with an explicit
excluded partition for samples that produced no URL payload.

Prerequisites:
  - run_gates.py check passes (both gates valid)
  - data/processed/features_primary.parquet exists (Phase 2 Task 3)
  - results/excluded_samples.json exists

Design:
  - Split at the **domain-group** level, not sample level.
  - GroupShuffleSplit twice: first 70% train vs 30% temp, then 30% temp into
    15% val + 15% test. Both with random_state=42.
  - Post-split assertions (hard-fail on violation):
      * set(train_domains) ∩ set(val_domains) == ∅
      * set(train_domains) ∩ set(test_domains) == ∅
      * set(val_domains) ∩ set(test_domains) == ∅
      * set(sample_ids_train) ∩ set(sample_ids_val) == ∅ (and same for test)
      * set(sample_ids_train) ∩ set(sample_ids_test) == ∅ (and same for val)
      * train/val/test sample_id union == full features_primary sample_ids
  - Class distribution per split reported (may deviate from global by <±5%
    because grouped splitting cannot preserve exact proportions at domain level).
  - Excluded partition preserves every non-URL sample from Task 3.

Outputs:
  data/processed/split_train.parquet
  data/processed/split_val.parquet
  data/processed/split_test.parquet
  data/processed/split_excluded.parquet
  data/processed/split_external.parquet
  results/dataset_split_report.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dataset_config as cfg                      # noqa: E402
import feature_extractor as fx                    # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def assert_gates_pass(dataset: str, external: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_gates.py"),
           "check", "--dataset", dataset, "--external", external]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit(
            "FATAL: gates not passed or stale. Re-run gate1-overlap and/or gate2-decode."
        )
    print("gates OK — proceeding to split.")


def _stats(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_benign": int((df["label"] == 0).sum()),
        "n_malicious": int((df["label"] == 1).sum()),
        "n_domains": int(df["registrable_domain"].nunique()),
        "n_empty_domain": int((df["registrable_domain"] == "").sum()),
        "n_unique_sample_ids": int(df["sample_id"].nunique()),
    }


def _group_shuffle_split_3way(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 grouped split. First split 70 vs 30, then 30 -> 15/15."""
    groups = df["registrable_domain"]
    gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))
    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()

    # Split temp 50/50 -> 15/15 of original
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED + 1)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["registrable_domain"]))
    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()
    return df_train, df_val, df_test


def _assert_disjoint_domains(df_train: pd.DataFrame, df_val: pd.DataFrame,
                             df_test: pd.DataFrame) -> dict:
    dt = set(df_train["registrable_domain"]) - {""}
    dv = set(df_val["registrable_domain"]) - {""}
    dte = set(df_test["registrable_domain"]) - {""}
    out = {
        "train_val_domain_overlap": len(dt & dv),
        "train_test_domain_overlap": len(dt & dte),
        "val_test_domain_overlap": len(dv & dte),
        "domain_overlap_examples": sorted(list((dt & dv) | (dt & dte) | (dv & dte)))[:20],
    }
    if any(out[k] > 0 for k in ("train_val_domain_overlap",
                                "train_test_domain_overlap",
                                "val_test_domain_overlap")):
        raise AssertionError(f"Domain leakage detected: {out}")
    return out


def _assert_disjoint_samples(df_train: pd.DataFrame, df_val: pd.DataFrame,
                             df_test: pd.DataFrame) -> dict:
    st = set(df_train["sample_id"])
    sv = set(df_val["sample_id"])
    ste = set(df_test["sample_id"])
    out = {
        "train_val_sample_overlap": len(st & sv),
        "train_test_sample_overlap": len(st & ste),
        "val_test_sample_overlap": len(sv & ste),
    }
    if any(v > 0 for v in out.values()):
        raise AssertionError(f"Sample ID leakage detected: {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--external", default=cfg.DEFAULT_EXTERNAL)
    args = ap.parse_args()

    dataset_key = args.dataset or cfg.DEFAULT_PRIMARY
    external_key = args.external or cfg.DEFAULT_EXTERNAL

    assert_gates_pass(dataset_key, external_key)

    primary_path = PROCESSED / "features_primary.parquet"
    external_path = PROCESSED / "features_external.parquet"
    excluded_path = RESULTS / "excluded_samples.json"
    for p in (primary_path, external_path, excluded_path):
        if not p.exists():
            raise SystemExit(
                f"FATAL: {p} missing. Run scripts/prepare_features.py first."
            )

    df = pd.read_parquet(primary_path)
    df_ext = pd.read_parquet(external_path)
    excluded_raw = json.loads(excluded_path.read_text(encoding="utf-8"))

    # Sanity: no missing domains in the training slice
    if (df["registrable_domain"] == "").sum() > 0:
        raise SystemExit(
            f"FATAL: {int((df['registrable_domain'] == '').sum())} rows have empty "
            "registrable_domain. Cannot form domain-grouped splits. Investigate "
            "extract_registrable_domain on these payloads before proceeding."
        )

    print(f"primary: {df.shape}  unique domains: {df['registrable_domain'].nunique()}")
    df_train, df_val, df_test = _group_shuffle_split_3way(df)
    print(f"split: train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    # The critical invariants
    domain_assert = _assert_disjoint_domains(df_train, df_val, df_test)
    sample_assert = _assert_disjoint_samples(df_train, df_val, df_test)

    # Union must equal the original feature matrix exactly
    all_sample_ids = set(df["sample_id"])
    split_sample_ids = set(df_train["sample_id"]) | set(df_val["sample_id"]) | set(df_test["sample_id"])
    if all_sample_ids != split_sample_ids:
        missing = all_sample_ids - split_sample_ids
        extra = split_sample_ids - all_sample_ids
        raise AssertionError(
            f"Split does not cover the full feature matrix. "
            f"missing={len(missing)}  extra={len(extra)}"
        )

    # Excluded partition
    excluded_rows = []
    for slot in ("primary", "external"):
        for rec in excluded_raw.get(slot, []):
            excluded_rows.append({**rec, "source_dataset": slot})
    df_excl = pd.DataFrame(excluded_rows)

    # Persist
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(PROCESSED / "split_train.parquet", index=False)
    df_val.to_parquet(PROCESSED / "split_val.parquet", index=False)
    df_test.to_parquet(PROCESSED / "split_test.parquet", index=False)
    df_ext.to_parquet(PROCESSED / "split_external.parquet", index=False)
    df_excl.to_parquet(PROCESSED / "split_excluded.parquet", index=False)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "ratios_target": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "splitting_method": "GroupShuffleSplit by registrable_domain (2 steps: 70/30, then 30 -> 15/15)",
        "splitting_seed_gss1": SEED,
        "splitting_seed_gss2": SEED + 1,
        "primary": {
            "source": str(primary_path.relative_to(PROJECT_ROOT)),
            "full_stats": _stats(df),
            "train_stats": _stats(df_train),
            "val_stats": _stats(df_val),
            "test_stats": _stats(df_test),
            "domain_disjointness": domain_assert,
            "sample_disjointness": sample_assert,
            "full_coverage_verified": True,
        },
        "external": {
            "source": str(external_path.relative_to(PROJECT_ROOT)),
            "role": "external_validation_only — never enters train/val/test",
            "stats": _stats(df_ext),
        },
        "excluded": {
            "source": str(excluded_path.relative_to(PROJECT_ROOT)),
            "n_rows": int(len(df_excl)),
            "categories": df_excl["category"].value_counts().to_dict()
                          if len(df_excl) > 0 else {},
            "note": "Not part of any ML split. Documented in docs/feature_engineering.md.",
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dataset_split_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\ntrain: {len(df_train)}  ({len(df_train)/len(df)*100:.1f}%)  "
          f"class {df_train['label'].value_counts().to_dict()}  "
          f"domains {df_train['registrable_domain'].nunique()}")
    print(f"val:   {len(df_val)}  ({len(df_val)/len(df)*100:.1f}%)  "
          f"class {df_val['label'].value_counts().to_dict()}  "
          f"domains {df_val['registrable_domain'].nunique()}")
    print(f"test:  {len(df_test)}  ({len(df_test)/len(df)*100:.1f}%)  "
          f"class {df_test['label'].value_counts().to_dict()}  "
          f"domains {df_test['registrable_domain'].nunique()}")
    print(f"excluded: {len(df_excl)} rows (categories: {report['excluded']['categories']})")
    print(f"external (untouched): {len(df_ext)} rows")
    print(f"\nall assertions passed:")
    print(f"  domain overlaps: {domain_assert}")
    print(f"  sample overlaps: {sample_assert}")
    print(f"  coverage:        full feature matrix covered exactly")
    print(f"\nwrote {RESULTS/'dataset_split_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())