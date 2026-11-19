#!/usr/bin/env python3
"""
scripts/recompute_train_constants.py — Phase 2b, Check 1.

Recompute the "constant features" set from the TRAIN PARTITION ONLY and write
the result into results/feature_summary.json as a new field. Then verify whether
the train-only list matches the existing full-data list.

Why this exists:
  Phase 2 Task 3 computed constant features from features_primary.parquet (the
  whole primary dataset, before Task 4's split). A reviewer legitimately flags
  that as selection-scope leakage: the feature-drop decision should be made on
  training data only.

What this script does:
  - Loads data/processed/split_train.parquet.
  - Computes nunique() for each of the 24 raw features on train only.
  - Any feature with nunique <= 1 is a "train-only constant."
  - Compares to results/feature_summary.json['primary']['feature_summary']['constant_features'].
  - Writes results/feature_summary.json['primary']['train_only_constant_features']
    and ['constant_features_match_train_only'].
  - If the lists DIFFER: exit non-zero by default (so a human decides whether to
    re-run downstream tasks). Pass --accept-difference to write the new list and
    exit zero, but ALSO writes a warning file.

Safe to re-run. No writes to models/, data/processed/*.parquet, or any other
artifact consumed by PredictionPipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import feature_extractor as fx                    # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-difference", action="store_true",
                    help="write the train-only list even if it differs from the "
                         "full-data list (do NOT use unless you plan to re-run "
                         "PCA, train, eval, calculus, prediction)")
    args = ap.parse_args()

    train_path = PROCESSED / "split_train.parquet"
    if not train_path.exists():
        raise SystemExit(f"FATAL: {train_path} missing. Run split_dataset.py first.")

    train = pd.read_parquet(train_path)
    all_feats = list(fx.feature_names)

    # features_primary was built from all 24 raw features; split_train contains them
    missing = [f for f in all_feats if f not in train.columns]
    if missing:
        raise SystemExit(f"FATAL: train split missing features {missing}.")

    train_nunique = {f: int(train[f].nunique(dropna=False)) for f in all_feats}
    train_only_constants = [f for f, n in train_nunique.items() if n <= 1]

    full_summary_path = RESULTS / "feature_summary.json"
    summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    full_constants = list(
        summary["primary"]["feature_summary"].get("constant_features", [])
    )

    match = (sorted(train_only_constants) == sorted(full_constants))

    print(f"all features: {len(all_feats)}")
    print(f"train-only constants ({len(train_only_constants)}): {sorted(train_only_constants)}")
    print(f"full-data constants ({len(full_constants)}):       {sorted(full_constants)}")
    print(f"match: {match}")

    # Write back into feature_summary.json (never overwrites the full-data list)
    summary["primary"]["train_only_constant_features"] = sorted(train_only_constants)
    summary["primary"]["constant_features_match_train_only"] = match
    summary["primary"]["train_only_constant_detection"] = {
        "fit_on": "data/processed/split_train.parquet",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(len(train)),
        "nunique_per_feature": train_nunique,
    }
    full_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {full_summary_path}")

    if not match and not args.accept_difference:
        print()
        print("STOP: train-only constants differ from full-data constants.")
        print("Downstream artifacts (PCA / models / eval / calculus / predictions) "
              "were derived using the full-data list and are still consistent with it.")
        print("Decide whether to:")
        print("  (a) accept full-data list as-is and re-run with --accept-difference "
              "(writes the new list; downstream artifacts then STALE) — NO, do not do this "
              "without re-running Tasks 5-9.")
        print("  (b) re-run Task 5 (pca_analysis.py) onward with the train-only list, "
              "regenerating models/eval/calculus/predictions.")
        print("  (c) revert (delete the two new fields) and open a Phase 2 review task.")
        return 2

    if not match and args.accept_difference:
        warn = RESULTS / "feature_constants_difference.json"
        warn.write_text(json.dumps({
            "warning": "train-only constants differ from full-data constants",
            "train_only": sorted(train_only_constants),
            "full_data": sorted(full_constants),
            "action": "downstream artifacts now STALE — re-run pca_analysis.py onward",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
        print(f"WROTE WARNING: {warn}")
        print("You MUST re-run: pca_analysis.py, train_models.py, compare_models.py, "
              "evaluate_models.py, run_calculus_experiment.py, and refresh model_manifest.")

    return 0 if match else (0 if args.accept_difference else 2)


if __name__ == "__main__":
    sys.exit(main())