#!/usr/bin/env python3
"""
scripts/compare_models.py — Phase 2 Task 6 (part B).

Reads results/model_comparison.json (validation metrics only), ranks the 8
model/feature combos, and freezes the selection by writing:
  results/model_selection.json — the frozen selected model, ranking, and rationale.

The selection rule is documented below and matches docs/experimental_protocol.md §9:

  selection_score = mean(F1, ROC-AUC)  on VALIDATION set only.
  Tiebreak 1: higher precision on validation.
  Tiebreak 2: lower Brier score on validation (better calibration).
  Tiebreak 3: alphabetical model_name (deterministic).

The selected model is NOT the final submission model until it is retrained on
train+val in a later step (Task 7/8). This task freezes the SELECTION, not the
final artifact — the test split stays untouched, per protocol.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"


def _score(m: dict) -> tuple:
    f1 = m.get("f1") or 0.0
    roc = m.get("roc_auc") or 0.0
    prec = m.get("precision") or 0.0
    brier = m.get("brier")
    brier_term = -brier if brier is not None else -999.0   # lower brier is better
    return (-(f1 + roc) / 2.0, -prec, brier_term, m["model_name"])


def main() -> int:
    src = RESULTS / "model_comparison.json"
    if not src.exists():
        raise SystemExit(f"FATAL: {src} missing. Run train_models.py first.")
    data = json.loads(src.read_text(encoding="utf-8"))

    models = {k: v for k, v in data["metrics"].items() if "error" not in v}
    if not models:
        raise SystemExit("FATAL: no successful models to compare.")

    ranked = sorted(models.values(), key=_score)
    best = ranked[0]

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": {
            "primary": "mean(F1, ROC-AUC) on validation",
            "tiebreak_1": "higher validation precision",
            "tiebreak_2": "lower validation Brier",
            "tiebreak_3": "alphabetical model_name (deterministic)",
        },
        "test_split_used": False,
        "n_models_compared": len(models),
        "ranking": [
            {
                "rank": i + 1,
                "model_name": m["model_name"],
                "feature_kind": m["feature_kind"],
                "family": m["family"],
                "val_accuracy": m["accuracy"],
                "val_precision": m["precision"],
                "val_recall": m["recall"],
                "val_f1": m["f1"],
                "val_roc_auc": m["roc_auc"],
                "val_pr_auc": m["pr_auc"],
                "val_brier": m["brier"],
                "selection_score": (m["f1"] + (m["roc_auc"] or 0)) / 2.0,
            }
            for i, m in enumerate(ranked)
        ],
        "selected_model": {
            "model_name": best["model_name"],
            "feature_kind": best["feature_kind"],
            "family": best["family"],
            "val_metrics": {
                "accuracy": best["accuracy"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "roc_auc": best["roc_auc"],
                "pr_auc": best["pr_auc"],
                "brier": best["brier"],
                "confusion_matrix": best["confusion_matrix"],
            },
            "joblib_path": f"models/{best['model_name']}.joblib",
        },
        "test_split_evaluation_plan": (
            "Deferred to Phase 2 Task 7 — this selection is frozen; test is evaluated "
            "ONCE with this model. Any change to the selection rule after seeing test "
            "would invalidate the test result."
        ),
    }
    (RESULTS / "model_selection.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print("RANKING (by selection_score = mean(F1, ROC-AUC) on validation):")
    print(f"{'rank':>4} {'model':12} {'kind':4} {'acc':>6} {'f1':>6} {'roc_auc':>8} {'pr_auc':>7} {'brier':>7}")
    for r in out["ranking"]:
        roc = r["val_roc_auc"] if r["val_roc_auc"] is not None else float("nan")
        pr = r["val_pr_auc"] if r["val_pr_auc"] is not None else float("nan")
        br = r["val_brier"] if r["val_brier"] is not None else float("nan")
        print(f"{r['rank']:>4} {r['model_name']:12} {r['feature_kind']:4} "
              f"{r['val_accuracy']:>6.4f} {r['val_f1']:>6.4f} {roc:>8.4f} {pr:>7.4f} {br:>7.4f}")

    print(f"\nSELECTED: {best['model_name']}  (feature_kind={best['feature_kind']}, "
          f"family={best['family']})")
    print(f"wrote {RESULTS/'model_selection.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())