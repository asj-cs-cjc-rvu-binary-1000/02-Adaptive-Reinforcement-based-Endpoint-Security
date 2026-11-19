#!/usr/bin/env python3
"""
scripts/evaluate_models.py — Phase 2 Task 7.

Evaluates the FROZEN selected model on:
  (a) the test split, exactly once, and
  (b) the external (Mendeley) split, exactly once.

Never re-trains, never re-selects, never touches validation. Selection was frozen
in Task 6 (`results/model_selection.json`). This script is deterministic and
should produce identical results on any re-run.

Outputs:
  results/final_metrics.json
  results/confusion_matrices/confusion_{test,external}.png
  results/model_accuracy_comparison.png    (val accuracies for all 8 models)
  results/model_precision_recall_comparison.png
  results/threshold_sweep.json             (FPR/FNR/Review-rate vs threshold; input for Task 8)
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dataset_config as cfg                      # noqa: E402
import feature_extractor as fx                    # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
CONF_DIR = RESULTS / "confusion_matrices"
CONF_DIR.mkdir(parents=True, exist_ok=True)


def assert_gates_pass(dataset: str, external: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_gates.py"),
           "check", "--dataset", dataset, "--external", external]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout); sys.stderr.write(r.stderr)
        raise SystemExit("FATAL: gates not passed or stale.")
    print("gates OK — proceeding to evaluation.")


def _metrics(y_true, y_pred, y_proba) -> dict:
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    return out


def _plot_confusion(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred benign", "Pred malicious"])
    ax.set_yticklabels(["True benign", "True malicious"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_model_bars(mc: dict, out_path: Path):
    names = [k for k, v in mc["metrics"].items() if "error" not in v]
    accs = [mc["metrics"][k]["accuracy"] for k in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color=["tab:blue" if "raw" in n else "tab:orange" for n in names])
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation accuracy — 4 families × 2 feature representations")
    ax.set_ylim(0.75, 0.92)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width()/2, a + 0.002, f"{a:.3f}",
                ha="center", fontsize=9)
    ax.axhline(0.5, color="grey", linestyle=":", label="Random")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_precision_recall_bars(mc: dict, out_path: Path):
    names = [k for k, v in mc["metrics"].items() if "error" not in v]
    prec = [mc["metrics"][k]["precision"] for k in names]
    rec = [mc["metrics"][k]["recall"] for k in names]
    f1 = [mc["metrics"][k]["f1"] for k in names]
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, prec, w, label="Precision", color="tab:blue")
    ax.bar(x, rec, w, label="Recall", color="tab:green")
    ax.bar(x + w, f1, w, label="F1", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score (validation)")
    ax.set_ylim(0.70, 0.90)
    ax.set_title("Precision / Recall / F1 — validation, all 8 models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _threshold_sweep(y_true, y_proba) -> dict:
    """Sweep thresholds from 0.05 to 0.95 (step 0.01). For each threshold report
    FPR (benign flagged >= t), FNR (malicious < t), and Review-rate
    (t_review <= p < t_block). But here we do a simpler 2-threshold sweep:
    report for each candidate review-threshold (assume block = review + 0.35),
    what the resulting FPR, FNR, and Review-rate would be. This is the
    data Task 8 needs to run the calculus optimiser.

    Simplification: 1-D sweep over a single threshold t. Task 8 will add the
    2-threshold (t_review, t_block) optimisation from main's calculus_threshold.py.
    """
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2).tolist()
    rows = []
    benign_mask = (y_true == 0)
    malicious_mask = (y_true == 1)
    n_benign = int(benign_mask.sum()); n_malicious = int(malicious_mask.sum())
    for t in grid:
        # single threshold: p >= t -> malicious
        fpr = float(((y_proba[benign_mask] >= t).sum()) / max(n_benign, 1))
        fnr = float(((y_proba[malicious_mask] < t).sum()) / max(n_malicious, 1))
        rows.append({
            "threshold": t,
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "tpr": round(1.0 - fnr, 4),
            "tnr": round(1.0 - fpr, 4),
        })
    return {"thresholds": rows, "n_benign": n_benign, "n_malicious": n_malicious}


def main() -> int:
    assert_gates_pass(cfg.DEFAULT_PRIMARY, cfg.DEFAULT_EXTERNAL)

    sel = json.loads((RESULTS / "model_selection.json").read_text(encoding="utf-8"))
    selected = sel["selected_model"]["model_name"]
    model_path = PROJECT_ROOT / sel["selected_model"]["joblib_path"]
    if not model_path.exists():
        raise SystemExit(
            f"FATAL: {model_path} missing. Re-run scripts/train_models.py."
        )
    print(f"evaluating frozen selected model: {selected}")
    print(f"  joblib: {model_path}")

    model = joblib.load(model_path)
    kept = json.loads((RESULTS / "pca_report.json").read_text())["features"]["kept"]

    kind = sel["selected_model"]["feature_kind"]
    suffix = "" if kind == "raw" else "_pca"

    # ---- test ----
    df_test = pd.read_parquet(PROCESSED / f"split_test{suffix}.parquet")
    pcs = [c for c in df_test.columns if c.startswith("PC")]
    X_test = df_test[kept].astype(float).values if kind == "raw" else df_test[pcs].astype(float).values
    y_test = df_test["label"].astype(int).values
    p_test = model.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)
    test_metrics = _metrics(y_test, pred_test, p_test)
    print(f"\n[test]  n={len(y_test)}  acc={test_metrics['accuracy']:.4f}  "
          f"f1={test_metrics['f1']:.4f}  roc_auc={test_metrics['roc_auc']:.4f}")

    # ---- external ----
    df_ext = pd.read_parquet(PROCESSED / f"split_external{suffix}.parquet")
    pcs_e = [c for c in df_ext.columns if c.startswith("PC")]
    X_ext = df_ext[kept].astype(float).values if kind == "raw" else df_ext[pcs_e].astype(float).values
    y_ext = df_ext["label"].astype(int).values
    p_ext = model.predict_proba(X_ext)[:, 1]
    pred_ext = (p_ext >= 0.5).astype(int)
    ext_metrics = _metrics(y_ext, pred_ext, p_ext)
    print(f"[external] n={len(y_ext)}  acc={ext_metrics['accuracy']:.4f}  "
          f"f1={ext_metrics['f1']:.4f}  roc_auc={ext_metrics['roc_auc']:.4f}")

    # ---- plots ----
    _plot_confusion(np.array(test_metrics["confusion_matrix"]),
                    f"Confusion matrix — test ({selected})",
                    CONF_DIR / "confusion_test.png")
    _plot_confusion(np.array(ext_metrics["confusion_matrix"]),
                    f"Confusion matrix — external ({selected})",
                    CONF_DIR / "confusion_external.png")
    mc = json.loads((RESULTS / "model_comparison.json").read_text(encoding="utf-8"))
    _plot_model_bars(mc, RESULTS / "model_accuracy_comparison.png")
    _plot_precision_recall_bars(mc, RESULTS / "model_precision_recall_comparison.png")

    # ---- threshold sweep (input for Task 8) ----
    sweep = _threshold_sweep(y_test, p_test)
    (RESULTS / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    # ---- final_metrics.json ----
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected,
        "selection_source": "results/model_selection.json (frozen in Task 6; not re-selected here)",
        "test": test_metrics,
        "external": ext_metrics,
        "deltas_external_minus_test": {
            "accuracy": round(ext_metrics["accuracy"] - test_metrics["accuracy"], 4),
            "f1": round(ext_metrics["f1"] - test_metrics["f1"], 4),
            "roc_auc": round(ext_metrics["roc_auc"] - test_metrics["roc_auc"], 4),
            "pr_auc": round(ext_metrics["pr_auc"] - test_metrics["pr_auc"], 4),
            "brier": round(ext_metrics["brier"] - test_metrics["brier"], 4),
        },
        "decision_threshold": 0.5,
        "confusion_matrices": {
            "test": test_metrics["confusion_matrix"],
            "external": ext_metrics["confusion_matrix"],
        },
    }
    (RESULTS / "final_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nwrote {RESULTS/'final_metrics.json'}")
    print(f"wrote {CONF_DIR/'confusion_test.png'}")
    print(f"wrote {CONF_DIR/'confusion_external.png'}")
    print(f"wrote {RESULTS/'model_accuracy_comparison.png'}")
    print(f"wrote {RESULTS/'model_precision_recall_comparison.png'}")
    print(f"wrote {RESULTS/'threshold_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())