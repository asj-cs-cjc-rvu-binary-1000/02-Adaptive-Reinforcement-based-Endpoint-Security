#!/usr/bin/env python3
"""
scripts/run_calculus_experiment.py — Phase 2 Task 8, part B.

Loads the frozen raw_xgb model, computes validation predictions, fits
(t_review, t_block) via the smoothed cost optimizer, and compares:
  (a) default policy (t_r=0.35, t_b=0.70), and
  (b) optimized policy (from calculus optimization).
Reports test metrics for both as a sanity check but never uses test during
optimization.

Runtime: ~3 s (no re-fit, no re-decode).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dataset_config as cfg                      # noqa: E402
import calculus_threshold as ct                   # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
DEFAULT_T = (0.35, 0.70)


def assert_gates_pass(dataset: str, external: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_gates.py"),
           "check", "--dataset", dataset, "--external", external]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout); sys.stderr.write(r.stderr)
        raise SystemExit("FATAL: gates not passed or stale.")
    print("gates OK — proceeding to calculus experiment.")


def _split_predictions(kind: str, kept: list[str], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    suffix = "" if kind == "raw" else "_pca"
    df = pd.read_parquet(PROCESSED / f"split_{split}{suffix}.parquet")
    sel_path = RESULTS / "model_selection.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    model = joblib.load(PROJECT_ROOT / sel["selected_model"]["joblib_path"])
    pcs = [c for c in df.columns if c.startswith("PC")]
    X = df[kept].astype(float).values if kind == "raw" else df[pcs].astype(float).values
    y = df["label"].astype(int).values
    p = model.predict_proba(X)[:, 1]
    return y, p, df["sample_id"].values


def _policy_metrics(y: np.ndarray, p: np.ndarray, t_r: float, t_b: float) -> dict:
    pred = np.where(p >= t_b, 1, np.where(p < t_r, 0, -1))   # -1 = review
    benign_mask = (y == 0); malignant_mask = (y == 1)
    fpr = float(((p[benign_mask] >= t_b).sum()) / max(benign_mask.sum(), 1))
    fnr = float(((p[malignant_mask] < t_r).sum()) / max(malignant_mask.sum(), 1))
    review_rate = float(((p >= t_r) & (p < t_b)).sum() / len(p))
    # treat review as "advisory correct" from a cost perspective; report separately
    allow = (pred == 0)
    block = (pred == 1)
    tp = int((block & (y == 1)).sum()); fp = int((block & (y == 0)).sum())
    tn = int((allow & (y == 0)).sum()); fn = int((allow & (y == 1)).sum())
    return {
        "t_r": t_r, "t_b": t_b,
        "fpr": fpr, "fnr": fnr, "review_rate": review_rate,
        "confusion_at_thresholds": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def main() -> int:
    assert_gates_pass(cfg.DEFAULT_PRIMARY, cfg.DEFAULT_EXTERNAL)

    pca_report = json.loads((RESULTS / "pca_report.json").read_text(encoding="utf-8"))
    sel = json.loads((RESULTS / "model_selection.json").read_text(encoding="utf-8"))
    kind = sel["selected_model"]["feature_kind"]
    kept = pca_report["features"]["kept"]

    y_val, p_val, _ = _split_predictions(kind, kept, "val")
    y_test, p_test, _ = _split_predictions(kind, kept, "test")
    print(f"validation: n={len(y_val)}  benign={int((y_val == 0).sum())}  "
          f"malicious={int((y_val == 1).sum())}")
    print(f"test:       n={len(y_test)}  benign={int((y_test == 0).sum())}  "
          f"malicious={int((y_test == 1).sum())}")

    p_val_benign = p_val[y_val == 0]
    p_val_mal = p_val[y_val == 1]

    # Optimize on VALIDATION only
    opt = ct.optimize(p_val_benign, p_val_mal, p_val)
    print(f"\noptimized on validation: t_r={opt['t_r']:.4f}  t_b={opt['t_b']:.4f}")
    print(f"  cost_smooth_final={opt['cost_smooth_final']:.4f}  "
          f"cost_hard_final={opt['cost_hard_final']:.4f}")
    print(f"  gradient_final={opt['gradient_final']}")
    print(f"  hessian_det={opt['hessian_det']:.4f}  is_minimum={opt['is_minimum']}  "
          f"is_stationary={opt['is_stationary']}  boundary_hit={opt['boundary_hit']}")

    # Compare default vs optimized on VALIDATION (sanity) and TEST (report only)
    val_default = _policy_metrics(y_val, p_val, *DEFAULT_T)
    val_opt = _policy_metrics(y_val, p_val, opt["t_r"], opt["t_b"])
    test_default = _policy_metrics(y_test, p_test, *DEFAULT_T)
    test_opt = _policy_metrics(y_test, p_test, opt["t_r"], opt["t_b"])

    print(f"\nvalidation default (0.35, 0.70): fpr={val_default['fpr']:.4f}  "
          f"fnr={val_default['fnr']:.4f}  review={val_default['review_rate']:.4f}")
    print(f"validation optimized:          fpr={val_opt['fpr']:.4f}  "
          f"fnr={val_opt['fnr']:.4f}  review={val_opt['review_rate']:.4f}")
    print(f"test default (0.35, 0.70):     fpr={test_default['fpr']:.4f}  "
          f"fnr={test_default['fnr']:.4f}  review={test_default['review_rate']:.4f}")
    print(f"test optimized:                fpr={test_opt['fpr']:.4f}  "
          f"fnr={test_opt['fnr']:.4f}  review={test_opt['review_rate']:.4f}")

    # Cost landscape plot (grid over t_r, t_b)
    tr_grid = np.linspace(0.10, 0.65, 40)
    tb_grid = np.linspace(0.40, 0.95, 40)
    Z = np.zeros((len(tb_grid), len(tr_grid)))
    for i, tb in enumerate(tb_grid):
        for j, tr in enumerate(tr_grid):
            if tb - tr < 0.15: continue
            Z[i, j] = ct._cost_smooth(p_val_benign, p_val_mal, p_val, tr, tb,
                                      {**ct.DEFAULT_COST})
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(tr_grid, tb_grid, Z, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="smoothed cost C̃")
    # history path
    hist = np.array([[h["t_r"], h["t_b"]] for h in opt["history"]])
    ax.plot(hist[:, 0], hist[:, 1], "r.-", label="GD path")
    ax.scatter(*DEFAULT_T, marker="x", c="white", s=80, label="default (0.35, 0.70)")
    ax.scatter([opt["t_r"]], [opt["t_b"]], marker="*", c="red", s=150, label="optimized")
    ax.set_xlabel("t_review")
    ax.set_ylabel("t_block")
    ax.set_title("ARES calculus threshold optimisation — smoothed cost landscape (fit on validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "calculus_cost_landscape.png", dpi=120)
    plt.close(fig)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fit_on": "validation split only",
        "test_used_for_fit": False,
        "default_thresholds": {"t_r": DEFAULT_T[0], "t_b": DEFAULT_T[1]},
        "optimized_thresholds": {"t_r": opt["t_r"], "t_b": opt["t_b"]},
        "optimization": {
            "cost_params": opt["cost_params"],
            "bounds": opt["bounds"],
            "cost_smooth_final": opt["cost_smooth_final"],
            "cost_hard_final": opt["cost_hard_final"],
            "gradient_final": opt["gradient_final"],
            "hessian": opt["hessian"],
            "hessian_det": opt["hessian_det"],
            "is_minimum": opt["is_minimum"],
            "is_stationary": opt["is_stationary"],
            "boundary_hit": opt["boundary_hit"],
            "is_kkt_constrained": opt["is_kkt_constrained"],
            "history": opt["history"],
        },
        "validation_metrics": {
            "default": val_default,
            "optimized": val_opt,
            "delta": {
                "fpr": round(val_opt["fpr"] - val_default["fpr"], 4),
                "fnr": round(val_opt["fnr"] - val_default["fnr"], 4),
                "review_rate": round(val_opt["review_rate"] - val_default["review_rate"], 4),
            },
        },
        "test_metrics": {
            "default": test_default,
            "optimized": test_opt,
            "delta": {
                "fpr": round(test_opt["fpr"] - test_default["fpr"], 4),
                "fnr": round(test_opt["fnr"] - test_default["fnr"], 4),
                "review_rate": round(test_opt["review_rate"] - test_default["review_rate"], 4),
            },
            "note": "test_metrics reported for the manuscript; NEVER used to fit thresholds",
        },
    }
    (RESULTS / "calculus_experiment.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nwrote {RESULTS/'calculus_experiment.json'}")
    print(f"wrote {RESULTS/'calculus_cost_landscape.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())