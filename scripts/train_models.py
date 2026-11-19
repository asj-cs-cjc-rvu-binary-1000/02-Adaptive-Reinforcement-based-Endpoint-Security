#!/usr/bin/env python3
"""
scripts/train_models.py — Phase 2 Task 6 (part A).

Trains 4 model families x 2 feature representations = 8 model/feature combos,
evaluates each on the VALIDATION split only. Never touches the test split.

Feature representations:
  raw  — the 20 kept features (constant-dropped) from split_{train,val}.parquet
  pca  — the 13 principal components from split_{train,val}_pca.parquet

Model families:
  LogisticRegression   (StandardScaler + LR for raw; LR alone for pca, since PCA is
                        already centered; but for consistency we also fit a scaler
                        on raw — LR needs scale; for pca the PCs are already in
                        unit-variance-after-scaling space, so no additional scaler)
  DecisionTree         (no scaling needed; max_depth=10, min_samples_leaf=5)
  RandomForest         (300 trees, class_weight="balanced_subsample", n_jobs=-1)
  XGBoost              (400 estimators, max_depth=6, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=balanced)

All fits are seed=42.  No hyperparameter search — fixed parameters documented in
docs/model_experiments.md.  Hyperparameter tuning is a Phase 2 extension and would
require an inner GroupKFold CV on train, which is a separate task.

Output: results/model_comparison.json (raw structured) + results/model_comparison.csv
        + models/{raw,pca}_{lr,dt,rf,xgb}.joblib
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dataset_config as cfg                      # noqa: E402
import feature_extractor as fx                    # noqa: E402

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SEED = 42


def assert_gates_pass(dataset: str, external: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_gates.py"),
           "check", "--dataset", dataset, "--external", external]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit("FATAL: gates not passed or stale.")
    print("gates OK — proceeding to model training.")


def _kept_features() -> list[str]:
    """Which features survived the drop in Task 5."""
    p = RESULTS / "pca_report.json"
    if not p.exists():
        raise SystemExit(f"FATAL: {p} missing. Run pca_analysis.py first.")
    r = json.loads(p.read_text(encoding="utf-8"))
    return list(r["features"]["kept"])


def _load_split(name: str, kind: str) -> pd.DataFrame:
    path = PROCESSED / f"split_{name}{'' if kind == 'raw' else '_pca'}.parquet"
    if not path.exists():
        raise SystemExit(f"FATAL: {path} missing. Run prior tasks.")
    return pd.read_parquet(path)


def _build_feature_matrix(df: pd.DataFrame, kind: str, kept: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if kind == "raw":
        X = df[kept].astype(float).values
    elif kind == "pca":
        pcs = [c for c in df.columns if c.startswith("PC")]
        X = df[pcs].astype(float).values
    else:
        raise ValueError(f"unknown feature kind {kind}")
    y = df["label"].astype(int).values
    return X, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            out["pr_auc"] = None
        try:
            out["brier"] = float(brier_score_loss(y_true, y_proba))
        except ValueError:
            out["brier"] = None
    return out


def _fit_model(family: str, kind: str, X_train: np.ndarray, y_train: np.ndarray):
    """Return (fitted_estimator, model_name)."""
    if family == "lr":
        if kind == "raw":
            est = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, random_state=SEED))
        else:
            est = LogisticRegression(max_iter=1000, random_state=SEED)
        return est, f"{kind}_lr"
    if family == "dt":
        est = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5,
                                     class_weight="balanced", random_state=SEED)
        return est, f"{kind}_dt"
    if family == "rf":
        est = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     random_state=SEED, n_jobs=-1)
        return est, f"{kind}_rf"
    if family == "xgb":
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost not installed")
        neg = int((y_train == 0).sum()); pos = int((y_train == 1).sum())
        scale_pos_weight = neg / max(pos, 1)
        est = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED, n_jobs=-1,
            eval_metric="logloss", tree_method="hist", verbosity=0,
        )
        return est, f"{kind}_xgb"
    raise ValueError(f"unknown family {family}")


def _proba_of(est, X) -> np.ndarray | None:
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else None
    return None


def main() -> int:
    assert_gates_pass(cfg.DEFAULT_PRIMARY, cfg.DEFAULT_EXTERNAL)
    kept = _kept_features()

    families = ["lr", "dt", "rf", "xgb"] if XGB_AVAILABLE else ["lr", "dt", "rf"]
    kinds = ["raw", "pca"]
    all_metrics: dict = {}

    for kind in kinds:
        train_df = _load_split("train", kind)
        val_df = _load_split("val", kind)
        X_train, y_train = _build_feature_matrix(train_df, kind, kept)
        X_val, y_val = _build_feature_matrix(val_df, kind, kept)
        print(f"\n[{kind}] X_train={X_train.shape}  X_val={X_val.shape}  "
              f"pos_train={int(y_train.sum())}  pos_val={int(y_val.sum())}")

        for family in families:
            t0 = time.time()
            try:
                est, name = _fit_model(family, kind, X_train, y_train)
                est.fit(X_train, y_train)
                y_pred = est.predict(X_val)
                y_proba = _proba_of(est, X_val)
                m = _metrics(y_val, y_pred, y_proba)
                m["fit_seconds"] = round(time.time() - t0, 2)
                m["feature_kind"] = kind
                m["family"] = family
                m["model_name"] = name
                all_metrics[name] = m
                joblib.dump(est, MODELS / f"{name}.joblib")
                print(f"  {name:10s}  val_acc={m['accuracy']:.4f}  "
                      f"val_f1={m['f1']:.4f}  val_roc_auc="
                      f"{(m['roc_auc'] if m['roc_auc'] is not None else float('nan')):.4f}  "
                      f"({m['fit_seconds']}s)")
            except Exception as e:
                print(f"  {family} ({kind}) FAILED: {type(e).__name__}: {e}")
                all_metrics[f"{kind}_{family}"] = {
                    "error": f"{type(e).__name__}: {e}",
                    "feature_kind": kind,
                    "family": family,
                    "model_name": f"{kind}_{family}",
                }

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "features_kept_count": len(kept),
        "feature_kinds": kinds,
        "families": families,
        "test_set_used": False,
        "selection_policy": "best mean(F1, ROC-AUC) on VALIDATION only; test never touched in this task",
        "metrics": all_metrics,
    }
    (RESULTS / "model_comparison.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    # CSV
    rows = []
    for name, m in all_metrics.items():
        if "error" in m:
            rows.append({"model_name": name, "error": m["error"]})
            continue
        rows.append({
            "model_name": name, "feature_kind": m["feature_kind"],
            "family": m["family"], "n": m["n"],
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "roc_auc": round(m["roc_auc"], 4) if m["roc_auc"] is not None else None,
            "pr_auc": round(m["pr_auc"], 4) if m["pr_auc"] is not None else None,
            "brier": round(m["brier"], 4) if m["brier"] is not None else None,
            "fit_seconds": m["fit_seconds"],
        })
    pd.DataFrame(rows).to_csv(RESULTS / "model_comparison.csv", index=False)
    print(f"\nwrote {RESULTS/'model_comparison.json'}")
    print(f"wrote {RESULTS/'model_comparison.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())