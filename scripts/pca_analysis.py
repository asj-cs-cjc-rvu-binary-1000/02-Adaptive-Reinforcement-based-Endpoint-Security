#!/usr/bin/env python3
"""
scripts/pca_analysis.py — Phase 2 Task 5.

Fit StandardScaler + PCA on the TRAIN split only. Transform val, test, and
external with the same fitted scaler and PCA. Report explained variance,
component count (95 % rule), loadings of every retained component, and per-feature
scaler parameters.

Hard prerequisites:
  - run_gates.py check passes
  - split_{train,val,test,external}.parquet exist (Task 4)
  - feature_summary.json exists (Task 3) — used for constant-feature detection

Design:
  - Feature set is derived from feature_names (24 features) **minus** the constant
    features listed in results/feature_summary.json. Constant features carry no
    signal and break StandardScaler (std = 0). The exact drop list is
    recorded in results/pca_report.json.
  - NO data leakage: scaler.fit and PCA.fit both see ONLY train.
  - Component count: smallest k such that cumulative_explained_variance >= 0.95.
  - All outputs are reproducible from results/pca_scaler_params.json + the parquet
    files, without needing to pickle the model objects.
  - Random state: not needed for PCA, but pass through in case of future variants.

Outputs:
  data/processed/split_{train,val,test,external}_pca.parquet
    columns: PC1..PCk + label + sample_id + registrable_domain + dataset
    (registrable_domain needed downstream to keep group assignments intact)
  results/pca_report.json
  results/pca_explained_variance.png
  results/pca_feature_loadings.csv
  results/pca_scaler_params.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dataset_config as cfg                      # noqa: E402
import feature_extractor as fx                    # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"

VARIANCE_TARGET = 0.95


def assert_gates_pass(dataset: str, external: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_gates.py"),
           "check", "--dataset", dataset, "--external", external]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit("FATAL: gates not passed or stale.")
    print("gates OK — proceeding to PCA.")


def _load_splits() -> dict:
    names = {
        "train": PROCESSED / "split_train.parquet",
        "val": PROCESSED / "split_val.parquet",
        "test": PROCESSED / "split_test.parquet",
        "external": PROCESSED / "split_external.parquet",
    }
    out = {}
    for name, path in names.items():
        if not path.exists():
            raise SystemExit(
                f"FATAL: {path} missing. Run scripts/split_dataset.py first."
            )
        out[name] = pd.read_parquet(path)
    return out


def _select_features(train_df: pd.DataFrame, feature_summary_path: Path) -> dict:
    """Return {kept_features, dropped_features, drop_reasons}."""
    all_features = list(fx.feature_names)
    summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    const_features = set(
        summary.get("primary", {})
               .get("feature_summary", {})
               .get("constant_features", [])
    )
    kept = [f for f in all_features if f not in const_features]
    dropped = [f for f in all_features if f in const_features]
    return {
        "all_features": all_features,
        "kept_features": kept,
        "dropped_features": dropped,
        "drop_reasons": {f: "constant on primary training feature matrix "
                            "(see results/feature_summary.json)" for f in dropped},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--external", default=cfg.DEFAULT_EXTERNAL)
    args = ap.parse_args()

    dataset_key = args.dataset or cfg.DEFAULT_PRIMARY
    external_key = args.external or cfg.DEFAULT_EXTERNAL
    assert_gates_pass(dataset_key, external_key)

    splits = _load_splits()
    train_df = splits["train"]
    feature_summary_path = RESULTS / "feature_summary.json"
    if not feature_summary_path.exists():
        raise SystemExit(f"FATAL: {feature_summary_path} missing. Run prepare_features.py.")

    feat_sel = _select_features(train_df, feature_summary_path)
    kept = feat_sel["kept_features"]
    dropped = feat_sel["dropped_features"]
    print(f"features: {len(kept)} kept, {len(dropped)} dropped (constant)")
    print(f"  dropped: {dropped}")

    X_train = train_df[kept].astype(float).values

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Guard: if any kept feature has std=0 after scaling, something is wrong
    kept_variances = scaler.var_
    zero_std = [f for f, v in zip(kept, kept_variances) if v <= 1e-12]
    if zero_std:
        raise SystemExit(
            f"FATAL: features {zero_std} have ~zero variance despite being absent "
            f"from constant_features. Feature-summary detection may be stale — "
            f"re-run prepare_features.py."
        )

    # Fit PCA on train only
    pca_full = PCA(n_components=None, svd_solver="full")
    pca_full.fit(X_train_scaled)
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    k = int(np.searchsorted(cumulative, VARIANCE_TARGET) + 1)
    k = max(1, min(k, len(kept)))  # guard bounds

    # Refit with k components so we get a single fitted object for transforms
    pca = PCA(n_components=k, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Transform every split with the SAME scaler + pca
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        X = df[kept].astype(float).values
        Xs = scaler.transform(X)
        Xp = pca.transform(Xs)
        cols = [f"PC{i+1}" for i in range(Xp.shape[1])]
        out = pd.DataFrame(Xp, columns=cols, index=df.index)
        out["label"] = df["label"].values
        out["sample_id"] = df["sample_id"].values
        out["registrable_domain"] = df["registrable_domain"].values
        out["dataset"] = df["dataset"].values
        return out

    train_pca = _transform(splits["train"])
    val_pca = _transform(splits["val"])
    test_pca = _transform(splits["test"])
    external_pca = _transform(splits["external"])

    # Persist PCA-transformed splits
    train_pca.to_parquet(PROCESSED / "split_train_pca.parquet", index=False)
    val_pca.to_parquet(PROCESSED / "split_val_pca.parquet", index=False)
    test_pca.to_parquet(PROCESSED / "split_test_pca.parquet", index=False)
    external_pca.to_parquet(PROCESSED / "split_external_pca.parquet", index=False)

    # Loadings table (kept features x k components)
    loadings = pd.DataFrame(pca.components_.T, index=kept,
                            columns=[f"PC{i+1}" for i in range(k)])
    loadings.to_csv(RESULTS / "pca_feature_loadings.csv")

    # Scaler params (so the transform is reproducible without pickling)
    scaler_params = {
        "features_kept": kept,
        "features_dropped": dropped,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "variance": scaler.var_.tolist(),
        "n_samples_fit": int(scaler.n_samples_seen_),
    }
    (RESULTS / "pca_scaler_params.json").write_text(
        json.dumps(scaler_params, indent=2), encoding="utf-8")

    # Report
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "variance_target": VARIANCE_TARGET,
        "features": {
            "all": feat_sel["all_features"],
            "kept": kept,
            "dropped": dropped,
            "drop_reasons": feat_sel["drop_reasons"],
        },
        "n_features_kept": len(kept),
        "n_components_selected": k,
        "explained_variance_ratio": explained.tolist(),
        "explained_variance_ratio_selected": explained[:k].tolist(),
        "cumulative_explained_variance": cumulative.tolist(),
        "cumulative_explained_variance_selected": float(cumulative[k-1]),
        "singular_values": pca.singular_values_.tolist(),
        "fit_on": "split_train.parquet (grouped 70 % split, seed 42)",
        "transform_splits": ["train", "val", "test", "external"],
        "scaler": {
            "type": "StandardScaler",
            "fit_on_train_only": True,
            "n_samples_fit": int(scaler.n_samples_seen_),
        },
        "pca": {
            "type": "sklearn.decomposition.PCA",
            "svd_solver": "full",
            "fit_on_train_only": True,
        },
        "split_shapes": {
            "train": list(train_pca.shape),
            "val": list(val_pca.shape),
            "test": list(test_pca.shape),
            "external": list(external_pca.shape),
        },
    }
    (RESULTS / "pca_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # Plot: scree + cumulative explained variance
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 5))
        xs = np.arange(1, len(explained) + 1)
        ax1.bar(xs, explained, alpha=0.5, label="Per-component variance")
        ax1.set_xlabel("Principal component index")
        ax1.set_ylabel("Explained variance ratio (per component)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(xs, cumulative, color="tab:red", marker="o",
                 label="Cumulative variance")
        ax2.axhline(VARIANCE_TARGET, linestyle="--", color="black",
                    label=f"{int(VARIANCE_TARGET*100)} % target")
        ax2.axvline(k, linestyle=":", color="green",
                    label=f"k = {k} ({cumulative[k-1]*100:.2f} %)")
        ax2.set_ylabel("Cumulative explained variance", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_ylim(0, 1.05)
        fig.suptitle("ARES PCA — explained variance (fit on train split)")
        fig.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(RESULTS / "pca_explained_variance.png", dpi=120)
        plt.close(fig)
        print(f"wrote {RESULTS/'pca_explained_variance.png'}")
    except Exception as e:                            # noqa: BLE001
        print(f"WARN: could not write plot: {type(e).__name__}: {e}")

    print(f"\nkept {len(kept)} features, PCA selected k = {k} components "
          f"({cumulative[k-1]*100:.2f} % cumulative variance)")
    print(f"train PCA shape: {train_pca.shape}")
    print(f"val PCA shape:   {val_pca.shape}")
    print(f"test PCA shape:  {test_pca.shape}")
    print(f"external PCA shape: {external_pca.shape}")
    print(f"\nPC1 variance: {explained[0]*100:.2f} %")
    print(f"PC2 variance: {explained[1]*100:.2f} %" if len(explained) > 1 else "")
    print(f"wrote {RESULTS/'pca_report.json'}")
    print(f"wrote {RESULTS/'pca_feature_loadings.csv'}")
    print(f"wrote {RESULTS/'pca_scaler_params.json'}")
    print("\n--- PC1 loadings (sorted by |loading|) ---")
    pc1 = loadings["PC1"].copy()
    for feat, val in pc1.reindex(pc1.abs().sort_values(ascending=False).index).items():
        print(f"  {feat:30s}  {val:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())