#!/usr/bin/env python3
"""
scripts/model_manifest.py — Phase 2b, Check 4.

Writes results/model_manifest.json: a single document recording everything
PredictionPipeline depends on.

Fields:
  git_commit_hash — HEAD when manifest was written
  python_version  — sys.version_info
  packages        — sklearn, xgboost, numpy, pandas, pyarrow (versions)
  artifacts       — path + sha256 + shape/size for every file PredictionPipeline reads
  prediction_pipeline_reads — explicit list of artifact paths

Also exposes `check_manifest()` used by PredictionPipeline to fail closed if the
artifacts have drifted since the manifest was written.

Usage:
  python scripts/model_manifest.py              # write manifest
  python scripts/model_manifest.py --check      # verify; exit 1 on mismatch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _collect() -> dict:
    # Import versions with graceful fallback
    try:
        import sklearn; skl = sklearn.__version__
    except Exception: skl = "unknown"
    try:
        import xgboost; xgb = xgboost.__version__
    except Exception: xgb = "unknown"
    try:
        import numpy; npv = numpy.__version__
    except Exception: npv = "unknown"
    try:
        import pandas; pdv = pandas.__version__
    except Exception: pdv = "unknown"
    try:
        import pyarrow; pyv = pyarrow.__version__
    except Exception: pyv = "unknown"

    # PredictionPipeline's consumed artifacts — keep in sync with predict_pipeline.py
    artifact_paths = {
        "model_selection": RESULTS / "model_selection.json",
        "pca_report": RESULTS / "pca_report.json",
        "calculus_experiment": RESULTS / "calculus_experiment.json",
    }
    # The selected model joblib
    sel_path = RESULTS / "model_selection.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        joblib_rel = sel["selected_model"]["joblib_path"]
        artifact_paths["model_joblib"] = PROJECT_ROOT / joblib_rel

    artifacts = {}
    for name, path in artifact_paths.items():
        if not path.exists():
            artifacts[name] = {"path": str(path.relative_to(PROJECT_ROOT)),
                               "missing": True}
            continue
        artifacts[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_head(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {
            "scikit-learn": skl,
            "xgboost": xgb,
            "numpy": npv,
            "pandas": pdv,
            "pyarrow": pyv,
        },
        "artifacts": artifacts,
        "prediction_pipeline_reads": sorted(artifact_paths.keys()),
    }


def write_manifest() -> Path:
    manifest = _collect()
    out = RESULTS / "model_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def check_manifest(strict: bool = True) -> tuple[bool, list[str]]:
    """Return (ok, mismatch_list). ok=True iff every artifact's sha256 matches."""
    manifest_path = RESULTS / "model_manifest.json"
    if not manifest_path.exists():
        return False, ["manifest missing"]
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = _collect()

    mismatches: list[str] = []
    for name, rec in stored["artifacts"].items():
        cur = current["artifacts"].get(name)
        if cur is None:
            mismatches.append(f"{name}: missing in current")
            continue
        if rec.get("missing") or cur.get("missing"):
            mismatches.append(f"{name}: missing file")
            continue
        if rec["sha256"] != cur["sha256"]:
            mismatches.append(
                f"{name}: sha256 changed "
                f"(stored={rec['sha256'][:12]}…, now={cur['sha256'][:12]}…)"
            )
    for name in current["artifacts"]:
        if name not in stored["artifacts"]:
            mismatches.append(f"{name}: new artifact not in manifest")

    pkg_mismatches = []
    for pkg, ver in stored["packages"].items():
        cur_ver = current["packages"].get(pkg)
        if cur_ver != ver:
            pkg_mismatches.append(f"{pkg}: {ver} → {cur_ver}")
    if pkg_mismatches:
        mismatches.extend(pkg_mismatches)

    return (len(mismatches) == 0), mismatches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify manifest (exit 1 on mismatch) instead of writing")
    args = ap.parse_args()

    if args.check:
        ok, mismatches = check_manifest()
        if ok:
            print("manifest OK — artifacts and packages match")
            return 0
        print("manifest MISMATCH:")
        for m in mismatches:
            print(f"  {m}")
        return 1

    out = write_manifest()
    manifest = json.loads(out.read_text(encoding="utf-8"))
    print(f"wrote {out}")
    print(f"git HEAD: {manifest['git_commit_hash']}")
    print(f"python:   {manifest['python_version']}")
    print(f"packages: {manifest['packages']}")
    print("artifacts:")
    for name, rec in manifest["artifacts"].items():
        print(f"  {name:22s} {rec.get('path')}  "
              f"sha256={rec.get('sha256', 'MISSING')[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())