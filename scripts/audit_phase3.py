#!/usr/bin/env python3
"""
scripts/audit_phase3.py — Phase 3 Task 0.

Read-only repository audit for Phase 3 integration. Produces
results/phase3_audit.json with:

  1. File inventory — which Phase 3 files exist, which are missing, their sizes.
  2. Authoritative-inference confirmation — the frozen model, manifest, and
     feature list used by scripts/predict_pipeline.py.
  3. Imports of scripts/predict_pipeline.py — what inference code it depends on.
  4. Test files present in tests/ (or absence of tests/).
  5. Manifest integrity (model_manifest.check_manifest).
  6. Gate status (run_gates.check via subprocess — read-only).
  7. Smoke test smoke_test_pipeline.py — run it read-only and capture pass count.
  8. Duplicated-logic risks — grep source files for inline model loading,
     predict_proba calls, or feature extraction that bypasses predict_pipeline.

Never writes to data/raw, data/processed, models/, or any Python module.
Only writes results/phase3_audit.json.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Files the Phase 3 brief names explicitly.
PHASE3_FILES = [
    "app.py",
    "adaptive_agent.py",
    "calculus_threshold.py",
    "feature_extractor.py",
    "qr_utils.py",
    "scripts/predict_pipeline.py",
    "train_model.py",
    "requirements.txt",
    "results/model_manifest.json",
    "scripts/model_manifest.py",
    "scripts/run_gates.py",
    "scripts/smoke_test_pipeline.py",
    "scripts/evaluate_models.py",
    "scripts/run_calculus_experiment.py",
]

CONSUMED_ARTIFACTS = [
    "results/model_selection.json",
    "results/pca_report.json",
    "results/calculus_experiment.json",
    "models/raw_xgb.joblib",
    "results/model_manifest.json",
]


def _file_info(rel_path: str) -> dict:
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return {"path": rel_path, "exists": False}
    return {
        "path": rel_path,
        "exists": True,
        "bytes": p.stat().st_size,
        "sha256": _sha256(p),
    }


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _imports_of(rel_path: str) -> list[str]:
    """Return module names imported by a file (best-effort; no execution)."""
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="replace")
    mods: set[str] = set()
    for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", text, re.MULTILINE):
        mods.add(m.group(1).split(".")[0])
    return sorted(mods)


def _grep_for(rel_path: str, pattern: str) -> list[dict]:
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return []
    hits = []
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if re.search(pattern, line):
            hits.append({"line": i, "text": line.rstrip()})
    return hits


def _check_manifest() -> dict:
    try:
        import model_manifest
        ok, mismatches = model_manifest.check_manifest()
        return {"ok": ok, "mismatches": mismatches}
    except Exception as e:                     # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _run_gates() -> dict:
    try:
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_gates.py"),
             "check", "--dataset", "trad_chehab", "--external", "mendeley"],
            capture_output=True, text=True, timeout=30,
        )
        return {"returncode": r.returncode, "stdout_tail": r.stdout[-800:]}
    except Exception as e:                     # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _run_smoke() -> dict:
    try:
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "smoke_test_pipeline.py")],
            capture_output=True, text=True, timeout=60,
        )
        tail = r.stdout[-600:]
        return {"returncode": r.returncode, "stdout_tail": tail}
    except Exception as e:                     # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _inspect_predict_pipeline() -> dict:
    """Extract the public contract that app.py will consume."""
    pp = PROJECT_ROOT / "scripts" / "predict_pipeline.py"
    if not pp.exists():
        return {"error": "scripts/predict_pipeline.py missing"}
    text = pp.read_text(encoding="utf-8")
    out: dict = {"imports": _imports_of("scripts/predict_pipeline.py")}

    # Public class
    m = re.search(r"^class\s+(\w+)", text, re.MULTILINE)
    out["public_class"] = m.group(1) if m else None

    # Public methods
    methods = re.findall(r"^\s+def\s+(\w+)\s*\(", text, re.MULTILINE)
    out["methods"] = methods

    # Output keys — grep for `"payload"`, `"action"`, etc. in the returned dict
    keys = sorted(set(re.findall(r'"(payload|payload_category|features|prob_phishing|action|reason|policy)"', text)))
    out["output_keys"] = keys
    return out


def _duplication_risks() -> dict:
    """Grep for code that duplicates what predict_pipeline already does."""
    risks = {}
    for f in ["app.py", "adaptive_agent.py", "calculus_threshold.py",
              "train_model.py"]:
        risks[f] = {
            "exists": (PROJECT_ROOT / f).exists(),
            "inline_joblib_load": _grep_for(f, r"joblib\.load|pickle\.load"),
            "inline_predict_proba": _grep_for(f, r"\.predict_proba\s*\("),
            "inline_extract_features": _grep_for(f, r"extract_url_features\s*\("),
            "inline_xgb": _grep_for(f, r"XGBClassifier\s*\("),
        }
    return risks


def _tests_inventory() -> dict:
    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.exists():
        return {"tests_dir_exists": False, "files": []}
    files = sorted(
        str(p.relative_to(PROJECT_ROOT))
        for p in tests_dir.rglob("test_*.py")
    )
    images = sorted(
        str(p.relative_to(PROJECT_ROOT))
        for p in tests_dir.rglob("*.png")
    ) if (tests_dir / "images").exists() else []
    return {"tests_dir_exists": True, "test_files": files, "test_images": images}


def main() -> int:
    audit: dict = {
        "phase": "3",
        "task": "0 — repository and integration audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_inventory": {f: _file_info(f) for f in PHASE3_FILES},
        "consumed_artifacts": {f: _file_info(f) for f in CONSUMED_ARTIFACTS},
        "predict_pipeline_contract": _inspect_predict_pipeline(),
        "manifest_check": _check_manifest(),
        "gate_check": _run_gates(),
        "smoke_test": _run_smoke(),
        "duplication_risks": _duplication_risks(),
        "tests_inventory": _tests_inventory(),
    }
    out = RESULTS / "phase3_audit.json"
    out.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    print()

    # Human-readable summary
    print("=== File inventory ===")
    for f, info in audit["file_inventory"].items():
        status = "MISSING" if not info.get("exists") else f"{info['bytes']:>10,} B"
        print(f"  {status:>12}  {f}")

    print()
    print("=== Consumed artifacts ===")
    for f, info in audit["consumed_artifacts"].items():
        status = "MISSING" if not info.get("exists") else info["sha256"][:16] + "…"
        print(f"  {status:>20}  {f}")

    print()
    print("=== PredictPipeline contract ===")
    pp = audit["predict_pipeline_contract"]
    print(f"  class:       {pp.get('public_class')}")
    print(f"  methods:     {pp.get('methods')}")
    print(f"  output keys: {pp.get('output_keys')}")
    print(f"  imports:     {pp.get('imports')}")

    print()
    print("=== Manifest & gates & smoke ===")
    print(f"  manifest ok: {audit['manifest_check'].get('ok')}")
    print(f"  gates rc:    {audit['gate_check'].get('returncode')}")
    print(f"  smoke rc:    {audit['smoke_test'].get('returncode')}")

    print()
    print("=== Duplication risks ===")
    for f, r in audit["duplication_risks"].items():
        if not r["exists"]:
            print(f"  {f}: FILE MISSING (Phase 3 must create it)")
            continue
        issues = [k for k in r if k != "exists" and r[k]]
        print(f"  {f}: {issues if issues else 'clean'}")

    print()
    print("=== Tests inventory ===")
    ti = audit["tests_inventory"]
    print(f"  tests/ exists: {ti['tests_dir_exists']}")
    print(f"  test files:    {ti.get('test_files', [])}")
    print(f"  test images:   {len(ti.get('test_images', []))} found")
    return 0


if __name__ == "__main__":
    sys.exit(main())