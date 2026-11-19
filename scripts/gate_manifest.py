#!/usr/bin/env python3
"""
scripts/gate_manifest.py — content-addressed gate manifests.

Every gate writes a manifest recording the sha256s of:
  - the dataset file(s) that were audited
  - the production decoder / extractor that performed the audit
  - the gate script itself
  - the audit-result JSON (full_overlap_audit.json / full_decode_audit.json)

`scripts/run_gates.py check` compares current sha256s to the manifest. If ANY
sha256 differs, the gate is stale and MUST be re-run (data changed, decoder
changed, or the gate script changed).

This is what makes a 52-minute gate a *one-time* cost instead of a per-task cost.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
RESULTS = PROJECT_ROOT / "results"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_inputs(primary_files: list[Path]) -> dict:
    """Return the sha256 map of everything a gate depends on."""
    manifest = {
        "dataset_files": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
                          for p in primary_files if p.exists()},
        "qr_utils_sha256": sha256_file(SCRIPTS / "qr_utils.py"),
        "feature_extractor_sha256": sha256_file(SCRIPTS / "feature_extractor.py"),
        "run_gates_sha256": sha256_file(SCRIPTS / "run_gates.py"),
        "dataset_config_sha256": sha256_file(SCRIPTS / "dataset_config.py"),
    }
    return manifest


def write_manifest(gate: str, inputs: dict, result_json: Path) -> Path:
    """Write results/.gateN_manifest.json."""
    manifest = dict(inputs)
    manifest["result_json_sha256"] = sha256_file(result_json)
    out = RESULTS / f".{gate}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def load_manifest(gate: str) -> dict | None:
    p = RESULTS / f".{gate}_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def verify_gate(gate: str, primary_files: list[Path]) -> dict:
    """Return {ok, stale_fields, missing_manifest, marker_present}.

    ok == True means:
      - marker results/.<gate>_pass exists
      - manifest exists and every sha256 matches the current files
      - result_json_sha256 still matches the actual result JSON
    """
    marker = RESULTS / f".{gate}_pass"
    manifest = load_manifest(gate)
    result_json = RESULTS / (
        "full_overlap_audit.json" if gate == "gate1" else "full_decode_audit.json"
    )
    out = {
        "gate": gate,
        "marker_present": marker.exists(),
        "manifest_present": manifest is not None,
        "result_json_present": result_json.exists(),
        "ok": False,
        "stale_fields": [],
        "reason": None,
    }
    if not out["marker_present"]:
        out["reason"] = f"marker {marker.name} missing — gate never ran or failed"
        return out
    if not out["manifest_present"]:
        out["reason"] = f"manifest .{gate}_manifest.json missing — cannot verify"
        return out
    if not out["result_json_present"]:
        out["reason"] = f"result JSON {result_json.name} missing — cannot verify"
        return out

    current = current_inputs(primary_files)
    for k, v_expected in current.items():
        if k == "dataset_files":
            for path, sha in v_expected.items():
                if manifest.get("dataset_files", {}).get(path) != sha:
                    out["stale_fields"].append(f"dataset_files.{path}")
        else:
            if manifest.get(k) != v_expected:
                out["stale_fields"].append(k)

    expected_result_sha = manifest.get("result_json_sha256")
    actual_result_sha = sha256_file(result_json)
    if expected_result_sha != actual_result_sha:
        out["stale_fields"].append("result_json_sha256")

    out["ok"] = len(out["stale_fields"]) == 0
    if not out["ok"]:
        out["reason"] = (
            "manifest stale: " + ", ".join(out["stale_fields"])
            + " — re-run the gate (data, decoder, or gate script changed)"
        )
    return out