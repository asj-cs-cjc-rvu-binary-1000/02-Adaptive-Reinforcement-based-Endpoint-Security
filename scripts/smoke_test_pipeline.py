#!/usr/bin/env python3
"""
scripts/smoke_test_pipeline.py — Phase 2 Task 9, part B.

End-to-end smoke test of the frozen prediction pipeline.
Loads 5 real Trad & Chehab images (3 benign, 2 malicious) + 2 Mendeley images
(1 benign, 1 malicious) + 1 intentionally corrupted image, and verifies that
each runs through the pipeline without exception and produces a well-formed
decision.
"""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import predict_pipeline as pp

RESULTS = PROJECT_ROOT / "results"
RAW = PROJECT_ROOT / "data" / "raw"


def _trad_chehab_image(idx: int) -> np.ndarray:
    imgs = pickle.load(open(RAW / "qr_codes_29.pickle", "rb"))
    labs = pickle.load(open(RAW / "qr_codes_29_labels.pickle", "rb"))
    arr = (imgs[idx].astype(np.uint8) * 255)
    pil = Image.fromarray(arr, mode="L")
    border = 20
    n = pil.size[0] + 2 * border
    canvas = Image.new("L", (n, n), 255)
    canvas.paste(pil, (border, border))
    canvas = canvas.resize((n * 4, n * 4), Image.NEAREST)
    return np.array(canvas.convert("L")), int(labs[idx])


def _mendeley_image(cls: str, idx: int) -> Path:
    files = sorted((RAW / "qr-images" / cls).glob("*.png"))
    return files[idx]


def main() -> int:
    pipeline = pp.PredictionPipeline(policy="default")
    print(f"pipeline: feature_kind={pipeline.feature_kind}, "
          f"t_review={pipeline.t_review}, t_block={pipeline.t_block}")

    # 3 benign + 2 malicious from Trad & Chehab, 1 benign + 1 malicious from Mendeley
    imgs_tc = pickle.load(open(RAW / "qr_codes_29.pickle", "rb"))
    labs_tc = pickle.load(open(RAW / "qr_codes_29_labels.pickle", "rb"))

    benign_idx = np.where(labs_tc == 0)[0][:3].tolist()
    mal_idx = np.where(labs_tc == 1)[0][:2].tolist()

    cases = []
    for i in benign_idx:
        arr, label = _trad_chehab_image(i)
        cases.append({"source": f"trad_chehab:benign:{i}", "label": "benign",
                      "image": arr, "kind": "ndarray"})
    for i in mal_idx:
        arr, label = _trad_chehab_image(i)
        cases.append({"source": f"trad_chehab:malicious:{i}", "label": "malicious",
                      "image": arr, "kind": "ndarray"})
    cases.append({"source": "mendeley:benign",
                  "label": "benign", "image": str(_mendeley_image("benign", 0)),
                  "kind": "path"})
    cases.append({"source": "mendeley:malicious",
                  "label": "malicious", "image": str(_mendeley_image("malicious", 0)),
                  "kind": "path"})

    # 1 corrupted image — pipeline should return Undecodable, not crash
    corrupted = np.zeros((100, 100), dtype=np.uint8)  # white noise
    cases.append({"source": "corrupted", "label": None,
                  "image": corrupted, "kind": "ndarray"})

    results = []
    n_passed = 0
    for case in cases:
        try:
            r = pipeline.run(case["image"])
            r["case_source"] = case["source"]
            r["case_label"] = case["label"]
            # Verify the shape of the output
            ok = (
                r.get("action") in ("Allow", "Review", "Block", "Undecodable")
                and (
                    r.get("prob_phishing") is None
                    or (0.0 <= r["prob_phishing"] <= 1.0)
                )
            )
            r["well_formed"] = ok
            if ok:
                n_passed += 1
            results.append(r)
            print(f"  [{'PASS' if ok else 'FAIL'}] {case['source']:30s} → "
                  f"{r['action']:12s} p={r.get('prob_phishing')}")
        except Exception as e:
            results.append({"case_source": case["source"],
                            "case_label": case["label"],
                            "error": f"{type(e).__name__}: {e}",
                            "well_formed": False})
            print(f"  [FAIL] {case['source']:30s} → ERROR {type(e).__name__}: {e}")

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": {"feature_kind": pipeline.feature_kind,
                     "t_review": pipeline.t_review,
                     "t_block": pipeline.t_block,
                     "model_path": "results/model_selection.json → models/*.joblib"},
        "n_cases": len(cases),
        "n_well_formed": n_passed,
        "all_passed": n_passed == len(cases),
        "results": results,
    }
    (RESULTS / "smoke_test_results.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n{n_passed}/{len(cases)} cases produced well-formed output")
    print(f"wrote {RESULTS/'smoke_test_results.json'}")
    return 0 if n_passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())