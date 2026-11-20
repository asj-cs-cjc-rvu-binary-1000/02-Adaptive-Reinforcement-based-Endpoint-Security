#!/usr/bin/env python3
"""
scripts/predict_pipeline.py — Phase 2 Task 9, part A.

Reusable inference module. This is what Phase 3+ app code imports.

Given:
  - a QR image (file path or numpy array),
  - the frozen selected model (`results/model_selection.json`),
  - the fitted scaler/PCA (`results/pca_scaler_params.json` — but we do NOT need
    it for `raw_xgb`, because raw feature kind means the model consumes raw
    20-feature vectors directly; the scaler only exists for future `pca_*` models),
  - a threshold policy (default or optimized from Task 8),

Produces:
  {
    "payload": str | None,          # decoded URL text
    "payload_category": str,         # "url" | "plain_text" | "non_url_scheme" | "undecodable"
    "features": dict | None,         # 20 features (constant-dropped) if payload is URL
    "prob_phishing": float | None,   # model's *estimated* phishing probability
                                     # (raw XGBoost output; NOT isotonic-calibrated)
    "action": str,                   # "Allow" | "Review" | "Block" | "Undecodable"
    "reason": str,
  }

No network. No side effects. Deterministic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import qr_utils                                   # noqa: E402
import feature_extractor as fx                    # noqa: E402

RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"


class PredictionPipeline:
    """Frozen ARES inference pipeline — model + feature policy + decision thresholds.

    Note on probability semantics: the returned `prob_phishing` field is the
    *estimated* phishing probability from the raw XGBoost `predict_proba` output.
    It is **not** isotonic-calibrated (the Phase 2 model is raw XGBoost, not
    CalibratedClassifierCV). Brier scores measure how well this estimate behaves
    empirically, but do not assert formal calibration.
    """

    def __init__(self, policy: str = "default"):
        """policy ∈ {"default", "optimized"}; thresholds read from Task 8 output.

        Fails closed if the on-disk artifacts have drifted from
        results/model_manifest.json (checks sha256 of model_selection.json,
        pca_report.json, calculus_experiment.json, and the selected joblib).
        """
        from model_manifest import check_manifest
        ok, mismatches = check_manifest()
        if not ok:
            raise SystemExit(
                "FATAL: PredictionPipeline artifact mismatch vs results/model_manifest.json:\n"
                + "\n".join(f"  - {m}" for m in mismatches)
                + "\nRe-run scripts/model_manifest.py to refresh the manifest after "
                  "intentional changes, or restore the drifted artifacts."
            )

        sel = json.loads((RESULTS / "model_selection.json").read_text(encoding="utf-8"))
        pca = json.loads((RESULTS / "pca_report.json").read_text(encoding="utf-8"))
        self.feature_kind = sel["selected_model"]["feature_kind"]
        self.feature_names = list(pca["features"]["kept"])           # 20 kept features
        self.model = joblib.load(PROJECT_ROOT / sel["selected_model"]["joblib_path"])

        calc = json.loads((RESULTS / "calculus_experiment.json").read_text(encoding="utf-8"))
        if policy == "default":
            th = calc["default_thresholds"]
        elif policy == "optimized":
            th = calc["optimized_thresholds"]
        else:
            raise ValueError(f"unknown policy {policy!r}; use 'default' or 'optimized'")
        self.t_review = float(th["t_r"])
        self.t_block = float(th["t_b"])

    # ---- payload extraction ----
    def decode_image(self, image_input) -> tuple[str | None, str]:
        """Return (payload, category). image_input: file path (str | Path) or np.ndarray.

        Missing or unreadable files map to (None, "undecodable") — fail closed,
        never raise. The low-level qr_utils.decode_qr_codes still raises
        FileNotFoundError for direct callers (tests/test_qr.py depends on that).
        """
        if isinstance(image_input, np.ndarray):
            # Write to a temp PNG for the production decoder (path-based API)
            import tempfile, cv2
            tmp = Path(tempfile.mkstemp(suffix=".png", dir=RESULTS / ".tmp_decode")[1])
            try:
                arr = image_input
                if arr.ndim == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                cv2.imwrite(str(tmp), np.ascontiguousarray(arr))
                try:
                    payloads = qr_utils.decode_qr_codes(tmp)
                except (qr_utils.QRDecodeError, FileNotFoundError, OSError):
                    payloads = []
            finally:
                tmp.unlink(missing_ok=True)
        else:
            path = Path(image_input)
            if not path.exists():
                return None, "undecodable"
            try:
                payloads = qr_utils.decode_qr_codes(path)
            except (qr_utils.QRDecodeError, FileNotFoundError, OSError):
                payloads = []

        if not payloads:
            return None, "undecodable"
        return payloads[0], self._categorize(payloads[0])

    @staticmethod
    def _categorize(payload: str) -> str:
        s = payload.strip()
        if not s:
            return "empty"
        low = s.lower()
        if any(low.startswith(p + ":") for p in
               ("javascript", "data", "mailto", "tel", "sms", "wifi", "ftp",
                "file", "geo", "bitcoin", "smsto", "skype", "otpauth", "vcard")):
            return "non_url_scheme"
        if fx.is_valid_url(s):
            return "url"
        return "plain_text"

    # ---- feature extraction & prediction ----
    def predict(self, payload: str) -> dict:
        feat = fx.extract_url_features(payload)
        # Keep only the 20 kept features (dropping the 4 constants)
        X = np.array([[feat[name] for name in self.feature_names]], dtype=float)
        p = float(self.model.predict_proba(X)[0, 1])
        action, reason = self._decide(p)
        return {
            "features": {name: feat[name] for name in self.feature_names},
            "prob_phishing": p,
            "action": action,
            "reason": reason,
        }

    @staticmethod
    def decide(prob: float, t_r: float, t_b: float) -> tuple[str, str]:
        """Map an estimated phishing probability to an advisory action.

        Pure function — the single source of truth for the Allow/Review/Block
        mapping across the pipeline and the Streamlit app. Callers that need a
        threshold policy other than the pipeline's baked-in (t_r, t_b) pass
        those thresholds explicitly.

        Raises ValueError if t_r > t_b or either value is outside [0, 1].
        """
        if not (0.0 <= t_r <= 1.0) or not (0.0 <= t_b <= 1.0):
            raise ValueError(
                f"thresholds must be in [0, 1]; got t_r={t_r}, t_b={t_b}"
            )
        if t_r > t_b:
            raise ValueError(
                f"t_review ({t_r}) must be <= t_block ({t_b})"
            )
        if prob >= t_b:
            return "Block", f"p={prob:.4f} ≥ t_block={t_b:.4f}"
        if prob >= t_r:
            return ("Review",
                    f"t_review={t_r:.4f} ≤ p={prob:.4f} < t_block={t_b:.4f}")
        return "Allow", f"p={prob:.4f} < t_review={t_r:.4f}"

    def _decide(self, p: float) -> tuple[str, str]:
        return self.decide(p, self.t_review, self.t_block)

    # ---- end-to-end ----
    def run(self, image_input) -> dict:
        payload, category = self.decode_image(image_input)
        out = {
            "payload": payload,
            "payload_category": category,
            "features": None,
            "prob_phishing": None,
            "action": None,
            "reason": None,
            "policy": {"t_review": self.t_review, "t_block": self.t_block},
        }
        if category == "url":
            r = self.predict(payload)
            out.update(r)
        elif category == "undecodable":
            out["action"] = "Undecodable"
            out["reason"] = "QR decoder could not find a valid QR code"
        else:
            out["action"] = "Review"
            out["reason"] = f"non-URL payload ({category}); route to human review"
        return out


# ---- CLI ----
def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to QR image")
    ap.add_argument("--policy", default="default", choices=["default", "optimized"])
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()
    p = PredictionPipeline(policy=args.policy)
    r = p.run(args.image)
    print(json.dumps(r, indent=2 if args.pretty else None, default=str))


if __name__ == "__main__":
    _cli()