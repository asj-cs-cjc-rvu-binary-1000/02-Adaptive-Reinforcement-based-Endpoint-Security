# ARES Application Guide

**Version:** Phase 3 (2026-11)
**Model:** `models/raw_xgb.joblib` (frozen, sha256 `648438e2…c26a21`)
**Entry point:** `app.py` (Streamlit)

---

## 1. What this app is (and is not)

ARES is an **offline** QR-phishing risk analyser. It decodes a QR image, inspects
the payload, and — when the payload is a URL — extracts 20 lexical/structural
features and returns an **estimated** phishing probability from a frozen XGBoost
model. Based on that probability and two thresholds `(t_review, t_block)` it
displays an advisory action: **Allow**, **Review**, **Block**, or **Undecodable**.

ARES is **not**:

- A live threat-intelligence scanner. It does **not** visit or fetch URLs.
- A browser sandbox. It has no JavaScript engine and does not render pages.
- A formal reinforcement-learning agent. The threshold policy is a **heuristic,
  feedback-driven controller** with two scalars — not an MDP, not Q-learning.
- A guarantee. `Allow` means "the frozen model assigned a low estimated
  probability under the current threshold policy". It is advisory, not a safety
  certification. HTTPS alone is not evidence of legitimacy.

---

## 2. Launching

From the repository root:

```bash
bash run.sh
```

run.sh will:

Prefer .venv/bin/python if present; otherwise fall back to python3 from PATH.

Check that models/raw_xgb.joblib and the four results/*.json artifacts exist.

Run python scripts/model_manifest.py --check — fails closed on drift.

Check that streamlit is importable.

Launch Streamlit on port ${ARES_PORT:-8501}.

Typical session:

$ bash run.sh
Using .venv/bin/python
Verifying model manifest…
manifest OK — artifacts and packages match
Launching ARES (Streamlit)…
  You can now view your Streamlit app in your browser.
  URL: http://localhost:8501

Note on --server.headless true: in a GitHub Codespace (or any remote
environment), Streamlit does not auto-open a browser; you forward the port
or click the Codespace's "Open in browser" prompt. On a laptop running locally,
open http://localhost:8501 manually. This is intentional — it avoids
auto-opening a browser from a remote shell.

If run.sh fails at the manifest check, do not delete the manifest or
the model. The manifest is the fail-closed guard. See §7 below.

3. The four tabs
3.1 1-Click Sample Gallery
Displays the first two benign and first two malicious samples from
data/raw/qr_codes_29.pickle (Trad & Chehab dataset). Each thumbnail has an
Analyse this sample button that routes the decoded image through the
authoritative PredictionPipeline.run() — same code path as any uploaded file.

Requires: data/raw/qr_codes_29.pickle and data/raw/qr_codes_29_labels.pickle.

3.2 Upload QR
Accepts png, jpg, jpeg up to 5 MB. The uploaded image is converted to
grayscale, displayed, and passed to PredictionPipeline.run().

What you'll see: decoded payload, payload category (url, plain_text,
non_url_scheme, undecodable, empty), advisory action, and — if URL —
the estimated phishing probability, URL anatomy (scheme, host, registrable
domain, path, query, port), and the 20 kept features.

3.3 Camera Snapshot
Uses st.camera_input — requires a device with a webcam or a browser that
supports camera access. Same downstream path as Upload.

3.4 Direct URL Analyzer & QR Encoder
Two utilities:

Analyse URL — runs PredictionPipeline.predict(url) on a URL string
directly, bypassing QR decoding. Same feature extraction and same frozen
model. Result rendering is identical to Upload.

Encode to QR PNG — generates a QR image from the input text using
qr_utils.generate_qr_image(). Downloads as PNG. This is a testing
utility, not a safety filter: encoding an unsafe URL does not make it safe.

4. The sidebar
4.1 Policy selector
Three radio options, all using the same frozen models/raw_xgb.joblib:

Option	Thresholds	Where they come from
Heuristic policy (feedback-driven)	Live state	results/heuristic_policy_state.json, updated by feedback
Default (0.35, 0.70)	(0.35, 0.70)	results/calculus_experiment.json → default_thresholds
Optimized (calculus)	(0.60, 0.7911…)	results/calculus_experiment.json → optimized_thresholds
The active thresholds are shown in a caption: t_review = … · t_block = … · policy = ….

4.2 Reset heuristic policy
Visible only when the heuristic policy is active. Resets
results/heuristic_policy_state.json to defaults (0.35, 0.70) and reruns the page.

4.3 Calculus panel
Read-only display of the Phase 2 threshold-optimization results:

Default and optimized thresholds.

Hessian determinant and is_minimum flag.

boundary_hit and is_kkt_constrained flags.

A green success banner only when is_optimized_cost_lower() returns True.

Otherwise (as on this dataset): a warning banner —

Optimized policy has higher hard cost than default on this dataset. The
calculus result demonstrates the optimizer machinery, not a policy
improvement.

That warning is correct. The optimized policy has higher hard cost on both
validation and test splits. It is not a bug, and the app must not be edited to
hide it.

5. Feedback
Below every URL prediction are three buttons:

✓ Correct — records n_correct += 1; no threshold change.

⚠ False positive — records n_false_positive += 1; raises both thresholds by 0.01.

✗ False negative — records n_false_negative += 1; lowers both thresholds by 0.01.

Every click:

Appends a row to results/feedback.csv (audit log).

Persists results/heuristic_policy_state.json.

Emits a small success toast — nothing else.

The frozen XGBoost model is never retrained, mutated, or re-serialized by
feedback. This is enforced by the absence of any joblib.dump, fit(, or
predict_proba call inside adaptive_agent.py, and by
tests/test_adaptive_agent.py::test_adaptive_agent_imports_no_ml.

Sample ID stability: the sample_id in feedback.csv is derived from
hashlib.sha256(payload | probability) — stable across app restarts. Two
feedback events on the same payload + probability will share an app:<hash>
prefix.

6. Probability and calibration
prob_phishing is the raw predict_proba output of models/raw_xgb.joblib.
It is not isotonic-calibrated. Brier scores on validation and test (0.0880,
0.0841) describe empirical behaviour; they are not a formal calibration
claim.

If ARES ever ships a calibrated model, that will be a separately documented
experiment with its own model artifact, manifest entry, and validation metrics.

7. Troubleshooting
Symptom	Cause	Fix
run.sh: manifest MISMATCH: scripts__predict_pipeline: sha256 changed	A source file was intentionally edited	Run python scripts/model_manifest.py (no flag) to refresh. Verify models/raw_xgb.joblib sha256 remains 648438e2…c26a21. Commit both.
run.sh: required artifact missing: models/raw_xgb.joblib	Model artifact absent	Restore from git. Never retrain as a workaround.
run.sh: streamlit not installed	venv missing the package	pip install -r requirements.txt in the venv.
Sidebar calculus panel: Calculus results unavailable: …	results/calculus_experiment.json missing or malformed	Restore from git. Do not re-run optimization as a shortcut — the JSON is Phase 2 evidence.
Sample gallery: Sample gallery requires data/raw/qr_codes_29*.pickle	Dataset not downloaded	See docs/testing_guide.md for the dataset acquisition steps.
Camera tab shows no image	Browser blocked camera or no webcam	Grant camera permission, or use Upload / Sample tabs.
Feedback state stuck	results/heuristic_policy_state.json corrupt	Click Reset heuristic policy in the sidebar, or delete the file — the agent falls back to defaults safely.
PredictionPipeline failed closed: FATAL: … artifact mismatch when launching direct	Manifest drift caught by pipeline at runtime	Same fix as the first row.
8. Reproducibility
Any result produced through this app can be reproduced with the pytest suite
(pytest -q — 79 tests, deterministic) plus the manual scenarios recorded in
results/phase3_e2e_test_report.md. The model artifact, feature order, and
threshold policy are all hashed and pinned. Nothing in this app depends on
network access at inference time.