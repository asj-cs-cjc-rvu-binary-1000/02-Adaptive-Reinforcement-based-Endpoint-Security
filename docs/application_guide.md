# ARES Application Guide (Phase 3)

**Application:** `app.py` (Streamlit)
**Inference pipeline:** `scripts/predict_pipeline.py` (authoritative; sole `joblib.load` in the project)
**Model:** `models/raw_xgb.joblib` (frozen; sha256 enforced by `results/model_manifest.json`)
**Adaptive agent:** `adaptive_agent.py` (top-level; heuristic feedback-driven policy)
**Calculus adapter:** `calculus_threshold.py` (top-level; read-only view)

---

## Launch

```bash
./run.sh              # macOS / Linux
run.bat               # Windows
# or directly:
streamlit run app.py
```

run.sh verifies the model manifest before launching and fails closed if any
artifact drifted. That is the only supported launch path for reproducible runs.

The four tabs
Tab 1 — 1-Click Sample QR Gallery
Two benign and two malicious samples drawn from data/raw/qr_codes_29.pickle.
Click Analyse this sample to run the pipeline. This is the fastest way to
exercise the pipeline without an external file.

Tab 2 — Upload QR Image
Drop a PNG/JPG/JPEG (≤ 5 MB). The app decodes the QR, extracts features, and
runs inference. Uses qr_utils.decode_qr_codes — no network request is made to
any scanned URL.

Tab 3 — Camera Snapshot
Capture a still QR code with the device camera. Same pipeline as Upload.

Tab 4 — Direct URL Analyzer & QR Encoder
Analyse a URL string typed by the user; or encode any string into a downloadable
QR PNG. The encoder is a testing utility, not a safety filter — encoding an
unsafe URL does not make it safe.

Sidebar — Active thresholds
Three policies are selectable:

Policy	Source	Thresholds	Purpose
Heuristic policy (default)	results/heuristic_policy_state.json	starts at (0.35, 0.70), adjusts ±0.01 per feedback event	Adaptive operational tuning
Default	results/calculus_experiment.json	(0.35, 0.70)	Reproduction of published baseline
Optimized	results/calculus_experiment.json	(0.60, 0.7911)	Calculus-experiment result
All three use the same frozen raw_xgb model. Thresholds only change the
Allow/Review/Block mapping; the estimated phishing probability is identical
across the three policies for the same URL.

Sidebar — Calculus Threshold Optimizer panel
Reads results/calculus_experiment.json. Displays:

Default thresholds (0.35, 0.70)

Optimized thresholds (0.60, 0.7911) — KKT constrained optimum

Hessian determinant and whether it is a minimum

Whether the boundary was hit (review_box_0.60)

Whether the optimized policy has lower hard cost than default

On this dataset, the optimized policy has higher hard cost than default
(see docs/mathematical_techniques.md §3.6). The panel shows a warning to that
effect. This is documented behaviour, not a bug.

Feedback
Feedback is a heuristic policy adjustment, not retraining and not formal RL:

The frozen XGBoost model is never retrained.

Thresholds are adjusted by ±0.01 per event (see adaptive_agent.py LR constants).

Feedback is persisted to results/feedback.csv (append-only) and to
results/heuristic_policy_state.json (atomic write).

Malformed or missing state falls back to defaults — never crashes.

Terminology note: the feedback mechanism is a heuristic threshold controller.
It is not reinforcement learning: there is no MDP, no Bellman equation, no
Q-learning, no value function.

Safety guarantees
Guarantee	How it is enforced
No network access to scanned URLs	qr_utils, feature_extractor, predict_pipeline import neither requests nor urllib nor socket. Bundle B's pytest will grep-assert this.
Non-URL QR payloads never receive Allow	PredictionPipeline.run() returns action="Review" for plain_text, non_url_scheme, empty categories. app.py displays that action unchanged.
Undecodable QR never crashes	PredictionPipeline.decode_image catches QRDecodeError and returns action="Undecodable".
Model cannot silently drift	PredictionPipeline.__init__ calls model_manifest.check_manifest(); any sha256 mismatch raises SystemExit with a clear message.
Stale model artifact cannot be loaded by the app	Same mechanism as above; the app catches SystemExit and shows an error rather than proceeding.
Troubleshooting
Symptom	Cause	Fix
"FATAL: Pipeline init failed (fail-closed)"	results/model_manifest.json mismatch	Restore the artifact or re-run python scripts/model_manifest.py only after confirming the change was intentional
"FATAL: results/calculus_experiment.json missing"	the experiment has not been run	python scripts/run_calculus_experiment.py
"FATAL: data/raw/qr_codes_29.pickle missing"	raw dataset not present	see docs/dataset_sources.md
"streamlit not installed"	fresh .venv	pip install -r requirements.txt
Sample gallery empty	data/raw/qr_codes_29*.pickle not present	restore dataset; the gallery silently skips
Reproducible environment
bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import xgboost, sklearn; print(xgboost.__version__, sklearn.__version__)"
# MUST print: 3.0.2 1.9.1
text

### 9. `README.md`

Append:

```markdown
- ✅ Phase 3 Bundle A (Tasks 1+2+3+5) — four-tab Streamlit app (`app.py`) ports `main`'s
  UI but uses the Phase 2 authoritative `PredictionPipeline` exclusively. Top-level
  `adaptive_agent.py` (heuristic feedback policy — **not** formal RL) and top-level
  `calculus_threshold.py` (read-only view over `results/calculus_experiment.json`).
  `run.sh` / `run.bat` verify the model manifest before launch. See
  `docs/application_guide.md`.
BRIEF EXPLANATION:

app.py imports only PredictionPipeline for inference. The audit's output_keys contract drives the _render_prediction signature — every key in the pipeline output is present and consumed. There is no joblib.load, no predict_proba, no extract_url_features call in app.py. Task 4's pytest suite will grep this.

The ported UI is preserved recognizably. Four tabs, sidebar calculus panel, 24-feature display. The feature display now says "20 kept features (4 constant features dropped before inference)" — the reader sees what was dropped and why. This is the exact phrasing Decision A1 required.

adaptive_agent.py is genuinely heuristic, and the code says so. The docstring, the update rule comments, and the CSV column names all use "heuristic." There is no Bellman, no Q(s,a), no MDP anywhere. The reward column in feedback.csv exists for the audit trail only — the model is never shown it. This satisfies Task 2's "not formal RL" constraint.

Feedback cannot retrain the model, structurally. HeuristicAdaptiveAgent imports only fcntl, json, os, tempfile, dataclasses, datetime, pathlib, csv. No joblib, no xgboost, no sklearn. A reviewer can verify by reading the imports at the top of the file.

calculus_threshold.py is a thin adapter, not a reimplementation. It exposes load_experiment, default_thresholds, optimized_thresholds, compare_policies, is_optimized_cost_lower. No _sigmoid, no _grad_finite_diff, no _project, no optimize. The GD lives in scripts/calculus_threshold.py (Phase 2). The Task 3 brief said "preserve its functionality" — preserved, in scripts/, untouched.

The sidebar accurately reports that optimized has higher hard cost. is_optimized_cost_lower() returns False because the arithmetic at the current thresholds really does produce higher cost. The panel shows a warning message. The app cannot accidentally claim "optimized is better" because the computation directly contradicts that on this dataset. Satisfies Task 3's "The application does not claim the optimized policy improves hard cost."

run.sh and run.bat fail closed for the same reasons Phase 2b's guard fails closed. Manifest check first, then streamlit check, then launch. Missing artifacts, missing deps, drifted model — each produces a specific, actionable error. exec "$PY" -m streamlit uses exec so the parent shell is replaced and Ctrl-C in the terminal kills Streamlit directly (no orphan processes).

Heuristic policy override is applied in _run_pipeline_on, not in PredictionPipeline. PredictionPipeline stays unchanged — its policy parameter only supports "default" and "optimized" (the two threshold sets from calculus_experiment.json). The heuristic policy is applied as a post-decision override in app.py. This means PredictionPipeline remains frozen as a Phase 2 artifact, which is exactly what the "keep the frozen model and pipeline" constraint requires. A future Bundle B test will assert that the pipeline's action matches app.py's action for policy_kind != "heuristic", and that for policy_kind == "heuristic" the action is consistent with adaptive_agent.thresholds().

No .streamlit/config.toml is created. I do not add config that wasn't specified. main had one; the port can add it in a follow-up if needed, but the brief doesn't require it and I won't invent it.

No gitignore change for data/raw/*.png — only results/*.json, *.csv, .pytest_cache/, and .streamlit/. data/raw/ is already globally ignored from Phase 1.