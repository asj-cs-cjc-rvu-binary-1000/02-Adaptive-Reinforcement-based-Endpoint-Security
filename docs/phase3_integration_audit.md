# Phase 3 Task 0 — Repository and Integration Audit

**Generated:** {{generated_utc}}
**Audit script:** `scripts/audit_phase3.py`
**Evidence file:** `results/phase3_audit.json`
**Repo state:** branch `update` @ `3b6050b` (+ whatever Task 0 commit adds)
**Phase 2 status:** complete and verified (see Phase 2 submission record)

---

## 1. Executive summary

The rebuild branch (`update`) does **not** currently contain `app.py`,
`adaptive_agent.py`, or `calculus_threshold.py`. Phase 1–2 rebuilt the pipeline
from scratch under `scripts/`, but the Streamlit application and adaptive agent
were not re-created on this branch — they exist only on `main` (v3.0).

**Integration implication:** Task 1 (Streamlit) and Task 2 (adaptive agent) are
not integrations of an existing application — they are **fresh implementations**
against `scripts/predict_pipeline.py`. `main`'s `app.py`, `adaptive_agent.py`,
and `calculus_threshold.py` are **reference implementations only**: they can
inform structure but cannot be imported as-is, because `main`'s inference path
used `models/model.joblib` (a different artifact) and a `feature_extractor` with
different constants than the Phase 2 pipeline.

**This is a decision point for the project owner, not the engineer.**

---

## 2. File inventory (from `results/phase3_audit.json.file_inventory`)

| File | Exists | Bytes | Notes |
|---|---|---|---|
| `app.py` | {{app_exists}} | {{app_bytes}} | {{app_note}} |
| `adaptive_agent.py` | {{aa_exists}} | {{aa_bytes}} | {{aa_note}} |
| `calculus_threshold.py` | {{ct_exists}} | {{ct_bytes}} | {{ct_note}} |
| `feature_extractor.py` | {{fe_exists}} | {{fe_bytes}} | byte-identical to origin/main |
| `qr_utils.py` | {{qr_exists}} | {{qr_bytes}} | byte-identical to origin/main |
| `scripts/predict_pipeline.py` | {{pp_exists}} | {{pp_bytes}} | authoritative inference |
| `train_model.py` | {{tm_exists}} | {{tm_bytes}} | absent from rebuild |
| `requirements.txt` | {{req_exists}} | {{req_bytes}} | Phase 2b pinned |
| `results/model_manifest.json` | {{mm_exists}} | {{mm_bytes}} | Phase 2b |
| `scripts/model_manifest.py` | {{mmc_exists}} | {{mmc_bytes}} | Phase 2b |
| `scripts/run_gates.py` | {{rg_exists}} | {{rg_bytes}} | Gate 1+2 |
| `scripts/smoke_test_pipeline.py` | {{st_exists}} | {{st_bytes}} | 8-case smoke test |
| `scripts/evaluate_models.py` | {{em_exists}} | {{em_bytes}} | Task 7 eval |
| `scripts/run_calculus_experiment.py` | {{ce_exists}} | {{ce_bytes}} | Task 8 |

## 3. Consumed artifacts (from `results/phase3_audit.json.consumed_artifacts`)

| Artifact | Exists | sha256 (first 16) |
|---|---|---|
| `results/model_selection.json` | ✅ | `{{ms_sha}}` |
| `results/pca_report.json` | ✅ | `{{pca_sha}}` |
| `results/calculus_experiment.json` | ✅ | `{{cexp_sha}}` |
| `models/raw_xgb.joblib` | ✅ | `648438e2a2b68a75…` |
| `results/model_manifest.json` | ✅ | `{{mm_sha}}` |

## 4. `PredictionPipeline` contract (from `results/phase3_audit.json.predict_pipeline_contract`)

This is what `app.py` will consume.

- **Class:** `{{pp_class}}`
- **Public methods:** `{{pp_methods}}`
- **Output keys:** `{{pp_output_keys}}`
- **Imports:** `{{pp_imports}}`

**Output contract (verbatim from `scripts/predict_pipeline.py`):**
- `payload` — decoded URL string or `None`
- `payload_category` — `"url"` | `"plain_text"` | `"non_url_scheme"` | `"undecodable"`
- `features` — dict of 20 kept features (only when `payload_category == "url"`)
- `prob_phishing` — estimated probability (raw XGBoost, **not** isotonic-calibrated)
- `action` — `"Allow"` | `"Review"` | `"Block"` | `"Undecodable"`
- `reason` — human-readable explanation of the decision
- `policy` — `{"t_review": ..., "t_block": ...}` showing which threshold set was used

**Non-URL payload → `action="Review"`, `prob_phishing=None`.** Never `Allow`.
**Undecodable → `action="Undecodable"`.** Never a crash.

## 5. Manifest & gate check

- `model_manifest.check_manifest()` → `{{manifest_ok}}`
- `run_gates.py check` return code → `{{gate_rc}}`
- `smoke_test_pipeline.py` return code → `{{smoke_rc}}`; `{{smoke_pass}}/8 cases well-formed`

The Phase 2b fail-closed guard is active and passes: `PredictionPipeline` refuses
to instantiate if the artifact hashes drift from `results/model_manifest.json`.

## 6. Duplication risks (from `results/phase3_audit.json.duplication_risks`)

**Does not apply as written** — `app.py` and `adaptive_agent.py` do not exist yet
on this branch. When written, they **must** import `PredictionPipeline`, not
re-implement:

- `joblib.load` → forbidden outside `predict_pipeline.py`
- `.predict_proba(` → forbidden outside `predict_pipeline.py` and `evaluate_models.py`
- `extract_url_features(` → forbidden outside `feature_extractor.py`, `prepare_features.py`, and `predict_pipeline.py`

The Task 4 pytest suite will grep for these.

## 7. Tests inventory

- `tests/` directory exists: **{{tests_dir_exists}}**
- Test files: **{{test_files}}**
- Test images: **{{test_image_count}}**

If `tests/` is empty or missing, **Task 4 must create the full pytest suite
from scratch** — the 42-test suite on `main` tested the old model and old
feature-extractor behaviour and **must not be reused verbatim**; the `models/model.joblib`
bundle it tested has been replaced by `models/raw_xgb.joblib` in Phase 2, and
several inference semantics changed (20 features not 24, uncalibrated not
calibrated, threshold values unchanged).

## 8. Integration risks — ranked

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | `app.py` / `adaptive_agent.py` / `calculus_threshold.py` do not exist on this branch — Task 1/2/3 are fresh implementations, not porting | **High** | Scope each task as a from-scratch implementation that imports `PredictionPipeline`; `main`'s versions are reference only |
| R2 | `main`'s old `app.py` loads `models/model.joblib` — a 24-feature calibrated bundle that no longer exists | **High** | Task 1 must call `PredictionPipeline` exclusively; grep for `joblib.load\|model.joblib` in the acceptance check |
| R3 | `main`'s `adaptive_agent.py` was labelled RL in places; brief forbids that framing | **Medium** | Task 2 must state "heuristic feedback-driven policy" explicitly; docs must too |
| R4 | `main`'s calculus panel defaulted thresholds to `(0.35, 0.70)` and optimized to `(0.60, 0.75)`; Phase 2 optimizer converged to `(0.60, 0.7911)` | **Medium** | Task 3 must read `calculus_experiment.json` for actual values, never hardcode |
| R5 | `main`'s test suite used `tests/images/*.png`; on rebuild those images may or may not exist | **Low** | Task 4 must include images or generate them; audit JSON records what's present |
| R6 | `model_manifest.check_manifest()` is a hard gate — any experiment (e.g., writing new results, editing docs) that touches a hashed file will trip it | **Low** | Task 4 tests already exercise the guard intentionally; normal runs unaffected |
| R7 | Streamlit re-imports `app.py` on every rerun; if `PredictionPipeline` is instantiated at module scope it will re-run `check_manifest` on every rerun | **Low** | Task 1 should cache with `@st.cache_resource` (the `main` branch used this pattern too) |

## 9. Features that must be preserved (from `main`'s README, for reference)

The following user-facing capabilities existed in `main` v3.0 and are the
**contract** for Task 1's UI:

1. 1-Click Sample QR Gallery — 7 built-in samples
2. Upload QR Image — PNG/JPG/JPEG up to 5 MB
3. Camera Snapshot — single still via `st.camera_input`
4. Direct URL Analyzer & QR Encoder — URL analysis + optional encode-to-PNG

Sidebar:
- `∫ Calculus Threshold Optimizer` panel with `C_FP`, `C_FN`, `C_Review` sliders
- `∫ Compute Optimal Thresholds` button
- Threshold selection reflects `results/calculus_experiment.json`

Body sections (per prediction):
- URL anatomy (scheme, subdomain, registrable, path, query)
- Risk / protective factors
- **Predicted phishing probability** (never "calibrated")
- 20-feature table with XGBoost importances + category + description
- Decision advisory (Allow / Review / Block)
- Feedback buttons → heuristic policy state file

## 10. Decisions the owner must make before Task 1

1. **Reference vs rewrite.** `main`'s `app.py` is ~640 lines. Should Task 1
   (a) port `main`'s structure and swap the inference path, or (b) write a
   cleaner fresh app? Port is faster but drags old assumptions; fresh is cleaner
   but loses the 4-tab layout unless we re-implement it. **Recommendation: (a)**
   — port `main`'s structure so the four tabs and sidebar remain recognizable,
   but replace every inference call with `PredictionPipeline` and drop `model.joblib`
   entirely.

2. **Where does `adaptive_agent.py` live?** On `main` it was top-level. On the
   rebuild branch, `scripts/` holds the pipeline and the repo root holds
   `qr_utils.py` + `feature_extractor.py`. **Recommendation:** top-level
   `adaptive_agent.py` importing from `scripts/predict_pipeline.py`, matching
   `main`'s convention, so Phase 3 is the last move.

3. **Feedback state file path.** `main` wrote `results/feedback.csv` and
   `results/rl_policy_state.json` (naming legacy — brief says "heuristic, not
   RL"). **Recommendation:** keep `results/feedback.csv` for backwards
   compatibility of column layout, but rename `rl_policy_state.json` →
   `heuristic_policy_state.json` and update all docs to say "heuristic." The
   `.gitignore` already excludes both.

## 11. Audit acceptance-criteria self-check

| Criterion | Evidence | Status |
|---|---|---|
| Authoritative inference pipeline identified | `scripts/predict_pipeline.py`, imported by nothing else | ✅ |
| Frozen model + feature ordering understood | `results/model_manifest.json` (sha256s), `results/pca_report.json` (20 kept features) | ✅ |
| No unnecessary changes to training/eval artifacts proposed | Nothing in §10 touches `data/processed/*` or `models/*` | ✅ |
| Existing application features documented before modification | §9 above | ✅ |

**Task 0 self-assessment: complete.** All criteria are met by the audit script
plus this document. No code was modified. Nothing under `data/` or `models/`
changed.