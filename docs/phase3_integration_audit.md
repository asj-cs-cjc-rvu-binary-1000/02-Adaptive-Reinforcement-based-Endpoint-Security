# Phase 3 Task 0 — Repository and Integration Audit

**Author:** ARES project (DeepSeek output, user-reviewed)
**Date:** 2026-11-20
**Branch:** `update`  ·  **HEAD at audit:** `701aa345315685a300b52fc4076f3f4ea1d11996`
**Model:** `models/raw_xgb.joblib` sha256 `648438e2a2b68a753bd7491862dfc57d604f9d8efeea6f777d75ae4669c26a21` (unchanged)
**Manifest:** `results/model_manifest.json` — check clean at time of audit

---

## 1. Scope and method

Before Phase 3 code changes, the following files were inspected line-by-line:

| File | Role |
|---|---|
| `app.py` | Streamlit interface (four-tab layout) |
| `adaptive_agent.py` | Feedback-driven heuristic threshold controller |
| `calculus_threshold.py` | Read-only adapter exposing `results/calculus_experiment.json` |
| `scripts/calculus_threshold.py` | The Phase 2 optimizer itself (sigmoid-smoothed GD) |
| `scripts/predict_pipeline.py` | **Authoritative inference module** |
| `scripts/feature_extractor.py` | 24-URL-feature extractor (20 kept) |
| `scripts/qr_utils.py` | Offline QR decode + encode |
| `scripts/model_manifest.py` | Fail-closed artifact hasher |
| `train_model.py` | *Not present in this branch; training lives in `scripts/train_models.py`* |
| `requirements.txt` | Pinned to `xgboost==3.0.2`, `scikit-learn==1.9.1` |
| `results/model_manifest.json` | Artifact-and-package pin |
| `run.sh` | One-command launcher |
| `tests/` (9 modules + conftest) | Regression suite (79 tests, all passing) |
| `pytest.ini` | Test configuration |

The audit was performed against on-disk source, not against prior-conversation memory.

---

## 2. Authoritative inference path (confirmed)

`scripts/predict_pipeline.PredictionPipeline` is the **only** place in the project that

1. calls `joblib.load` on `models/raw_xgb.joblib`,
2. calls `model.predict_proba`,
3. calls `feature_extractor.extract_url_features`,
4. applies the frozen 20-feature selection (`self.feature_names`, from `results/pca_report.json`).

`app.py` **imports** `PredictionPipeline` and calls `pipe.run(...)` / `pipe.predict(...)`. It does not load models, call `predict_proba`, or run `feature_extractor` inline. This is enforced by `tests/test_app_integration.py::test_app_has_no_inference_calls_ast`.

After the D1 fix below, `app.py` also no longer re-implements the threshold mapping — `PredictionPipeline.decide(prob, t_r, t_b)` is the single source of truth for Allow/Review/Block.

---

## 3. Frozen contract (confirmed, not modified)

- Model artifact `models/raw_xgb.joblib` — sha256 `648438e2…c26a21`.
- 20 kept features, ordering from `results/pca_report.json → features.kept`.
- 4 constant features dropped pre-inference: `has_https`, `has_ip`, `has_userinfo`, `embedded_url_in_query`.
- Pinned packages: `xgboost==3.0.2`, `scikit-learn==1.9.1`.
- Non-URL QR payloads → `Review`. Undecodable → `Undecodable`. Undecodable and non-URL never `Allow`.
- No network calls to scanned URLs anywhere in the inference chain. Enforced by `tests/test_safety.py::test_no_network_in_inference_chain` (regex over inference source files).

---

## 4. Files requiring modification (Phase 3)

| File | Reason | Resolved? |
|---|---|---|
| `app.py` | 3× duplicated threshold logic (Findings 2, 3); unstable `hash()` sample_id (Finding 11); dead `SAMPLES_DIR` constant (Finding 1) | **Yes — Batch A** |
| `scripts/predict_pipeline.py` | Add public `decide()` classmethod so `app.py` can call the single source of truth | **Yes — Batch A** |
| `run.sh` | Cosmetic: `app.py` not checked before launch (Finding 6); headless note (Finding 7) | **Documented, no code change** |
| `docs/mathematical_techniques.md` | Duplicated `§3.6` heading (Finding 8); provenance footers (Finding 9) | **Deferred to Batch C** |
| `README.md` | Refresh to reflect current pipeline | **Deferred to Batch C** |
| `docs/application_guide.md` | Create | **Yes — Batch A** |
| `docs/testing_guide.md` | Extend with failure modes and gap analysis | **Yes — Batch A** |
| `results/phase3_e2e_test_report.md` | Fill in from Batch B evidence | **Deferred to Batch C** |
| `results/model_manifest.json` | Regenerate after source-file edits | **User runs `python scripts/model_manifest.py` post-Batch-A** |

---

## 5. Files that must **not** be modified

- `scripts/feature_extractor.py` — the 24-feature contract. Any change invalidates the recorded external-validation result (`Mendeley` acc 42.73%, ROC-AUC 0.526).
- `scripts/qr_utils.py` — offline decoder. `tests/test_qr.py` and the record of benign/malicious decode rates depend on its byte-identical behaviour.
- `scripts/calculus_threshold.py` — Phase 2 optimizer. Its `results/calculus_experiment.json` output is cited in `docs/mathematical_techniques.md §3.5`. Modifying it would require re-running the experiment, not a "quick tweak".
- `models/raw_xgb.joblib` — frozen.
- `results/final_metrics.json`, `results/threshold_sweep.json`, `results/calculus_experiment.json`, `results/feature_drift_diagnosis.json` — Phase 2 evidence.
- `data/` — raw dataset pickles.
- `tests/` — the 79-test suite is the contract. New tests may be added, but existing tests must not be weakened to make a fix pass.

---

## 6. Integration findings

Findings from this audit. Each has a resolution.

| # | File / location | Finding | Severity | Resolution |
|---|---|---|---|---|
| 1 | `app.py` line 44 | Dead `SAMPLES_DIR` constant | Cosmetic | **Removed in Batch A** |
| 2 | `app.py` `_run_pipeline_on` | Duplicated Allow/Review/Block logic (was re-implemented inline instead of calling pipeline) | **Contract risk** | **Fixed in Batch A** — calls `PredictionPipeline.decide` |
| 3 | `app.py` `tab_direct_url` | Same duplication in the direct-URL path (a third copy) | **Contract risk** | **Fixed in Batch A** — calls `PredictionPipeline.decide` |
| 4 | `results/heuristic_policy_state.json` and `results/feedback.csv` | Not hashed by `model_manifest.py` | Design — **intentional** | **Documented here (§6.1)** |
| 5 | `adaptive_agent.py` `_load_or_default` | Silent fallback to defaults on malformed state | Cosmetic | Documented; no change |
| 6 | `run.sh` | `app.py` existence not checked before launch | Cosmetic | Documented; no change |
| 7 | `run.sh` | `--server.headless true` means no browser auto-open on a laptop | Environment | Documented in `docs/application_guide.md` |
| 8 | `docs/mathematical_techniques.md` | Duplicated `### 3.6` heading | **Documentation bug** | **Batch C** |
| 9 | `docs/mathematical_techniques.md §3.5.1` | Table values re-derived from `results/calculus_experiment.json` — **match** | Verified | Provenance footer added in Batch C |
| 10 | `app.py` calculus panel | Correctly warns when optimized policy has higher hard cost | ✅ | No change |
| 11 | `app.py` `_render_prediction` | Python `hash()` for `sample_id` is per-process unstable | **Integrity issue** | **Fixed in Batch A** — `hashlib.sha256` |

### 6.1 Deliberate manifest exclusions (D3 resolution)

`results/heuristic_policy_state.json` and `results/feedback.csv` are **deliberately not hashed** by `model_manifest.py`.

Rationale: the manifest's contract is "the frozen artifacts the recorded experimental results depend on". Feedback state is (a) operator-specific, (b) mutated on every feedback click, and (c) never read by the inference path. Hashing it would:
- make `model_manifest.py --check` fail every time a user gives feedback,
- break `run.sh`,
- and falsely imply the model itself drifts with feedback, which it does not.

The compensating control is this document. Anyone reproducing the ARES run should reset the heuristic state to defaults before measuring — `agent.reset_to_default()` via the sidebar, or delete the state file.

---

## 7. Features to preserve

- **Four-tab layout**: samples, upload, camera, direct URL/QR encoder. Every tab currently functional.
- **Sidebar policy selector**: Heuristic / Default / Optimized. All three must remain selectable.
- **Calculus panel**: read-only display of default and optimized thresholds, Hessian diagnostic, and the honest "optimized is worse on this data" warning.
- **Feedback flow**: Correct / False positive / False negative buttons. Must adjust thresholds, must never touch the model.
- **Fail-closed on manifest drift**: `PredictionPipeline.__init__` raises `SystemExit`; `run.sh` checks the manifest first.
- **No-network invariant**: no scanned-URL fetches anywhere.
- **Safe non-URL and undecodable handling**: Review and Undecodable actions respectively.

---

## 8. Confirmations (what Phase 3 does **not** do)

- **No retraining.** `scripts/train_models.py` is not invoked by `app.py`, `predict_pipeline.py`, or any Phase 3 test.
- **No test-set tuning.** Optimized thresholds were fit on validation only (`results/calculus_experiment.json → fit_on: "validation split only"`).
- **No RL terminology.** `adaptive_agent.py` calls itself a "heuristic, feedback-driven threshold policy" and explicitly disclaims MDP/Bellman/Q-learning.
- **No calibration claim.** `prob_phishing` is described everywhere as raw `predict_proba`; not isotonic-calibrated.
- **No calculus-improves-cost claim.** `is_optimized_cost_lower()` returns `False` on this data; both the app panel and `docs/mathematical_techniques.md §3.6` state this.

---

## 9. Sign-off

Findings 1, 2, 3, 11 resolved in Batch A. Findings 8, 9 deferred to Batch C. Findings 4, 5, 6, 7, 10 documented; no code change required. The authoritative inference path is confirmed and will remain `scripts/predict_pipeline.PredictionPipeline` after the D1 fix.