# ARES Testing Guide (Phase 3)

## Quick start
```bash
cd /workspaces/02-Adaptive-Reinforcement-based-Endpoint-Security
source .venv/bin/activate
pytest -q
```

What the suite covers (79 tests)
File	Purpose
tests/test_adaptive_agent.py	Threshold policy, feedback state persistence, no-ML / no-RL guarantees
tests/test_app_integration.py	Streamlit app imports pipeline (AST check), no inline inference, no hard-coded thresholds
tests/test_calculus.py	Defaults 0.35/0.70, optimized 0.60/0.7911, Hessian positive, fit-on-validation only
tests/test_error_handling.py	Missing file / corrupt image / malformed state
tests/test_features.py	24-feature canonical order, 20-feature PCA kept, URL validity
tests/test_manifest.py	Model sha256, pinned packages, fail-closed on drift
tests/test_predict_pipeline.py	Output contract, non-URL → Review, undecodable → Undecodable, no network, determinism
tests/test_qr.py	Decode one benign + one malicious from each dataset, missing file raises
tests/test_safety.py	Safety invariants across the inference chain
Frozen artifact contract
models/raw_xgb.joblib sha256 must remain 648438e2…c26a21.

python scripts/model_manifest.py --check must exit 0.

Never bypass the manifest guard.

Marking a task verified
Only after pasting the pytest output AND the manifest check output — not before.

---

## BRIEF EXPLANATION

- **Failure 1** is a contract mismatch: the test expects `predict_pipeline.run()` to fail *closed* on a missing file, but `decode_image()` was letting `qr_utils.decode_qr_codes` raise through. Wrapping that call restores the documented `Undecodable` behaviour without weakening `test_qr.py::test_missing_file_raises_filenotfound` (the low-level utility still raises).
- **Failure 2** is a false positive in the safety audit: the test forbids the literal substring `from urllib`, but `feature_extractor.py` uses `urllib.parse` (pure string parsing — no sockets, no `urlopen`). Refining the pattern to actual network calls (`urllib.request`, `urlopen`, `requests`, `http.client`, `httpx`, `aiohttp`, `urllib3`, `socket.socket`) keeps the safety contract intact and stops flagging the legitimate parser.
- `docs/testing_guide.md` was scheduled in your previous task list but the `git` command errored, so it never existed.

## 5. Frozen artifact contract (recap)

- `models/raw_xgb.joblib` sha256 **must** remain `648438e2…c26a21`.
- `python scripts/model_manifest.py --check` must exit 0.
- Never bypass the manifest guard.
- `results/heuristic_policy_state.json` and `results/feedback.csv` are **deliberately not hashed** — see `docs/phase3_integration_audit.md §6.1`.

## 6. Failure-mode playbook

### 6.1 Manifest drift fires

`pytest` shows `PredictionPipeline … FATAL: … artifact mismatch` OR
`run.sh` prints `manifest MISMATCH`.

1. Identify the drifted entry in the error message (`scripts__predict_pipeline`, `model_joblib`, etc.).
2. If drift is **intentional** (a source file was edited as part of an approved change):
   - Confirm `sha256sum models/raw_xgb.joblib` still equals `648438e2…c26a21`. If not, **stop** — the model changed, which requires explicit re-approval.
   - Run `python scripts/model_manifest.py` (no flag) to refresh.
   - Run `python scripts/model_manifest.py --check` — must say `manifest OK`.
   - Commit both the source change and the regenerated `results/model_manifest.json` together, in one commit.
3. If drift is **unintentional** (a source file was edited by accident, or an artifact was regenerated):
   - `git restore` the drifted file(s).
   - Re-run `pytest -q` — must be 79 passed.

### 6.2 A test skips

Skips are acceptable only for **missing local data** — never for model, manifest, or source integrity. Current skip sources:

- `trad_chehab_paths` — skips if `data/raw/qr_codes_29*.pickle` is missing.
- `mendeley_paths` — skips if `data/raw/qr-images/{benign,malicious}/` is missing.
- `manifest_path` — skips if `results/model_manifest.json` is missing.

If any of these skips fire on a machine that is meant to be a **release** machine, treat it as a failure — the artifacts are required.

### 6.3 Model hash check fails after an unrelated edit

Never happens if you follow rule 6.1. If it does, restore from git (`git checkout models/raw_xgb.joblib`) and re-run `pytest -q`. Do **not** re-run any training script.

## 7. What the 79 tests do **not** cover

The suite is dense at the unit / integration / safety level but has explicit gaps:

| Gap | Why | Mitigation |
|---|---|---|
| Streamlit rendering (widgets, columns, HTML injection) | Pytest does not run a browser | `results/phase3_e2e_test_report.md` (manual walkthrough) |
| Camera hardware path | Requires a real camera | Documented in `docs/application_guide.md §3.3` |
| `run.sh` end-to-end launch | Shell script, not importable | Manual test in e2e report row 1 |
| Feedback CSV accumulation across restarts | Needs two process launches | Manual test in e2e report rows 11, 14 |
| `results/heuristic_policy_state.json` persistence across restarts | Same | Manual test in e2e report row 14 |
| Duplicated threshold logic in `app.py` | AST tests confirm imports, not internal structure | **Fixed in Batch A** — single source of truth via `PredictionPipeline.decide()`; tests can now assert the classmethod's boundaries |

## 9. Cosmetic warnings you will see (and can ignore)

These are environment or framework notices, not failures. None affect test
results or application behaviour.

| Warning | Source | Meaning |
|---|---|---|
| `Inotify queue overflowed; filesystem events were lost` | GitHub Codespace watcher | The Codespace filesystem watcher lost events during Streamlit's hot-reload polling. Streamlit re-reconciles on its own. Harmless. |
| `Please replace use_container_width with width` | Streamlit ≥1.30 deprecation | `st.dataframe(..., use_container_width=True)` is deprecated in favour of `width="stretch"`. The old form still works until 2025-12-31; the framework says so in its own message. Cosmetic. A future commit can migrate the parameter when Streamlit's removal date approaches. Tracked in `docs/phase3_completion_report.md` under "Known follow-ups". |
| `Uncaught app execution` following a traceback in a script you *just edited* | Streamlit caching | Streamlit's script runner shows the last error until the next file save triggers a rerun. Saving the file (or pressing `R` in the browser) clears it. |

None of these require changes for Phase 3 completion.