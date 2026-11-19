# Experimental Protocol (Phase 1 — Task 5)

**Status:** Draft — pre-filled with fixed rules from Phase 1 measurements; primary/external-dataset slots are placeholders pending `dataset_decision.md` §5.
**Purpose:** One unambiguous, reproducible specification for Phase 2 model training, so the split, seeds, and metrics are locked **before** a single model is fit.

---

## 1. Datasets and role assignment

| Slot | Assigned to | When filled |
|---|---|---|
| Primary training set | ⬜ Trad & Chehab **or** ⬜ Mendeley | §5 of `dataset_decision.md` |
| External validation set | ⬜ Mendeley **or** ⬜ None | same |

Where the protocol below says "primary", it applies to whichever dataset is picked. External-validation-specific rules are in §7.

## 2. Random seed and reproducibility

- **Global `RANDOM_SEED = 42`** for: split generation, model `random_state`, XGBoost subsample, and preprocessing shuffles.
- **`numpy.random.default_rng(42)`** for any NumPy-side shuffling.
- Every reported metric must be reproduced by re-running a single documented command; deviations require updating this document.
- Artifact checksums are recorded in `results/*.json` — a metric cannot be cited in the manuscript without a matching JSON.

## 3. Split ratios (primary dataset)

**Grouped by registrable domain, not by image.** Any split that puts two images from the same registrable domain into different splits leaks the domain identity across train/val/test and inflates metrics. This is a hard requirement.

- **Train : Val : Test = 70 : 15 : 15** by **registrable-domain groups**, using `sklearn.model_selection.GroupShuffleSplit` (or `GroupKFold` for cross-validation) with `groups = extract_registrable_domain(decoded_payload)`.
- **Registrable-domain extraction** uses the byte-identical `feature_extractor.extract_registrable_domain` from `scripts/feature_extractor.py` (sha256 `45e13436…`). No re-implementation, no shortcuts.
- **Every image whose decoded payload does not yield a registrable domain** (undecodable, non-URL schemes, empty) is **excluded from the split** and counted separately in a "pipeline-rejects" table for the manuscript.

## 4. Duplicate-removal rules (applied before split)

Applied to the **decoded payload**, not the image — two images encoding the same URL are one sample.

1. **Hash the decoded URL payload** with `sha256(payload.encode()).hexdigest()`.
2. **Within each label class**, drop later occurrences, keeping the first by deterministic `(sort_by_filename_sha256)` order. Result: no two training samples share an exact URL.
3. **Across labels** — if the same payload appears as both benign and malicious, that is a **labelling contradiction**. Drop it from both classes and add it to a `results/label_conflicts.json` audit file. **Do not** keep the majority vote; a contradiction is a defect, not a training sample.

Sample-level measurement in Phase 1 found **0 duplicates within either class, 0 overlaps across classes** on the 200/class sample. **Full-dataset application of these rules is a Phase 2 pre-training gate (§8).**

## 5. Class balance

- Trad & Chehab: 5,005 benign / 4,982 malicious — **already balanced (1.005:1)**. No resampling.
- Mendeley: 500 / 500 — already balanced. No resampling.
- **No SMOTE, no class weighting, no oversampling.** The dataset is balanced; introducing resampling would be an unjustified degree of freedom.

## 6. Grouping strategy (hard rule)

- **Group = registrable domain.** Not TLD, not host (subdomain is *not* enough — the split must be subdomain-invariant), not IP.
- **Precondition for any split:** `set(train_domains) ∩ set(val_domains) ∩ set(test_domains) == ∅`. Enforced by an assertion in the Phase 2 training script; failure aborts the run.
- **Repeated cross-validation** for model selection: **5-fold GroupKFold** on the training split; the holdout test set is touched **once** at the end.
- **Sample-level coverage warning:** the Phase 1 overlap check was 200/class only. The **full-dataset overlap check is §8 gate #1** — no split will be generated before it passes.

## 7. External validation protocol

**If `dataset_decision.md` §5 chooses Mendeley as external validation:**

- **Never** merge Mendeley into the training, validation, or test splits — it is **only** touched after the primary-trained model is frozen.
- Metrics reported on the external set are recorded in a **separate section** of the results, labeled "External validation — not used for model selection or tuning."
- **Mendeley resolution normalization:** before external validation, resize every image so the **short side = 512 px** using `PIL.Image.resize(..., Image.LANCZOS)`. This neutralizes the documented size-class confounding in Mendeley (`dataset_comparison.md` §"Mendeley").
- **Do not tune `C_FP`/`C_FN`/`C_Review` or the review/block thresholds on the external set** — that is evaluation, not tuning.

**If `dataset_decision.md` §5 chooses "None":**

- External validation must be substituted by acquiring **CIC-Trap4Phish** (deferred candidate) before manuscript submission. The IEEE Access reviewer will require it for a phishing detector.
- Until acquired, the manuscript reports only primary-dataset metrics with an explicit limitation statement: *"no independent external validation; interpret reported generalisation cautiously."*

## 8. Phase 2 pre-training gates (both must PASS before any model is trained)

**Gate 1 — full-dataset duplicate + domain-overlap audit.**
Re-run the same duplicate-payload and domain-overlap logic used in `scripts/check_qr_compatibility.py`, but on **the full dataset** (not 200/class). Outputs `results/full_overlap_audit.json`.
- PASS if: within-class duplicates = 0 **after** §4 deduplication, cross-class payload conflicts = 0 **after** §4 filtering, and any resulting excluded counts are recorded.
- FAIL blocks training. If FAIL, resolve per §4 and re-audit.

**Gate 2 — full-dataset decode confirm.**
Re-run decode on **all** primary images (not 200/class), the same way `scripts/check_qr_compatibility.py` does. Outputs `results/full_decode_audit.json`.
- PASS if decode rate ≥ **0.95** in **both** classes (Phase 1 measured 0.990 / 0.975 on 200/class; a full-dataset drop below 0.95 would signal a specific subgroup the sample missed).
- FAIL blocks training. If FAIL, drill into failing images and either fix preprocessing or record the affected subgroup as a documented exclusion.

**Runtime expectation for Gate 2:** Trad & Chehab at 9987 images × ~0.29 s ≈ **~48 min**. Schedule accordingly (background job or overnight). Do not reduce the sample "to save time" — a sampling shortcut on a gate defeats the gate.

## 9. Metrics to report (fixed before training)

All metrics computed on the **holdout test set** of the primary dataset, and separately on the external validation set if chosen.

| Category | Metric | Why |
|---|---|---|
| Discrimination | **ROC-AUC**, **PR-AUC** | ROC for balance-free discrimination, PR for reviewer-expected phishing-detector sensitivity |
| Calibration | **Brier score**, **reliability diagram** | ARES uses thresholds `Review`/`Block` on calibrated probabilities — miscalibration directly harms operations |
| Point-operating | **Precision, Recall, F1 at threshold 0.50** | Convention for comparability with prior work |
| Operational | **FPR, FNR, Review-rate** at the **primary-policy thresholds** (`0.35/0.70` heuristic baseline and, separately, `0.60/0.75` calculus-optimised) | The calculus module in `main` optimises for cost, and this is the metric that actually reflects operational cost |
| Cost | **`C(t_r,t_b)`** under default `C_FP=1.0, C_FN=5.0, C_Review=0.5` | Direct comparison to Phase-2 calculus policy |
| Per-class breakdown | confusion matrix (TN/FP/FN/TP) | Required for the manuscript |
| Robustness | **Decision-stability across seeds** (5 runs of `RANDOM_SEED ∈ {42, 43, 44, 45, 46}`) | Single-seed metrics are unreliable at n≈10,000; report mean ± std |

**Reported as mean ± std across the 5 GroupKFold folds** (for CV metrics) **and** on the holdout test set once.

**Explicitly excluded from the metric set** (so there are no surprises later):
- No accuracy as primary metric (misleading under imbalance and at operating points).
- No "interpretability score" invented ad hoc.
- No SHAP values as headline metric — feature importances may be reported descriptively but are not a decision metric.

## 10. Artifact checklist (Phase 1 handoff to Phase 2)

| Artifact | Path | Status |
|---|---|---|
| Dataset decision | `docs/dataset_decision.md` | ✅ exists (choice pending) |
| This protocol | `docs/experimental_protocol.md` | ✅ exists |
| Reproducible inspection | `scripts/inspect_datasets.py` → `results/dataset_inspection.json` | ✅ |
| Reproducible decode report | `scripts/check_qr_compatibility.py` → `results/qr_decode_report.json` | ✅ |
| Full-dataset overlap audit | `results/full_overlap_audit.json` | ⬜ Phase 2 gate 1 |
| Full-dataset decode audit | `results/full_decode_audit.json` | ⬜ Phase 2 gate 2 |
| `scripts/qr_utils.py` byte-identical | `d6510490…` | ✅ |
| `scripts/feature_extractor.py` byte-identical | `45e13436…` | ✅ |

### Gate reuse — content-addressed, not time-based

Both gates write a **manifest** (`results/.gateN_manifest.json`) recording the
sha256 of:

- the dataset files that were audited (Trad & Chehab pickles; Mendeley PNGs);
- `scripts/qr_utils.py`, `scripts/feature_extractor.py`, `scripts/dataset_config.py`;
- `scripts/run_gates.py` (the gate script itself);
- the resulting audit JSON (`full_overlap_audit.json` / `full_decode_audit.json`).

Any task requiring gate status (e.g. `train_models.py`, `split_dataset.py`) must call:

```bash
python scripts/run_gates.py check --dataset trad_chehab --external mendeley

## 11. What is deliberately *not* in this protocol

- **No model architecture choice** — Phase 2 decides XGBoost vs. LR vs. baseline, using the CV protocol in §6.
- **No hyperparameter values** — Phase 2 grid/CV decides, with the same grouping rule.
- **No calculus threshold optimisation spec** — that is `main`'s `calculus_threshold.py` (documented in the original README) and will be re-integrated in a Phase 2 task, not this protocol.
- **No app / deployment spec** — out of scope for Phase 1.

The protocol fixes **how to split, deduplicate, gate, evaluate, and report** — nothing else. Everything else remains a Phase 2 decision made against these fixed rules, not a re-litigation of them.

---

## Phase 2 Task 4 — Split execution record

**Split method:** `GroupShuffleSplit` at the `registrable_domain` group level,
two stages (70 % vs 30 %, then 30 % → 15 % + 15 %), `random_state=42` and
`random_state=43` respectively. Reproducible.

**Script:** `scripts/split_dataset.py`. Requires both gate markers
(`.gate1_pass`, `.gate2_pass`) — aborts without them.

**Outputs:** `data/processed/split_{train,val,test}.parquet`
plus `split_excluded.parquet` (non-URL samples) and `split_external.parquet`
(Mendeley, never part of any ML split).

**Assertions enforced (hard-fail on violation):**
- Registrable-domain sets across `train`, `val`, `test` are pairwise disjoint
- Sample-ID sets across `train`, `val`, `test` are pairwise disjoint
- `set(train_ids) ∪ set(val_ids) ∪ set(test_ids) == set(full_feature_ids)`
- No row with empty `registrable_domain` enters the split

**Actual counts (from `results/dataset_split_report.json`):** *(filled after run)*

**Deviation from 70/15/15:** domain-grouped splits cannot preserve exact
proportions at the row level; deviations are reported in the split report and
must remain within ±5 % of target. Exceeding ±5 % requires re-seeding — documented
as a design constraint, not silently accepted.

---

## Phase 2 Task 5 — PCA execution record

**Method:** `StandardScaler` + `PCA` (sklearn, `svd_solver="full"`).
**Fit data:** `split_train.parquet` **only** — never val, test, or external.
**Component count rule:** smallest `k` with cumulative explained variance ≥ 95 %.

**Feature policy:** 4 constant features dropped before standardization
(`has_https`, `has_ip`, `has_userinfo`, `embedded_url_in_query`) — zero variance
on the primary training feature matrix; drop list is in `results/pca_report.json`.

**Transformed splits:** `data/processed/split_{train,val,test,external}_pca.parquet`.

**Reported values (from `results/pca_report.json`):** *(filled after run)*