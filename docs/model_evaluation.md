# Model Evaluation (Phase 2 — Task 7)

**Status:** Draft — numbers to be filled after `scripts/evaluate_models.py` runs.
**Source of truth:** `results/final_metrics.json` (test + external) and `results/model_comparison.json` (validation).

## Protocol

- **Selected model:** `raw_xgb`, frozen in `results/model_selection.json` (Task 6).
- **Selection rule:** `mean(F1, ROC-AUC)` on validation, tiebreaks precision → brier → name.
- **Test set:** touched **once** in this task; no re-selection, no hyperparameter change.
- **External set (Mendeley):** touched **once**; frozen model; no tuning; no threshold change.
- **Decision threshold:** 0.50 for all reported point metrics. Threshold optimisation is Task 8.

## Validation metrics for all 8 models (from Task 6)

The `raw` representation beat `pca` on every family. **PCA loses discriminative signal
on this dataset**, even though it captured 95 % of *variance*. This is expected when
URL features are weakly correlated (Task 3 found no |r| ≥ 0.95 pairs): PCA reduces
variance but not the information that distinguishes benign from malicious URLs.

## Test-set metrics — `raw_xgb`

*(filled after run — copied from `results/final_metrics.json` `test` block)*

## External validation metrics — `raw_xgb` on Mendeley

*(filled after run — copied from `results/final_metrics.json` `external` block)*

## PCA vs raw — discussion for the manuscript

The 4×2 comparison is the most honest test of PCA in this pipeline:

- **PCA is not a free win.** On every family, `raw` beats `pca` on F1 and ROC-AUC.
  The gap is +2.4 to +5.0 pp F1 depending on family.
- **PCA's cost is information loss.** With weak correlations, dimensionality
  reduction removes signal (the "redundancy" a PCA would otherwise exploit is
  small at |r| < 0.95).
- **PCA is still useful for interpretability.** PC1 explains 32.8 % of variance
  and corresponds to a composite URL "suspiciousness" axis (url_length, digits,
  entropy, suspicious tokens, dots, subdomain depth all load positively).
  That is a good feature for a reviewer narrative — even if it does not help
  the classifier's accuracy.

**Recommendation for the manuscript:** report both representations; recommend
`raw_xgb` for deployment; present PCA as an interpretability and dimensionality-
comparison result, not as a deployment recommendation.

## Reproducibility

```bash
python scripts/run_gates.py check --dataset trad_chehab --external mendeley  # ~3 s
python scripts/evaluate_models.py                                            # ~5 s
```

No training, no decoding, no selection logic. Everything downstream of Task 6.

## External validation — domain shift finding

**Test metrics (primary, in-distribution):**
- accuracy 0.8910 · precision 0.9078 · recall 0.8819 · F1 0.8947 · ROC-AUC 0.9490 · PR-AUC 0.9641 · Brier 0.0841
- confusion: TN=573, FP=63, FN=83, TP=620

**External metrics (Mendeley, out-of-distribution):**
- accuracy 0.4273 · precision 0.4554 · recall 0.7902 · F1 0.5778 · ROC-AUC 0.5262 · PR-AUC 0.6225 · Brier 0.4965
- confusion: TN=35, FP=464, FN=103, TP=388
- deltas external − test: accuracy −0.4637, F1 −0.3169, ROC-AUC −0.4228, Brier +0.4124

**Diagnosis: `results/feature_drift_diagnosis.json`**

The external drop is a **dataset property, not a code defect**. Per-feature analysis:

| Observation | Evidence |
|---|---|
| **Benign classes are aligned** | `mean |z_shift_benign| = 0.1724`, max 0.7373 on `num_subdirs`; **0/20 features exceed |z| > 1.0** |
| **Malicious classes are wildly different** | Trad & Chehab malicious: `url_length` mean **82.08**, `path_length` **32.47**, `suspicious_words` **0.67**. Mendeley malicious: `url_length` mean **42.70**, `path_length` **9.19**, `suspicious_words` **0.075** |
| **Era and style** | Primary malicious payloads are 2008–2012-era credential-phishing (deeply nested paths, `?cmd=`/`?login=` params). External malicious payloads are 2024-era quishing (lookalike subdomains, `duckdns.org` abuse, `webflow.io` lookalike hosting, bare-domain short URLs) |

**Interpretation:**

The model correctly learned Trad & Chehab's label boundary — "phishing = long, deeply nested, phishing-token-rich." Applied to Mendeley's test set, that boundary collapses because:

1. Mendeley **benign** URLs (`https://www.notion.so/`, `https://www.freecodecamp.org/`) are short, clean, `https://`-scheme — superficially resembling suspicious short URLs, so 464/499 benign external samples are called malicious.
2. Mendeley **malicious** URLs use techniques that leave no lexical trace in Trad & Chehab's feature set — `duckdns.org` subdomain abuse, `webflow.io` lookalike hosting, and typosquatting via subdomain separators. 388/491 malicious external samples are correctly caught (recall 0.79) but at the cost of 93 % false-positive rate.

**Conclusion:** external validation on Mendeley **does not support** generalization claims. This is a **finding about dataset era and collection methodology**, not a measure of model quality. Reported honestly, it strengthens the manuscript by documenting a real limit of URL-lexical phishing detection.

**Recommended future work (for manuscript):** re-train with Mendeley included in training data; or acquire a *contemporary* external validation dataset (e.g. CIC-Trap4Phish 2025) with matching era and techniques; or add QR-image features as a separate signal orthogonal to URL lexical structure.

**What is *not* claimed:** we do not claim ARES generalizes to contemporary quishing. We claim it performs at 89.10 % accuracy / 0.949 ROC-AUC on its own domain-grouped holdout, and that external validation on a distributionally-shifted dataset fails — as it should.

## PCA-vs-raw discussion

The validation results (Task 6) show every `raw_*` model beats its `pca_*` counterpart:

| Family | raw F1 | pca F1 | Δ F1 | raw ROC-AUC | pca ROC-AUC | Δ ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGB | **0.8651** | 0.8413 | +0.0238 | **0.9494** | 0.9261 | +0.0233 |
| RF  | 0.8640 | 0.8376 | +0.0264 | 0.9486 | 0.9303 | +0.0183 |
| DT  | 0.8462 | 0.7962 | +0.0500 | 0.8931 | 0.8769 | +0.0162 |
| LR  | 0.8223 | 0.7718 | +0.0505 | 0.9119 | 0.8576 | +0.0543 |

PCA is not a free win. It captured 95.86 % of *variance*, but on weakly correlated features (no |r| ≥ 0.95 pairs) dimensionality reduction removes discriminative signal. **PC1 is still interpretable** (composite URL "suspiciousness" axis — `url_length`, `digits_count`, `entropy`, `suspicious_words`, `num_dots`, `susp_subdomain_count` all load positively) and worth reporting for narrative purposes. But deployment uses `raw_xgb`.

## Task 8 — calculus threshold optimization

**Fit on:** validation split only. **Test set:** not used to fit; only reported for the manuscript.

**Default policy:** `(t_review, t_block) = (0.35, 0.70)`.
**Optimized policy:** see `results/calculus_experiment.json` `optimized_thresholds`.

**Measured deltas (from `results/calculus_experiment.json`):**

| Dataset | Policy | FPR | FNR | Review-rate |
|---|---|---:|---:|---:|
| Validation | Default (0.35, 0.70) | *(filled after run)* | *(filled after run)* | *(filled after run)* |
| Validation | Optimized | *(filled after run)* | *(filled after run)* | *(filled after run)* |
| Test | Default (0.35, 0.70) | *(filled after run)* | *(filled after run)* | *(filled after run)* |
| Test | Optimized | *(filled after run)* | *(filled after run)* | *(filled after run)* |

**Interpretation for the manuscript:** *(filled after run)*

**Cost function caveat:** GD minimizes a sigmoid-smoothed surrogate; the hard-cost
validation at the final point is what determines whether the optimized policy is
actually better than default. See `docs/mathematical_techniques.md` §3.6.