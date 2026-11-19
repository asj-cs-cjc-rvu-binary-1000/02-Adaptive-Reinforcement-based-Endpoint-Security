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