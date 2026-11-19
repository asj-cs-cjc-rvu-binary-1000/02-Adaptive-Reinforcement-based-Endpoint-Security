# Model Experiments (Phase 2 — Task 6)

**Status:** Draft — val metrics and selection will be filled after `train_models.py`
and `compare_models.py` run. Every number below will be copied from
`results/model_comparison.json` and `results/model_selection.json` — nothing
pre-entered.

## Protocol

- **Feature representations:** 2 — `raw` (20 features, 4 constants dropped) and
  `pca` (13 principal components at 95 % cumulative variance).
- **Model families:** 4 — Logistic Regression, Decision Tree, Random Forest, XGBoost.
- **Total combinations:** 8.
- **Training data:** `split_train.parquet` / `split_train_pca.parquet` (6,658 rows).
- **Validation data:** `split_val.parquet` / `split_val_pca.parquet` (1,425 rows).
- **Test data:** **NOT touched in this task.** Test evaluation is Phase 2 Task 7
  with the frozen selected model.

## Model configurations (fixed, no hyperparameter search in this task)

| Family | Configuration |
|---|---|
| Logistic Regression | `StandardScaler → LogisticRegression(max_iter=1000)` for raw; LR alone for pca (PCs already in unit variance). |
| Decision Tree | `max_depth=10, min_samples_leaf=5, class_weight="balanced"` |
| Random Forest | `n_estimators=300, class_weight="balanced_subsample", n_jobs=-1` |
| XGBoost | `n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=balanced, tree_method="hist"` |

**Hyperparameter search (inner GroupKFold CV on train) is deliberately deferred to
a Phase 2 extension.** Fixed parameters keep the 8-way comparison an apples-to-apples
test of feature representation and family, not of tuning effort.

## Selection rule (from `results/model_selection.json`)

1. **Primary:** `mean(F1, ROC-AUC)` on validation — balances threshold-quality and
   ranking-quality, and neither dominates the other.
2. **Tiebreak 1:** higher validation precision.
3. **Tiebreak 2:** lower validation Brier (better calibration).
4. **Tiebreak 3:** alphabetical model name (deterministic tiebreak for reproducibility).

**This rule is fixed before any test evaluation.** Changing it after seeing test
metrics would invalidate the test result — that is a hard rule in
`docs/experimental_protocol.md`.

## Measured validation metrics

*(filled after `train_models.py` and `compare_models.py` run)*

## Selected model

*(filled after `compare_models.py` runs)*