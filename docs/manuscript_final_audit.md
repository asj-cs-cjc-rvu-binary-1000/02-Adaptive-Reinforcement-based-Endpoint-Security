# ARES Manuscript Final Audit (Phase 4 — Task 7)

**Status:** PASS — Ready for DOCX assembly and submission formatting.
**Evidence Base:** All numbers extracted from committed `results/*.json` at commit `fdb875e`.

## 1. Scientific Integrity Checks
- [x] **No overclaiming of generalization:** Abstract and §VII explicitly report the Mendeley 0.4273 failure and attribute it to domain shift.
- [x] **No formal RL claims:** §III-G and §VIII-C explicitly define the feedback policy as a heuristic controller with no MDP/Bellman equation.
- [x] **No calibration claims:** §VIII-A explicitly states raw `predict_proba` is not isotonic-calibrated.
- [x] **Calculus honestly reported:** §VI-D and §VII-B explicitly state the optimized policy had higher hard cost (0.9162 vs 0.6076 val; 0.8666 vs 0.5941 test).
- [x] **No fabricated references:** All 12 references verified against publisher/arXiv records.

## 2. Formatting Readiness
- [x] **Title:** "ARES: An Offline Quishing Risk Analyser Using Domain-Grouped XGBoost and a Feedback-Driven Threshold Policy"
- [x] **Abstract:** 249 words (under 250 limit).
- [x] **Index Terms:** 7 terms.
- [x] **Figures:** 7 figures planned (1 architecture, 1 decoding detail, 1 PCA, 1 model comparison, 2 confusion matrices, 1 cost landscape).
- [x] **Tables:** 10 tables planned (Payload categories, Dataset comparison, Decode stats, URL features, Split distribution, 8-model comp, Test metrics, External metrics, Threshold comparison, Environment).
- [x] **Equations:** 13 numbered equations.

## 3. Pending Author Actions Before Submission
1. Add author names, affiliations, and emails to the title block.
2. Verify the exact author list and conference name for Reference [3] (IEEE Xplore 2026).
3. Insert the generated figures at the indicated placement markers.
4. Apply the IEEE Access Word template (single-column for review, double-column if final).