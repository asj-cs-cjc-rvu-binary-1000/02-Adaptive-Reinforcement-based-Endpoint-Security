# ARES — Adaptive Reinforcement-based Endpoint Security

**Offline QR-code phishing risk analysis — research prototype, R V University, AIML Semester 3.**

ARES decodes QR images offline, classifies the payload (URL / non-URL / undecodable),
and — for URLs only — estimates a phishing probability with a frozen XGBoost model
trained on lexical and structural URL features. The estimate is mapped to an advisory
action (**Allow** / **Review** / **Block** / **Undecodable**) using two configurable
thresholds.

> **ARES is advisory only. It does not visit or fetch scanned URLs, does not
> inspect page content, and does not guarantee safety. HTTPS is not proof of
> legitimacy. `Allow` is a model output, not a safety certification.**

---

## Team

| Name | Role |
|---|---|
| Aryan S Jain | XGBoost migration, calculus module, adaptive agent |
| Ashwin Chandrashekar Naik | Feature extractor and QR utilities |
| Ashwin Rajesha Bhat | Streamlit app and UI/UX |
| Ashwin Subramaniam | Experiments, baseline metrics, report |

---

## What this repository contains

- **Phase 1** — dataset selection, audit, decode compatibility, protocol — all in `docs/`.
- **Phase 2** — feature engineering, model comparison, frozen `raw_xgb` artifact, calculus threshold experiment, inference pipeline — in `scripts/` and `results/`.
- **Phase 2b** — reproducibility audit, model manifest, package pins.
- **Phase 3** — Streamlit application, adaptive heuristic feedback agent, pytest regression suite (80 tests), `run.sh` launcher.

---

## The pipeline (authoritative — `scripts/predict_pipeline.py`)
