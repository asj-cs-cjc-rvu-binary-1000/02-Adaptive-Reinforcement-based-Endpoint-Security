# Dataset Decision (Phase 1 — Task 4)

**Status:** Draft — awaiting **final primary-dataset choice** (see §5, this one decision is yours; the rest of the document is evidence-based and fixed).
**Evidence base:** `results/dataset_inspection.json` (Task 1+2), `results/qr_decode_report.json` (Task 3), `docs/dataset_comparison.md`.

---

## 1. What was evaluated

Four candidates were screened against the Phase 1 acceptance criteria (*QR-image native, provenance verifiable, license usable, labels defensible, no domain/duplicate leakage across splits, ARES-compatible decode + feature extraction*).

| Candidate | Downloaded? | Decode rate (benign / malicious) | Feature-extraction failures | Decision |
|---|---|---|---|---|
| **Trad & Chehab (Quishing 2025/26)** | **Yes** | **0.990 / 0.975** (200/class sample) | **0** | Eligible |
| **Mendeley 1,000 QR Images (Feb 2025)** | **Yes** | **1.000 / 0.990** (200/class sample) | **0** | Eligible |
| CIC-Trap4Phish 2025 | No | — | — | Deferred (see §4) |
| QRïS | No | — | — | Rejected (see §4) |

## 2. Evidence for the two eligible candidates

### 2.1 Trad & Chehab — primary candidate

| Criterion | Value | Source |
|---|---|---|
| Size | **9,987 images** (5,005 benign / 4,982 malicious) | `dataset_inspection.json` |
| Format | NumPy pickle, `(9987, 69, 69) int16 {0,1}` | same |
| Label semantics | `0=benign`, `1=malicious` (empirically derived vs. paper's 5005/4982 split) | same |
| Decode rate (200/class) | **0.990 benign / 0.975 malicious** | `qr_decode_report.json` |
| `plain_text` payloads (malicious) | **15 / 200 = 7.5%** — must be handled in Phase 2 (filter or route to Review) | same |
| Feature-extraction failures | **0 / 393** | same |
| Domain overlap (sample) | **0** | same |
| Duplicate payloads (sample) | **0** | same |
| Provenance | Research publication 2025 / conference 2026; data repository | `dataset_sources.md` |
| License | **CC BY 4.0 claimed — UNVERIFIED IN REPO** | `dataset_sources.md` |

**Strengths:** balanced, largest, QR-native, real malicious payloads (paypal-lure, roguesecurityhosts), 7.5% `plain_text` in malicious class is actually *useful* — teaches the pipeline to reject non-URL payloads. 

**Weaknesses:** 69×69 source matrices are below the decoder's "native sweep" range, requiring our 4× upscale + 20px border preprocessing (documented in `dataset_comparison.md`). ~44 min full-dataset decode extrapolated.

### 2.2 Mendeley — secondary candidate / external validation

| Criterion | Value | Source |
|---|---|---|
| Size | **1,000 images** (500 / 500) | `dataset_inspection.json` |
| Format | PNG, mode `1` (1-bit), sizes 290²–1010² | same |
| Decode rate (200/class) | **1.000 benign / 0.990 malicious** | `qr_decode_report.json` |
| `plain_text` payloads (malicious) | 1 / 200 = 0.5% | same |
| Feature-extraction failures | **0 / 398** | same |
| Domain overlap (sample) | **0** | same |
| Duplicate payloads (sample) | **0** | same |
| Provenance | Mendeley Data, 20 Feb 2025 | `dataset_sources.md` |
| License | **CC BY 4.0 claimed — UNVERIFIED IN REPO** | `dataset_sources.md` |

**Strengths:** highest decode rate, clean payloads, second independent provenance → ideal **external validation set** (different collection campaign, different QR-generation tooling).

**Weaknesses:** **size-class confounding** — benign is 87% at 330², malicious spread up to 1010². A pixel-statistics model could shortcut on size alone. Requires resolution normalization in Phase 2. Only 1,000 images — too small for standalone training with a domain-grouped split, fine for a held-out external validation set.

## 3. Task 4 acceptance criteria — mapping

| Criterion | Status |
|---|---|
| Exactly one primary dataset named | **⬜ PENDING — §5 (your decision)** |
| External validation either selected or excluded | **⬜ PENDING — §5 (your decision)** |
| All decisions evidence-based | ✅ every table above cites a JSON |
| Limitations documented | ✅ §2.1, §2.2 "weaknesses"; §6 below |

## 4. Rejected / deferred candidates — justification

**CIC-Trap4Phish 2025 — DEFERRED (not rejected).**
Reported as 430,000 benign + 575,000 malicious QR images; access, license, and exact sample counts cannot be independently confirmed. Not downloaded in Phase 1. **If access is granted in Phase 2 or 3, this becomes a strong large-scale validation candidate.** No decision depends on its absence.

**QRïS — REJECTED.**
400,000 synthetic QR layouts. **Synthetic-data-only** does not match ARES's threat model (real-world quishing captures). Its presence in the manuscript as a "comparison dataset" would weaken, not strengthen, the evaluation.

## 5. ⬜ Decision (your call — see note)

**Primary dataset chosen:**

> ⬜ **Trad & Chehab** *or* ⬜ **Mendeley**

**External validation dataset:**

> ⬜ **Mendeley** (as external set, if primary is Trad & Chehab)  
> ⬜ **None** (if primary is Mendeley — it has only 1,000 images; external validation would require acquiring CIC-Trap4Phish)

**Rationale (you fill in):** ………………………………………………………………………………………………………

**The two configurations and their consequences (documented factually — not a recommendation):**

| Config | Primary | External val | Strengths | Costs |
|---|---|---|---|---|
| **A** | Trad & Chehab (9,987) | Mendeley (1,000) | Largest training set + truly independent external set; both balanced; both decode ≥0.975 | `plain_text` 7.5% in malicious class must be handled; ~44 min full decode; Mendeley size-class confound in *external* set is acceptable (it's validation, not training) |
| **B** | Mendeley (1,000) | None (or CIC-Trap4Phish if acquired) | Cleanest payloads, highest decode rate | 1,000 images is very small for a domain-grouped split (5-fold × 20% ≈ 5 validation domains/class/fold); would need CIC-Trap4Phish for external validation to satisfy IEEE Access reviewers |

**Manuscript consequence:** IEEE Access reviewers expect external validation. **Config B without CIC-Trap4Phish** would be flagged. **Config A** satisfies external validation by construction, at the cost of preprocessing Trad & Chehab's 69×69 matrices (our already-documented 4× upscale + 20px border).

## 6. Known limitations (recorded for the manuscript)

1. **License files UNVERIFIED IN REPO** for both datasets — external-capture step required before submission (see `dataset_sources.md`).
2. **Sample-level overlap check** — 200/class measured, not full-dataset. **Phase 2 pre-training gate** required (see `experimental_protocol.md`).
3. **Decoder cost** — Trad & Chehab full-dataset decode ≈ ~44 min at observed 0.29 s/image.
4. **`plain_text` payloads** — 7.5% of Trad & Chehab malicious class; must be routed explicitly.
5. **Mendeley size-class confounding** — benign 87% at 330², malicious spread to 1010²; Phase 2 must normalize resolution before training.
6. **Trad & Chehab 69×69 source matrices** — below decoder's native sweep; **our 4× upscale + 20px border preprocessing is a documented part of the pipeline**, not a dataset property.
7. **No live threat-intelligence verification** of malicious labels — we take the authors' labels at face value (this is normal for offline dataset work and must be stated).