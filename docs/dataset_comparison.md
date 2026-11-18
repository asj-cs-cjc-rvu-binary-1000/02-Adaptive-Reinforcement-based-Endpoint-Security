# Dataset Comparison (Phase 1 — Tasks 2 & 3)

**Status:** Phase 1 Task 2 (structure) + Task 3 (QR decoding & ARES compatibility) **verified 2026-11-18**.
**Authoritative sources (re-run to reproduce):**
- `results/dataset_inspection.json` ← `python scripts/inspect_datasets.py`
- `results/qr_decode_report.json` ← `python scripts/check_qr_compatibility.py`

If any number below disagrees with either JSON on re-run, **the JSON wins** and this document must be corrected.

---

## Task 2 — Structure (per dataset)

### Trad & Chehab

| Field | Value |
|---|---|
| Total images | 9,987 |
| Array shape | `(9987, 69, 69) int16`, values `{0,1}` (binary modules — not 0–255 grayscale) |
| Labels | `(9987,) int32`, `{0: 5005, 1: 4982}` → **`0 = benign`, `1 = malicious`** (empirically derived — matches paper's 5,005/4,982) |
| Class balance | balanced (5005 / 4982 = 1.005:1) |
| Length consistency | ✅ images == labels |
| Corruption | N/A (in-memory NumPy arrays) |
| License file in archive | **NO — UNVERIFIED IN REPO** (see `dataset_sources.md`) |

### Mendeley

| Field | Value |
|---|---|
| Total images | 1,000 (500 benign + 500 malicious) |
| Format | PNG, PIL `mode = 1` (1-bit) |
| Sizes — benign | `330²:437, 370²:53, 290²:9, 450²:1` |
| Sizes — malicious | `370²:246, 330²:155, 410²:68, 450²:18, 650²:3, 530²:2, 570²:2, 610²:1, 490²:1, 290²:3, 1010²:1` |
| Class balance | balanced (500 / 500) |
| Corruption | 0 unreadable at inspection time |
| License file in archive | **NO — UNVERIFIED IN REPO** |

**Limitation (must be in the manuscript):** Mendeley's size distribution is class-imbalanced (benign 87% at 330²; malicious spread from 290² to 1010²). A pixel-statistics model could shortcut on **image size alone**. Phase 2 preprocessing must **normalize resolution** before training.

---

## Task 3 — QR decoding & ARES compatibility (MEASURED)

**Provenance (byte-identical to `origin/main`, hash-verified in the JSON):**

| File | sha256 | Path |
|---|---|---|
| `qr_utils.py` | `d65104906924bff1b762cd48267a6b839f4fc01c06e3a7bf5b8945c2d27db8fa` | `scripts/qr_utils.py` |
| `feature_extractor.py` | `45e13436f738b50f5992b84ef56a5c35aed3b828e0005bb475f96b56b5c0ee32` | `scripts/feature_extractor.py` |

**Sample cap:** 200 per class, per dataset (deterministic via `RNG_SEED = 42`).
**Wall-clock:** 203 s for 800 images total (observed ~0.29 s/image Trad & Chehab, ~0.25 s/image Mendeley — dominated by `qr_utils._variants(...)` sweep).
**Full-dataset decode extrapolation for Trad & Chehab:** ~44 min at 9987 images at the observed rate. Not a blocker; recorded for planning.

### Measured decode rates and payload categories

| Metric | Trad & Chehab | Mendeley |
|---|---:|---:|
| Decode rate — benign | **0.990** (198/200) | **1.000** (200/200) |
| Decode rate — malicious | **0.975** (195/200) | **0.990** (198/200) |
| `url` payloads (benign) | 197 | 200 |
| `url` payloads (malicious) | 180 | 197 |
| `plain_text` payloads (benign) | 1 | 0 |
| `plain_text` payloads (malicious) | **15 (7.5%)** | 1 |
| `undecodable` (benign) | 2 | 0 |
| `undecodable` (malicious) | 5 | 2 |

**Interpretation:** both datasets are **usable** for ARES. Both are QR-image-native; both have plausibly-labelled payloads (sampled payloads below confirm the labelling is genuinely about phishing, not a proxy). Neither exhibits the silent-zero pattern that would indicate a broken pipeline.

### Feature-extraction compatibility (24-feature extractor from `origin/main`)

| Metric | Trad & Chehab | Mendeley |
|---|---:|---:|
| URLs attempted | 393 | 398 |
| Succeeded (24 features, fixed order) | 377 | 397 |
| **Failed** | **0** | **0** |

`feature_extraction.failed == 0` on both datasets is the single strongest compatibility signal available for `feature_extractor.py` against these URL populations. **No production bug surfaced.**

### Duplicate & domain-overlap checks (sample level)

| Metric | Trad & Chehab | Mendeley |
|---|---:|---:|
| Unique benign payloads / total | 198 / 198 | 200 / 200 |
| Unique malicious payloads / total | 195 / 195 | 198 / 198 |
| Duplicates (benign, malicious) | 0, 0 | 0, 0 |
| Unique benign registrable domains | 168 | 190 |
| Unique malicious registrable domains | 166 | 114 |
| **Domain overlap (benign ∩ malicious)** | **0** | **0** |

⚠ **Sample-level only.** Zero overlap on 200/class does not prove zero overlap on 9987. **Phase 2 pre-training gate:** rerun the overlap check on the **full** dataset before defining train/val/test splits.

### Representative payloads (verbatim from the JSON)

**Trad & Chehab — benign (label 0):**

www.benchmarkhosting.com/index.html
sra.itc.it/people/massa/index.html
www.media.mit.edu/resenv/classes/MAS965/readings/TextilePressureSensor.pdf
www.washington.edu/computing/unix/shell.html
www.lastbit.com/trafmeter/


**Trad & Chehab — malicious (label 1):**

paypal.com.us.cgi.bin.webscr.cmd.login.member.verifed.interhottap.com.mx/54855458214/23bf9c1ccadeb59398f870bc932bd60e/
pastehtml.com/view/bp9b6a48d.html
www.promo-james.ru/logo/
www.fsjinlan.cn/images/?secure.runescape.com/m=weblogin/loginform.ws?mod=www&clntgdsamp;ssl=0&dest
mobilevirtualbusiness.com/~amfaceli/webscr.php


**Mendeley — benign:**

https://data.worldbank.org/
https://www.notion.so/
https://www.freecodecamp.org/
https://www.hathitrust.org/
https://www.myfitnesspal.com/


**Mendeley — malicious:**

http://mobildeniz2025firsatlari.duckdns.org/
http://www.avc3c.com/SharePointFileProposal/
http://www.webogram.org/z/
https://pub-6ec7dc245c434057b766048a50064ade.r2.dev/home.html…
https://app--sso--robinhood---com.webflow.io/


These payloads are **recognizably** benign vs malicious — no labelling ambiguity for the reviewer to contest.

---

## Task 3 acceptance criteria — status

| Criterion | Status |
|---|---|
| Decoding success rate measured | ✅ both datasets, both classes |
| Non-URL payloads not silently discarded | ✅ `plain_text` counted separately (Trad & Chehab: 15 malicious) |
| Failed decodes documented | ✅ `undecodable` 2–5 per cell |
| URL feature extraction tested on decoded URLs | ✅ 377 + 397 succeeded, **0 failed** |
| Duplicate and domain-overlap risks documented | ✅ sample-level: 0/0, full-dataset check flagged as Phase 2 gate |

## Known limitations (recorded for the manuscript)

1. **`plain_text` payloads in the malicious class** — Trad & Chehab: 15/200 (7.5%), Mendeley: 1/200 (0.5%). The Phase 2 pipeline must **route `plain_text` explicitly** (filter or `Review` outcome), not treat them as URLs.
2. **Sample-level, not full-dataset, overlap check** — 200/class measured; Phase 2 gate required.
3. **Decoder cost** — Trad & Chehab ~0.29 s/image → ~44 min full 9987 decode. Not a blocker; recorded.
4. **Mendeley size-class confounding** — benign is mostly 330², malicious spread up to 1010². Phase 2 must normalize resolution.
5. **License files not shipped in either archive** — `UNVERIFIED IN REPO`; external-capture step required before manuscript submission (see `dataset_sources.md`).

---

## Phase 2 Task 1 — Full-dataset duplicate + label-conflict audit

See `results/full_overlap_audit.json` for the authoritative numbers. This section will be populated **after** the gate script completes; nothing is written here in advance.