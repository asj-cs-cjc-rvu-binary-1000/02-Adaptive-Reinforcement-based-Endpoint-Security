#!/usr/bin/env python3
"""
app.py — ARES Streamlit application (Phase 3).

Four-tab layout preserved from main:
  Tab 1: 1-Click Sample QR Gallery
  Tab 2: Upload QR Image
  Tab 3: Camera Snapshot
  Tab 4: Direct URL Analyzer & QR Encoder

Sidebar:
  - Policy selector (heuristic thresholds from adaptive_agent)
  - Calculus Threshold Optimizer panel (read-only results from calculus_threshold)

Inference is done EXCLUSIVELY via scripts/predict_pipeline.PredictionPipeline.
No joblib.load, no predict_proba, no feature extraction in this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure both scripts/ and repo root are importable
_HERE = Path(__file__).resolve().parent
for _p in (_HERE / "scripts", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import streamlit as st

# Authoritative inference — the ONLY inference path in this file
from predict_pipeline import PredictionPipeline

# Adaptive heuristic agent (top-level module per Decision 2)
from adaptive_agent import HeuristicAdaptiveAgent

# Calculus adapter — read-only view over results/calculus_experiment.json
import calculus_threshold

# These are used only for URL-anatomy display, not for inference
import feature_extractor as fx

# ---- Page config ----
st.set_page_config(
    page_title="ARES — Offline QR Phishing Analyser",
    page_icon="🛡️",
    layout="wide",
)

RESULTS = _HERE / "results"
SAMPLES_DIR = _HERE / "tests" / "images"  # fallback if created by tests; otherwise generated


# ---- Cached resources (called at most once per Streamlit process) ----
@st.cache_resource
def _pipeline_default():
    return PredictionPipeline(policy="default")


@st.cache_resource
def _pipeline_optimized():
    return PredictionPipeline(policy="optimized")


@st.cache_resource
def _agent():
    return HeuristicAdaptiveAgent()


# ---- Sidebar: policy + calculus panel ----
def _render_sidebar() -> dict:
    st.sidebar.header("Decision policy")

    agent = _agent()
    heuristic_thr = agent.thresholds()
    default_thr = calculus_threshold.default_thresholds()
    opt_thr = calculus_threshold.optimized_thresholds()

    policy_choice = st.sidebar.radio(
        "Active thresholds",
        options=["Heuristic policy (feedback-driven)",
                 "Default (0.35, 0.70) — publication reference",
                 "Optimized (calculus) — experimental"],
        index=0,
        help="All three policies use the same frozen raw_xgb model. Thresholds "
             "only change the Allow/Review/Block mapping.",
    )

    if policy_choice.startswith("Heuristic"):
        t_r, t_b = heuristic_thr
        policy_kind = "heuristic"
    elif policy_choice.startswith("Default"):
        t_r, t_b = default_thr
        policy_kind = "default"
    else:
        t_r, t_b = opt_thr
        policy_kind = "optimized"

    st.sidebar.caption(
        f"t_review = {t_r:.4f}  ·  t_block = {t_b:.4f}  ·  policy = {policy_kind}"
    )

    if policy_kind == "heuristic":
        if st.sidebar.button("Reset heuristic thresholds to defaults"):
            agent.reset_to_default()
            st.rerun()

    # Calculus panel — read-only
    with st.sidebar.expander("∫ Calculus Threshold Optimizer", expanded=False):
        try:
            cmp = calculus_threshold.compare_policies()
            st.markdown(
                f"**Default:** ({cmp['default_thresholds'][0]:.2f}, "
                f"{cmp['default_thresholds'][1]:.2f})\n\n"
                f"**Optimized:** ({cmp['optimized_thresholds'][0]:.2f}, "
                f"{cmp['optimized_thresholds'][1]:.4f})"
            )
            opt = cmp["optimization"]
            st.markdown(
                f"**Hessian det:** {opt['hessian_det']:.4f}  "
                f"({'minimum' if opt['is_minimum'] else 'not minimum'})"
            )
            st.markdown(
                f"**Boundary hit:** {opt['boundary_hit']}  "
                f"(KKT constrained: {opt['is_kkt_constrained']})"
            )
            if calculus_threshold.is_optimized_cost_lower():
                st.success("Optimized policy has lower hard cost on both splits.")
            else:
                st.warning(
                    "Optimized policy has **higher hard cost** than default on this "
                    "dataset. The calculus result demonstrates the optimizer machinery, "
                    "not a policy improvement. See docs/mathematical_techniques.md §3.6."
                )
        except Exception as e:                # noqa: BLE001
            st.error(f"Calculus results unavailable: {e}")

    return {"t_r": t_r, "t_b": t_b, "policy_kind": policy_kind}


# ---- Result rendering ----
def _render_prediction(result: dict, thresholds: dict, agent: HeuristicAdaptiveAgent):
    """Render the PredictionPipeline.run() output."""
    payload = result.get("payload")
    category = result.get("payload_category")
    action = result.get("action")
    prob = result.get("prob_phishing")
    reason = result.get("reason")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Decoded payload")
        if payload is None:
            st.warning("No QR payload could be decoded from this image.")
        else:
            st.code(payload, language="text")
        st.caption(f"Category: **{category}**")

    with col2:
        st.subheader("Action")
        color = {"Allow": "green", "Review": "orange",
                 "Block": "red", "Undecodable": "grey"}.get(action, "grey")
        st.markdown(
            f"<div style='font-size:2rem;font-weight:bold;color:{color}'>{action}</div>",
            unsafe_allow_html=True,
        )
        if reason:
            st.caption(reason)

    with col3:
        st.subheader("Estimated phishing probability")
        if prob is None:
            st.markdown("_N/A — non-URL payload does not enter the URL classifier._")
        else:
            st.markdown(f"**{prob:.4f}**")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(
                "Raw XGBoost `predict_proba` output — **not** isotonic-calibrated. "
                "See `docs/mathematical_techniques.md` §3.7."
            )

    # URL anatomy + features — only when the payload is a URL
    if category == "url" and result.get("features"):
        feat = result["features"]
        try:
            anatomy = fx.parse_host_structure(payload)
            st.subheader("URL anatomy")
            st.json({
                "scheme": anatomy["parsed"].scheme,
                "hostname": anatomy["hostname"],
                "registrable_domain": anatomy["registrable_domain"],
                "subdomains": anatomy["subdomain_labels"],
                "path": anatomy["parsed"].path,
                "query": anatomy["parsed"].query,
                "port": anatomy["port"],
            })
        except Exception:                     # noqa: BLE001
            pass

        with st.expander("20 kept features (4 constant features dropped before inference)",
                         expanded=False):
            df = pd.DataFrame(
                [{"feature": k, "value": v} for k, v in feat.items()]
            )
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption(
                "The 4 features `has_https`, `has_ip`, `has_userinfo`, "
                "`embedded_url_in_query` are constant (zero variance) on this dataset "
                "and are dropped before model inference. See "
                "`results/feature_summary.json`."
            )

    # Feedback
    st.subheader("Feedback (heuristic policy — not formal RL)")
    st.caption(
        "Your feedback adjusts the two decision thresholds only. The frozen XGBoost "
        "model is **never retrained** by this action."
    )
    if payload is not None and prob is not None:
        sample_id = f"app:{hash((payload, round(prob, 6))) & 0xFFFFFFFF:08x}"
        c1, c2, c3 = st.columns(3)
        if c1.button("✓ Correct", key=f"fb_correct_{sample_id}"):
            agent.submit_feedback(sample_id, prob, action, "correct")
            st.success("Feedback recorded.")
        if c2.button("⚠ False positive", key=f"fb_fp_{sample_id}"):
            agent.submit_feedback(sample_id, prob, action, "false_positive")
            st.success("Feedback recorded. Thresholds moved slightly up.")
        if c3.button("✗ False negative", key=f"fb_fn_{sample_id}"):
            agent.submit_feedback(sample_id, prob, action, "false_negative")
            st.success("Feedback recorded. Thresholds moved slightly down.")
    else:
        st.info("Feedback requires a URL payload with a probability — this payload does not have one.")


# ---- Tab bodies ----
def _run_pipeline_on(image_input, thresholds: dict, agent) -> None:
    """Run PredictionPipeline on a file path or ndarray, then render."""
    try:
        # Use whichever pipeline corresponds to the active policy
        if thresholds["policy_kind"] == "optimized":
            pipe = _pipeline_optimized()
        else:
            pipe = _pipeline_default()          # also used for heuristic + default
        result = pipe.run(image_input)
    except SystemExit as e:
        st.error(f"Pipeline init failed (fail-closed): {e}")
        return
    except Exception as e:                    # noqa: BLE001
        st.error(f"Prediction failed: {type(e).__name__}: {e}")
        return

    # Override action decision if heuristic policy is active AND payload is a URL
    if thresholds["policy_kind"] == "heuristic" and result["payload_category"] == "url":
        p = result["prob_phishing"]
        t_r, t_b = thresholds["t_r"], thresholds["t_b"]
        if p >= t_b:
            result["action"] = "Block"
            result["reason"] = f"p={p:.4f} ≥ t_block={t_b:.4f}"
        elif p >= t_r:
            result["action"] = "Review"
            result["reason"] = f"t_review={t_r:.4f} ≤ p={p:.4f} < t_block={t_b:.4f}"
        else:
            result["action"] = "Allow"
            result["reason"] = f"p={p:.4f} < t_review={t_r:.4f}"
        result["policy"] = {"t_review": t_r, "t_block": t_b}

    _render_prediction(result, thresholds, agent)


def tab_sample_gallery(thresholds, agent):
    st.header("1-Click Sample QR Gallery")
    st.write(
        "These samples are drawn from the Trad & Chehab dataset. "
        "Upload your own image in the next tab for evaluation of arbitrary QR codes."
    )
    # Build a tiny in-session gallery from the dataset (first 3 per class)
    import pickle
    imgs_path = _HERE / "data" / "raw" / "qr_codes_29.pickle"
    labs_path = _HERE / "data" / "raw" / "qr_codes_29_labels.pickle"
    if not (imgs_path.exists() and labs_path.exists()):
        st.warning("Sample gallery requires data/raw/qr_codes_29*.pickle")
        return
    with open(imgs_path, "rb") as f:
        imgs = pickle.load(f)
    with open(labs_path, "rb") as f:
        labs = pickle.load(f)
    cols = st.columns(4)
    shown = 0
    for lbl_display, cls in ((0, "benign"), (1, "malicious")):
        idxs = np.where(labs == lbl_display)[0][:2]
        for i in idxs:
            with cols[shown % 4]:
                arr = (imgs[int(i)].astype(np.uint8) * 255)
                st.image(arr, caption=f"sample idx={i} ({cls})", width=180)
                if st.button("Analyse this sample", key=f"sample_{i}"):
                    _run_pipeline_on(arr, thresholds, agent)
            shown += 1


def tab_upload(thresholds, agent):
    st.header("Upload QR Image")
    uploaded = st.file_uploader(
        "PNG / JPG / JPEG (≤ 5 MB)", type=["png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        if uploaded.size > 5 * 1024 * 1024:
            st.error("File exceeds 5 MB.")
            return
        from PIL import Image as PILImage
        img = PILImage.open(uploaded).convert("L")
        st.image(img, caption="Uploaded image", width=240)
        arr = np.array(img)
        _run_pipeline_on(arr, thresholds, agent)


def tab_camera(thresholds, agent):
    st.header("Camera Snapshot")
    st.caption("Take a still photo of a QR code using your webcam / phone camera.")
    img_bytes = st.camera_input("Capture QR")
    if img_bytes is not None:
        from PIL import Image as PILImage
        img = PILImage.open(img_bytes).convert("L")
        st.image(img, caption="Captured", width=240)
        arr = np.array(img)
        _run_pipeline_on(arr, thresholds, agent)


def tab_direct_url(thresholds, agent):
    st.header("Direct URL Analyzer & QR Encoder")
    st.caption(
        "Two utilities: analyse a URL string directly, or encode any text/URL into "
        "a downloadable QR PNG. The QR encoder is a testing utility — **not** a safety "
        "filter; encoding an unsafe URL does not make it safe."
    )
    url = st.text_input(
        "URL or text", value="https://secure-login.paypal.com.evil.xyz/verify?id=1"
    )
    colA, colB = st.columns(2)
    with colA:
        if st.button("Analyse URL"):
            result = {
                "payload": url,
                "payload_category": "url" if fx.is_valid_url(url) else "plain_text",
                "features": None,
                "prob_phishing": None,
                "action": None,
                "reason": None,
                "policy": {"t_review": thresholds["t_r"], "t_block": thresholds["t_b"]},
            }
            if result["payload_category"] == "url":
                try:
                    pipe = (_pipeline_optimized()
                            if thresholds["policy_kind"] == "optimized"
                            else _pipeline_default())
                    pred = pipe.predict(url)
                    result["features"] = pred["features"]
                    result["prob_phishing"] = pred["prob_phishing"]
                    # Override decision if heuristic policy
                    p = pred["prob_phishing"]
                    if thresholds["policy_kind"] == "heuristic":
                        t_r, t_b = thresholds["t_r"], thresholds["t_b"]
                        if p >= t_b:
                            result["action"] = "Block"
                            result["reason"] = f"p={p:.4f} ≥ t_block={t_b:.4f}"
                        elif p >= t_r:
                            result["action"] = "Review"
                            result["reason"] = f"t_review={t_r:.4f} ≤ p={p:.4f} < t_block={t_b:.4f}"
                        else:
                            result["action"] = "Allow"
                            result["reason"] = f"p={p:.4f} < t_review={t_r:.4f}"
                    else:
                        result["action"] = pred["action"]
                        result["reason"] = pred["reason"]
                except SystemExit as e:
                    st.error(f"Pipeline init failed (fail-closed): {e}")
                    return
            else:
                result["action"] = "Review"
                result["reason"] = (
                    f"Payload not a valid http(s) URL — category={result['payload_category']}"
                )
            _render_prediction(result, thresholds, agent)
    with colB:
        if st.button("Encode to QR PNG"):
            try:
                import qr_utils
                arr = qr_utils.generate_qr_image(url, scale=8)
                st.image(arr, caption="Encoded QR", width=240)
                from io import BytesIO
                from PIL import Image as PILImage
                buf = BytesIO()
                PILImage.fromarray(arr).save(buf, format="PNG")
                st.download_button(
                    "Download QR PNG",
                    data=buf.getvalue(),
                    file_name="ares_qr.png",
                    mime="image/png",
                )
            except Exception as e:            # noqa: BLE001
                st.error(f"Encode failed: {type(e).__name__}: {e}")


def main():
    st.title("ARES — Offline QR Phishing Risk Analyser")
    st.caption(
        "Research prototype — R V University. Advisory-only. Never use as the sole basis "
        "for trusting an unknown link. ARES does not visit or fetch scanned URLs."
    )

    thresholds = _render_sidebar()
    agent = _agent()

    tabs = st.tabs([
        "🖼 1-Click Samples",
        "⬆ Upload QR",
        "📷 Camera Snapshot",
        "🔗 URL Analyser & QR Encoder",
    ])
    with tabs[0]:
        tab_sample_gallery(thresholds, agent)
    with tabs[1]:
        tab_upload(thresholds, agent)
    with tabs[2]:
        tab_camera(thresholds, agent)
    with tabs[3]:
        tab_direct_url(thresholds, agent)

    st.divider()
    st.caption(
        "Model: `models/raw_xgb.joblib` (frozen; sha256 enforced by "
        "`results/model_manifest.json`). "
        "Probability is raw XGBoost — not isotonic-calibrated. "
        "Threshold policy is a **heuristic, feedback-driven** controller — not formal RL. "
        "See `docs/mathematical_techniques.md` §3.6–3.7 and `docs/application_guide.md`."
    )


if __name__ == "__main__":
    main()