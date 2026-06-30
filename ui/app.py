"""
Operator interface — Streamlit app for spacecraft telemetry health assessment.

Implements PDD Section 7.3 (operator interface requirements):
  - Single-glance severity badge per pass (colour-coded)
  - Expandable reasoning trace (hard-limit vs model-scored vs note-derived)
  - Visible confidence indicator (low-confidence visually distinct)
  - Operator override control (logged separately)
  - Operator's original note displayed alongside the trace

Usage:
    streamlit run ui/app.py
"""

import json
import os
import sys
import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from pipeline.hard_limits import evaluate_hard_limits
from pipeline.numeric_scoring import (
    score_pass, aggregate_subsystem_scores, load_models
)
from pipeline.text_scoring import score_note
from pipeline.fusion import fuse_scores
from pipeline.severity import SEVERITY_COLOURS, SEVERITY_ORDER


# ===========================================================================
# Data loading
# ===========================================================================

@st.cache_data
def load_passes():
    """Load the labelled pass dataset."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
    with open(os.path.join(data_dir, "labelled_passes.json")) as f:
        passes = json.load(f)
    with open(os.path.join(data_dir, "labels.json")) as f:
        labels = json.load(f)
    return passes, labels


@st.cache_resource
def load_ml_models():
    """Load trained numeric scoring models."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
    return load_models(os.path.join(data_dir, "trained_models.pkl"))


# ===========================================================================
# Override log (Step 8.4)
# ===========================================================================

OVERRIDE_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "generated", "override_log.json"
)


def load_override_log():
    """Load the operator override log."""
    if os.path.exists(OVERRIDE_LOG_PATH):
        with open(OVERRIDE_LOG_PATH) as f:
            return json.load(f)
    return []


def save_override(pass_id, system_severity, operator_severity, reason):
    """Append an override entry to the log."""
    log = load_override_log()
    log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "pass_id": pass_id,
        "system_severity": system_severity,
        "operator_severity": operator_severity,
        "reason": reason,
    })
    with open(OVERRIDE_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


# ===========================================================================
# Sample operator notes (paired with passes for the demo)
# ===========================================================================

DEMO_NOTES = {
    "none": [
        "All systems nominal. Clean pass.",
        "Routine housekeeping pass. All parameters nominal.",
        "Standard contact. No concerns. All looks good.",
        "Clean downlink. All subsystems healthy.",
        "Nominal pass. Stable telemetry across the board.",
    ],
    "point": [
        "Noticed a brief glitch in readings. Watching it.",
        "Unexpected spike observed. Checking further.",
        "Momentary dropout in signal. Within tolerance overall.",
    ],
    "contextual": [
        "Battery behaviour slightly off for current eclipse state. Monitoring.",
        "Temperature not tracking eclipse as expected. Keep an eye on it.",
    ],
    "trend": [
        "Values drifting slowly. Trending toward limit. Will monitor.",
        "Gradual change across the pass. Watching it closely.",
        "Slow drift noticed. Still in range but monitoring trend.",
    ],
    "hard_limit": [
        "Battery failure. Voltage at critical levels. Emergency.",
        "Loss of attitude control. Critical anomaly detected.",
        "Complete signal loss. Satellite unresponsive.",
        "Temperature breach detected. Critical thermal event.",
        "Major SEU spike. Possible radiation event.",
    ],
}


def get_note_for_pass(label, idx):
    """Assign a synthetic operator note to a pass based on its anomaly type."""
    atype = label["anomaly_type"]
    notes = DEMO_NOTES.get(atype, DEMO_NOTES["none"])
    return notes[idx % len(notes)]


# ===========================================================================
# Pipeline runner
# ===========================================================================

def assess_pass(pass_data, operator_note, models):
    """Run the full pipeline on a single pass."""
    breaches = evaluate_hard_limits(pass_data)
    sub_scores = score_pass(pass_data, models, "iforest")
    agg = aggregate_subsystem_scores(sub_scores)
    text = score_note(operator_note)
    result = fuse_scores(breaches, agg, text, operator_note)
    result["subsystem_scores"] = sub_scores
    return result


# ===========================================================================
# Streamlit UI
# ===========================================================================

def main():
    st.set_page_config(
        page_title="Spacecraft Telemetry Health Assessment",
        page_icon="🛰️",
        layout="wide",
    )

    st.title("Spacecraft Telemetry Health Assessment")
    st.caption("Ground Segment Decision Support — 3U-6U CubeSat LEO Mission")

    # Load data and models
    passes, labels = load_passes()
    models = load_ml_models()

    # ------------------------------------------------------------------
    # 8.2 — Pass list view
    # ------------------------------------------------------------------
    st.sidebar.header("Pass List")

    # Assess all passes (cached)
    if "assessments" not in st.session_state:
        assessments = []
        for i, (p, l) in enumerate(zip(passes, labels)):
            note = get_note_for_pass(l, i)
            result = assess_pass(p, note, models)
            assessments.append(result)
        st.session_state.assessments = assessments

    assessments = st.session_state.assessments

    # Severity filter
    filter_sev = st.sidebar.multiselect(
        "Filter by severity",
        SEVERITY_ORDER,
        default=SEVERITY_ORDER,
    )

    # Pass list
    for i, (p, l, a) in enumerate(zip(passes, labels, assessments)):
        if a["severity"] not in filter_sev:
            continue
        colour = SEVERITY_COLOURS[a["severity"]]
        conf_str = f"{a['confidence']:.0%}"
        label_text = f"{'🔴' if a['severity'] == 'CRITICAL' else '🟠' if a['severity'] == 'CAUTION' else '🟡' if a['severity'] == 'WATCH' else '🟢'} **{p['pass_id']}** — {a['severity']} ({conf_str})"
        if st.sidebar.button(label_text, key=f"pass_{i}", use_container_width=True):
            st.session_state.selected_pass = i

    # ------------------------------------------------------------------
    # 8.3 — Pass detail view
    # ------------------------------------------------------------------
    sel = st.session_state.get("selected_pass", 0)
    p = passes[sel]
    l = labels[sel]
    a = assessments[sel]

    # Severity badge (colour-coded)
    sev = a["severity"]
    colour = SEVERITY_COLOURS[sev]
    conf = a["confidence"]

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.markdown(
            f'<div style="background-color:{colour}; color:white; '
            f'padding:20px; border-radius:10px; text-align:center;">'
            f'<h1 style="margin:0; color:white;">{sev}</h1>'
            f'<p style="margin:5px 0 0 0; color:white;">Pass {p["pass_id"]}</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        # Confidence indicator (low confidence visually distinct)
        conf_colour = "#e74c3c" if conf < 0.4 else "#f39c12" if conf < 0.7 else "#2ecc71"
        conf_label = "LOW" if conf < 0.4 else "MEDIUM" if conf < 0.7 else "HIGH"
        st.markdown(
            f'<div style="background-color:{conf_colour}; color:white; '
            f'padding:20px; border-radius:10px; text-align:center;">'
            f'<h2 style="margin:0; color:white;">Confidence: {conf_label}</h2>'
            f'<p style="margin:5px 0 0 0; color:white;">{conf:.1%}</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div style="background-color:#34495e; color:white; '
            f'padding:20px; border-radius:10px; text-align:center;">'
            f'<h2 style="margin:0; color:white;">Ground Truth</h2>'
            f'<p style="margin:5px 0 0 0; color:white;">{l["expected_severity"]} '
            f'({l["anomaly_type"]})</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Summary
    st.subheader("Assessment Summary")
    st.info(a["summary"])

    # Operator's original note (displayed alongside the trace)
    st.subheader("Operator Note")
    st.text_area("Original note", a["operator_note"], disabled=True, key="note_display")

    # Expandable reasoning trace
    st.subheader("Reasoning Trace")
    for t in a["reasoning_trace"]:
        source = t["source"]
        if source == "hard_limit":
            st.error(f"**HARD LIMIT** — {t['detail']} | Rule: `{t['rule']}` | "
                     f"Threshold: {t['threshold']} | Actual: {t['actual_value']}")
        elif source == "numeric":
            st.warning(f"**Numeric Scoring** — {t['detail']}")
        elif source == "text":
            st.info(f"**Note Scoring** — {t['detail']}")
            if t.get("matched_phrases"):
                phrases = ", ".join(f"'{ph}' ({wt:+.2f})"
                                   for ph, wt in t["matched_phrases"])
                st.caption(f"Matched phrases: {phrases}")
        elif source == "fusion_escalation":
            st.warning(f"**Fusion** — {t['detail']}")

    # Subsystem breakdown
    st.subheader("Subsystem Scores")
    if "subsystem_scores" in a:
        cols = st.columns(5)
        for i, (sub, scores) in enumerate(a["subsystem_scores"].items()):
            with cols[i]:
                score = scores["score"]
                sub_colour = (SEVERITY_COLOURS["CRITICAL"] if score > 0.85
                              else SEVERITY_COLOURS["CAUTION"] if score > 0.6
                              else SEVERITY_COLOURS["WATCH"] if score > 0.3
                              else SEVERITY_COLOURS["NOMINAL"])
                st.metric(sub.upper(), f"{score:.2f}", delta=None)

    st.divider()

    # ------------------------------------------------------------------
    # 8.4 — Override control (logged separately)
    # ------------------------------------------------------------------
    st.subheader("Operator Override")
    st.caption("The system's recommendation is decision support. "
               "A human operator confirms every action.")

    override_col1, override_col2 = st.columns([1, 2])
    with override_col1:
        override_sev = st.selectbox(
            "Override severity to:",
            SEVERITY_ORDER,
            index=SEVERITY_ORDER.index(sev),
            key=f"override_{sel}",
        )
    with override_col2:
        override_reason = st.text_input(
            "Reason for override (required):",
            key=f"override_reason_{sel}",
        )

    if st.button("Submit Override", key=f"submit_override_{sel}"):
        if override_reason.strip():
            save_override(p["pass_id"], sev, override_sev, override_reason)
            st.success(f"Override logged: {sev} -> {override_sev}")
        else:
            st.error("Please provide a reason for the override.")

    # Show override log
    override_log = load_override_log()
    if override_log:
        with st.expander("Override Log"):
            for entry in reversed(override_log):
                st.text(f"{entry['timestamp']} | {entry['pass_id']} | "
                        f"{entry['system_severity']} -> {entry['operator_severity']} | "
                        f"{entry['reason']}")


if __name__ == "__main__":
    main()
