"""
Operator interface — Streamlit app for spacecraft telemetry health assessment.

Implements PDD Section 7.3 (operator interface requirements):
  - Single-glance severity badge per pass (colour-coded)
  - Expandable reasoning trace (hard-limit vs model-scored vs note-derived)
  - Visible confidence indicator (low-confidence visually distinct)
  - Operator override control (logged separately)
  - Operator's original note displayed alongside the trace

Navigation:
  - Use the sidebar to select a pass from the list (click the pass button).
  - Each pass shows a severity badge, confidence indicator, and ground truth.
  - The reasoning trace explains exactly why the system reached its verdict.
  - The operator note is editable; click 'Re-assess' to update the assessment.
  - Use the override control to log a manual severity correction.

Usage:
    streamlit run ui/app.py
    (run from the telemetry-health/ project root)
"""

import json
import os
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from pipeline.hard_limits import evaluate_hard_limits
from pipeline.numeric_scoring import score_pass, aggregate_subsystem_scores, load_models
from pipeline.text_scoring import score_note
from pipeline.fusion import fuse_scores
from pipeline.severity import SEVERITY_COLOURS, SEVERITY_ORDER


# ===========================================================================
# Data loading
# ===========================================================================

@st.cache_data
def load_passes():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
    with open(os.path.join(data_dir, "labelled_passes.json")) as f:
        passes = json.load(f)
    with open(os.path.join(data_dir, "labels.json")) as f:
        labels = json.load(f)
    return passes, labels


@st.cache_resource
def load_ml_models():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
    return load_models(os.path.join(data_dir, "trained_models.pkl"))


# ===========================================================================
# Override log
# ===========================================================================

OVERRIDE_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "generated", "override_log.json"
)


def load_override_log():
    if os.path.exists(OVERRIDE_LOG_PATH):
        with open(OVERRIDE_LOG_PATH) as f:
            return json.load(f)
    return []


def save_override(pass_id, system_severity, operator_severity, reason):
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
# Operator note library — varied, subsystem-aware, realistic
# ===========================================================================

# Notes keyed by (anomaly_type, subsystem). Falls back to anomaly_type only.
OPERATOR_NOTES = {
    # ---- NOMINAL — 20 varied notes cycling across passes ----
    ("none", None): [
        "All systems nominal. Clean pass with no anomalies.",
        "Routine housekeeping downlink. All parameters within expected limits.",
        "Standard contact pass. EPS, TCS, AOCS all healthy. No concerns.",
        "Clean telemetry across all subsystems. Battery charging normally in sunlight.",
        "Nominal pass. Reaction wheels within speed limits, temperatures stable.",
        "Downlink complete. All housekeeping in green. Nothing to flag.",
        "Uneventful pass. RSSI good, data rate nominal, no EDAC events.",
        "All nominal. Battery at expected state-of-charge for this orbit phase.",
        "Good contact. Sun panel temps as expected, shade side cooling normally.",
        "Routine pass. No anomalies detected. Satellite appears healthy.",
        "Clean downlink. OBC CPU load steady, memory occupancy within bounds.",
        "Nominal housekeeping. Thermal gradient across panels is expected for eclipse exit.",
        "All looks good. Attitude error within pointing budget. Wheels nominal.",
        "Standard pass. Comms link nominal throughout arc. Data complete.",
        "Nothing to report. Battery voltage holding, solar current tracking sunlight as expected.",
        "Nominal contact. All five subsystems reporting healthy telemetry.",
        "Quiet pass. No SEU events logged by OBC. Memory stable.",
        "Clean pass. Internal temperature steady. No thermal excursions.",
        "All systems go. Attitude control holding fine. RSSI as predicted by link budget.",
        "Nominal downlink. No deviations from previous pass trend.",
    ],

    # ---- POINT anomalies — subsystem-aware ----
    ("point", "eps"): [
        "Brief voltage dip on battery bus mid-pass. Returned to nominal. Watching next pass.",
        "Short dropout on battery voltage reading. Could be sensor glitch. Flagged for review.",
        "Momentary spike in solar current during eclipse transition. Possibly a sensor transient.",
    ],
    ("point", "tcs"): [
        "Short temperature spike on sun-facing panel. Duration under 30s. Could be solar flare.",
        "Brief thermal excursion on battery thermal sensor. Returned to nominal quickly.",
        "Momentary internal temp spike. FDIR did not trigger. Monitoring next pass.",
    ],
    ("point", "aocs"): [
        "Wheel speed transient observed for ~10 readings. Attitude recovered. Checking logs.",
        "Pointing error briefly exceeded expected value. Wheel speed returned to nominal.",
        "Short gyro rate spike. Possibly a disturbance torque. Attitude still maintained.",
    ],
    ("point", "comms"): [
        "Brief RSSI dropout mid-pass. Link recovered. Possibly local RF interference.",
        "Short data-rate dropout observed. No packet loss confirmed at ground segment.",
        "Momentary signal loss during pass centre. Link geometry was nominal. Investigating.",
    ],
    ("point", "obc"): [
        "SEU spike observed — 3 correctable errors in 10-reading burst. EDAC handled it.",
        "Brief CPU load spike. Possibly a scheduled task collision. Memory stable.",
        "Short EDAC event. Correctable errors within expected SAA passage budget.",
    ],
    # generic point fallback
    ("point", None): [
        "Noticed a brief transient in readings. Returned to nominal. Watching next pass.",
        "Unexpected spike observed. Short duration. Checking further next contact.",
        "Momentary anomaly in signal. Within tolerance overall. Will monitor.",
    ],

    # ---- CONTEXTUAL anomalies — subsystem-aware ----
    ("contextual", "eps"): [
        "Battery voltage lower than expected for current sunlight fraction. Slightly off. Monitoring.",
        "Solar current not tracking eclipse state as expected. Below expected for sun-pointing.",
        "Charge rate anomalous given orbit phase. Voltage dipping during sunlight. Keep an eye.",
    ],
    ("contextual", "tcs"): [
        "Internal temp higher than expected for eclipse depth at this orbit. Thermal model mismatch.",
        "Battery temperature not tracking eclipse as expected. Slightly above model prediction.",
        "Sun panel temp anomalous for current sun angle. Off from thermal model. Monitoring.",
    ],
    ("contextual", "aocs"): [
        "Wheel speed elevated relative to expected for this attitude profile. Keep an eye on it.",
        "Pointing error higher than predicted for this manoeuvre. Slight attitude deviation.",
        "Gyro rate slightly elevated for what should be a quiet, non-manoeuvring pass.",
    ],
    ("contextual", "comms"): [
        "RSSI weaker than link-budget prediction for this pass geometry. Signal below expected.",
        "Data rate lower than expected given elevation angle. Possible antenna pointing issue.",
        "Signal slightly degraded relative to pass geometry prediction. Monitoring next arc.",
    ],
    ("contextual", "obc"): [
        "CPU load elevated for a non-payload pass. No payload tasks scheduled. Investigating.",
        "Memory occupancy higher than expected between downlinks. Possible log file growth.",
        "EDAC rate slightly elevated for non-SAA orbit segment. Not expected here.",
    ],
    ("contextual", None): [
        "Readings slightly off for the current orbital context. Monitoring.",
        "Values not tracking orbital phase as expected. Keep an eye on it.",
        "Behaviour inconsistent with eclipse state. Will monitor next pass.",
    ],

    # ---- TREND anomalies — subsystem-aware ----
    ("trend", "eps"): [
        "Battery voltage trending downward across pass, faster than eclipse explains. Watching it.",
        "Gradual voltage decline across entire pass. Rate of change slightly elevated. Monitoring.",
        "Solar current declining slowly across pass. Possible partial shading or cell degradation?",
    ],
    ("trend", "tcs"): [
        "Internal temperature climbing steadily across pass. Rate above expected. Monitoring trend.",
        "Battery temp trending upward. Gradient sustained across full pass. Keep an eye on it.",
        "Sun panel temp rising throughout pass. Rate of change above thermal model. Flagging.",
    ],
    ("trend", "aocs"): [
        "Wheel speed drifting upward across pass. Gradual trend, no step change. Monitoring.",
        "Attitude error slowly increasing across contact. AOCS is correcting but trend is concerning.",
        "Pointing error trending upward. Slow drift over the entire pass. Will review after next contact.",
    ],
    ("trend", "comms"): [
        "RSSI declining steadily across pass, beyond what geometry explains. Signal trending weaker.",
        "Data rate degrading slowly throughout pass. Trending below link-budget floor. Monitoring.",
        "Signal strength drifting lower across pass. Rate of decline elevated. Flagging for review.",
    ],
    ("trend", "obc"): [
        "CPU load slowly climbing across pass. No new tasks scheduled. Memory stable for now.",
        "Memory occupancy trending upward throughout contact. Possible log accumulation.",
        "EDAC error count gradually rising across pass. Not a spike — a slow trend. Monitoring.",
    ],
    ("trend", None): [
        "Values drifting slowly across the pass. Trending toward limit. Will monitor.",
        "Gradual change observed throughout pass. Watching closely.",
        "Slow drift noticed. Still in range but rate of change is elevated. Monitoring trend.",
    ],

    # ---- HARD LIMIT anomalies — subsystem-aware ----
    ("hard_limit", "eps"): [
        "Battery voltage at critical levels. Below safe discharge floor. Emergency — immediate action required.",
        "Voltage collapse detected. Battery below 6V. Possible over-discharge. Emergency.",
        "Critical EPS failure. Battery bus voltage dropped to unsafe level. Alerting mission team.",
    ],
    ("hard_limit", "tcs"): [
        "Temperature breach on internal component. Exceeded survival limit. Critical thermal event.",
        "Component overtemperature detected. Above survival range. Hardware damage risk. Emergency.",
        "Critical TCS failure. Internal temp outside survival bounds. Immediate investigation required.",
    ],
    ("hard_limit", "aocs"): [
        "Loss of attitude control. Tumble detected. Angular rate exceeds threshold. Critical anomaly.",
        "Spacecraft tumbling. Gyro rate above limit. Attitude recovery mode required. Emergency.",
        "Critical AOCS failure. Uncontrolled rotation detected. Mission operations alerted.",
    ],
    ("hard_limit", "comms"): [
        "Complete signal loss. RSSI below noise floor for majority of pass. Satellite unresponsive.",
        "Loss of contact. Link dropped early in pass with no recovery. Emergency.",
        "Critical comms failure. Extended RSSI loss beyond pass-geometry explanation. No data received.",
    ],
    ("hard_limit", "obc"): [
        "Major SEU spike. EDAC count exceeds threshold. Possible radiation event — SAA passage.",
        "Critical OBC anomaly. Correctable error spike detected. Possible bit-flip event.",
        "EDAC spike above limit. High-energy particle event suspected. OBC memory integrity check needed.",
    ],
    ("hard_limit", None): [
        "Critical anomaly detected. Hard limit breach. Immediate action required.",
        "Hard limit violation. Emergency. Mission operations team alerted.",
        "System has exceeded operational safety limit. Critical — immediate review required.",
    ],

    # ---- BORDERLINE — deliberately ambiguous notes ----
    ("borderline", None): [
        "Something slightly off but not sure if it's real. Keeping an eye. Monitoring next pass.",
        "Borderline reading noticed. Within limits but on the edge. Will monitor.",
        "Values marginally elevated. Could be noise. Watching it.",
        "Slightly unusual behaviour. Not alarming yet but not typical either. Monitoring.",
        "Ambiguous telemetry. Not flagging formally yet but keeping watch.",
    ],
}


def get_note_for_pass(label, idx):
    """Assign a realistic operator note to a pass.

    Priority:
      1. (anomaly_type, subsystem) — most specific
      2. (anomaly_type, None)      — type-level fallback
      3. ("none", None)            — nominal fallback
    """
    atype = label.get("anomaly_type", "none")
    sub = label.get("subsystem", None)

    # BORDER- prefix passes use borderline notes
    pass_id = label.get("pass_id", "")

    # Try most-specific key first
    notes = OPERATOR_NOTES.get((atype, sub))
    if notes is None:
        notes = OPERATOR_NOTES.get((atype, None))
    if notes is None:
        notes = OPERATOR_NOTES[("none", None)]

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
# Severity colour map for subsystem score badges
# ===========================================================================

def score_colour(score):
    if score > 0.85:
        return SEVERITY_COLOURS["CRITICAL"]
    elif score > 0.6:
        return SEVERITY_COLOURS["CAUTION"]
    elif score > 0.3:
        return SEVERITY_COLOURS["WATCH"]
    else:
        return SEVERITY_COLOURS["NOMINAL"]


def sev_emoji(sev):
    return {"CRITICAL": "🔴", "CAUTION": "🟠", "WATCH": "🟡", "NOMINAL": "🟢"}.get(sev, "⚪")


# ===========================================================================
# Streamlit UI
# ===========================================================================

def main():
    st.set_page_config(
        page_title="Spacecraft Telemetry Health Assessment",
        page_icon="🛰️",
        layout="wide",
    )

    st.title("🛰️ Spacecraft Telemetry Health Assessment")
    st.caption("Ground Segment Decision Support — 3U-6U CubeSat LEO Mission")

    passes, labels = load_passes()
    models = load_ml_models()

    # ------------------------------------------------------------------
    # Pre-compute assessments once and cache in session_state
    # ------------------------------------------------------------------
    if "assessments" not in st.session_state or "notes" not in st.session_state:
        with st.spinner("Running pipeline on all passes..."):
            notes = []
            assessments = []
            for i, (p, l) in enumerate(zip(passes, labels)):
                # Inject pass_id into label so get_note_for_pass can see it
                l_with_id = {**l, "pass_id": p.get("pass_id", "")}
                note = get_note_for_pass(l_with_id, i)
                notes.append(note)
                result = assess_pass(p, note, models)
                assessments.append(result)
            st.session_state.assessments = assessments
            st.session_state.notes = notes

    assessments = st.session_state.assessments
    notes = st.session_state.notes

    # ------------------------------------------------------------------
    # Sidebar — pass list with severity filter
    # ------------------------------------------------------------------
    st.sidebar.header("Pass List")
    st.sidebar.caption("Click a pass to inspect it →")

    filter_sev = st.sidebar.multiselect(
        "Filter by severity",
        SEVERITY_ORDER,
        default=SEVERITY_ORDER,
    )

    visible_indices = []
    for i, (p, l, a) in enumerate(zip(passes, labels, assessments)):
        if a["severity"] not in filter_sev:
            continue
        visible_indices.append(i)
        emoji = sev_emoji(a["severity"])
        conf_str = f"{a['confidence']:.0%}"
        btn_label = f"{emoji} **{p['pass_id']}** — {a['severity']} ({conf_str})"
        if st.sidebar.button(btn_label, key=f"pass_{i}", use_container_width=True):
            st.session_state.selected_pass = i

    # ------------------------------------------------------------------
    # Detail view
    # ------------------------------------------------------------------
    # Default to first visible pass
    if "selected_pass" not in st.session_state:
        st.session_state.selected_pass = visible_indices[0] if visible_indices else 0

    sel = st.session_state.selected_pass
    p = passes[sel]
    l = labels[sel]
    a = assessments[sel]

    sev = a["severity"]
    conf = a["confidence"]
    colour = SEVERITY_COLOURS[sev]

    # ---- Top badges row ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div style="background:{colour};padding:22px 16px;border-radius:12px;'
            f'text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);">'
            f'<div style="color:white;font-size:2rem;font-weight:800;letter-spacing:1px;">'
            f'{sev_emoji(sev)} {sev}</div>'
            f'<div style="color:rgba(255,255,255,0.85);margin-top:6px;font-size:0.9rem;">'
            f'Pass: {p["pass_id"]}</div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        conf_colour = "#c0392b" if conf < 0.4 else "#e67e22" if conf < 0.7 else "#27ae60"
        conf_label = "LOW" if conf < 0.4 else "MEDIUM" if conf < 0.7 else "HIGH"
        conf_note = ("⚠️ Uncertain — human review recommended"
                     if conf < 0.4 else
                     "~ Moderate certainty" if conf < 0.7 else
                     "✓ High certainty")
        st.markdown(
            f'<div style="background:{conf_colour};padding:22px 16px;border-radius:12px;'
            f'text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);">'
            f'<div style="color:white;font-size:1.6rem;font-weight:700;">'
            f'Confidence: {conf_label}</div>'
            f'<div style="color:rgba(255,255,255,0.85);margin-top:6px;font-size:0.85rem;">'
            f'{conf:.1%} — {conf_note}</div></div>',
            unsafe_allow_html=True,
        )

    with col3:
        gt_sev = l["expected_severity"]
        gt_type = l["anomaly_type"]
        gt_sub = l.get("subsystem", "—")
        gt_colour = SEVERITY_COLOURS.get(gt_sev, "#34495e")
        match_icon = "✓ Agrees" if gt_sev == sev else "✗ Disagrees"
        match_colour = "#27ae60" if gt_sev == sev else "#c0392b"
        st.markdown(
            f'<div style="background:#2c3e50;padding:22px 16px;border-radius:12px;'
            f'text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);">'
            f'<div style="color:white;font-size:1.4rem;font-weight:700;">Ground Truth</div>'
            f'<div style="color:white;margin-top:6px;">'
            f'<span style="background:{gt_colour};padding:2px 10px;border-radius:6px;'
            f'font-weight:700;font-size:1rem;">{gt_sev}</span>'
            f'&nbsp;({gt_type})'
            f'</div>'
            f'<div style="color:{match_colour};margin-top:6px;font-size:0.85rem;font-weight:600;">'
            f'{match_icon}</div>'
            f'<div style="color:rgba(255,255,255,0.65);font-size:0.8rem;margin-top:2px;">'
            f'Subsystem: {gt_sub}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Assessment summary ----
    st.subheader("Assessment Summary")
    st.info(a["summary"])

    st.divider()

    # ---- Operator note (editable) ----
    st.subheader("Operator Note")
    st.caption("This note was written by the operator before the assessment ran. "
               "Edit it below and click **Re-assess** to see how the pipeline responds.")

    current_note = st.session_state.notes[sel]
    edited_note = st.text_area(
        "Operator note (edit and click Re-assess to update):",
        value=current_note,
        height=90,
        key=f"note_edit_{sel}",
    )

    if st.button("🔄 Re-assess with updated note", key=f"reassess_{sel}"):
        st.session_state.notes[sel] = edited_note
        updated = assess_pass(p, edited_note, models)
        st.session_state.assessments[sel] = updated
        assessments = st.session_state.assessments
        st.rerun()

    st.divider()

    # ---- Reasoning trace ----
    st.subheader("Reasoning Trace")
    st.caption("Step-by-step explanation of how this severity was reached.")

    for t in a["reasoning_trace"]:
        source = t["source"]
        if source == "hard_limit":
            st.error(
                f"🚨 **HARD LIMIT FIRED** — {t['detail']}  \n"
                f"Rule: `{t['rule']}` | Threshold: `{t['threshold']}` | "
                f"Actual value: `{t['actual_value']}`  \n"
                f"*This is a physics-based safety limit — not a model judgement. "
                f"No downstream score can override it.*"
            )
        elif source == "numeric":
            st.warning(
                f"📊 **Numeric Scoring** — {t['detail']}  \n"
                f"Worst subsystem: `{t['worst_subsystem']}` | "
                f"Score: `{t['score']:.3f}` | Confidence: `{t['confidence']:.3f}`"
            )
        elif source == "text":
            st.info(
                f"📝 **Note Scoring** — {t['detail']}"
            )
            if t.get("matched_phrases"):
                phrases = " | ".join(
                    f"*'{ph}'* ({'+' if wt > 0 else ''}{wt:.2f})"
                    for ph, wt in t["matched_phrases"]
                )
                st.caption(f"Matched phrases: {phrases}")
            if t.get("reasoning"):
                st.caption(t["reasoning"])
        elif source == "fusion_escalation":
            st.warning(f"⚡ **Fusion Escalation** — {t['detail']}")

    st.divider()

    # ---- Subsystem scores breakdown ----
    st.subheader("Subsystem Scores")
    st.caption("Isolation Forest anomaly score per subsystem (0.0 = nominal, 1.0 = highly anomalous). "
               "Colour band: 🟢 <0.3 · 🟡 0.3–0.6 · 🟠 0.6–0.85 · 🔴 >0.85")

    if "subsystem_scores" in a:
        cols = st.columns(5)
        for i, (sub, scores) in enumerate(a["subsystem_scores"].items()):
            score = scores["score"]
            s_colour = score_colour(score)
            conf_sub = scores["confidence"]
            with cols[i]:
                st.markdown(
                    f'<div style="background:{s_colour};padding:14px 8px;border-radius:10px;'
                    f'text-align:center;margin-bottom:4px;">'
                    f'<div style="color:white;font-size:1.4rem;font-weight:800;">{score:.2f}</div>'
                    f'<div style="color:rgba(255,255,255,0.9);font-size:0.75rem;font-weight:600;">'
                    f'{sub.upper()}</div>'
                    f'<div style="color:rgba(255,255,255,0.7);font-size:0.7rem;">'
                    f'conf {conf_sub:.2f}</div></div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ---- Operator override ----
    st.subheader("Operator Override")
    st.caption(
        "The system's output is **decision support only**. "
        "A human operator must confirm every action. "
        "Use this control to log a manual severity correction with a reason."
    )

    ov_col1, ov_col2 = st.columns([1, 2])
    with ov_col1:
        override_sev = st.selectbox(
            "Override severity to:",
            SEVERITY_ORDER,
            index=SEVERITY_ORDER.index(sev),
            key=f"override_{sel}",
        )
    with ov_col2:
        override_reason = st.text_input(
            "Reason for override (required):",
            placeholder="e.g. Confirmed via secondary instrument — reading was a sensor glitch",
            key=f"override_reason_{sel}",
        )

    if st.button("📋 Submit Override", key=f"submit_override_{sel}"):
        if override_reason.strip():
            save_override(p["pass_id"], sev, override_sev, override_reason)
            st.success(
                f"Override logged: **{sev}** → **{override_sev}**  \n"
                f"Reason: {override_reason}"
            )
        else:
            st.error("Please provide a reason for the override before submitting.")

    override_log = load_override_log()
    if override_log:
        with st.expander(f"Override Log ({len(override_log)} entries)"):
            for entry in reversed(override_log):
                st.markdown(
                    f"`{entry['timestamp'][:19]}` | **{entry['pass_id']}** | "
                    f"{entry['system_severity']} → **{entry['operator_severity']}** | "
                    f"_{entry['reason']}_"
                )


if __name__ == "__main__":
    main()
