"""
Compares late fusion (the default pipeline) vs early fusion (concatenated
numeric + text features into a single classifier).
Builds an early-fusion variant (logistic regression on a concatenated numeric + keyword-text feature vector) and runs both against the full labelled set. 
Headline case: a genuine battery anomaly paired with a reassuring note — late fusion preserves the high severity, early fusion lets the reassuring text dilute it. 
Demonstrates why late fusion is architecturally safer.
"""

import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.numeric_scoring import (
    extract_all_features, load_models, score_pass, aggregate_subsystem_scores,
    SUBSYSTEMS,
)
from pipeline.text_scoring import score_note, CONCERN_PHRASES, REASSURANCE_PHRASES
from pipeline.hard_limits import evaluate_hard_limits
from pipeline.fusion import fuse_scores
from pipeline.severity import score_to_severity, SEVERITY_ORDER


# ===========================================================================
# Early fusion: concatenate numeric + text features into one vector
# ===========================================================================

def extract_text_features(note_text):
    """Convert a note into a numeric feature vector for early fusion.

    Uses keyword presence as binary features, matching the keyword classifier's
    vocabulary. This is the text equivalent of the numeric feature extraction.
    """
    text = note_text.lower().strip()
    features = []
    # One binary feature per concern phrase
    for phrase in sorted(CONCERN_PHRASES.keys()):
        features.append(1.0 if phrase in text else 0.0)
    # One binary feature per reassurance phrase
    for phrase in sorted(REASSURANCE_PHRASES.keys()):
        features.append(1.0 if phrase in text else 0.0)
    return features


def extract_early_fusion_features(pass_data, note_text):
    """Concatenate all numeric features + text features into one vector."""
    numeric_feats = []
    for sub in SUBSYSTEMS:
        numeric_feats.extend(extract_all_features(pass_data)[sub])
    text_feats = extract_text_features(note_text)
    return numeric_feats + text_feats


# Severity → numeric label for the classifier
SEVERITY_TO_LABEL = {"NOMINAL": 0, "WATCH": 1, "CAUTION": 2, "CRITICAL": 3}
LABEL_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_LABEL.items()}


# ===========================================================================
# Demo note assignment (same logic as the UI)
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
    atype = label["anomaly_type"]
    notes = DEMO_NOTES.get(atype, DEMO_NOTES["none"])
    return notes[idx % len(notes)]


# ===========================================================================
# Main comparison
# ===========================================================================

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")

    # Load data
    with open(os.path.join(data_dir, "labelled_passes.json")) as f:
        passes = json.load(f)
    with open(os.path.join(data_dir, "labels.json")) as f:
        labels = json.load(f)
    models = load_models(os.path.join(data_dir, "trained_models.pkl"))

    # Assign notes to passes
    notes = [get_note_for_pass(l, i) for i, l in enumerate(labels)]

    # ------------------------------------------------------------------
    # Build early-fusion classifier
    # ------------------------------------------------------------------
    print("=== Building early-fusion classifier ===")

    X_early = []
    y_early = []
    for p, l, note in zip(passes, labels, notes):
        feats = extract_early_fusion_features(p, note)
        X_early.append(feats)
        y_early.append(SEVERITY_TO_LABEL[l["expected_severity"]])

    X_early = np.array(X_early)
    y_early = np.array(y_early)

    scaler = MinMaxScaler()
    X_early_scaled = scaler.fit_transform(X_early)

    # Train logistic regression on the combined feature vector
    early_clf = LogisticRegression(max_iter=1000, random_state=42)
    early_clf.fit(X_early_scaled, y_early)
    print(f"  Early fusion classifier trained on {len(X_early)} passes")
    print(f"  Feature vector length: {X_early.shape[1]} "
          f"(numeric: {X_early.shape[1] - len(extract_text_features(''))}, "
          f"text: {len(extract_text_features(''))})")

    # ------------------------------------------------------------------
    # Run both topologies against all passes
    # ------------------------------------------------------------------
    print("\n=== Comparing late fusion vs early fusion ===\n")

    late_results = []
    early_results = []

    for i, (p, l, note) in enumerate(zip(passes, labels, notes)):
        # Late fusion (default pipeline)
        breaches = evaluate_hard_limits(p)
        sub_scores = score_pass(p, models, "iforest")
        agg = aggregate_subsystem_scores(sub_scores)
        text = score_note(note)
        late = fuse_scores(breaches, agg, text, note)
        late_results.append(late["severity"])

        # Early fusion
        x = scaler.transform(np.array(extract_early_fusion_features(p, note)).reshape(1, -1))
        early_pred = early_clf.predict(x)[0]
        early_sev = LABEL_TO_SEVERITY[early_pred]
        early_results.append(early_sev)

    # ------------------------------------------------------------------
    # Hard-limit-adjacent numeric signal + reassuring note
    # ------------------------------------------------------------------
    print("=== KEY COMPARISON: Hard-limit-adjacent signal + reassuring note ===\n")

    # Create the specific test case
    from data.synthetic_generator import generate_one_pass
    from data.anomaly_injection import inject_trend_eps

    rng = np.random.default_rng(999)
    base_pass = generate_one_pass(rng, "FUSION-TEST-001")

    # Inject a trend that pushes battery voltage close to the 6.0V hard limit
    # severity_scale=0.8 → strong drift but just above hard-limit floor
    anomalous_pass, _ = inject_trend_eps(base_pass, rng, severity_scale=0.8)
    reassuring_note = "All looks fine. Nominal pass. Battery stable, no concerns."

    # Late fusion on this case
    breaches = evaluate_hard_limits(anomalous_pass)
    sub_scores = score_pass(anomalous_pass, models, "iforest")
    agg = aggregate_subsystem_scores(sub_scores)
    text = score_note(reassuring_note)
    late_case = fuse_scores(breaches, agg, text, reassuring_note)

    # Early fusion on this case
    x_case = scaler.transform(
        np.array(extract_early_fusion_features(anomalous_pass, reassuring_note)).reshape(1, -1)
    )
    early_case_pred = early_clf.predict(x_case)[0]
    early_case_sev = LABEL_TO_SEVERITY[early_case_pred]

    bv = anomalous_pass["eps"]["battery_voltage"]
    print(f"Pass: FUSION-TEST-001")
    print(f"  Battery voltage: min={min(bv):.2f}V, max={max(bv):.2f}V")
    print(f"  Note: \"{reassuring_note}\"")
    print(f"")
    print(f"  LATE FUSION result:  {late_case['severity']} "
          f"(confidence: {late_case['confidence']:.2f})")
    print(f"    Numeric says: {score_to_severity(agg['overall_score'])} "
          f"(score={agg['overall_score']:.3f})")
    print(f"    Text says:    {score_to_severity(text['score'])} "
          f"(score={text['score']:.3f})")
    print(f"    -> Late fusion takes the HIGHER of the two = {late_case['severity']}")
    print(f"")
    print(f"  EARLY FUSION result: {early_case_sev}")
    print(f"    -> Single classifier trained on concatenated features")
    print(f"    -> Note's reassuring features can dilute the alarming numeric signal")

    if SEVERITY_ORDER.index(late_case["severity"]) > SEVERITY_ORDER.index(early_case_sev):
        print(f"\n  ** PDD PREDICTION CONFIRMED: Late fusion preserves the high severity")
        print(f"     ({late_case['severity']}), while early fusion dilutes it to ({early_case_sev}).")
        print(f"     This is why the PDD recommends late fusion — a reassuring note")
        print(f"     should never downgrade a genuine numeric anomaly. **")
    elif late_case["severity"] == early_case_sev:
        print(f"\n  Both topologies agree at {late_case['severity']}.")
        print(f"  The dilution effect may not appear at this severity level,")
        print(f"  but late fusion is still architecturally safer because it")
        print(f"  guarantees the note can never override the numeric signal.")
    else:
        print(f"\n  Unexpected: early fusion gave a HIGHER severity than late fusion.")
        print(f"  This can happen when the early classifier learned different boundaries.")

    # ------------------------------------------------------------------
    # Overall accuracy comparison
    # ------------------------------------------------------------------
    print("\n=== Overall accuracy comparison ===\n")
    expected = [l["expected_severity"] for l in labels]

    late_correct = sum(1 for a, e in zip(late_results, expected) if a == e)
    early_correct = sum(1 for a, e in zip(early_results, expected) if a == e)

    print(f"  Late fusion accuracy:  {late_correct}/{len(expected)} "
          f"({late_correct/len(expected)*100:.1f}%)")
    print(f"  Early fusion accuracy: {early_correct}/{len(expected)} "
          f"({early_correct/len(expected)*100:.1f}%)")

    # Per-severity breakdown
    for sev in SEVERITY_ORDER:
        mask = [e == sev for e in expected]
        n = sum(mask)
        if n == 0:
            continue
        late_match = sum(1 for m, a, e in zip(mask, late_results, expected) if m and a == e)
        early_match = sum(1 for m, a, e in zip(mask, early_results, expected) if m and a == e)
        print(f"  {sev:10s}: late={late_match}/{n}, early={early_match}/{n}")

    # Save results
    results = {
        "key_case": {
            "pass_id": "FUSION-TEST-001",
            "battery_voltage_range": [round(min(bv), 2), round(max(bv), 2)],
            "note": reassuring_note,
            "late_fusion_severity": late_case["severity"],
            "late_fusion_confidence": late_case["confidence"],
            "early_fusion_severity": early_case_sev,
        },
        "overall_accuracy": {
            "late_fusion": round(late_correct / len(expected), 4),
            "early_fusion": round(early_correct / len(expected), 4),
        },
    }
    results_path = os.path.join(data_dir, "fusion_topology_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
