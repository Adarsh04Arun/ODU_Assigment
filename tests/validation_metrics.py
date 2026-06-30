"""
Step 10 - Validation and Metrics.

Implements PDD requirements for validation:
10.1 - Compute precision/recall/F1 per subsystem for numeric models.
     - Compute overall accuracy/precision/recall for fused severity output.
10.2 - Validate hard-limit layer (100% detection, correct attribution).
10.3 - Validate calibration on borderline/uncertain cases (low confidence surface).
"""

import json
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.hard_limits import evaluate_hard_limits
from pipeline.numeric_scoring import load_models, score_pass, aggregate_subsystem_scores, SUBSYSTEMS
from pipeline.text_scoring import score_note
from pipeline.fusion import fuse_scores
from pipeline.severity import score_to_severity

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")

    # Load data
    with open(os.path.join(data_dir, "labelled_passes.json")) as f:
        passes = json.load(f)
    with open(os.path.join(data_dir, "labels.json")) as f:
        labels = json.load(f)
    models = load_models(os.path.join(data_dir, "trained_models.pkl"))

    # Demo notes assigned based on anomaly_type, to feed the full pipeline
    from ui.app import get_note_for_pass
    notes = [get_note_for_pass(l, i) for i, l in enumerate(labels)]

    print("=== Step 10: Validation and Metrics ===\n")

    # =========================================================================
    # 10.2 Hard-Limit Layer Validation
    # =========================================================================
    print("--- 10.2 Hard-Limit Layer Validation ---")
    hl_passes = [(p, l) for p, l in zip(passes, labels) if l["anomaly_type"] == "hard_limit"]
    hl_caught = 0
    hl_correct_subsystem = 0
    for p, l in hl_passes:
        breaches = evaluate_hard_limits(p)
        if len(breaches) > 0:
            hl_caught += 1
            if any(b["subsystem"] == l["subsystem"] for b in breaches):
                hl_correct_subsystem += 1

    print(f"Total hard-limit passes: {len(hl_passes)}")
    print(f"Caught by hard-limit rules: {hl_caught}/{len(hl_passes)} ({hl_caught/len(hl_passes)*100:.1f}%)")
    print(f"Correctly attributed to subsystem: {hl_correct_subsystem}/{len(hl_passes)} ({hl_correct_subsystem/len(hl_passes)*100:.1f}%)")
    if hl_caught == len(hl_passes) and hl_correct_subsystem == len(hl_passes):
        print("-> VALIDATION PASSED: 100% of hard-limit breaches caught and correctly attributed.")
    else:
        print("-> VALIDATION FAILED on hard-limit breaches.")
    print()


    # =========================================================================
    # 10.1 Numeric Model & Fused Pipeline Validation
    # =========================================================================
    print("--- 10.1 Numeric Scoring & Pipeline Validation ---")
    
    # We will evaluate Subsystem Detection (Numeric Score > 0.5 means anomaly in that subsystem)
    # Ground truth: if pass is anomalous, l["subsystem"] is the anomaly location.
    
    y_true_subsystems = {sub: [] for sub in SUBSYSTEMS}
    y_pred_subsystems_iforest = {sub: [] for sub in SUBSYSTEMS}
    y_pred_subsystems_ocsvm = {sub: [] for sub in SUBSYSTEMS}

    y_true_fused = []
    y_pred_fused = []
    
    for p, l, note in zip(passes, labels, notes):
        # 1. Ground truth for subsystem anomaly (binary: anomaly or not in this sub)
        # Note: 'hard_limit' and 'none' are handled properly
        for sub in SUBSYSTEMS:
            is_anomaly = (l["anomaly_type"] != "none" and l.get("subsystem") == sub)
            y_true_subsystems[sub].append(1 if is_anomaly else 0)
        
        # Numeric predictions (IForest and OCSVM)
        sub_scores_if = score_pass(p, models, "iforest")
        sub_scores_oc = score_pass(p, models, "ocsvm")
        
        for sub in SUBSYSTEMS:
            # Threshold 0.5 to declare anomaly in numeric model
            y_pred_subsystems_iforest[sub].append(1 if sub_scores_if[sub]["score"] >= 0.5 else 0)
            y_pred_subsystems_ocsvm[sub].append(1 if sub_scores_oc[sub]["score"] >= 0.5 else 0)

        # 2. Pipeline Fused output vs Expected Severity
        breaches = evaluate_hard_limits(p)
        agg_if = aggregate_subsystem_scores(sub_scores_if)
        text = score_note(note)
        fused = fuse_scores(breaches, agg_if, text, note)
        
        y_true_fused.append(l["expected_severity"])
        y_pred_fused.append(fused["severity"])

    print("Numeric Models (Binary Detection >= 0.5 score):")
    for sub in SUBSYSTEMS:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_subsystems[sub], y_pred_subsystems_iforest[sub], average='binary', zero_division=0)
        print(f"  Isolation Forest - {sub.upper()}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

        precision_oc, recall_oc, f1_oc, _ = precision_recall_fscore_support(y_true_subsystems[sub], y_pred_subsystems_ocsvm[sub], average='binary', zero_division=0)
        print(f"  One-Class SVM    - {sub.upper()}: Precision={precision_oc:.2f}, Recall={recall_oc:.2f}, F1={f1_oc:.2f}")
    
    print("\nFused Pipeline (Expected Severity vs Predicted Severity):")
    # Note: As discussed, expected_severity from injector is arbitrary for non-hard-limits,
    # so we expect accuracy here to be low, but we provide it as requested in Step 10.1
    print(classification_report(y_true_fused, y_pred_fused, zero_division=0))
    print()


    # =========================================================================
    # 10.3 Calibration on Borderline/Uncertain Cases
    # =========================================================================
    print("--- 10.3 Calibration Validation (Borderline Cases) ---")
    
    borderline_passes = [(p, l, note) for p, l, note in zip(passes, labels, notes) if p["pass_id"].startswith("BORDER-")]
    
    low_conf_count = 0
    print(f"Total Borderline Passes: {len(borderline_passes)}")
    for p, l, note in borderline_passes:
        breaches = evaluate_hard_limits(p)
        sub_scores = score_pass(p, models, "iforest")
        agg = aggregate_subsystem_scores(sub_scores)
        text = score_note(note)
        fused = fuse_scores(breaches, agg, text, note)
        
        conf = fused["confidence"]
        is_low_conf = (conf <= 0.4)
        if is_low_conf:
            low_conf_count += 1
            
        print(f"  Pass: {p['pass_id']} | Type: {l['anomaly_type']} | Expected: {l['expected_severity']}")
        print(f"    Fused Sev: {fused['severity']} | Conf: {conf:.2f} -> {'Low/Medium Confidence' if conf <= 0.7 else 'High Confidence'}")
        
    # We want to see lower confidence for borderline cases
    print(f"Low/Medium confidence surfaced in: {low_conf_count}/{len(borderline_passes)} borderline cases.")
    print("-> VALIDATION CHECK: Are borderline passes correctly surfaced with uncertainty? (Check confidence values above).")

if __name__ == "__main__":
    main()
