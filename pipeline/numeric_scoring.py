"""
Numeric anomaly scoring layer — classical ML per-subsystem anomaly detection.

Implements PDD Section 6.1 (hybrid detection) and Section 6.2 (continuous
severity scoring, not binary flags).

Architecture: one Isolation Forest (primary) and one One-Class SVM (comparison)
per subsystem, trained on nominal-only passes. Outputs a continuous anomaly
score (0.0 = fully normal, 1.0 = maximally anomalous) per subsystem per pass,
plus a confidence band.

PDD Section 7.2.2 decided per-subsystem scoring (not unified) because it gives
free subsystem-level attribution in the reasoning trace.
"""

import json
import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler


# ===========================================================================
# 5.1 — Feature engineering per subsystem
# For each pass, compute summary statistics over the channel time series.
# These are "windowed" features (summary over the full pass window).
# ===========================================================================

def extract_features_eps(pass_data):
    """EPS features: voltage stats, solar current stats, charge rate."""
    v = np.array(pass_data["eps"]["battery_voltage"])
    sc = np.array(pass_data["eps"]["solar_current"])
    cr = np.array(pass_data["eps"]["charge_rate"])
    return [
        np.mean(v), np.std(v), np.min(v), np.max(v), v[-1],  # end-of-pass
        np.mean(np.diff(v)),  # rate of change (voltage trend)
        np.mean(sc), np.std(sc),
        np.mean(cr),
    ]


def extract_features_tcs(pass_data):
    """TCS features: temperature stats for all four channels."""
    sp = np.array(pass_data["tcs"]["sun_panel_temp"])
    sh = np.array(pass_data["tcs"]["shade_panel_temp"])
    bt = np.array(pass_data["tcs"]["battery_temp"])
    it = np.array(pass_data["tcs"]["internal_temp"])
    return [
        np.mean(sp), np.std(sp), np.max(sp),
        np.mean(sh), np.std(sh),
        np.mean(bt), np.std(bt),
        np.mean(it), np.std(it), np.max(it),
        np.mean(np.diff(it)),  # internal temp trend
    ]


def extract_features_aocs(pass_data):
    """AOCS features: wheel speed, pointing, gyro stats."""
    ws = np.array(pass_data["aocs"]["wheel_speed"])
    pe = np.array(pass_data["aocs"]["pointing_error"])
    gr = np.array(pass_data["aocs"]["gyro_rate"])
    ae = np.array(pass_data["aocs"]["attitude_error"])
    return [
        np.mean(ws), np.std(ws), np.max(ws),
        np.mean(pe), np.max(pe),
        np.mean(np.abs(gr)), np.max(np.abs(gr)),  # abs because direction irrelevant
        np.mean(ae), np.max(ae),
    ]


def extract_features_comms(pass_data):
    """Comms features: RSSI and data rate stats."""
    rssi = np.array(pass_data["comms"]["rssi"])
    dr = np.array(pass_data["comms"]["data_rate"])
    return [
        np.mean(rssi), np.std(rssi), np.min(rssi),
        np.mean(dr), np.std(dr), np.min(dr),
    ]


def extract_features_obc(pass_data):
    """OBC features: CPU, memory, and SEU count stats."""
    cpu = np.array(pass_data["obc"]["cpu_load"])
    mem = np.array(pass_data["obc"]["memory_occupancy"])
    seu = np.array(pass_data["obc"]["seu_count"])
    return [
        np.mean(cpu), np.std(cpu), np.max(cpu),
        np.mean(mem), np.std(mem), np.max(mem),
        np.sum(seu), np.max(seu),  # total and peak SEU
    ]


# Map subsystem name to its feature extractor
FEATURE_EXTRACTORS = {
    "eps": extract_features_eps,
    "tcs": extract_features_tcs,
    "aocs": extract_features_aocs,
    "comms": extract_features_comms,
    "obc": extract_features_obc,
}

SUBSYSTEMS = list(FEATURE_EXTRACTORS.keys())


def extract_all_features(pass_data):
    """Extract feature vectors for all subsystems from one pass.

    Returns:
        dict of subsystem_name -> feature list
    """
    return {sub: fn(pass_data) for sub, fn in FEATURE_EXTRACTORS.items()}


# ===========================================================================
# 5.2 / 5.3 — Model training: Isolation Forest (primary) + One-Class SVM
# ===========================================================================

def train_models(nominal_passes):
    """Train one Isolation Forest and one One-Class SVM per subsystem.

    Args:
        nominal_passes: list of pass dicts (nominal only, no anomalies).

    Returns:
        dict with keys:
            "iforest": {subsystem: fitted IsolationForest}
            "ocsvm":   {subsystem: fitted OneClassSVM}
            "scalers": {subsystem: fitted MinMaxScaler}
    """
    # Build feature matrices per subsystem
    features_by_sub = {sub: [] for sub in SUBSYSTEMS}
    for p in nominal_passes:
        feats = extract_all_features(p)
        for sub in SUBSYSTEMS:
            features_by_sub[sub].append(feats[sub])

    models = {"iforest": {}, "ocsvm": {}, "scalers": {}, "baselines": {}}

    for sub in SUBSYSTEMS:
        X = np.array(features_by_sub[sub])

        # 5.4 — Normalise onto common scale
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        models["scalers"][sub] = scaler

        # Isolation Forest (primary detector)
        # contamination=0.05: we expect ~5% of nominal data might be borderline
        iforest = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        iforest.fit(X_scaled)
        models["iforest"][sub] = iforest

        # One-Class SVM (comparison model)
        ocsvm = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,  # similar to contamination
        )
        ocsvm.fit(X_scaled)
        models["ocsvm"][sub] = ocsvm

        # Store training score baselines for calibrated normalisation
        # We'll convert test scores to percentile rank against these
        iforest_train_scores = iforest.score_samples(X_scaled)
        ocsvm_train_scores = ocsvm.decision_function(X_scaled)
        models["baselines"][sub] = {
            "iforest_scores": np.sort(iforest_train_scores),
            "ocsvm_scores": np.sort(ocsvm_train_scores),
        }

    return models



# ===========================================================================
# 5.4 / 5.5 — Scoring: continuous anomaly score + confidence
# ===========================================================================

def score_pass(pass_data, models, model_type="iforest"):
    """Score a single pass across all subsystems.

    Args:
        pass_data: a single pass dict.
        models: dict from train_models().
        model_type: "iforest" or "ocsvm".

    Returns:
        dict with keys per subsystem, each containing:
            "score":      float in [0, 1] (0 = normal, 1 = anomalous)
            "confidence": float in [0, 1] (how far from the decision boundary)
            "features":   the raw feature values (for reasoning trace)
    """
    feats = extract_all_features(pass_data)
    results = {}

    for sub in SUBSYSTEMS:
        x = np.array(feats[sub]).reshape(1, -1)
        x_scaled = models["scalers"][sub].transform(x)

        model = models[model_type][sub]
        baseline_key = f"{model_type}_scores"
        train_scores = models["baselines"][sub][baseline_key]

        if model_type == "iforest":
            raw_score = model.score_samples(x_scaled)[0]
        else:
            raw_score = model.decision_function(x_scaled)[0]

        # Percentile-rank normalisation against the training distribution.
        # Lower raw scores = more anomalous, so we convert:
        #   percentile 0 (below all training) → anomaly_score 1.0
        #   percentile 100 (above all training) → anomaly_score 0.0
        percentile = np.searchsorted(train_scores, raw_score) / len(train_scores)
        anomaly_score = np.clip(1.0 - percentile, 0.0, 1.0)

        # 5.5 — Confidence: how far from the decision boundary (0.5 score)
        # Near the boundary -> low confidence; far from it -> high confidence
        confidence = min(1.0, abs(anomaly_score - 0.5) * 2.0)

        results[sub] = {
            "score": round(float(anomaly_score), 4),
            "confidence": round(float(confidence), 4),
            "features": [round(float(f), 4) for f in feats[sub]],
        }


    return results


def aggregate_subsystem_scores(subsystem_scores):
    """Combine per-subsystem scores into an overall numeric severity score.

    Uses max-of-subsystems: the overall score is the worst (highest) subsystem.
    This is the simplest rule and gives free attribution — the subsystem with
    the highest score is the one driving the severity.

    Returns:
        dict with "overall_score", "overall_confidence", "worst_subsystem",
        and the full per-subsystem breakdown.
    """
    worst_sub = max(subsystem_scores, key=lambda s: subsystem_scores[s]["score"])
    overall_score = subsystem_scores[worst_sub]["score"]

    # Overall confidence: minimum across subsystems (PDD Section 7.2.2)
    overall_confidence = min(s["confidence"] for s in subsystem_scores.values())

    return {
        "overall_score": overall_score,
        "overall_confidence": overall_confidence,
        "worst_subsystem": worst_sub,
        "subsystem_scores": subsystem_scores,
    }


def save_models(models, path):
    """Save trained models to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)


def load_models(path):
    """Load trained models from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ===========================================================================
# CLI: train and evaluate
# ===========================================================================

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.severity import score_to_severity

    # Load nominal-only training data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
    with open(os.path.join(data_dir, "nominal_passes.json")) as f:
        nominal = json.load(f)
    print(f"Training on {len(nominal)} nominal passes...")

    # Train both models
    models = train_models(nominal)
    model_path = os.path.join(data_dir, "trained_models.pkl")
    save_models(models, model_path)
    print(f"Models saved to {model_path}")

    # Load labelled test set
    with open(os.path.join(data_dir, "labelled_passes.json")) as f:
        test_passes = json.load(f)
    with open(os.path.join(data_dir, "labels.json")) as f:
        test_labels = json.load(f)

    # Evaluate both models
    for model_type in ["iforest", "ocsvm"]:
        print(f"\n=== {model_type.upper()} Results ===")
        correct = 0
        total = 0
        for p, l in zip(test_passes, test_labels):
            sub_scores = score_pass(p, models, model_type)
            agg = aggregate_subsystem_scores(sub_scores)
            predicted_sev = score_to_severity(agg["overall_score"])
            expected = l["expected_severity"]

            # For hard-limit passes, the hard-limit layer handles them,
            # so we only evaluate non-hard-limit passes here
            if l["anomaly_type"] == "hard_limit":
                continue

            total += 1
            if predicted_sev == expected:
                correct += 1

        print(f"  Accuracy (non-hard-limit): {correct}/{total} ({correct/total*100:.1f}%)")

        # Per-subsystem breakdown for anomalous passes
        print(f"\n  Per-subsystem scores on anomalous passes:")
        for p, l in zip(test_passes, test_labels):
            if l["anomaly_type"] in ("none", "hard_limit"):
                continue
            sub_scores = score_pass(p, models, model_type)
            target_sub = l["subsystem"]
            if target_sub in sub_scores:
                s = sub_scores[target_sub]
                agg = aggregate_subsystem_scores(sub_scores)
                pred = score_to_severity(agg["overall_score"])
                print(f"    {p['pass_id']:25s} "
                      f"type={l['anomaly_type']:12s} "
                      f"sub={target_sub:6s} "
                      f"score={s['score']:.3f} "
                      f"conf={s['confidence']:.3f} "
                      f"pred={pred:8s} "
                      f"exp={l['expected_severity']}")
