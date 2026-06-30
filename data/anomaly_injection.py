"""
Synthetic anomaly injection for the spacecraft telemetry health assessment.

Implements Step 3 of the Implementation Plan, resolving the open question
from PDD Section 9 (anomaly injection strategy not yet specified).

Anomaly taxonomy (grounded in SMAP/MSL literature, PDD Section 2):
  - Point anomalies:      single reading or short burst sharply out of band
  - Contextual anomalies: value in-range but wrong given current context
  - Trend anomalies:      gradual drift toward a limit across the pass
  - Hard-limit breaches:  values outside survival/safety thresholds (PDD 5.2)

Usage:
    python data/anomaly_injection.py            # generates the full labelled set
    python data/anomaly_injection.py --demo      # generates 3 demo passes only
"""

import copy
import json
import os
import sys
import argparse
import numpy as np

# Add project root to path so we can import the generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.synthetic_generator import generate_one_pass, READINGS_PER_PASS


# ===========================================================================
# 3.1 — Anomaly type definitions
# ===========================================================================

ANOMALY_TYPES = ["point", "contextual", "trend", "hard_limit"]

SUBSYSTEMS = ["eps", "tcs", "aocs", "comms", "obc"]

# Ground-truth severity each anomaly type should map to
EXPECTED_SEVERITY = {
    "point": "CAUTION",       # sharp out-of-band → CAUTION
    "contextual": "WATCH",    # subtle, context-dependent → WATCH
    "trend": "WATCH",         # gradual drift → WATCH (may escalate)
    "hard_limit": "CRITICAL", # hard-limit breach → always CRITICAL
}


# ===========================================================================
# 3.2 — Injector functions: one per anomaly-type × subsystem combination
# Each takes a nominal pass (deep-copied) and returns the modified pass
# plus a ground-truth label dict.
# ===========================================================================

def inject_point_eps(pass_data, rng, severity_scale=1.0):
    """Point anomaly: sudden battery voltage spike/drop for a few readings."""
    p = copy.deepcopy(pass_data)
    n = len(p["eps"]["battery_voltage"])
    # Pick a random short burst (3-10 readings)
    burst_len = rng.integers(3, 11)
    start = rng.integers(0, n - burst_len)
    # Drop voltage sharply (scale controls how far outside normal)
    drop = 1.0 + 0.5 * severity_scale  # 1.0V to 1.5V drop
    for i in range(start, start + burst_len):
        p["eps"]["battery_voltage"][i] -= drop

    label = {
        "anomaly_type": "point",
        "subsystem": "eps",
        "channel": "battery_voltage",
        "description": f"Sudden voltage drop of {drop:.1f}V for {burst_len} readings at index {start}",
        "expected_severity": "CAUTION",
        "affected_indices": list(range(start, start + burst_len)),
    }
    return p, label


def inject_point_comms(pass_data, rng, severity_scale=1.0):
    """Point anomaly: brief RSSI dropout (momentary loss of signal)."""
    p = copy.deepcopy(pass_data)
    n = len(p["comms"]["rssi"])
    burst_len = rng.integers(5, 15)
    start = rng.integers(0, n - burst_len)
    # Drop RSSI to noise floor
    for i in range(start, start + burst_len):
        p["comms"]["rssi"][i] = -100 + rng.normal(0, 1)
        p["comms"]["data_rate"][i] = max(0.1, rng.normal(1.0, 0.5))

    label = {
        "anomaly_type": "point",
        "subsystem": "comms",
        "channel": "rssi",
        "description": f"RSSI dropout to noise floor for {burst_len} readings at index {start}",
        "expected_severity": "CAUTION",
        "affected_indices": list(range(start, start + burst_len)),
    }
    return p, label


def inject_point_obc(pass_data, rng, severity_scale=1.0):
    """Point anomaly: sudden SEU/EDAC error spike."""
    p = copy.deepcopy(pass_data)
    n = len(p["obc"]["seu_count"])
    burst_len = rng.integers(3, 8)
    start = rng.integers(0, n - burst_len)
    spike_val = 3 + int(severity_scale * 5)  # 3-8 errors per reading
    for i in range(start, start + burst_len):
        p["obc"]["seu_count"][i] = float(spike_val)

    label = {
        "anomaly_type": "point",
        "subsystem": "obc",
        "channel": "seu_count",
        "description": f"SEU spike of {spike_val} errors/reading for {burst_len} readings at index {start}",
        "expected_severity": "CAUTION",
        "affected_indices": list(range(start, start + burst_len)),
    }
    return p, label


def inject_contextual_eps(pass_data, rng, severity_scale=1.0):
    """Contextual anomaly: battery voltage dropping during sunlight (should be charging).

    Value stays in-range but contradicts the eclipse context.
    """
    p = copy.deepcopy(pass_data)
    eclipse = p["context"]["eclipse_flag"]
    voltage = p["eps"]["battery_voltage"]

    # Find sunlit readings and make voltage decline (wrong direction)
    for i in range(len(voltage)):
        if not eclipse[i]:
            # Instead of charging, slowly discharge
            voltage[i] -= 0.3 * severity_scale * (i / len(voltage))
    # Keep within overall bus range so it doesn't trigger hard limits
    p["eps"]["battery_voltage"] = [max(6.5, v) for v in voltage]

    label = {
        "anomaly_type": "contextual",
        "subsystem": "eps",
        "channel": "battery_voltage",
        "description": "Battery voltage declining during sunlight (should be charging)",
        "expected_severity": "WATCH",
        "affected_indices": [i for i in range(len(eclipse)) if not eclipse[i]],
    }
    return p, label


def inject_contextual_tcs(pass_data, rng, severity_scale=1.0):
    """Contextual anomaly: sun panel temperature rising during eclipse (should be cooling)."""
    p = copy.deepcopy(pass_data)
    eclipse = p["context"]["eclipse_flag"]
    temp = p["tcs"]["sun_panel_temp"]

    for i in range(len(temp)):
        if eclipse[i]:
            # Temperature rising instead of falling during eclipse
            temp[i] += 20 * severity_scale
    p["tcs"]["sun_panel_temp"] = [min(74.0, t) for t in temp]

    label = {
        "anomaly_type": "contextual",
        "subsystem": "tcs",
        "channel": "sun_panel_temp",
        "description": "Sun panel temperature rising during eclipse (should be cooling)",
        "expected_severity": "WATCH",
        "affected_indices": [i for i in range(len(eclipse)) if eclipse[i]],
    }
    return p, label


def inject_trend_eps(pass_data, rng, severity_scale=1.0):
    """Trend anomaly: battery voltage steadily declining across the full pass,
    faster than the eclipse cycle explains."""
    p = copy.deepcopy(pass_data)
    n = len(p["eps"]["battery_voltage"])
    drift = np.linspace(0, 1.2 * severity_scale, n)
    voltage = [v - d for v, d in zip(p["eps"]["battery_voltage"], drift)]
    # Keep above hard limit so this stays a WATCH, not CRITICAL
    p["eps"]["battery_voltage"] = [max(6.3, v) for v in voltage]

    label = {
        "anomaly_type": "trend",
        "subsystem": "eps",
        "channel": "battery_voltage",
        "description": "Battery voltage declining steadily across pass (faster than eclipse explains)",
        "expected_severity": "WATCH",
        "affected_indices": list(range(n)),
    }
    return p, label


def inject_trend_tcs(pass_data, rng, severity_scale=1.0):
    """Trend anomaly: internal component temperature steadily rising across pass."""
    p = copy.deepcopy(pass_data)
    n = len(p["tcs"]["internal_temp"])
    drift = np.linspace(0, 30 * severity_scale, n)
    temp = [t + d for t, d in zip(p["tcs"]["internal_temp"], drift)]
    # Stay inside survival range (-55 to +85) so it's WATCH not CRITICAL
    p["tcs"]["internal_temp"] = [min(80.0, t) for t in temp]

    label = {
        "anomaly_type": "trend",
        "subsystem": "tcs",
        "channel": "internal_temp",
        "description": "Internal temperature steadily rising across pass",
        "expected_severity": "WATCH",
        "affected_indices": list(range(n)),
    }
    return p, label


def inject_trend_aocs(pass_data, rng, severity_scale=1.0):
    """Trend anomaly: reaction wheel speed gradually increasing toward upper limit."""
    p = copy.deepcopy(pass_data)
    n = len(p["aocs"]["wheel_speed"])
    drift = np.linspace(0, 2000 * severity_scale, n)
    speed = [s + d for s, d in zip(p["aocs"]["wheel_speed"], drift)]
    # Stay below hard-limit tumble threshold
    p["aocs"]["wheel_speed"] = [min(7800.0, s) for s in speed]

    label = {
        "anomaly_type": "trend",
        "subsystem": "aocs",
        "channel": "wheel_speed",
        "description": "Reaction wheel speed gradually increasing toward upper limit",
        "expected_severity": "WATCH",
        "affected_indices": list(range(n)),
    }
    return p, label


def inject_hard_limit_eps(pass_data, rng, severity_scale=1.0):
    """Hard-limit breach: battery voltage drops below 6.0V safe discharge floor.
    PDD Section 5.2: over-discharge causes irreversible Li-ion cell damage."""
    p = copy.deepcopy(pass_data)
    n = len(p["eps"]["battery_voltage"])
    # Set voltage below 6.0V for a section of the pass
    start = rng.integers(n // 4, n // 2)
    for i in range(start, n):
        p["eps"]["battery_voltage"][i] = 5.0 + rng.normal(0, 0.2)

    label = {
        "anomaly_type": "hard_limit",
        "subsystem": "eps",
        "channel": "battery_voltage",
        "description": "Battery voltage below 6.0V safe discharge floor",
        "expected_severity": "CRITICAL",
        "affected_indices": list(range(start, n)),
    }
    return p, label


def inject_hard_limit_tcs(pass_data, rng, severity_scale=1.0):
    """Hard-limit breach: component temperature outside survival range.
    PDD Section 5.2: survival-range breach risks permanent hardware damage."""
    p = copy.deepcopy(pass_data)
    n = len(p["tcs"]["internal_temp"])
    start = rng.integers(n // 4, n // 2)
    # Push temperature above +100°C survival limit
    for i in range(start, n):
        p["tcs"]["internal_temp"][i] = 105.0 + rng.normal(0, 3)

    label = {
        "anomaly_type": "hard_limit",
        "subsystem": "tcs",
        "channel": "internal_temp",
        "description": "Component temperature above +100°C survival range",
        "expected_severity": "CRITICAL",
        "affected_indices": list(range(start, n)),
    }
    return p, label


def inject_hard_limit_aocs(pass_data, rng, severity_scale=1.0):
    """Hard-limit breach: angular velocity exceeds tumble-rate threshold.
    PDD Section 5.2: uncontrolled tumble is a loss-of-attitude precursor."""
    p = copy.deepcopy(pass_data)
    n = len(p["aocs"]["gyro_rate"])
    start = rng.integers(n // 4, n // 2)
    # Gyro rate exceeding 10 deg/s indicates tumble
    for i in range(start, n):
        p["aocs"]["gyro_rate"][i] = 15.0 + rng.normal(0, 2)

    label = {
        "anomaly_type": "hard_limit",
        "subsystem": "aocs",
        "channel": "gyro_rate",
        "description": "Angular velocity exceeds tumble-rate threshold (>10 deg/s)",
        "expected_severity": "CRITICAL",
        "affected_indices": list(range(start, n)),
    }
    return p, label


def inject_hard_limit_comms(pass_data, rng, severity_scale=1.0):
    """Hard-limit breach: RSSI loss for extended duration beyond pass geometry.
    PDD Section 5.2: distinguishes unexpected loss from expected end-of-pass."""
    p = copy.deepcopy(pass_data)
    n = len(p["comms"]["rssi"])
    # Signal loss for >40% of pass (well beyond normal edge dropout)
    loss_start = rng.integers(0, n // 4)
    loss_end = min(n, loss_start + int(n * 0.5))
    for i in range(loss_start, loss_end):
        p["comms"]["rssi"][i] = -105 + rng.normal(0, 1)  # below noise floor
        p["comms"]["data_rate"][i] = 0.0

    label = {
        "anomaly_type": "hard_limit",
        "subsystem": "comms",
        "channel": "rssi",
        "description": f"RSSI loss for {loss_end - loss_start} readings ({(loss_end-loss_start)/n*100:.0f}% of pass)",
        "expected_severity": "CRITICAL",
        "affected_indices": list(range(loss_start, loss_end)),
    }
    return p, label


def inject_hard_limit_obc(pass_data, rng, severity_scale=1.0):
    """Hard-limit breach: SEU/EDAC error count spikes sharply.
    PDD Section 5.2: sudden spike associated with radiation events (e.g. SAA)."""
    p = copy.deepcopy(pass_data)
    n = len(p["obc"]["seu_count"])
    # Sharp spike: 10+ errors per reading over a burst
    burst_len = rng.integers(10, 30)
    start = rng.integers(0, n - burst_len)
    for i in range(start, start + burst_len):
        p["obc"]["seu_count"][i] = float(10 + rng.integers(0, 10))

    label = {
        "anomaly_type": "hard_limit",
        "subsystem": "obc",
        "channel": "seu_count",
        "description": f"SEU spike of 10+ errors/reading for {burst_len} readings",
        "expected_severity": "CRITICAL",
        "affected_indices": list(range(start, start + burst_len)),
    }
    return p, label


# Registry of all injection functions
INJECTORS = {
    ("point", "eps"): inject_point_eps,
    ("point", "comms"): inject_point_comms,
    ("point", "obc"): inject_point_obc,
    ("contextual", "eps"): inject_contextual_eps,
    ("contextual", "tcs"): inject_contextual_tcs,
    ("trend", "eps"): inject_trend_eps,
    ("trend", "tcs"): inject_trend_tcs,
    ("trend", "aocs"): inject_trend_aocs,
    ("hard_limit", "eps"): inject_hard_limit_eps,
    ("hard_limit", "tcs"): inject_hard_limit_tcs,
    ("hard_limit", "aocs"): inject_hard_limit_aocs,
    ("hard_limit", "comms"): inject_hard_limit_comms,
    ("hard_limit", "obc"): inject_hard_limit_obc,
}


# ===========================================================================
# 3.3 — Generate the full labelled dataset
# ===========================================================================

def generate_labelled_dataset(n_nominal=60, n_per_injector=3,
                              n_borderline=10, seed=42):
    """Generate a mixed dataset of nominal and anomalous passes.

    Args:
        n_nominal:       number of clean nominal passes (majority)
        n_per_injector:  number of passes per anomaly injector
        n_borderline:    number of borderline (low-severity-scale) passes
        seed:            random seed

    Returns:
        list of (pass_data, label_or_None) tuples
    """
    rng = np.random.default_rng(seed)
    dataset = []

    # Nominal passes (no anomalies)
    for i in range(n_nominal):
        pass_data = generate_one_pass(rng, f"NOM-{i+1:04d}")
        label = {
            "anomaly_type": "none",
            "subsystem": "none",
            "channel": "none",
            "description": "Clean nominal pass",
            "expected_severity": "NOMINAL",
            "affected_indices": [],
        }
        dataset.append((pass_data, label))

    # Anomalous passes: one set per injector at normal severity
    for (atype, subsys), injector_fn in INJECTORS.items():
        for j in range(n_per_injector):
            base_pass = generate_one_pass(rng, f"{atype.upper()}-{subsys.upper()}-{j+1:03d}")
            modified, label = injector_fn(base_pass, rng, severity_scale=1.0)
            dataset.append((modified, label))

    # Borderline/uncertain passes: low severity_scale, right at soft-limit boundary
    # These are what the demo's "genuinely uncertain" pass will be drawn from
    borderline_injectors = [
        (inject_trend_eps, "eps"),
        (inject_trend_tcs, "tcs"),
        (inject_contextual_eps, "eps"),
        (inject_point_eps, "eps"),
        (inject_point_comms, "comms"),
    ]
    for i in range(n_borderline):
        injector_fn, subsys = borderline_injectors[i % len(borderline_injectors)]
        base_pass = generate_one_pass(rng, f"BORDER-{i+1:04d}")
        # Low severity scale = values right near the boundary
        modified, label = injector_fn(base_pass, rng, severity_scale=0.3)
        label["expected_severity"] = "WATCH"  # borderline → uncertain
        label["description"] += " (borderline — low severity scale)"
        dataset.append((modified, label))

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate labelled anomaly dataset")
    parser.add_argument("--demo", action="store_true",
                        help="Generate only 3 demo passes (CRITICAL, NOMINAL, borderline)")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    out_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(out_dir, exist_ok=True)

    if args.demo:
        # Deliverable check: produce exactly the 3 required demo passes
        print("=== Generating 3 demo passes ===\n")

        # 1. Clearly CRITICAL: hard-limit battery voltage breach
        base = generate_one_pass(rng, "DEMO-CRITICAL")
        critical_pass, critical_label = inject_hard_limit_eps(base, rng)
        print(f"CRITICAL pass: {critical_label['description']}")
        print(f"  Battery V range: [{min(critical_pass['eps']['battery_voltage']):.2f}, "
              f"{max(critical_pass['eps']['battery_voltage']):.2f}]")

        # 2. Clearly NOMINAL: clean pass with no anomalies
        nominal_pass = generate_one_pass(rng, "DEMO-NOMINAL")
        nominal_label = {
            "anomaly_type": "none", "subsystem": "none", "channel": "none",
            "description": "Clean nominal pass", "expected_severity": "NOMINAL",
            "affected_indices": [],
        }
        print(f"\nNOMINAL pass: {nominal_label['description']}")
        print(f"  Battery V range: [{min(nominal_pass['eps']['battery_voltage']):.2f}, "
              f"{max(nominal_pass['eps']['battery_voltage']):.2f}]")

        # 3. Borderline WATCH/CAUTION: trend anomaly at low severity
        base = generate_one_pass(rng, "DEMO-BORDERLINE")
        borderline_pass, borderline_label = inject_trend_eps(base, rng, severity_scale=0.3)
        borderline_label["expected_severity"] = "WATCH/CAUTION borderline"
        print(f"\nBORDERLINE pass: {borderline_label['description']}")
        print(f"  Battery V range: [{min(borderline_pass['eps']['battery_voltage']):.2f}, "
              f"{max(borderline_pass['eps']['battery_voltage']):.2f}]")

        # Save demo passes
        demo = [
            {"pass": critical_pass, "label": critical_label},
            {"pass": nominal_pass, "label": nominal_label},
            {"pass": borderline_pass, "label": borderline_label},
        ]
        demo_path = os.path.join(out_dir, "demo_passes.json")
        with open(demo_path, "w") as f:
            json.dump(demo, f, indent=2)
        print(f"\nSaved to {demo_path}")
        return

    # Full dataset generation
    dataset = generate_labelled_dataset()

    # Separate pass data and labels for saving
    all_passes = []
    all_labels = []
    for pass_data, label in dataset:
        all_passes.append(pass_data)
        all_labels.append(label)

    # Save
    passes_path = os.path.join(out_dir, "labelled_passes.json")
    labels_path = os.path.join(out_dir, "labels.json")
    with open(passes_path, "w") as f:
        json.dump(all_passes, f, indent=2)
    with open(labels_path, "w") as f:
        json.dump(all_labels, f, indent=2)

    # Summary
    from collections import Counter
    type_counts = Counter(l["anomaly_type"] for l in all_labels)
    sev_counts = Counter(l["expected_severity"] for l in all_labels)
    print(f"Generated {len(dataset)} labelled passes -> {passes_path}")
    print(f"Labels -> {labels_path}")
    print(f"\nBy anomaly type: {dict(type_counts)}")
    print(f"By expected severity: {dict(sev_counts)}")


if __name__ == "__main__":
    main()
