"""
Hard-limit rule engine — fully independent module with no learned parameters.
Stage 1, and the safety backbone. A rule engine with no learned parameters — every threshold is a fixed physics constant. 
Five rules, one per subsystem: battery voltage floor (< 6.0 V), component temp survival range (−40 to +100 °C), tumble rate (> 10 deg/s), RSSI loss (below noise floor for > 30% of pass), and SEU spike (≥ 5 errors/reading). 
Each check_* function is a pure function returning a breach record or None; 
evaluate_hard_limits(pass) runs all five and returns the list of breaches. 
If any rule fires, the final verdict is CRITICAL and cannot be overridden downstream.
"""

import numpy as np


# ===========================================================================
# Hard-limit thresholds
# Every value here is a physical safety limit, not a judgement call.
# ===========================================================================

# EPS: Battery bus voltage safe discharge floor
# irreversible Li-ion cell damage
BATTERY_VOLTAGE_FLOOR = 6.0

# TCS: Component survival temperature range
# Using -40°C to +100°C general equipment range 
COMPONENT_TEMP_MIN = -40.0
COMPONENT_TEMP_MAX = 100.0

# AOCS: Tumble-rate angular velocity threshold
# corresponding planned manoeuvre flag"
# 10 deg/s chosen as threshold — well above any nominal attitude manoeuvre
TUMBLE_RATE_THRESHOLD = 10.0  # deg/s

# Comms: RSSI loss duration threshold
# If RSSI is below -100 dBm for >30% of the pass, that's unexpected signal loss
RSSI_LOSS_FLOOR = -100.0  # dBm — below this is noise floor / no signal
RSSI_LOSS_FRACTION = 0.30  # fraction of pass readings that triggers the rule

# OBC: SEU/EDAC spike threshold
# A sudden burst of 5+ errors in any reading is a spike (vs. gradual baseline)
SEU_SPIKE_THRESHOLD = 5  # errors per reading


# ===========================================================================
# Rule functions — each is a pure function:
#   input:  a pass dict (same shape as synthetic_generator output)
#   output: a breach record dict, or None if no breach
# ===========================================================================

def check_battery_voltage(pass_data):
    """EPS hard limit: battery bus voltage below safe discharge floor.

    If any reading drops below 6.0V, the Li-ion cells risk irreversible
    damage. This is a physics limit, not a statistical threshold.
    """
    voltage = pass_data["eps"]["battery_voltage"]
    min_v = min(voltage)
    if min_v < BATTERY_VOLTAGE_FLOOR:
        breaching = [i for i, v in enumerate(voltage) if v < BATTERY_VOLTAGE_FLOOR]
        return {
            "rule": "BATTERY_VOLTAGE_FLOOR",
            "subsystem": "eps",
            "channel": "battery_voltage",
            "threshold": BATTERY_VOLTAGE_FLOOR,
            "actual_value": round(min_v, 3),
            "description": (f"Battery voltage {min_v:.2f}V is below the "
                            f"{BATTERY_VOLTAGE_FLOOR}V safe discharge floor"),
            "breaching_indices": breaching,
        }
    return None


def check_component_temperature(pass_data):
    """TCS hard limit: component temperature outside survival range.

    Survival-range breaches (-40°C to +100°C) risk permanent hardware damage
    regardless of mission phase.
    """
    temp = pass_data["tcs"]["internal_temp"]
    min_t = min(temp)
    max_t = max(temp)

    if max_t > COMPONENT_TEMP_MAX:
        breaching = [i for i, t in enumerate(temp) if t > COMPONENT_TEMP_MAX]
        return {
            "rule": "COMPONENT_TEMP_MAX",
            "subsystem": "tcs",
            "channel": "internal_temp",
            "threshold": COMPONENT_TEMP_MAX,
            "actual_value": round(max_t, 1),
            "description": (f"Component temperature {max_t:.1f}°C exceeds "
                            f"+{COMPONENT_TEMP_MAX}°C survival limit"),
            "breaching_indices": breaching,
        }

    if min_t < COMPONENT_TEMP_MIN:
        breaching = [i for i, t in enumerate(temp) if t < COMPONENT_TEMP_MIN]
        return {
            "rule": "COMPONENT_TEMP_MIN",
            "subsystem": "tcs",
            "channel": "internal_temp",
            "threshold": COMPONENT_TEMP_MIN,
            "actual_value": round(min_t, 1),
            "description": (f"Component temperature {min_t:.1f}°C is below "
                            f"{COMPONENT_TEMP_MIN}°C survival limit"),
            "breaching_indices": breaching,
        }
    return None


def check_tumble_rate(pass_data):
    """AOCS hard limit: angular velocity exceeds tumble-rate threshold.

    Uncontrolled tumble is a classic loss-of-attitude precursor and demands
    immediate attention. Threshold: 10 deg/s without a planned manoeuvre flag.
    """
    gyro = pass_data["aocs"]["gyro_rate"]
    # Use absolute value — tumble can be in either direction
    max_rate = max(abs(g) for g in gyro)
    if max_rate > TUMBLE_RATE_THRESHOLD:
        breaching = [i for i, g in enumerate(gyro) if abs(g) > TUMBLE_RATE_THRESHOLD]
        return {
            "rule": "TUMBLE_RATE",
            "subsystem": "aocs",
            "channel": "gyro_rate",
            "threshold": TUMBLE_RATE_THRESHOLD,
            "actual_value": round(max_rate, 2),
            "description": (f"Angular rate {max_rate:.1f} deg/s exceeds "
                            f"{TUMBLE_RATE_THRESHOLD} deg/s tumble threshold"),
            "breaching_indices": breaching,
        }
    return None


def check_rssi_loss(pass_data):
    """Comms hard limit: extended RSSI loss beyond pass geometry.

    Distinguishes an unexpected loss of contact from a normal end-of-pass
    signal drop. If RSSI is below the noise floor for >30% of the pass,
    this is not explained by pass geometry alone.
    """
    rssi = pass_data["comms"]["rssi"]
    n = len(rssi)
    loss_count = sum(1 for r in rssi if r < RSSI_LOSS_FLOOR)
    loss_fraction = loss_count / n

    if loss_fraction > RSSI_LOSS_FRACTION:
        breaching = [i for i, r in enumerate(rssi) if r < RSSI_LOSS_FLOOR]
        return {
            "rule": "RSSI_LOSS_DURATION",
            "subsystem": "comms",
            "channel": "rssi",
            "threshold": f">{RSSI_LOSS_FRACTION*100:.0f}% of pass below {RSSI_LOSS_FLOOR} dBm",
            "actual_value": f"{loss_fraction*100:.1f}% of pass",
            "description": (f"RSSI below noise floor for {loss_fraction*100:.0f}% of pass "
                            f"(threshold: {RSSI_LOSS_FRACTION*100:.0f}%)"),
            "breaching_indices": breaching,
        }
    return None


def check_seu_spike(pass_data):
    """OBC hard limit: SEU/EDAC correctable error count spikes sharply.

    A sudden spike (vs. gradual baseline drift) is associated with radiation
    events such as South Atlantic Anomaly passage.
    """
    seu = pass_data["obc"]["seu_count"]
    max_seu = max(seu)

    if max_seu >= SEU_SPIKE_THRESHOLD:
        breaching = [i for i, s in enumerate(seu) if s >= SEU_SPIKE_THRESHOLD]
        return {
            "rule": "SEU_SPIKE",
            "subsystem": "obc",
            "channel": "seu_count",
            "threshold": SEU_SPIKE_THRESHOLD,
            "actual_value": int(max_seu),
            "description": (f"SEU count {int(max_seu)} errors/reading exceeds "
                            f"spike threshold of {SEU_SPIKE_THRESHOLD}"),
            "breaching_indices": breaching,
        }
    return None


# All five rules in evaluation order
ALL_RULES = [
    check_battery_voltage,
    check_component_temperature,
    check_tumble_rate,
    check_rssi_loss,
    check_seu_spike,
]


def evaluate_hard_limits(pass_data):
    """Run all hard-limit rules against a pass. Return list of breach records.

    This is the pipeline's first stage. If any breach is returned, the final
    severity is CRITICAL regardless of what the numeric or text scoring
    layers say downstream.

    Args:
        pass_data: a single pass dict from the synthetic generator.

    Returns:
        list of breach dicts (empty list if no breaches).
    """
    breaches = []
    for rule_fn in ALL_RULES:
        result = rule_fn(pass_data)
        if result is not None:
            breaches.append(result)
    return breaches
