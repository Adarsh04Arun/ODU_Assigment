"""
Unit tests for the hard-limit rule engine (Step 4 deliverable check).

Tests each rule independently with synthetic values crafted to just barely
cross and just barely NOT cross each threshold.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.hard_limits import (
    check_battery_voltage,
    check_component_temperature,
    check_tumble_rate,
    check_rssi_loss,
    check_seu_spike,
    evaluate_hard_limits,
    BATTERY_VOLTAGE_FLOOR,
    COMPONENT_TEMP_MAX,
    COMPONENT_TEMP_MIN,
    TUMBLE_RATE_THRESHOLD,
    RSSI_LOSS_FLOOR,
    RSSI_LOSS_FRACTION,
    SEU_SPIKE_THRESHOLD,
)


def make_pass(overrides=None):
    """Build a minimal nominal pass dict. Override specific channels as needed."""
    n = 300
    base = {
        "pass_id": "TEST",
        "context": {"eclipse_flag": [False] * n, "pass_elapsed_time": list(range(n))},
        "eps": {
            "battery_voltage": [8.2] * n,
            "rail_3v3": [3.3] * n,
            "rail_5v0": [5.0] * n,
            "solar_current": [1.5] * n,
            "charge_rate": [0.0] * n,
        },
        "tcs": {
            "sun_panel_temp": [50.0] * n,
            "shade_panel_temp": [5.0] * n,
            "battery_temp": [25.0] * n,
            "internal_temp": [25.0] * n,
        },
        "aocs": {
            "wheel_speed": [4500.0] * n,
            "pointing_error": [0.5] * n,
            "gyro_rate": [0.1] * n,
            "attitude_error": [0.3] * n,
        },
        "comms": {
            "rssi": [-70.0] * n,
            "data_rate": [200.0] * n,
        },
        "obc": {
            "cpu_load": [40.0] * n,
            "memory_occupancy": [50.0] * n,
            "seu_count": [0.0] * n,
        },
    }
    if overrides:
        for key_path, value in overrides.items():
            parts = key_path.split(".")
            d = base
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = value
    return base


def test_battery_voltage():
    print("Testing check_battery_voltage...")

    # Just above threshold (5.99V) — should fire
    v_breach = [8.2] * 150 + [5.99] * 150
    p = make_pass({"eps.battery_voltage": v_breach})
    result = check_battery_voltage(p)
    assert result is not None, "Should fire at 5.99V"
    assert result["rule"] == "BATTERY_VOLTAGE_FLOOR"
    print(f"  [OK] Fires at 5.99V: {result['description']}")

    # Just below threshold (6.01V) — should NOT fire
    v_safe = [8.2] * 150 + [6.01] * 150
    p = make_pass({"eps.battery_voltage": v_safe})
    result = check_battery_voltage(p)
    assert result is None, "Should NOT fire at 6.01V"
    print("  [OK] Does not fire at 6.01V")

    # Exactly at threshold (6.0V) — should NOT fire (< is strict)
    v_exact = [6.0] * 300
    p = make_pass({"eps.battery_voltage": v_exact})
    result = check_battery_voltage(p)
    assert result is None, "Should NOT fire at exactly 6.0V"
    print("  [OK] Does not fire at exactly 6.0V (boundary)")


def test_component_temperature():
    print("Testing check_component_temperature...")

    # Just above max (100.1°C) — should fire
    t_hot = [25.0] * 150 + [100.1] * 150
    p = make_pass({"tcs.internal_temp": t_hot})
    result = check_component_temperature(p)
    assert result is not None, "Should fire at 100.1°C"
    assert result["rule"] == "COMPONENT_TEMP_MAX"
    print(f"  [OK] Fires at 100.1°C: {result['description']}")

    # Just below max (99.9°C) — should NOT fire
    t_warm = [25.0] * 150 + [99.9] * 150
    p = make_pass({"tcs.internal_temp": t_warm})
    result = check_component_temperature(p)
    assert result is None, "Should NOT fire at 99.9°C"
    print("  [OK] Does not fire at 99.9°C")

    # Just below min (-40.1°C) — should fire
    t_cold = [25.0] * 150 + [-40.1] * 150
    p = make_pass({"tcs.internal_temp": t_cold})
    result = check_component_temperature(p)
    assert result is not None, "Should fire at -40.1°C"
    assert result["rule"] == "COMPONENT_TEMP_MIN"
    print(f"  [OK] Fires at -40.1°C: {result['description']}")

    # Just above min (-39.9°C) — should NOT fire
    t_cool = [25.0] * 150 + [-39.9] * 150
    p = make_pass({"tcs.internal_temp": t_cool})
    result = check_component_temperature(p)
    assert result is None, "Should NOT fire at -39.9°C"
    print("  [OK] Does not fire at -39.9°C")


def test_tumble_rate():
    print("Testing check_tumble_rate...")

    # Just above threshold (10.1 deg/s) — should fire
    g_tumble = [0.1] * 150 + [10.1] * 150
    p = make_pass({"aocs.gyro_rate": g_tumble})
    result = check_tumble_rate(p)
    assert result is not None, "Should fire at 10.1 deg/s"
    assert result["rule"] == "TUMBLE_RATE"
    print(f"  [OK] Fires at 10.1 deg/s: {result['description']}")

    # Just below threshold (9.9 deg/s) — should NOT fire
    g_safe = [0.1] * 150 + [9.9] * 150
    p = make_pass({"aocs.gyro_rate": g_safe})
    result = check_tumble_rate(p)
    assert result is None, "Should NOT fire at 9.9 deg/s"
    print("  [OK] Does not fire at 9.9 deg/s")

    # Negative tumble (-10.1 deg/s) — should also fire (abs value)
    g_neg = [0.1] * 150 + [-10.1] * 150
    p = make_pass({"aocs.gyro_rate": g_neg})
    result = check_tumble_rate(p)
    assert result is not None, "Should fire at -10.1 deg/s (abs)"
    print("  [OK] Fires at -10.1 deg/s (negative tumble)")


def test_rssi_loss():
    print("Testing check_rssi_loss...")

    # >30% of pass below noise floor — should fire
    n = 300
    rssi_loss = [-70.0] * 200 + [-101.0] * 100  # 33% below floor
    p = make_pass({"comms.rssi": rssi_loss})
    result = check_rssi_loss(p)
    assert result is not None, "Should fire at 33% loss"
    assert result["rule"] == "RSSI_LOSS_DURATION"
    print(f"  [OK] Fires at 33% loss: {result['description']}")

    # Just under 30% — should NOT fire
    rssi_ok = [-70.0] * 220 + [-101.0] * 80  # 27% below floor
    p = make_pass({"comms.rssi": rssi_ok})
    result = check_rssi_loss(p)
    assert result is None, "Should NOT fire at 27% loss"
    print("  [OK] Does not fire at 27% loss")


def test_seu_spike():
    print("Testing check_seu_spike...")

    # Spike of 5 — should fire (>= threshold)
    seu_spike = [0.0] * 290 + [5.0] * 10
    p = make_pass({"obc.seu_count": seu_spike})
    result = check_seu_spike(p)
    assert result is not None, "Should fire at SEU count 5"
    assert result["rule"] == "SEU_SPIKE"
    print(f"  [OK] Fires at SEU=5: {result['description']}")

    # Count of 4 — should NOT fire
    seu_safe = [0.0] * 290 + [4.0] * 10
    p = make_pass({"obc.seu_count": seu_safe})
    result = check_seu_spike(p)
    assert result is None, "Should NOT fire at SEU count 4"
    print("  [OK] Does not fire at SEU=4")


def test_evaluate_all():
    print("Testing evaluate_hard_limits (integration)...")

    # Nominal pass — no breaches
    p = make_pass()
    breaches = evaluate_hard_limits(p)
    assert len(breaches) == 0, "Nominal pass should have no breaches"
    print("  [OK] Nominal pass: 0 breaches")

    # Multiple simultaneous breaches
    p = make_pass({
        "eps.battery_voltage": [5.5] * 300,
        "tcs.internal_temp": [105.0] * 300,
    })
    breaches = evaluate_hard_limits(p)
    assert len(breaches) == 2, f"Expected 2 breaches, got {len(breaches)}"
    rules = {b["rule"] for b in breaches}
    assert "BATTERY_VOLTAGE_FLOOR" in rules
    assert "COMPONENT_TEMP_MAX" in rules
    print(f"  [OK] Double breach: {rules}")


if __name__ == "__main__":
    test_battery_voltage()
    test_component_temperature()
    test_tumble_rate()
    test_rssi_loss()
    test_seu_spike()
    test_evaluate_all()
    print("\n=== ALL HARD-LIMIT TESTS PASSED ===")
