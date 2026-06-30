"""
Synthetic telemetry generator for a 3U-6U CubeSat-class LEO satellite,
single-string COTS EPS architecture.

Mission assumption (PDD Section 3): all ranges below are for a CubeSat-class
LEO satellite. Several ranges — particularly bus voltage — are architecture-
dependent and would be wrong for a larger GEO-class platform.

Implements PDD Section 3 (parameter schema) and Section 3.7 (orbital context).
Each generated pass is a dict containing per-channel time series across a
single ground-station contact window.

Usage:
    python data/synthetic_generator.py          # generates 100 nominal passes
    python data/synthetic_generator.py --n 200  # generates 200 nominal passes
"""

import json
import os
import sys
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Constants: one pass = 300 readings at 1-second intervals (~5-minute contact)
# ---------------------------------------------------------------------------
READINGS_PER_PASS = 300
SAMPLE_INTERVAL_S = 1.0  # seconds between readings


# ===========================================================================
# 2.2 — Orbital / pass context (PDD Section 3.7)
# Generated first because physical channels depend on eclipse state.
# ===========================================================================

def generate_pass_context(rng, pass_id):
    """Generate eclipse flag and pass timing for one ground-station pass.

    Eclipse state is modelled as a simple binary: each pass is either fully
    sunlit or partially eclipsed.  About 35% of LEO orbits include eclipse
    (typical ~90-min period, ~30-min eclipse), so we sample accordingly.

    Returns:
        dict with keys: pass_id, eclipse_fraction, eclipse_flag (per-reading
        bool array), pass_elapsed_time (array in seconds).
    """
    t = np.arange(READINGS_PER_PASS) * SAMPLE_INTERVAL_S

    # Decide if this pass includes eclipse (~35% chance)
    has_eclipse = rng.random() < 0.35

    if has_eclipse:
        # Eclipse starts at a random point in the pass, lasts 30-60% of it
        eclipse_start = rng.uniform(0.0, 0.4)  # fraction of pass
        eclipse_duration = rng.uniform(0.3, 0.6)  # fraction of pass
        eclipse_flag = np.array([
            eclipse_start <= (i / READINGS_PER_PASS) <= (eclipse_start + eclipse_duration)
            for i in range(READINGS_PER_PASS)
        ])
    else:
        eclipse_flag = np.zeros(READINGS_PER_PASS, dtype=bool)

    eclipse_fraction = float(np.mean(eclipse_flag))

    return {
        "pass_id": pass_id,
        "eclipse_fraction": round(eclipse_fraction, 3),
        "eclipse_flag": eclipse_flag,
        "pass_elapsed_time": t,
    }


# ===========================================================================
# 2.1 — Per-subsystem generators (PDD Section 3.2–3.6)
# Each function takes the pass context and returns channel arrays.
# ===========================================================================

def generate_eps(rng, ctx):
    """Electrical Power System (PDD Section 3.2).

    Battery bus voltage: ~8.2V nominal, 6.2-8.4V across charge cycle [Tier 1]
        Source: AAC Clyde STARBUCK-NANO EPS, Aalto-1 flight telemetry.
    Regulated rails: 3.3V, 5.0V [Tier 1]
        Source: NASA small-spacecraft power systems reference.
    Solar array current: scaled to panel area and eclipse state [Tier 3]
        Design assumption — no single CubeSat-wide figure found.
    Battery charge/discharge rate: derived from voltage trend [Tier 3]
        Computed, not sourced independently.
    """
    n = READINGS_PER_PASS
    eclipse = ctx["eclipse_flag"]

    # Battery bus voltage: dips during eclipse (discharge), rises in sunlight
    base_voltage = 8.2
    voltage = np.full(n, base_voltage)
    for i in range(n):
        if eclipse[i]:
            # Discharge: voltage drops ~0.3-0.8V over eclipse
            elapsed_in_eclipse = np.sum(eclipse[:i+1])
            voltage[i] = base_voltage - 0.5 * (elapsed_in_eclipse / n)
        else:
            # Charge: voltage slowly rises toward 8.4V
            elapsed_in_sun = np.sum(~eclipse[:i+1])
            voltage[i] = base_voltage + 0.2 * (elapsed_in_sun / n)
    voltage = np.clip(voltage, 6.2, 8.4)
    # Sensor noise [Tier 2 — generic measurement noise]
    voltage += rng.normal(0, 0.02, n)

    # Regulated rails: stable with small noise
    rail_3v3 = 3.3 + rng.normal(0, 0.005, n)
    rail_5v0 = 5.0 + rng.normal(0, 0.008, n)

    # Solar array current: ~1.5A in sunlight, ~0A in eclipse [Tier 3]
    solar_current = np.where(eclipse, rng.normal(0.01, 0.005, n),
                             rng.normal(1.5, 0.1, n))
    solar_current = np.clip(solar_current, 0.0, 3.0)

    # Battery charge rate: derived from voltage derivative [Tier 3]
    charge_rate = np.gradient(voltage)

    return {
        "battery_voltage": voltage,
        "rail_3v3": rail_3v3,
        "rail_5v0": rail_5v0,
        "solar_current": solar_current,
        "charge_rate": charge_rate,
    }


def generate_tcs(rng, ctx):
    """Thermal Control System (PDD Section 3.3).

    Sun-facing panel temp: -20C to +75C [Tier 1]
        Source: MinXSS CubeSat thermal-balance flight data.
    Shaded panel temp: -16C to +17C [Tier 1]
        Source: MinXSS radiator plate flight data.
    Battery temperature: comfort band -5C to +40C [Tier 2]
        Source: common Li-ion safe-operation guidance.
    Internal component temp: -40C to +100C survival [Tier 1]
        Source: DeMi CubeSat published thermal budget.
    """
    n = READINGS_PER_PASS
    eclipse = ctx["eclipse_flag"]

    # Sun panel: hot in sunlight (~40-60C), cold in eclipse (~-10 to +5C)
    sun_panel = np.where(eclipse,
                         rng.normal(-5, 5, n),
                         rng.normal(50, 8, n))
    # Thermal inertia: smooth transitions
    for i in range(1, n):
        sun_panel[i] = 0.9 * sun_panel[i-1] + 0.1 * sun_panel[i]
    sun_panel = np.clip(sun_panel, -20, 75)

    # Shaded panel: relatively stable, correlated with eclipse
    shade_panel = np.where(eclipse,
                           rng.normal(-10, 3, n),
                           rng.normal(5, 4, n))
    for i in range(1, n):
        shade_panel[i] = 0.9 * shade_panel[i-1] + 0.1 * shade_panel[i]
    shade_panel = np.clip(shade_panel, -16, 17)

    # Battery temp: lags sun panel slightly, stays in comfort band
    battery_temp = sun_panel * 0.3 + rng.normal(20, 3, n)
    for i in range(1, n):
        battery_temp[i] = 0.95 * battery_temp[i-1] + 0.05 * battery_temp[i]
    battery_temp = np.clip(battery_temp, -5, 40)

    # Internal component temp: stable around 25C
    internal_temp = rng.normal(25, 3, n)
    for i in range(1, n):
        internal_temp[i] = 0.95 * internal_temp[i-1] + 0.05 * internal_temp[i]

    return {
        "sun_panel_temp": sun_panel,
        "shade_panel_temp": shade_panel,
        "battery_temp": battery_temp,
        "internal_temp": internal_temp,
    }


def generate_aocs(rng, ctx):
    """Attitude & Orbit Control System (PDD Section 3.4).

    Reaction wheel speed: 3000-8000 RPM [Tier 1]
        Source: CADRE flight wheels (3400 RPM), UWE-3 (8000 RPM nominal).
    Pointing accuracy: 0.003-1.0 deg [Tier 1]
        Source: CADRE (1.0 deg), Blue Canyon XACT (0.003 deg).
    Gyro angular rate: MEMS-class, tens of deg/s full-scale [Tier 1]
        Source: AAC Clyde PG400 gyroscope datasheet.
    Attitude error: sub-degree to ~2 deg during nominal tracking [Tier 1]
        Derived from pointing accuracy range.
    """
    n = READINGS_PER_PASS

    # Reaction wheel: nominal ~4500 RPM with slow drift
    wheel_speed = rng.normal(4500, 200, n)
    for i in range(1, n):
        wheel_speed[i] = 0.95 * wheel_speed[i-1] + 0.05 * wheel_speed[i]
    wheel_speed = np.clip(wheel_speed, 3000, 8000)

    # Pointing error: small, stable around 0.5 deg
    pointing_error = np.abs(rng.normal(0.5, 0.1, n))
    pointing_error = np.clip(pointing_error, 0.003, 1.0)

    # Gyro rate: near zero in stable pointing
    gyro_rate = rng.normal(0.0, 0.3, n)

    # Attitude error: stable around 0.3 deg
    attitude_error = np.abs(rng.normal(0.3, 0.1, n))
    attitude_error = np.clip(attitude_error, 0.0, 2.0)

    return {
        "wheel_speed": wheel_speed,
        "pointing_error": pointing_error,
        "gyro_rate": gyro_rate,
        "attitude_error": attitude_error,
    }


def generate_comms(rng, ctx):
    """Communications (PDD Section 3.5).

    RSSI: approx -100 to -60 dBm at typical LEO pass ranges [Tier 3]
        Derived from UHF link-budget path-loss figures, not a single named spec.
    Downlink data rate: low kbps to ~1 Mbps class [Tier 1]
        Source: CubeSat ground-station link-budget literature.
    """
    n = READINGS_PER_PASS
    t = ctx["pass_elapsed_time"]
    pass_mid = t[-1] / 2.0

    # RSSI: strongest at mid-pass (closest approach), weaker at edges
    # Parabolic profile centred on pass midpoint
    distance_factor = -20 * ((t - pass_mid) / pass_mid) ** 2
    rssi = -70 + distance_factor + rng.normal(0, 2, n)
    rssi = np.clip(rssi, -100, -60)

    # Data rate: tracks RSSI roughly (higher RSSI → higher rate)
    # Normalise RSSI to 0-1 range, map to 9.6 kbps – 500 kbps
    rssi_norm = (rssi - (-100)) / ((-60) - (-100))
    data_rate = 9.6 + rssi_norm * 490.4  # kbps
    data_rate += rng.normal(0, 5, n)
    data_rate = np.clip(data_rate, 1.0, 1000.0)

    return {
        "rssi": rssi,
        "data_rate": data_rate,
    }


def generate_obc(rng, ctx):
    """On-Board Computer / CDH (PDD Section 3.6).

    CPU load: 0-100% [Tier 3]
        Generic computing telemetry, no spacecraft-specific nominal band found.
    Memory occupancy: 0-100% [Tier 3]
        Same as CPU load — direction grounded in real OBC practice.
    SEU/EDAC corrected error count: low single digits nominal [Tier 1 direction / Tier 3 count]
        Source: Flying Laptop satellite SEU studies (practice is real,
        per-pass count threshold is a synthetic placeholder).
    """
    n = READINGS_PER_PASS

    # CPU load: nominal ~30-50%, slow-varying
    cpu_load = rng.normal(40, 5, n)
    for i in range(1, n):
        cpu_load[i] = 0.9 * cpu_load[i-1] + 0.1 * cpu_load[i]
    cpu_load = np.clip(cpu_load, 0, 100)

    # Memory occupancy: nominal ~40-60%, very slowly trending up
    memory = rng.normal(50, 3, n)
    memory += np.linspace(0, 2, n)  # slight upward trend over pass
    for i in range(1, n):
        memory[i] = 0.95 * memory[i-1] + 0.05 * memory[i]
    memory = np.clip(memory, 0, 100)

    # SEU/EDAC count: mostly 0, occasional single correction
    seu_count = np.zeros(n)
    # Sprinkle 1-3 single corrections randomly in nominal
    n_events = rng.integers(0, 4)
    if n_events > 0:
        event_indices = rng.choice(n, size=n_events, replace=False)
        seu_count[event_indices] = 1

    return {
        "cpu_load": cpu_load,
        "memory_occupancy": memory,
        "seu_count": seu_count,
    }


# ===========================================================================
# 2.3 — Cross-channel correlation checks are embedded above:
#   - Solar current tracks eclipse flag (generate_eps)
#   - Battery voltage dips during eclipse (generate_eps)
#   - Sun panel temp tracks eclipse (generate_tcs)
#   - Battery temp lags sun panel temp (generate_tcs)
#   - RSSI follows pass geometry (generate_comms)
#   - Data rate tracks RSSI (generate_comms)
# ===========================================================================


def generate_one_pass(rng, pass_id):
    """Generate all subsystem telemetry for a single nominal pass.

    Args:
        rng: numpy random Generator instance.
        pass_id: string identifier for this pass.

    Returns:
        dict with keys: pass_id, context, eps, tcs, aocs, comms, obc.
        Each subsystem value is a dict of channel_name → list of floats.
    """
    ctx = generate_pass_context(rng, pass_id)

    eps = generate_eps(rng, ctx)
    tcs = generate_tcs(rng, ctx)
    aocs = generate_aocs(rng, ctx)
    comms = generate_comms(rng, ctx)
    obc = generate_obc(rng, ctx)

    # Convert numpy arrays to plain lists for JSON serialisation
    def to_lists(d):
        return {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in d.items()}

    return {
        "pass_id": pass_id,
        "readings_per_pass": READINGS_PER_PASS,
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "context": {
            "eclipse_fraction": ctx["eclipse_fraction"],
            "eclipse_flag": ctx["eclipse_flag"].tolist(),
            "pass_elapsed_time": ctx["pass_elapsed_time"].tolist(),
        },
        "eps": to_lists(eps),
        "tcs": to_lists(tcs),
        "aocs": to_lists(aocs),
        "comms": to_lists(comms),
        "obc": to_lists(obc),
    }


def generate_dataset(n_passes=100, seed=42):
    """Generate a full dataset of nominal passes.

    Args:
        n_passes: number of passes to generate.
        seed: random seed for reproducibility.

    Returns:
        list of pass dicts.
    """
    rng = np.random.default_rng(seed)
    passes = []
    for i in range(n_passes):
        pass_id = f"PASS-{i+1:04d}"
        passes.append(generate_one_pass(rng, pass_id))
    return passes


# ===========================================================================
# 2.4 — Output: save as JSON under data/generated/
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CubeSat telemetry passes")
    parser.add_argument("--n", type=int, default=100, help="Number of passes to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    passes = generate_dataset(n_passes=args.n, seed=args.seed)

    # Ensure output directory exists
    out_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(out_dir, exist_ok=True)

    # Save full dataset
    out_path = os.path.join(out_dir, "nominal_passes.json")
    with open(out_path, "w") as f:
        json.dump(passes, f, indent=2)
    print(f"Generated {len(passes)} nominal passes -> {out_path}")

    # Save one sample pass separately for repo inspection (Step 2.4)
    sample_path = os.path.join(out_dir, "sample_pass.json")
    with open(sample_path, "w") as f:
        json.dump(passes[0], f, indent=2)
    print(f"Sample pass -> {sample_path}")

    # Quick sanity check: print summary of first pass
    p = passes[0]
    print(f"\n--- Sample pass: {p['pass_id']} ---")
    print(f"  Eclipse fraction: {p['context']['eclipse_fraction']}")
    print(f"  Battery voltage: min={min(p['eps']['battery_voltage']):.3f}, "
          f"max={max(p['eps']['battery_voltage']):.3f}, "
          f"mean={np.mean(p['eps']['battery_voltage']):.3f}")
    print(f"  Sun panel temp:  min={min(p['tcs']['sun_panel_temp']):.1f}, "
          f"max={max(p['tcs']['sun_panel_temp']):.1f}")
    print(f"  Wheel speed:     min={min(p['aocs']['wheel_speed']):.0f}, "
          f"max={max(p['aocs']['wheel_speed']):.0f}")
    print(f"  RSSI:            min={min(p['comms']['rssi']):.1f}, "
          f"max={max(p['comms']['rssi']):.1f}")
    print(f"  CPU load:        min={min(p['obc']['cpu_load']):.1f}, "
          f"max={max(p['obc']['cpu_load']):.1f}")
    print(f"  SEU events:      {int(sum(p['obc']['seu_count']))}")


if __name__ == "__main__":
    main()
