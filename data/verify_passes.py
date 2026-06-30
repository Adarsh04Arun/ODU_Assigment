"""Quick verification of generated passes for physical plausibility."""
import json
import numpy as np

with open("data/generated/nominal_passes.json") as f:
    passes = json.load(f)

print(f"Total passes: {len(passes)}")
print()

# Check a range of passes for physical plausibility
for idx in [0, 10, 30, 60, 99]:
    p = passes[idx]
    pid = p["pass_id"]
    ecl = p["context"]["eclipse_fraction"]
    bv = p["eps"]["battery_voltage"]
    sc = p["eps"]["solar_current"]
    sp = p["tcs"]["sun_panel_temp"]
    rssi = p["comms"]["rssi"]
    print(f"{pid}  eclipse={ecl:.2f}  "
          f"batt_v=[{min(bv):.2f},{max(bv):.2f}]  "
          f"solar_i=[{min(sc):.2f},{max(sc):.2f}]  "
          f"sun_T=[{min(sp):.1f},{max(sp):.1f}]  "
          f"rssi=[{min(rssi):.1f},{max(rssi):.1f}]")

# Verify cross-channel correlations
print()
print("--- Cross-channel correlation checks ---")

# Find passes with and without eclipse
ecl_pass = None
sun_pass = None
for p in passes:
    if p["context"]["eclipse_fraction"] > 0.2 and ecl_pass is None:
        ecl_pass = p
    if p["context"]["eclipse_fraction"] == 0.0 and sun_pass is None:
        sun_pass = p
    if ecl_pass and sun_pass:
        break

if ecl_pass:
    pid = ecl_pass["pass_id"]
    ef = ecl_pass["context"]["eclipse_fraction"]
    bv = ecl_pass["eps"]["battery_voltage"]
    sc = ecl_pass["eps"]["solar_current"]
    sp = ecl_pass["tcs"]["sun_panel_temp"]
    print(f"{pid} (eclipse={ef}):")
    print(f"  Battery V: [{min(bv):.2f}, {max(bv):.2f}]")
    print(f"  Solar current: [{min(sc):.2f}, {max(sc):.2f}]")
    print(f"  Sun panel temp: [{min(sp):.1f}, {max(sp):.1f}]")

if sun_pass:
    pid = sun_pass["pass_id"]
    ef = sun_pass["context"]["eclipse_fraction"]
    bv = sun_pass["eps"]["battery_voltage"]
    sc = sun_pass["eps"]["solar_current"]
    sp = sun_pass["tcs"]["sun_panel_temp"]
    print(f"{pid} (eclipse={ef}):")
    print(f"  Battery V: [{min(bv):.2f}, {max(bv):.2f}]")
    print(f"  Solar current: [{min(sc):.2f}, {max(sc):.2f}]")
    print(f"  Sun panel temp: [{min(sp):.1f}, {max(sp):.1f}]")

# Count eclipse vs sunlit passes
n_eclipse = sum(1 for p in passes if p["context"]["eclipse_fraction"] > 0)
print(f"\nEclipse passes: {n_eclipse}/{len(passes)} ({n_eclipse/len(passes)*100:.0f}%)")
