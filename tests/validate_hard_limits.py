"""Validate hard-limit rule engine against the full labelled dataset."""
import json
import sys
sys.path.insert(0, ".")
from pipeline.hard_limits import evaluate_hard_limits

with open("data/generated/labelled_passes.json") as f:
    passes = json.load(f)
with open("data/generated/labels.json") as f:
    labels = json.load(f)

# Check all hard_limit passes are caught
hl_passes = [(p, l) for p, l in zip(passes, labels) if l["anomaly_type"] == "hard_limit"]
caught = 0
for p, l in hl_passes:
    breaches = evaluate_hard_limits(p)
    if len(breaches) > 0:
        caught += 1
    else:
        print(f"  MISSED: {p['pass_id']} - {l['description']}")

print(f"Hard-limit passes caught: {caught}/{len(hl_passes)}")

# Check nominal passes produce no breaches
nom_passes = [(p, l) for p, l in zip(passes, labels) if l["anomaly_type"] == "none"]
false_alarms = 0
for p, l in nom_passes:
    breaches = evaluate_hard_limits(p)
    if len(breaches) > 0:
        false_alarms += 1
        print(f"  FALSE ALARM: {p['pass_id']}")

print(f"Nominal false alarms: {false_alarms}/{len(nom_passes)}")
