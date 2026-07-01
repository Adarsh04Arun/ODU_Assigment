import sys
sys.path.insert(0, '.')
src = open('ui/app.py', encoding='utf-8').read()
exec(src.split('def main')[0])
import json
with open('data/generated/labels.json') as f:
    labels = json.load(f)
with open('data/generated/labelled_passes.json') as f:
    passes = json.load(f)

print('=== Anomalous pass notes ===')
seen = set()
for i, (p, l) in enumerate(zip(passes, labels)):
    if l['anomaly_type'] == 'none':
        continue
    l_with_id = {**l, 'pass_id': p.get('pass_id', '')}
    note = get_note_for_pass(l_with_id, i)
    unique = note not in seen
    seen.add(note)
    print(f"  [{l['anomaly_type']:12s} {l.get('subsystem','—'):6s}] {'NEW' if unique else '---'} {note[:75]}")
