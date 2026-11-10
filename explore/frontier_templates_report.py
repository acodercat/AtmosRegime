"""D-2 section 3 frontier group: template contrasts (diagnosis.template_summary) and S1-S3 for one v1-runner run.

    PYTHONPATH=. python explore/frontier_templates_report.py <families_v4>/templates.json <run file> <out.json>
"""
import json, sys
from pathlib import Path
from atmoscoder_rl.diagnosis import template_summary
from atmoscoder_rl.runs import is_gradable
from explore.d2_report import templates_part, load
items = json.loads(Path(sys.argv[1]).read_text())
rep = load(Path(sys.argv[2]))
assert len(rep) == len(items), (len(rep), len(items))
errors = sum(not is_gradable(r) for r in rep.values())
s = template_summary(items, rep)
part, _ = templates_part(items, rep)
usable = len({r['cluster'] for r in s['templates'].values() if r['usable']})
out = {'rows': len(rep), 'infra_errors': errors, 'usable_clusters': usable,
       'contrasts': s['all']['contrasts'], 'usable_only_contrasts': s['usable_only']['contrasts'],
       'variant_accuracy': {v: sum(c[v] for c in s['all']['clusters'].values()) / len(s['all']['clusters'])
                            for v in ('inside', 'outside', 'given', 'flag')}, **part}
Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + '\n')
f = lambda x: (round(100 * x['mean'], 1), [round(100 * v, 1) for v in x['ci95']])
print('errors', errors, 'usable clusters', usable)
print({k: round(100 * v, 1) for k, v in out['variant_accuracy'].items()})
for k, v in out['contrasts'].items(): print(k, f(v))
print('S1', f(part['S1_capture_outside_wrong']), 'S3', f(part['S3_given_minus_inside']), 'S2', part['S2_cue_rate'])
