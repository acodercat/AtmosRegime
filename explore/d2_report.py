"""D-2 secondary analyses (docs/PREREG_MAIN.md, revision D-2): S1 shortcut capture, S2 warning cues,
S3 given vs inside, S4 pooled textbook regime - geometry. Descriptive only; D-1 is untouched.

    python -m explore.d2_report --input-dir experiments/_inputs/families_v4 \
        --run-dir DIR [--run-dir DIR ...] --configs qwen3.5-9b.think-off gemma4-e4b.think-off ... \
        --out results/d2/report.json [--audit-out results/d2/s2_audit.tsv]

A configuration is used only when its three textbook runs and its template run are all present and full;
the caller passes the configurations that passed the D completion checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.capture import shortcut_match
from atmoscoder_rl.coverage_clusters import unit_of
from atmoscoder_rl.traps import family

DRAWS = 10_000
CUES = re.compile(r"not valid|invalid|breaks down|no longer|not applicable|does not apply|outside .*range|validity|"
                  r"regime|assumption fails|correction|instead of|must account|cannot use|should not use", re.I)


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    results = data['results']
    return results if isinstance(results, dict) else {r['id']: r for r in results}


def find(run_dirs: list[Path], name: str) -> Path:
    hits = [d / name for d in run_dirs if (d / name).exists()]
    if len(hits) != 1:
        raise SystemExit(f'{name}: expected exactly one file, found {len(hits)}')
    return hits[0]


def graded_wrong(record: dict) -> bool:
    if record.get('passed') is not False:
        return False
    return any(isinstance(d.get('actual'), (int, float)) for d in record.get('details') or [])


def final_text(record: dict) -> str:
    attempts = record.get('attempts') or []
    return (attempts[-1].get('response') or '') if attempts else ''


def cluster_ci(per_cluster: dict[str, float], seed: str) -> dict:
    values = [v for v in per_cluster.values() if v is not None]
    if not values:
        return {'mean': None, 'ci95': None, 'units': 0}
    rng = random.Random(seed)
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {'mean': fmean(values), 'ci95': [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], 'units': len(values)}


def templates_part(items: list[dict], rep: dict) -> dict:
    by_tpl = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_tpl[item['template']][item['variant']].append(item)
    s1 = defaultdict(lambda: defaultdict(list))   # cluster -> variant -> per-template capture rate
    s3 = defaultdict(list)                        # cluster -> per-template given - inside
    s2 = {'outside_right': [0, 0], 'outside_wrong': [0, 0]}
    audit = []
    for name, variants in by_tpl.items():
        cluster = unit_of(name)
        acc = {v: fmean(int(rep[i['id']]['passed']) for i in rows) for v, rows in variants.items()}
        s3[cluster].append(acc['given'] - acc['inside'])
        for variant in ('outside', 'inside'):
            hits = total = 0
            for item in variants[variant]:
                record = rep[item['id']]
                if variant == 'inside' and item.get('gap', 1) < .02:
                    continue
                if graded_wrong(record):
                    total += 1
                    hits += shortcut_match(record, item)
            if total:
                s1[cluster][variant].append(hits / total)
        for item in variants['outside']:
            record = rep[item['id']]
            text = final_text(record)
            cue = bool(CUES.search(text))
            key = 'outside_right' if record.get('passed') else 'outside_wrong'
            s2[key][0] += cue
            s2[key][1] += 1
            audit.append((item['id'], key, int(cue), text))
    def collapse(variant: str) -> dict[str, float]:
        return {c: fmean(v[variant]) for c, v in s1.items() if v.get(variant)}
    return {
        'S1_capture_outside_wrong': cluster_ci(collapse('outside'), 'd2:20261011:s1o'),
        'S1_capture_inside_wrong': cluster_ci(collapse('inside'), 'd2:20261011:s1i'),
        'S2_cue_rate': {k: {'cue': c, 'n': n, 'rate': c / n if n else None} for k, (c, n) in s2.items()},
        'S3_given_minus_inside': cluster_ci({c: fmean(v) for c, v in s3.items()}, 'd2:20261011:s3'),
    }, audit


def textbook_gaps(items: list[dict], reps: list[dict]) -> dict[str, dict[str, float]]:
    gaps = defaultdict(dict)
    for item in items:
        if 'parent_id' in item:
            gaps[family(item)][item['id']] = fmean(int(r[item['parent_id']]['passed']) - int(r[item['id']]['passed'])
                                                   for r in reps)
    return gaps


def pooled(per_config: list[dict[str, dict[str, float]]], seed: str) -> dict | None:
    if not per_config:
        return None
    out = {}
    for fam in ('regime:recognise', 'geometry_detail'):
        ids = sorted(per_config[0][fam])
        out[fam] = [fmean(cfg[fam][i] for cfg in per_config) for i in ids]
    # Same algorithm as diagnosis.contrast_ci (each family resampled independently by trap), with the seed D-2 names.
    a, b = out['regime:recognise'], out['geometry_detail']
    rng = random.Random(seed)
    samples = sorted(fmean(rng.choices(a, k=len(a))) - fmean(rng.choices(b, k=len(b))) for _ in range(DRAWS))
    return {'mean': fmean(a) - fmean(b), 'ci95': [samples[int(.025 * DRAWS)], samples[int(.975 * DRAWS) - 1]],
            'units': [len(a), len(b)], 'configurations': len(per_config)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, action='append', required=True)
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--audit-out', type=Path)
    args = ap.parse_args()
    tb_items = json.loads((args.input_dir / 'textbook.json').read_text())
    tpl_items = json.loads((args.input_dir / 'templates.json').read_text())
    result, gaps_all, audit_rows = {'configurations': {}}, {}, []
    for key in args.configs:
        tpl = load(find(args.run_dir, f'{key}.templates.run1.json'))
        tbs = [load(find(args.run_dir, f'{key}.textbook.run{n}.json')) for n in (1, 2, 3)]
        if len(tpl) != len(tpl_items) or any(len(t) != len(tb_items) for t in tbs):
            raise SystemExit(f'{key}: incomplete run files')
        part, audit = templates_part(tpl_items, tpl)
        result['configurations'][key] = part
        audit_rows += [(key, *row) for row in audit]
        gaps_all[key] = textbook_gaps(tb_items, tbs)
    result['S4_pooled'] = {
        'all': pooled(list(gaps_all.values()), 'd2-pool:20261011'),
        'think_off': pooled([g for k, g in gaps_all.items() if k.endswith('think-off')], 'd2-pool:20261011:off'),
        'think_on': pooled([g for k, g in gaps_all.items() if k.endswith('think-on')], 'd2-pool:20261011:on'),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    if args.audit_out:
        rng = random.Random('d2:20261011:audit')
        hit = [r for r in audit_rows if r[3]]
        miss = [r for r in audit_rows if not r[3]]
        sample = rng.sample(hit, min(40, len(hit))) + rng.sample(miss, min(40, len(miss)))
        with args.audit_out.open('w', newline='') as fh:
            w = csv.writer(fh, delimiter='\t')
            w.writerow(['config', 'id', 'outcome', 'cue', 'is_real_warning', 'text'])
            for key, iid, outcome, cue, text in sample:
                w.writerow([key, iid, outcome, cue, '', text.replace('\t', ' ')[:4000]])
    for key, part in result['configurations'].items():
        s1 = part['S1_capture_outside_wrong']
        print(key, 'S1', round(s1['mean'], 3) if s1['mean'] is not None else None, 'S3',
              round(part['S3_given_minus_inside']['mean'], 3))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
