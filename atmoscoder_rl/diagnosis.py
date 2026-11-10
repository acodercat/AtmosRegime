"""Frozen D run schedule and complete-case, paired diagnostic summaries.

Definitions implement PREREG_MAIN's D. Report both all and usable template clusters.
A missing file/API failure prevents a D-1 verdict, never becomes a wrong answer.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import fmean

from .coverage_clusters import unit_of
from .given_contrast import bootstrap_mean_ci
from .grading import TOLERANCE
from .frozen import require
from .runs import is_gradable
from .traps import family

SEED = 20261008
PAIRS = 20
EXP_ID = 'families_confirm'
# Names are protocol labels, not a claim that weights are installed or supported.
MODELS = (
    ('qwen3.5-9b', 'Qwen3.5', ('off', 'on')),
    ('qwen3.5-27b', 'Qwen3.5', ('off', 'on')),
    ('gemma4-e4b', 'Gemma 4', ('off', 'on')),
    ('gemma4-31b', 'Gemma 4', ('off', 'on')),
    ('granite4.2-8b', 'Granite 4.2', ('off', 'on')),
    ('granite4.2-30b', 'Granite 4.2', ('off', 'on')),
    ('ministral3-8b', 'Ministral 3', ('off',)),
    ('ministral3-14b', 'Ministral 3', ('off',)),
    ('qwen3-8b', 'bridge', ('off', 'on')),
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261008


def configs() -> list[dict]:
    return [{'key': f'{name}.think-{mode}', 'model': name, 'family': fam, 'mode': mode}
            for name, fam, modes in MODELS for mode in modes]


def schedule() -> list[dict]:
    return [{**config, 'instrument': instrument, 'run': rep,
             'filename': f"{config['key']}.{instrument}.run{rep}.json"}
            for config in configs() for instrument, reps in (('textbook', 3), ('templates', 1))
            for rep in range(1, reps+1)]


def mean_ci(values: list[float], draws: int = BOOTSTRAP_DRAWS) -> dict:
    if not values:
        return {'mean': None, 'ci95': None, 'units': 0}
    return {'mean': fmean(values), 'ci95': list(bootstrap_mean_ci(values, draws=draws,
             seed=BOOTSTRAP_SEED, level=.95)), 'units': len(values)}


def contrast_ci(a: list[float], b: list[float], draws: int) -> dict:
    """Trap-level stratified bootstrap: retain all repetitions of each sampled trap."""
    if not a or not b:
        return {'mean': None, 'ci95': None, 'units': [len(a), len(b)]}
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(fmean(rng.choices(a, k=len(a))) - fmean(rng.choices(b, k=len(b)))
                     for _ in range(draws))
    return {'mean': fmean(a)-fmean(b), 'ci95': [samples[int(.025*draws)], samples[int(.975*draws)-1]],
            'units': [len(a), len(b)]}


def load_complete(path: Path, items: list[dict], job: dict | None = None) -> tuple[dict, list[str]]:
    if not path.exists():
        return {}, ['missing run file']
    try:
        doc = json.loads(path.read_text())
        records = doc['results']
        require(isinstance(records, list), 'results must be a list')
        ids = [r['id'] for r in records]
        require(len(set(ids)) == len(ids), 'duplicate result ids')
        expected = {x['id'] for x in items}
        extra = set(ids)-expected
        problems = []
        if extra:
            problems.append(f'{len(extra)} unexpected ids')
        missing = expected-set(ids)
        if missing:
            problems.append(f'{len(missing)} missing ids')
        bad = [r['id'] for r in records if not is_gradable(r) or type(r.get('passed')) is not bool]
        if bad:
            problems.append(f'{len(bad)} infrastructure/malformed records')
        meta = doc.get('metrics', {})
        if meta.get('units') != 'benchmark' or meta.get('tolerance') != TOLERANCE or meta.get('mode') != 'code':
            problems.append('grading/protocol metadata mismatch')
        if job and (meta.get('exp_id') != EXP_ID or meta.get('run') != job['run'] or
                    meta.get('model') != job['key']+'.'+job['instrument']):
            problems.append('model/instrument/repetition metadata mismatch')
        return {r['id']: r for r in records}, problems
    except (ValueError, KeyError, TypeError) as error:
        return {}, [f'malformed run: {error}']


def textbook_summary(items: list[dict], reps: list[dict], draws: int = BOOTSTRAP_DRAWS) -> dict:
    groups = defaultdict(list)
    for item in items:
        if 'parent_id' in item:
            groups[family(item)].append(item)
    summary, gaps = {}, {}
    for label, traps in sorted(groups.items()):
        paired = [[(int(rep[t['parent_id']]['passed']), int(rep[t['id']]['passed'])) for rep in reps]
                  for t in traps]
        values = [fmean(p-t for p, t in pairs) for pairs in paired]
        gaps[label] = values
        flat = [pair for pairs in paired for pair in pairs]
        parents_right = sum(p for p, _ in flat)
        summary[label] = {'traps': len(traps), 'pairs': len(flat),
                          'parent_accuracy': fmean(p for p, _ in flat),
                          'trap_accuracy': fmean(t for _, t in flat),
                          'conditional_recognition': sum(p*t for p, t in flat)/parents_right if parents_right else None,
                          'conditional_denominator': parents_right, 'gap': mean_ci(values, draws)}
    parent_ids = sorted({t['parent_id'] for group in groups.values() for t in group})
    parent_accuracy = fmean(int(rep[p]['passed']) for rep in reps for p in parent_ids)
    regime_pairs = summary.get('regime:recognise', {}).get('pairs', 0)
    return {'families': summary, 'parent_accuracy_unique_parents': parent_accuracy,
            'regime_pairs': regime_pairs, 'usable': parent_accuracy >= .2 and regime_pairs >= 20,
            'regime_minus_geometry': contrast_ci(gaps.get('regime:recognise', []), gaps.get('geometry_detail', []), draws)}


def template_summary(items: list[dict], rep: dict, draws: int = BOOTSTRAP_DRAWS) -> dict:
    by_template = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_template[item['template']][item['variant']].append(int(rep[item['id']]['passed']))
    templates = {}
    for name, variants in sorted(by_template.items()):
        require(set(variants) == {'inside', 'outside', 'given', 'flag'}, 'incomplete template variants')
        require(all(len(v) == PAIRS for v in variants.values()), 'incorrect template pair count')
        templates[name] = {'cluster': unit_of(name), 'accuracy': {k: fmean(v) for k, v in variants.items()},
                           'usable': sum(variants['inside']) >= 10 and sum(variants['given']) >= 15}
    def aggregate(eligible_only: bool) -> dict:
        clusters = defaultdict(list)
        for row in templates.values():
            if not eligible_only or row['usable']:
                clusters[row['cluster']].append(row['accuracy'])
        cluster_means = {c: {v: fmean(r[v] for r in rows) for v in ('inside', 'outside', 'given', 'flag')}
                         for c, rows in sorted(clusters.items())}
        contrasts = {a+'_minus_outside': mean_ci([r[a]-r['outside'] for r in cluster_means.values()], draws)
                     for a in ('inside', 'given', 'flag')}
        return {'clusters': cluster_means, 'cluster_count': len(clusters), 'contrasts': contrasts}
    usable = aggregate(True)
    return {'templates': templates, 'all': aggregate(False), 'usable_only': usable,
            'usable': usable['cluster_count'] >= 15}


def verdict(textbook: dict, templates: dict) -> str:
    if not textbook['usable'] or not templates['usable']:
        return 'not_eligible'
    a = textbook['regime_minus_geometry']
    b = templates['all']['contrasts']['inside_minus_outside']
    if (a['mean'] is not None and b['mean'] is not None and
        a['mean'] >= .15-1e-12 and b['mean'] >= .20-1e-12 and
        a['ci95'][0] > 0 and b['ci95'][0] > 0):
        return 'deficit_established'
    return 'threshold_not_met'
