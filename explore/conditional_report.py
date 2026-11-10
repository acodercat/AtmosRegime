"""Exploratory (not pre-registered): template contrasts conditioned on the inside item of the same pair being solved.

The pre-registered template contrast is inside - outside accuracy. Models with different inside accuracy are then hard to
compare: a model that solves few inside items has little room to fall. Here, per template, only pairs whose inside item
was solved count; the rate is P(outside solved | inside solved) (and the same for given / flag). Templates with fewer than
MIN_PAIRS such pairs are left out. Templates are averaged within a cluster (`coverage_clusters.unit_of`, or the template
itself with --per-template, as for X5) and clusters are resampled 10,000 times (seed "cond:20261012").

    python -m explore.conditional_report --input <templates.json> --runs KEY=FILE [KEY=FILE ...] [--per-template] --out out.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.coverage_clusters import unit_of
from explore.d2_report import load

DRAWS, SEED, MIN_PAIRS = 10_000, "cond:20261012", 5


def ci(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(SEED)
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": len(values)}


def analyse(items: list[dict], rep: dict, per_template: bool) -> dict:
    by_pair = defaultdict(dict)
    for it in items:
        by_pair[(it["template"], it["pair"])][it["variant"]] = bool(rep[it["id"]].get("passed"))
    per_tpl = defaultdict(lambda: defaultdict(list))
    for (tpl, _), v in by_pair.items():
        if v.get("inside"):
            for arm in ("outside", "given", "flag"):
                if arm in v:
                    per_tpl[tpl][arm].append(int(v[arm]))
    kept = {t: {a: fmean(x) for a, x in arms.items()} for t, arms in per_tpl.items() if len(arms["outside"]) >= MIN_PAIRS}
    clusters = defaultdict(list)
    for t, rates in kept.items():
        clusters[t if per_template else unit_of(t)].append(rates)
    cl = {c: {a: fmean(r[a] for r in rows) for a in ("outside", "given", "flag")} for c, rows in clusters.items()}
    return {"templates_kept": len(kept), "units": len(cl),
            "outside_given_inside": ci([r["outside"] for r in cl.values()]),
            "given_given_inside": ci([r["given"] for r in cl.values()]),
            "flag_given_inside": ci([r["flag"] for r in cl.values()]),
            "per_template": kept}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--per-template", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    items = json.loads(args.input.read_text())
    out = {}
    for spec in args.runs:
        key, path = spec.split("=", 1)
        rep = load(Path(path))
        if len(rep) != len(items):
            raise SystemExit(f"{path}: {len(rep)} of {len(items)} rows")
        part = analyse(items, rep, args.per_template)
        f = lambda d: (round(100 * d["mean"]), [round(100 * x) for x in d["ci95"]]) if d["mean"] is not None else None
        print(f"{key:30s} kept {part['templates_kept']:2d} tpl / {part['units']:2d} units  "
              f"P(out|in) {f(part['outside_given_inside'])}  P(given|in) {f(part['given_given_inside'])}  P(flag|in) {f(part['flag_given_inside'])}")
        out[key] = part
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
