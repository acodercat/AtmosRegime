"""Audit (PREREG_DOSE_MINISTRAL DM-5): did any archived RL run have the training-side / rollout-side log-prob mismatch that
verl's padding-free packing caused for Ministral 3? Scans every verl training log under --logs and summarises, per run, the
rollout/actor consistency metrics verl logs each step.

    python -m explore.audit_rollout_consistency --logs $ATMOS_ROOT/logs --out results/audit/rollout_consistency.json

Per run: steps seen; min and median `training/rollout_actor_probs_pearson_corr`; max and median of
`rollout_corr/training_ppl / rollout_corr/rollout_ppl`; max `actor/pg_clipfrac`; median `actor/grad_norm`; and whether the
run passes the DM-6 gate (pearson >= 0.97 and ppl ratio <= 1.10 on every step); max and median `rollout_corr/kl` (DM-8). Runs
whose log has no such metric are listed as `no_metric` (older verl or not a PPO run).

    python -m explore.audit_rollout_consistency --steps $ATMOS_ROOT/logs/dose/<run>.attempt1.nohup   # one log, per step
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"step:(\d+) - ")
KEYS = {"pearson": "training/rollout_actor_probs_pearson_corr", "tppl": "rollout_corr/training_ppl",
        "rppl": "rollout_corr/rollout_ppl", "kl": "rollout_corr/kl", "clip": "actor/pg_clipfrac", "grad": "actor/grad_norm",
        "entropy": "actor/entropy", "resp_len": "response_length/mean", "trunc": "response_length/clip_ratio"}


def parse(path: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    with path.open(errors="replace") as f:
        for line in f:
            if "step:" not in line or "training/" not in line and "actor/" not in line:
                continue
            line = ANSI.sub("", line)
            m = STEP.search(line)
            if not m:
                continue
            vals = {}
            for k, key in KEYS.items():
                g = re.search(re.escape(key) + r":(?:np\.float64\()?([-0-9.eE+naif]+)", line)
                if g:
                    try:
                        vals[k] = float(g.group(1))
                    except ValueError:
                        pass
            if vals:
                steps[int(m.group(1))] = vals   # a resumed attempt overwrites the same step: keep the last
    return steps


def summarise(steps: dict[int, dict[str, float]]) -> dict:
    p = [v["pearson"] for v in steps.values() if "pearson" in v]
    r = [v["tppl"] / v["rppl"] for v in steps.values() if "tppl" in v and v.get("rppl")]
    k = [v["kl"] for v in steps.values() if "kl" in v]
    c = [v["clip"] for v in steps.values() if "clip" in v]
    g = [v["grad"] for v in steps.values() if "grad" in v]
    if not p and not r:
        return {"steps": len(steps), "no_metric": True}
    out = {"steps": len(steps),
           "pearson_min": min(p) if p else None, "pearson_median": median(p) if p else None,
           "ppl_ratio_max": max(r) if r else None, "ppl_ratio_median": median(r) if r else None,
           "kl_max": max(k) if k else None, "kl_median": median(k) if k else None,
           "pg_clipfrac_max": max(c) if c else None, "grad_norm_median": median(g) if g else None}
    out["passes_dm6_gate"] = (not p or min(p) >= 0.97) and (not r or max(r) <= 1.10)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--steps", type=Path, help="print the per-step table of this one log and exit")
    a = ap.parse_args()
    if a.steps:
        cols = list(KEYS)
        print("step " + " ".join(cols) + " ppl_ratio")
        for s, v in sorted(parse(a.steps).items()):
            r = v["tppl"] / v["rppl"] if "tppl" in v and v.get("rppl") else None
            print(s, " ".join("-" if c not in v else f"{v[c]:.4g}" for c in cols), "-" if r is None else f"{r:.3f}")
        return 0
    if not (a.logs and a.out):
        ap.error("--logs and --out are required without --steps")
    runs = {}
    for f in sorted(a.logs.glob("*.log")):
        try:
            if f.stat().st_size < 10_000:
                continue
        except OSError:
            continue
        steps = parse(f)
        if steps:
            runs[f.stem] = summarise(steps)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(runs, indent=1) + "\n")
    bad = {k: v for k, v in runs.items() if not v.get("no_metric") and not v["passes_dm6_gate"]}
    print(f"{len(runs)} runs with step metrics; {sum(v.get('no_metric', False) for v in runs.values())} without the consistency "
          f"metrics; {len(bad)} fail the DM-6 gate")
    for k, v in sorted(runs.items()):
        if v.get("no_metric"):
            continue
        f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
        print(f"{'FAIL' if k in bad else 'ok  '} {k[:52]:52s} steps {v['steps']:3d} pearson min {f(v['pearson_min'])} "
              f"med {f(v['pearson_median'])} ppl-ratio max {f(v['ppl_ratio_max'])} med {f(v['ppl_ratio_median'])} "
              f"clip max {f(v['pg_clipfrac_max'], 4)} grad med {f(v['grad_norm_median'], 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
