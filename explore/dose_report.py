"""PREREG_DOSE §5: the dose-gradient readings. Written and committed before any dose model was trained.

    python -m explore.dose_report --design results/dose/design.json --name qwen3-8b \
        --passk-dir $ATMOS_ROOT/experiments/_dose/qwen3-8b --eval-dir $ATMOS_ROOT/experiments/_x_eval \
        --final $ATMOS_ROOT/experiments/_inputs/kc/final.json --inputs $ATMOS_ROOT/experiments/_inputs/x_eval --out dose_qwen3-8b.json

An observation is (unit mechanism, seed): its dose, the outside accuracy (x_report's regraded final.json) of the seed's warm
start sft-dose-<name>-<fold>-s<seed> and of its RL endpoint dose-<name>-<fold>-WA-s<seed>, and the warm start's pass@64
(mean unbiased pass@64 over the mechanism's final.json outside rows, as P1). Intervals resample mechanisms (each with both
of its observations), 10,000 draws, random.Random("dose:20261027"); a draw without both groups of a contrast is dropped.
H1: mean RL accuracy of dose >= 1 observations - that of dose 0; holds iff >= 0.30 and the interval's lower end > 0.
H2: dose >= 1 observations, split by warm-start pass@64 = 0 / > 0; success = RL accuracy >= 0.5; holds iff the success share
    is <= 0.20 with pass@64 = 0 and >= 0.80 with pass@64 > 0; not testable if either group has fewer than 3 observations.
Secondary, numbers only: per dose level; RL - warm start; dose 0 against the A arm of the same seed (spillover) and the
warm start against the base; the existing WA (full dose) on the units; non-unit training-fold mechanisms, new RL against WA.
A verdict needs every label complete (x_report's missing()); otherwise it reads "incomplete".
Exploratory (PREREG_DOSE DOSE-2, added before any dose result was read; numbers only): H2 inside each dose level >= 1
(success counts by pass@64 = 0 / > 0, and their share difference pooled over the levels with both groups, weighted by the
level's observations, same interval), and the H1 difference inside each of the four runs (fold, seed).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean

from explore.x_report import Family, P1_EXCLUDED, passk_table

SEED = "dose:20261027"
DRAWS = 10_000
H1_MIN = 0.30
H2_LOW, H2_HIGH, H2_MIN_N = 0.20, 0.80, 3
SUCCESS = 0.5


def warm_label(name: str, fold: str, seed: int) -> str: return f"sft-dose-{name}-{fold}-s{seed}"
def rl_label(name: str, fold: str, seed: int) -> str: return f"dose-{name}-{fold}-WA-s{seed}"


def boot(obs: list[dict], stat) -> dict:
    """stat(observations) with an interval resampling mechanisms; draws where stat is None are dropped."""
    point = stat(obs)
    by_t: dict[str, list[dict]] = {}
    for o in obs:
        by_t.setdefault(o["mechanism"], []).append(o)
    mechs = sorted(by_t)
    rng = random.Random(SEED)
    vals = []
    for _ in range(DRAWS):
        v = stat([o for t in (rng.choice(mechs) for _ in mechs) for o in by_t[t]])
        if v is not None:
            vals.append(v)
    vals.sort()
    ci = [vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]] if vals else None
    return {"value": point, "ci95": ci, "draws_used": len(vals), "observations": len(obs), "mechanisms": len(mechs)}


def mean_of(obs: list[dict], key: str) -> float | None:
    v = [o[key] for o in obs if o.get(key) is not None]
    return fmean(v) if v else None


def h1_stat(obs: list[dict]) -> float | None:
    hi, lo = mean_of([o for o in obs if o["dose"] >= 1], "rl"), mean_of([o for o in obs if o["dose"] == 0], "rl")
    return None if hi is None or lo is None else hi - lo


def h2(obs: list[dict]) -> dict:
    dosed = [o for o in obs if o["dose"] >= 1 and o["rl"] is not None and o["warm_pass64"] is not None]
    groups = {"pass64_0": [o for o in dosed if o["warm_pass64"] == 0], "pass64_pos": [o for o in dosed if o["warm_pass64"] > 0]}
    out = {}
    for g, v in groups.items():
        succ = [o["mechanism"] + f"/s{o['seed']}" for o in v if o["rl"] >= SUCCESS]
        out[g] = {"n": len(v), "successes": len(succ), "share": len(succ) / len(v) if v else None, "successful": succ,
                  "observations": [f"{o['mechanism']}/s{o['seed']}" for o in v]}
    if min(out[g]["n"] for g in groups) < H2_MIN_N:
        out["verdict"] = "不可检验"
    else:
        ok = out["pass64_0"]["share"] <= H2_LOW and out["pass64_pos"]["share"] >= H2_HIGH
        out["verdict"] = "成立" if ok else "不成立"
    return out


def within_dose_stat(obs: list[dict]) -> float | None:
    """DOSE-2: success share (RL >= 0.5) with warm pass@64 > 0 minus with pass@64 = 0, inside each dose level >= 1 that has
    both groups, averaged over those levels weighted by the level's observations; None if no level has both groups."""
    num = den = 0
    for d in sorted({o["dose"] for o in obs if o["dose"] >= 1}):
        v = [o for o in obs if o["dose"] == d and o["rl"] is not None and o["warm_pass64"] is not None]
        pos, zero = [o for o in v if o["warm_pass64"] > 0], [o for o in v if o["warm_pass64"] == 0]
        if pos and zero:
            share = lambda g: sum(o["rl"] >= SUCCESS for o in g) / len(g)
            num += len(v) * (share(pos) - share(zero))
            den += len(v)
    return num / den if den else None


def exploratory(obs: list[dict]) -> dict:
    """DOSE-2 (added before any dose result was read; numbers only, no verdict): H2 inside each dose level, and H1 per run."""
    levels = {}
    for d in sorted({o["dose"] for o in obs if o["dose"] >= 1}):
        v = [o for o in obs if o["dose"] == d and o["rl"] is not None and o["warm_pass64"] is not None]
        levels[d] = {g: {"n": len(w), "successes": sum(o["rl"] >= SUCCESS for o in w)}
                     for g, w in (("pass64_0", [o for o in v if o["warm_pass64"] == 0]),
                                  ("pass64_pos", [o for o in v if o["warm_pass64"] > 0]))}
    runs = {}
    for fold, s in sorted({(o["fold"], o["seed"]) for o in obs}):
        v = [o for o in obs if o["fold"] == fold and o["seed"] == s]
        hi, lo = [o for o in v if o["dose"] >= 1], [o for o in v if o["dose"] == 0]
        d = h1_stat(v)
        runs[f"{fold}-s{s}"] = {"n_dosed": len(hi), "n_dose0": len(lo), "rl_dosed": mean_of(hi, "rl"),
                                "rl_dose0": mean_of(lo, "rl"), "difference": d}
    return {"h2_within_dose": {"levels": levels, "pooled_difference": boot(obs, within_dose_stat)}, "h1_per_run": runs}


def observations(fam: Family, design: dict, passk: dict) -> list[dict]:
    out = []
    for fold, by_seed in design["doses"].items():
        for s, doses in by_seed.items():
            s = int(s)
            w = fam.accuracy(warm_label(fam.name, fold, s), "outside")
            rl = fam.accuracy(rl_label(fam.name, fold, s), "outside")
            a = fam.accuracy(fam.arm("A", fold, s), "outside")
            wa = fam.accuracy(fam.arm("WA", fold, s), "outside")
            base = fam.accuracy(fam.base(), "outside")
            pk = passk.get((fold, s), {})
            for t, d in doses.items():
                out.append({"mechanism": t, "fold": fold, "seed": s, "dose": d, "warm": w.get(t), "rl": rl.get(t),
                            "warm_pass64": pk.get(t, {}).get(64), "A": a.get(t), "WA_full": wa.get(t), "base": base.get(t)})
    return out


def secondary(fam: Family, design: dict, obs: list[dict]) -> dict:
    levels = sorted({o["dose"] for o in obs})
    per_dose = {d: {"n": len(v), "warm": mean_of(v, "warm"), "rl": mean_of(v, "rl"),
                    "warm_pass64_pos": sum(o["warm_pass64"] > 0 for o in v if o["warm_pass64"] is not None),
                    "rl_at_or_above_50": sum(o["rl"] >= SUCCESS for o in v if o["rl"] is not None)}
                for d in levels for v in [[o for o in obs if o["dose"] == d]]}
    for o in obs:
        o["rl-warm"] = None if o["rl"] is None or o["warm"] is None else o["rl"] - o["warm"]
        o["rl-A"] = None if o["rl"] is None or o["A"] is None else o["rl"] - o["A"]
        o["warm-base"] = None if o["warm"] is None or o["base"] is None else o["warm"] - o["base"]
    low = [o for o in obs if o["dose"] >= 1 and o["warm"] is not None and o["warm"] < SUCCESS
           and (o["warm_pass64"] or 0) > 0]
    zero = [o for o in obs if o["dose"] == 0]
    m = lambda k: (lambda v: mean_of(v, k))
    nonunit = []
    for fold, by_seed in design["doses"].items():
        for s in by_seed:
            new = fam.accuracy(rl_label(fam.name, fold, int(s)), "outside")
            old = fam.accuracy(fam.arm("WA", fold, int(s)), "outside")
            for t in fam.templates(fold, "train"):
                if t not in by_seed[s] and t not in P1_EXCLUDED and t in new and t in old:
                    nonunit.append({"mechanism": t, "seed": int(s), "new": new[t], "old": old[t], "d": new[t] - old[t]})
    return {"per_dose": per_dose,
            "rl-warm": {"all_dosed": boot([o for o in obs if o["dose"] >= 1 and o["rl-warm"] is not None], m("rl-warm")),
                        "dosed_warm_below_50_pass64_pos": boot([o for o in low if o["rl-warm"] is not None], m("rl-warm"))},
            "dose0_spillover": {"rl-A": boot([o for o in zero if o["rl-A"] is not None], m("rl-A")),
                                "warm-base": boot([o for o in zero if o["warm-base"] is not None], m("warm-base"))},
            "WA_full_dose_on_units": mean_of(obs, "WA_full"),
            "non_unit_new_rl-WA": boot(nonunit, m("d")) if nonunit else None}


def report(args) -> dict:
    design = json.loads(args.design.read_text())
    fam = Family(args)
    passk, labels, missing = {}, [], []
    for fold, by_seed in design["doses"].items():
        for s in by_seed:
            s = int(s)
            labels += [warm_label(args.name, fold, s), rl_label(args.name, fold, s)]
            p = args.passk_dir / f"passk_final_outside.{warm_label(args.name, fold, s)}.jsonl"
            if p.exists():
                passk[(fold, s)] = passk_table(p)
            else:
                missing.append(str(p))
    for lab in labels:
        missing += fam.missing(lab)
    obs = observations(fam, design, passk)
    complete = not missing and all(o[k] is not None for o in obs for k in ("warm", "rl", "warm_pass64"))
    h1 = boot(obs, h1_stat)
    h1["verdict"] = ("成立" if h1["value"] is not None and h1["value"] >= H1_MIN and h1["ci95"] and h1["ci95"][0] > 0
                     else "不成立") if complete else "不完整"
    h2r = h2(obs)
    if not complete:
        h2r["verdict"] = "不完整"
    return {"family": args.name, "design_levels": design["levels"], "labels": labels, "missing": missing, "complete": complete,
            "infra_errors": {lab: e for lab in labels if (e := fam.infra_errors(lab))},
            "H1": h1, "H2": h2r, "secondary": secondary(fam, design, obs), "exploratory": exploratory(obs),
            "observations": obs}


def pts(x) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def show(r: dict) -> None:
    print(f"== dose gradient {r['family']}: complete {r['complete']}; missing {r['missing'] or 'none'}")
    h = r["H1"]
    print(f"H1 RL(dose>=1) - RL(dose 0) = {pts(h['value'])} {[pts(x) for x in h['ci95']] if h['ci95'] else ''} -> {h['verdict']}")
    h2r = r["H2"]
    for g in ("pass64_0", "pass64_pos"):
        print(f"H2 {g}: {h2r[g]['successes']} / {h2r[g]['n']} at >= 50 %")
    print(f"  -> {h2r['verdict']}")
    for d, v in r["secondary"]["per_dose"].items():
        print(f"dose {d}: n {v['n']}, warm {pts(v['warm'])}, warm pass@64 > 0 {v['warm_pass64_pos']}, RL {pts(v['rl'])}, "
              f"RL >= 50 % {v['rl_at_or_above_50']}")
    s = r["secondary"]
    for k in ("all_dosed", "dosed_warm_below_50_pass64_pos"):
        b = s["rl-warm"][k]
        print(f"RL - warm ({k}): {pts(b['value'])} {[pts(x) for x in b['ci95']] if b['ci95'] else ''} n {b['observations']}")
    for k, b in s["dose0_spillover"].items():
        print(f"dose 0 {k}: {pts(b['value'])} {[pts(x) for x in b['ci95']] if b['ci95'] else ''} n {b['observations']}")
    n = s["non_unit_new_rl-WA"]
    if n:
        print(f"non-unit mechanisms, new RL - WA: {pts(n['value'])} {[pts(x) for x in n['ci95']] if n['ci95'] else ''}")
    e = r["exploratory"]
    for d, v in e["h2_within_dose"]["levels"].items():
        print(f"[DOSE-2] dose {d}: pass@64 = 0 {v['pass64_0']['successes']} / {v['pass64_0']['n']}, "
              f"> 0 {v['pass64_pos']['successes']} / {v['pass64_pos']['n']} at >= 50 %")
    b = e["h2_within_dose"]["pooled_difference"]
    print(f"[DOSE-2] within-dose success share, pass@64 > 0 - = 0: {pts(b['value'])} "
          f"{[pts(x) for x in b['ci95']] if b['ci95'] else ''} ({b['draws_used']} draws)")
    for run, v in e["h1_per_run"].items():
        print(f"[DOSE-2] H1 {run}: {pts(v['rl_dosed'])} (n {v['n_dosed']}) - {pts(v['rl_dose0'])} (n {v['n_dose0']}) "
              f"= {pts(v['difference'])}")


def main() -> int:
    from atmoscoder_rl.paths import EXPERIMENTS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--passk-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.seeds = sorted({int(s) for by in json.loads(args.design.read_text())["doses"].values() for s in by})
    r = report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    show(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
