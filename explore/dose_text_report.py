"""PREREG_DOSE_TEXT §5: the readings of the text-injection experiment. Written and committed before any text model was trained.

    python -m explore.dose_text_report --design results/dose_text/design.json --name qwen3-8b \
        --passk-dir $ATMOS_ROOT/experiments/_dose/qwen3-8b --eval-dir $ATMOS_ROOT/experiments/_x_eval \
        --final $ATMOS_ROOT/experiments/_inputs/kc/final.json --inputs $ATMOS_ROOT/experiments/_inputs/x_eval \
        [--dose-posthoc results/dose/posthoc_qwen3-8b.json] --out dose_text_qwen3-8b.json

An observation is (unit mechanism, seed): whether the text was injected in that seed, the warm start's per-sample probability p
(mean pass@1 over the mechanism's final.json outside rows) and pass@64, the warm and RL outside accuracy (x_report's regrade),
and the number of the 20 outside rows the warm start / RL endpoint answer with the standard-formula value (2 %).
Intervals resample mechanisms (each with both observations), 10,000 draws, random.Random("dose-text:20261029").
H1-text: mean over mechanisms of RL(text seed) - RL(no-text seed); holds iff >= 0.30 and the lower bound > 0 (the unpaired
difference as DOSE's H1 is reported beside it). H2-p: injected observations by warm p (< 0.005, 0.005-0.02, >= 0.02),
success = RL >= 0.5; holds iff the >= 0.02 group's share is >= 0.80 and the < 0.005 group's <= 0.20; a group with fewer than 3
observations is untestable (numbers only). A verdict needs every label complete; otherwise "incomplete".
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean

from explore.x_report import Family, P1_EXCLUDED, passk_table

SEED = "dose-text:20261029"
DRAWS = 10_000
H1_MIN = 0.30
H2_LOW, H2_HIGH, H2_MIN_N = 0.20, 0.80, 3
P_BINS = (("p<0.005", 0.0, 0.005), ("0.005<=p<0.02", 0.005, 0.02), ("p>=0.02", 0.02, 2.0))
SUCCESS = 0.5


def warm_label(name: str, fold: str, seed: int) -> str: return f"sft-dtext-{name}-{fold}-s{seed}"
def rl_label(name: str, fold: str, seed: int) -> str: return f"dtext-{name}-{fold}-WA-s{seed}"


def close(a, b, tol: float = 0.02) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1e-12)
    except (TypeError, ValueError):
        return False


def shortcut_counts(eval_dir: Path, label: str, final: list[dict]) -> dict[str, int]:
    """mechanism -> outside rows the label answers with the standard-formula value; {} if the file is absent."""
    p = eval_dir / label / "final.json"
    if not p.exists():
        return {}
    res = {r["id"]: r for r in json.loads(p.read_text())["results"]}
    out: dict[str, int] = {row["template"]: 0 for row in final if row.get("variant") == "outside"}
    for row in final:
        r = res.get(row["id"])
        if row.get("variant") != "outside" or r is None or r.get("passed"):
            continue
        actual = [d.get("actual") for d in (r.get("details") or []) if isinstance(d, dict)]
        sv = (row.get("shortcut_values") or {}).values()
        out[row["template"]] = out.get(row["template"], 0) + any(close(a, s) for a in actual for s in sv)
    return out


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


def paired_stat(obs: list[dict]) -> float | None:
    """mean over mechanisms of RL(text seed) - RL(no-text seed); a mechanism needs both observations with RL read."""
    d = []
    by_t: dict[str, dict] = {}
    for o in obs:
        by_t.setdefault(o["mechanism"], {})[o["injected"]] = o
    for t, v in by_t.items():
        if True in v and False in v and v[True]["rl"] is not None and v[False]["rl"] is not None:
            d.append(v[True]["rl"] - v[False]["rl"])
    return fmean(d) if d else None


def unpaired_stat(obs: list[dict]) -> float | None:
    hi, lo = mean_of([o for o in obs if o["injected"]], "rl"), mean_of([o for o in obs if not o["injected"]], "rl")
    return None if hi is None or lo is None else hi - lo


def p_groups(obs: list[dict]) -> dict:
    out = {}
    for name, lo, hi in P_BINS:
        v = [o for o in obs if o["warm_p"] is not None and o["rl"] is not None and lo <= o["warm_p"] < hi]
        succ = [f"{o['mechanism']}/s{o['seed']}" for o in v if o["rl"] >= SUCCESS]
        out[name] = {"n": len(v), "successes": len(succ), "share": len(succ) / len(v) if v else None, "successful": succ,
                     "observations": [f"{o['mechanism']}/s{o['seed']}" for o in v]}
    return out


def h2(obs: list[dict]) -> dict:
    g = p_groups([o for o in obs if o["injected"]])
    low, high = g["p<0.005"], g["p>=0.02"]
    g["untestable_groups"] = [k for k, v in (("p<0.005", low), ("p>=0.02", high)) if v["n"] < H2_MIN_N]
    if g["untestable_groups"]:
        g["verdict"] = "不可检验"
    else:
        g["verdict"] = "成立" if high["share"] >= H2_HIGH and low["share"] <= H2_LOW else "不成立"
    return g


def observations(fam: Family, design: dict, passk: dict, final: list[dict], eval_dir: Path) -> list[dict]:
    out = []
    for fold, units in design["units"].items():
        for s in (42, 43):
            w, rl = warm_label(fam.name, fold, s), rl_label(fam.name, fold, s)
            acc_w, acc_rl = fam.accuracy(w, "outside"), fam.accuracy(rl, "outside")
            a, wa, base = fam.accuracy(fam.arm("A", fold, s), "outside"), fam.accuracy(fam.arm("WA", fold, s), "outside"), fam.accuracy(fam.base(), "outside")
            sc_w, sc_rl = shortcut_counts(eval_dir, w, final), shortcut_counts(eval_dir, rl, final)
            pk = passk.get((fold, s), {})
            for t in units:
                out.append({"mechanism": t, "fold": fold, "seed": s, "injected": t in design["text"][fold][str(s)],
                            "warm_p": pk.get(t, {}).get(1), "warm_pass64": pk.get(t, {}).get(64), "warm": acc_w.get(t), "rl": acc_rl.get(t),
                            "shortcut_warm": sc_w.get(t), "shortcut_rl": sc_rl.get(t), "A": a.get(t), "WA_full": wa.get(t), "base": base.get(t)})
    return out


def secondary(fam: Family, design: dict, obs: list[dict], dose_posthoc: dict | None) -> dict:
    for o in obs:
        o["rl-warm"] = None if o["rl"] is None or o["warm"] is None else o["rl"] - o["warm"]
        o["rl-A"] = None if o["rl"] is None or o["A"] is None else o["rl"] - o["A"]
    by_t: dict[str, dict] = {}
    for o in obs:
        by_t.setdefault(o["mechanism"], {})[o["injected"]] = o
    p_pairs = [{"mechanism": t, "p_text": v[True]["warm_p"], "p_none": v[False]["warm_p"]}
               for t, v in sorted(by_t.items()) if True in v and False in v]
    dose1 = {}
    if dose_posthoc:
        for d in dose_posthoc.get("observations", []):
            if d.get("dose") == 1:
                dose1[d["mechanism"]] = d["warm_p"]
    m = lambda k: (lambda v: mean_of(v, k))
    inj, non = [o for o in obs if o["injected"]], [o for o in obs if not o["injected"]]
    nonunit = []
    for fold, units in design["units"].items():
        for s in (42, 43):
            new, old = fam.accuracy(rl_label(fam.name, fold, s), "outside"), fam.accuracy(fam.arm("WA", fold, s), "outside")
            for t in fam.templates(fold, "train"):
                if t not in units and t not in P1_EXCLUDED and t in new and t in old:
                    nonunit.append({"mechanism": t, "seed": s, "new": new[t], "old": old[t], "d": new[t] - old[t]})
    return {"groups": {"injected": {"n": len(inj), "warm": mean_of(inj, "warm"), "rl": mean_of(inj, "rl"),
                                    "rl_at_or_above_50": sum(o["rl"] >= SUCCESS for o in inj if o["rl"] is not None),
                                    "p_at_or_above_0.02": sum(o["warm_p"] >= 0.02 for o in inj if o["warm_p"] is not None)},
                       "not_injected": {"n": len(non), "warm": mean_of(non, "warm"), "rl": mean_of(non, "rl"),
                                        "rl_at_or_above_50": sum(o["rl"] >= SUCCESS for o in non if o["rl"] is not None),
                                        "p_at_or_above_0.02": sum(o["warm_p"] >= 0.02 for o in non if o["warm_p"] is not None),
                                        "by_p": p_groups(non)}},
            "p_text_vs_none": p_pairs, "p_text_vs_dose1_program": [{"mechanism": t, "p_text": v[True]["warm_p"], "p_dose1": dose1.get(t)}
                                                                   for t, v in sorted(by_t.items()) if True in v] if dose1 else None,
            "rl-warm": {"injected": boot([o for o in inj if o["rl-warm"] is not None], m("rl-warm"))},
            "not_injected_rl-A": boot([o for o in non if o["rl-A"] is not None], m("rl-A")),
            "non_unit_new_rl-WA": boot(nonunit, m("d")) if nonunit else None}


def report(args) -> dict:
    design = json.loads(args.design.read_text())
    fam = Family(args)
    final = json.loads(args.final.read_text())
    passk, labels, missing = {}, [], []
    for fold in design["units"]:
        for s in (42, 43):
            labels += [warm_label(args.name, fold, s), rl_label(args.name, fold, s)]
            p = args.passk_dir / f"passk_final_outside.{warm_label(args.name, fold, s)}.jsonl"
            if p.exists():
                passk[(fold, s)] = passk_table(p)
            else:
                missing.append(str(p))
    for lab in labels:
        missing += fam.missing(lab)
    obs = observations(fam, design, passk, final, args.eval_dir)
    complete = not missing and all(o[k] is not None for o in obs for k in ("warm", "rl", "warm_p"))
    h1 = boot(obs, paired_stat)
    h1["unpaired"] = boot(obs, unpaired_stat)
    h1["verdict"] = ("成立" if h1["value"] is not None and h1["value"] >= H1_MIN and h1["ci95"] and h1["ci95"][0] > 0
                     else "不成立") if complete else "不完整"
    h2r = h2(obs)
    if not complete:
        h2r["verdict"] = "不完整"
    dose_posthoc = json.loads(args.dose_posthoc.read_text()) if args.dose_posthoc and args.dose_posthoc.exists() else None
    return {"family": args.name, "labels": labels, "missing": missing, "complete": complete,
            "infra_errors": {lab: e for lab in labels if (e := fam.infra_errors(lab))},
            "H1_text": h1, "H2_p": h2r, "secondary": secondary(fam, design, obs, dose_posthoc), "observations": obs}


def pts(x) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def show(r: dict) -> None:
    print(f"== dose-text {r['family']}: complete {r['complete']}; missing {r['missing'] or 'none'}")
    h = r["H1_text"]
    print(f"H1-text paired RL(text) - RL(none) = {pts(h['value'])} {[pts(x) for x in h['ci95']] if h['ci95'] else ''} "
          f"(unpaired {pts(h['unpaired']['value'])}) -> {h['verdict']}")
    g = r["H2_p"]
    for name, _, _ in P_BINS:
        print(f"H2-p injected {name}: {g[name]['successes']} / {g[name]['n']} at >= 50 %")
    print(f"  -> {g['verdict']}" + (f" (untestable: {g['untestable_groups']})" if g["untestable_groups"] else ""))
    s = r["secondary"]
    for k, v in s["groups"].items():
        print(f"{k}: n {v['n']}, warm {pts(v['warm'])}, RL {pts(v['rl'])}, RL >= 50 % {v['rl_at_or_above_50']}, p >= 0.02 {v['p_at_or_above_0.02']}")
    for o in sorted(r["observations"], key=lambda o: (o["warm_p"] is None, o["warm_p"] or 0)):
        print(f"  {o['fold']} s{o['seed']} {o['mechanism']:34s} {'text' if o['injected'] else 'none'} p {o['warm_p'] if o['warm_p'] is None else f'{o['warm_p']:.4f}'} "
              f"warm {pts(o['warm'])} RL {pts(o['rl'])} shortcut {o['shortcut_warm']}->{o['shortcut_rl']}")
    b = s["rl-warm"]["injected"]
    print(f"RL - warm (injected): {pts(b['value'])} {[pts(x) for x in b['ci95']] if b['ci95'] else ''} n {b['observations']}")
    b = s["not_injected_rl-A"]
    print(f"not injected, RL - A: {pts(b['value'])} {[pts(x) for x in b['ci95']] if b['ci95'] else ''} n {b['observations']}")
    n = s["non_unit_new_rl-WA"]
    if n:
        print(f"non-unit mechanisms, new RL - WA: {pts(n['value'])} {[pts(x) for x in n['ci95']] if n['ci95'] else ''}")


def main() -> int:
    from atmoscoder_rl.paths import EXPERIMENTS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--passk-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--dose-posthoc", type=Path, help="results/dose/posthoc_<family>.json, for the p of DOSE's dose-1 observations")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.seeds = [42, 43]
    r = report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    show(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
