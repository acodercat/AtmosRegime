"""The textbook regime traps after RL: every X endpoint on the 46 test-side regime traps and their parents (X's textbook.json
evaluation), by arm, with trap-resampled intervals; frontier runs on the same 46 traps as reference. Descriptive, numbers only.

    python -m explore.traps_rl_report --name qwen3-8b --seeds 42 43 --eval-dir experiments/x_textbook/eval \
        --inputs experiments/x_textbook/inputs --traps <Bench-Lab>/benchmark/traps.json \
        [--frontier results/paper/traps_frontier.json] --out results/traps_rl/report_qwen3-8b.json

Grading is the runner's verdict (as x_report.textbook). An arm's per-trap accuracy averages its endpoints (two folds x the
seeds); starts are the base (A, D, F) and W_f of each fold (WA, WD). Statistics per arm: trap and parent accuracy, the paired
within-trap gain over the start (trap bootstrap, 10,000 draws, random.Random("traps-rl:<family>:<arm>:20261029")), P(trap right |
parent right) pooled over endpoints, discordant counts against the start, shortcut capture among failed endpoint rows (every
sub-answer within 2 % of shortcut_values), and the same by category (five kept, the rest "other") and on the pilot subset.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

DRAWS = 10_000
ARMS = {"A": "base", "D": "base", "F": "base", "WA": "warm", "WD": "warm", "R": "base"}
KEEP = ("atmospheric_thermodynamics", "atmospheric_dynamics", "atmospheric_chemistry", "air_quality", "climate_dynamics")


def close(a, b, tol=0.02) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1e-12)
    except (TypeError, ValueError):
        return False


def pct(vals):
    if not vals:
        return None
    v = sorted(vals)
    return [v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--traps", type=Path, required=True)
    ap.add_argument("--frontier", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    groups = json.loads((args.inputs / "textbook_groups.json").read_text())
    parent_of = groups["parent_of"]
    ids = sorted(groups["regime_arm"])
    pilot = set(groups["regime_recognise_pilot"])
    meta = {r["id"]: r for r in json.loads(args.traps.read_text())}
    meta.update({r["id"]: r for r in json.loads((args.inputs / "textbook.json").read_text())})   # the twins (traps_regime_twins) live here only
    cat = {i: (meta[i]["category"] if meta[i]["category"] in KEEP else "other") for i in ids}

    def records(label):
        p = args.eval_dir / label / "textbook.json"
        return {r["id"]: r for r in json.loads(p.read_text())["results"] if "details" in r or r.get("passed") is not None} if p.exists() else None
    labels = {"base": [f"x-{args.name}-base"], "warm": [f"sft-x-{args.name}-F1", f"sft-x-{args.name}-F2"]}
    for a in ARMS:
        labels[a] = [l for f in ("F1", "F2") for s in args.seeds if (args.eval_dir / (l := f"x-{args.name}-{f}-{a}-s{s}") / "textbook.json").exists()]
    recs = {l: records(l) for ls in labels.values() for l in ls}
    present = {k: [l for l in ls if recs.get(l)] for k, ls in labels.items()}

    def per_trap(label_list):
        """trap -> (mean trap passed, mean parent passed, failed rows, captured rows) over the labels."""
        out = {}
        for i in ids:
            t, p, failed, cap = [], [], 0, 0
            for l in label_list:
                r, pr = recs[l].get(i), recs[l].get(parent_of[i])
                if r is None or pr is None:
                    continue
                t.append(float(bool(r["passed"]))); p.append(float(bool(pr["passed"])))
                if not r["passed"]:
                    failed += 1
                    actual = {d.get("sub"): d.get("actual") for d in (r.get("details") or []) if isinstance(d, dict)}
                    sv = meta[i].get("shortcut_values") or {}
                    cap += bool(sv) and all(s in actual and close(actual[s], v) for s, v in sv.items())
            if t:
                out[i] = {"trap": fmean(t), "parent": fmean(p), "failed": failed, "captured": cap, "n": len(t)}
        return out

    def summarise(name, pt, start_pt, subset=None):
        keys = [i for i in pt if (subset is None or i in subset) and (start_pt is None or i in start_pt)]
        rng = random.Random(f"traps-rl:{args.name}:{name}:20261029")
        boots_t = [fmean(pt[k]["trap"] for k in (rng.choice(keys) for _ in keys)) for _ in range(DRAWS)] if keys else []
        s = {"traps": len(keys), "trap_acc": fmean(pt[k]["trap"] for k in keys) if keys else None, "trap_acc_ci95": pct(boots_t),
             "parent_acc": fmean(pt[k]["parent"] for k in keys) if keys else None,
             "failed_rows": sum(pt[k]["failed"] for k in keys), "captured_rows": sum(pt[k]["captured"] for k in keys)}
        s["capture_share"] = s["captured_rows"] / s["failed_rows"] if s["failed_rows"] else None
        if start_pt is not None and keys:
            d = {k: pt[k]["trap"] - start_pt[k]["trap"] for k in keys}
            boots = [fmean(d[k] for k in (rng.choice(keys) for _ in keys)) for _ in range(DRAWS)]
            s["gain_over_start"] = {"mean": fmean(d.values()), "ci95": pct(boots)}
            s["parent_gain_over_start"] = fmean(pt[k]["parent"] - start_pt[k]["parent"] for k in keys)
            s["discordant_vs_start"] = {"trap_right_only_after": sum(pt[k]["trap"] > start_pt[k]["trap"] for k in keys),
                                        "trap_right_only_before": sum(pt[k]["trap"] < start_pt[k]["trap"] for k in keys)}
        return s

    def conditional(label_list, subset=None):
        num = den = 0
        for l in label_list:
            for i in ids:
                if subset is not None and i not in subset:
                    continue
                r, pr = recs[l].get(i), recs[l].get(parent_of[i])
                if r is not None and pr is not None and pr["passed"]:
                    den += 1; num += bool(r["passed"])
        return {"trap_right": num, "parent_right": den, "share": num / den if den else None}

    starts = {"base": per_trap(present["base"]), "warm": per_trap(present["warm"])}
    out = {"family": args.name, "seeds": args.seeds, "traps": len(ids), "labels": {k: v for k, v in present.items() if v}, "arms": {}}
    for name in ("base", "warm"):
        if present[name]:
            pt = starts[name]
            out["arms"][name] = {"all": summarise(name, pt, None), "pilot38": summarise(name + "-pilot", pt, None, pilot),
                                 "conditional": conditional(present[name]), "by_category": {c: summarise(f"{name}-{c}", {k: v for k, v in pt.items() if cat[k] == c}, None) for c in sorted(set(cat.values()))},
                                 "per_trap": pt}
    for a, start_name in ARMS.items():
        if not present.get(a):
            continue
        pt, st = per_trap(present[a]), starts[start_name]
        out["arms"][a] = {"endpoints": len(present[a]), "start": start_name, "all": summarise(a, pt, st), "pilot38": summarise(a + "-pilot", pt, st, pilot),
                          "conditional": conditional(present[a]),
                          "by_category": {c: summarise(f"{a}-{c}", {k: v for k, v in pt.items() if cat[k] == c}, {k: v for k, v in st.items() if cat[k] == c}) for c in sorted(set(cat.values()))},
                          "per_trap": pt}
    if args.frontier and args.frontier.exists():
        fr = json.loads(args.frontier.read_text())["configs"]
        out["frontier_on_same_traps"] = {}
        for m, v in fr.items():
            rows = {i: v["per_trap"][i] for i in ids if i in v["per_trap"]}
            if len(rows) >= 30:
                out["frontier_on_same_traps"][m] = {"traps": len(rows), "trap_acc": fmean(r["trap"] for r in rows.values()),
                                                   "parent_acc": fmean(r["parent_acc"] for r in rows.values() if r["parent_acc"] is not None),
                                                   "capture_share": (sum(r["captured_runs"] for r in rows.values()) / sum(r["failed_runs"] for r in rows.values())) if sum(r["failed_runs"] for r in rows.values()) else None}
    out["categories"] = {c: sorted(i for i in ids if cat[i] == c) for c in sorted(set(cat.values()))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"== {args.name}: {len(ids)} regime traps; labels {sum(len(v) for v in present.values())}")
    for a, v in out["arms"].items():
        s, c = v["all"], v["conditional"]
        g = s.get("gain_over_start")
        print(f"{a:5s} trap {100 * s['trap_acc']:5.1f} {[round(100 * x, 1) for x in s['trap_acc_ci95']]} parent {100 * s['parent_acc']:5.1f} "
              f"cond {c['trap_right']}/{c['parent_right']}={100 * (c['share'] or 0):.0f}% capture {s['capture_share'] if s['capture_share'] is None else round(s['capture_share'], 2)}"
              + (f" | gain {100 * g['mean']:+.1f} {[round(100 * x, 1) for x in g['ci95']]} parent gain {100 * s['parent_gain_over_start']:+.1f} discordant {s['discordant_vs_start']}" if g else ""))
        if a in ("A", "D", "F", "WA", "WD"):
            print("      by category: " + "; ".join(f"{cat_}: {w['traps']} traps {100 * w['trap_acc']:.0f}% (gain {100 * w['gain_over_start']['mean']:+.0f})" for cat_, w in v["by_category"].items() if w["traps"]))
    for m, v in out.get("frontier_on_same_traps", {}).items():
        print(f"frontier {m:28s} trap {100 * v['trap_acc']:5.1f} parent {100 * v['parent_acc']:5.1f} capture {v['capture_share']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
