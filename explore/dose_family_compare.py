"""PREREG_DOSE_MINISTRAL §5 "cross-family side-by-side table" + the DM-23 review tables: the two DOSE families side by side. Numbers only, no verdict.

    python -m explore.dose_family_compare \
        --name-a qwen3-8b --report-a results/dose/report_qwen3-8b.json --posthoc-a results/dose/posthoc_qwen3-8b.json \
        --name-b ministral3-8b --report-b results/dose_ministral/report_ministral3-8b.json --posthoc-b results/dose_ministral/posthoc_ministral3-8b.json \
        --eval-dirs experiments/dose/eval experiments/dose_ministral/eval experiments/x_base_eval \
        --train-log-dirs experiments/dose/logs/train experiments/dose_ministral/logs/train \
        --out results/dose_ministral/family_compare.json

Everything here is read from the two committed reports (dose_report / dose_posthoc) and, for the last two sections, from the
archived evaluations and training logs. Sections:
  paired_mechanisms   every mechanism of either design: per family and seed its dose, warm-start p / pass@64, warm and RL outside
                      accuracy, learned (RL >= 0.5); counts over the mechanisms both families dosed in both seeds / in any seed.
  dose_levels         per family and dose: n, learned, mean warm, mean RL (report secondary.per_dose).
  p_bins              dosed observations by warm-start p (posthoc bins) per family and pooled; the p values of the learned and
                      the not-learned observations in the lowest bin.
  same_dose_seeds     mechanisms whose two seeds had the same dose >= 1: the two warm-start p and their ratio, warm and RL outcomes.
  rl_failures         posthoc rl_failures per family (every failing endpoint gives the shortcut on all outside rows?).
  overall_accuracy    base / warm / RL accuracy on the four evaluation files (right / gradable), per family, if the labels are found.
  training            per RL run: logged steps, reward first / last / mean of the last 10, mean clip ratio, response length first / last.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

SUCCESS = 0.5
BINS = ((0.0, 0.005), (0.005, 0.02), (0.02, 2.0))
SEEDS = ("42", "43")
EVAL_FILES = ("final", "judge", "test_id", "textbook")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def learned(rl: float) -> bool:
    return rl >= SUCCESS


def by_mechanism(posthoc: dict) -> dict[str, dict[str, dict]]:
    """mechanism -> seed -> observation (dose, warm_p, warm_pass64, warm, rl, learned, fold)."""
    out: dict[str, dict[str, dict]] = {}
    for o in posthoc["observations"]:
        out.setdefault(o["mechanism"], {})[str(o["seed"])] = {
            "fold": o["fold"], "dose": o["dose"], "warm_p": o["warm_p"], "warm_pass64": o["warm_pass64"],
            "warm": o["warm"], "rl": o["rl"], "learned": learned(o["rl"]),
            "shortcut_warm": o.get("shortcut_warm"), "shortcut_rl": o.get("shortcut_rl")}
    return out


def paired(a: dict, b: dict, name_a: str, name_b: str) -> dict:
    mechs = sorted(set(a) | set(b))
    table = {m: {name_a: a.get(m), name_b: b.get(m)} for m in mechs}

    def dosed_all(fam: dict | None) -> bool:
        return bool(fam) and all(fam[s]["dose"] >= 1 for s in SEEDS if s in fam) and len(fam) == 2

    def learned_all(fam: dict) -> bool:
        return all(fam[s]["learned"] for s in fam)

    def learned_any(fam: dict) -> bool:
        return any(fam[s]["learned"] for s in fam)

    common = [m for m in mechs if m in a and m in b]
    both_dosed = [m for m in common if dosed_all(a[m]) and dosed_all(b[m])]
    any_dosed = [m for m in common if any(v["dose"] >= 1 for v in a[m].values()) and any(v["dose"] >= 1 for v in b[m].values())]
    dose0_both = [m for m in common if all(v["dose"] == 0 for v in a[m].values()) and all(v["dose"] == 0 for v in b[m].values())]
    summary = {
        "mechanisms_a": len(a), "mechanisms_b": len(b), "common": len(common), "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a)),
        "both_seeds_dosed_in_both_families": {
            "n": len(both_dosed),
            "learned_both_seeds_in_both": sorted(m for m in both_dosed if learned_all(a[m]) and learned_all(b[m])),
            "not_learned_any_seed_in_either": sorted(m for m in both_dosed if not learned_any(a[m]) and not learned_any(b[m])),
            "mixed": sorted(m for m in both_dosed if (learned_all(a[m]) and learned_all(b[m])) is False and (not learned_any(a[m]) and not learned_any(b[m])) is False)},
        "dosed_in_any_seed_in_both_families": {
            "n": len(any_dosed),
            "learned_in_some_seed_in_both": sorted(m for m in any_dosed if learned_any(a[m]) and learned_any(b[m])),
            "learned_in_neither": sorted(m for m in any_dosed if not learned_any(a[m]) and not learned_any(b[m])),
            "learned_in_one_family_only": sorted(m for m in any_dosed if learned_any(a[m]) != learned_any(b[m]))},
        "dose_0_in_both_seeds_of_both_families": sorted(dose0_both),
        "dose_0_observations": {n: {"n": sum(1 for fam in x.values() for v in fam.values() if v["dose"] == 0),
                                    "learned": sum(1 for fam in x.values() for v in fam.values() if v["dose"] == 0 and v["learned"]),
                                    "mechanisms_learned": sorted({m for m, fam in x.items() for v in fam.values() if v["dose"] == 0 and v["learned"]})}
                                for n, x in ((name_a, a), (name_b, b))},
    }
    return {"table": table, "summary": summary}


def dose_levels(report: dict) -> dict:
    return {d: {"n": v["n"], "learned": v["rl_at_or_above_50"], "warm_mean": v["warm"], "rl_mean": v["rl"], "warm_pass64_pos": v["warm_pass64_pos"]}
            for d, v in report["secondary"]["per_dose"].items()}


def p_bins(posthocs: dict[str, dict]) -> dict:
    out = {}
    pooled = {f"[{lo}, {hi})": {"n": 0, "learned": 0} for lo, hi in BINS}
    for name, p in posthocs.items():
        bins = {f"[{b['p_from']}, {b['p_to']})": {"n": b["n"], "learned": b["learned"]} for b in p["dosed_by_warm_p"]}
        for k, b in bins.items():
            pooled[k]["n"] += b["n"]; pooled[k]["learned"] += b["learned"]
        lo, hi = BINS[0]
        low = [o for o in p["observations"] if o["dose"] >= 1 and lo <= o["warm_p"] < hi]
        out[name] = {"dosed_by_warm_p": bins,
                     "dose0_by_warm_p": {f"[{b['p_from']}, {b['p_to']})": {"n": b["n"], "learned": b["learned"]} for b in p["dose0_by_warm_p"]},
                     "lowest_bin": {"learned": sorted((round(o["warm_p"], 5), o["mechanism"], o["seed"]) for o in low if learned(o["rl"])),
                                    "not_learned": sorted((round(o["warm_p"], 5), o["mechanism"], o["seed"]) for o in low if not learned(o["rl"]))}}
    out["pooled_dosed"] = pooled
    return out


def same_dose_seeds(mech: dict[str, dict[str, dict]]) -> list[dict]:
    out = []
    for m, fam in sorted(mech.items()):
        if len(fam) != 2 or fam["42"]["dose"] != fam["43"]["dose"] or fam["42"]["dose"] < 1:
            continue
        p = [fam[s]["warm_p"] for s in SEEDS]
        out.append({"mechanism": m, "dose": fam["42"]["dose"], "warm_p": p,
                    "p_ratio": (max(p) / min(p)) if min(p) > 0 else None,
                    "warm": [fam[s]["warm"] for s in SEEDS], "rl": [fam[s]["rl"] for s in SEEDS],
                    "learned": [fam[s]["learned"] for s in SEEDS]})
    return out


def accuracy(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    rows = d["results"] if isinstance(d, dict) and "results" in d else d
    graded = [r for r in rows if r.get("status") != "error" and not r.get("error")]
    return {"rows": len(rows), "graded": len(graded), "accuracy": sum(1 for r in graded if r.get("passed")) / len(graded) if graded else None}


def overall(name: str, eval_dirs: list[Path]) -> dict:
    labels = [f"x-{name}-base"] + [f"sft-dose-{name}-F{f}-s{s}" for f in (1, 2) for s in SEEDS] + [f"dose-{name}-F{f}-WA-s{s}" for f in (1, 2) for s in SEEDS]
    out = {}
    for label in labels:
        d = next((e / label for e in eval_dirs if (e / label / "final.json").exists()), None)
        out[label] = {"dir": str(d) if d else None, **{f: accuracy(d / f"{f}.json") if d else None for f in EVAL_FILES}}
    return out


def parse_training(text: str) -> dict:
    text = ANSI.sub("", text)
    score = [float(x) for x in re.findall(r"critic/score/mean:([0-9.]+)", text)]
    clip = [float(x) for x in re.findall(r"response_length/clip_ratio:([0-9.]+)", text)]
    length = [float(x) for x in re.findall(r"response_length/mean:([0-9.]+)", text)]
    steps = [int(x) for x in re.findall(r"\bstep:(\d+)", text)]
    return {"logged_steps": len(score), "max_step": max(steps) if steps else None,
            "score_first": score[0] if score else None, "score_last": score[-1] if score else None,
            "score_last10_mean": statistics.fmean(score[-10:]) if score else None, "score_max": max(score) if score else None,
            "clip_mean": statistics.fmean(clip) if clip else None,
            "response_length_first": length[0] if length else None, "response_length_last": length[-1] if length else None}


def training(name: str, log_dirs: list[Path]) -> dict:
    out = {}
    for f in (1, 2):
        for s in SEEDS:
            label = f"dose-{name}-F{f}-WA-s{s}"
            p = next((d / f"{label}.log" for d in log_dirs if (d / f"{label}.log").exists()), None)
            out[label] = {"log": str(p), **parse_training(p.read_text(errors="replace"))} if p else None
    return out


def build(args) -> dict:
    rep = {n: json.loads(p.read_text()) for n, p in ((args.name_a, args.report_a), (args.name_b, args.report_b))}
    post = {n: json.loads(p.read_text()) for n, p in ((args.name_a, args.posthoc_a), (args.name_b, args.posthoc_b))}
    mech = {n: by_mechanism(p) for n, p in post.items()}
    out = {
        "families": {n: {"report": str(p), "H1": rep[n]["H1"], "H2": {k: ({kk: vv for kk, vv in v.items() if kk != "observations"} if isinstance(v, dict) else v) for k, v in rep[n]["H2"].items()},
                         "rl_minus_warm": rep[n]["secondary"].get("rl-warm")} for n, p in ((args.name_a, args.report_a), (args.name_b, args.report_b))},
        "paired_mechanisms": paired(mech[args.name_a], mech[args.name_b], args.name_a, args.name_b),
        "dose_levels": {n: dose_levels(r) for n, r in rep.items()},
        "p_bins": p_bins(post),
        "same_dose_seeds": {n: same_dose_seeds(m) for n, m in mech.items()},
        "rl_failures": {n: p["rl_failures"] for n, p in post.items()},
        "overall_accuracy": {n: overall(n, args.eval_dirs) for n in rep} if args.eval_dirs else None,
        "training": {n: training(n, args.train_log_dirs) for n in rep} if args.train_log_dirs else None,
    }
    return out


def show(out: dict) -> None:
    names = list(out["families"])
    for n in names:
        h1 = out["families"][n]["H1"]
        print(f"{n}: H1 {h1['value'] * 100:.1f} [{h1['ci95'][0] * 100:.1f}, {h1['ci95'][1] * 100:.1f}] {h1['verdict']}; "
              + "; ".join(f"dose {d}: {v['learned']}/{v['n']} learned, warm {v['warm_mean'] * 100:.1f}, RL {v['rl_mean'] * 100:.1f}" for d, v in out["dose_levels"][n].items()))
    s = out["paired_mechanisms"]["summary"]
    print(f"common mechanisms {s['common']} (only {names[0]}: {len(s['only_a'])}, only {names[1]}: {len(s['only_b'])})")
    b = s["both_seeds_dosed_in_both_families"]
    print(f"  dosed in both seeds of both families: {b['n']}; learned everywhere {len(b['learned_both_seeds_in_both'])} {b['learned_both_seeds_in_both']}; "
          f"learned nowhere {b['not_learned_any_seed_in_either']}; mixed {b['mixed']}")
    a = s["dosed_in_any_seed_in_both_families"]
    print(f"  dosed in some seed of both families: {a['n']}; learned in both {len(a['learned_in_some_seed_in_both'])}; neither {a['learned_in_neither']}; one family only {a['learned_in_one_family_only']}")
    print(f"  dose 0: " + "; ".join(f"{n} {v['learned']}/{v['n']} learned {v['mechanisms_learned']}" for n, v in s["dose_0_observations"].items()))
    for n in names:
        pb = out["p_bins"][n]
        print(f"{n} dosed by warm p: " + " | ".join(f"{k}: {v['learned']}/{v['n']}" for k, v in pb["dosed_by_warm_p"].items())
              + f"; lowest bin learned p {[x[0] for x in pb['lowest_bin']['learned']]} vs not {[x[0] for x in pb['lowest_bin']['not_learned']]}")
    print("pooled dosed by warm p: " + " | ".join(f"{k}: {v['learned']}/{v['n']}" for k, v in out["p_bins"]["pooled_dosed"].items()))
    for n in names:
        for c in out["same_dose_seeds"][n]:
            print(f"{n} same dose {c['dose']} {c['mechanism']}: warm p {c['warm_p'][0]:.4f} / {c['warm_p'][1]:.4f}"
                  + (f" (x{c['p_ratio']:.0f})" if c["p_ratio"] else "") + f", warm {c['warm']}, RL {c['rl']}")
        f = out["rl_failures"][n]
        print(f"{n} RL failures {f['n']} ({f['dosed']} dosed): shortcut on all rows in {f['shortcut_rl_all_rows']}; warm shortcut counts {f['shortcut_warm']}")
    if out["overall_accuracy"]:
        for n in names:
            for label, v in out["overall_accuracy"][n].items():
                if v["dir"]:
                    print(f"  {label:34s} " + " ".join(f"{f} {v[f]['accuracy'] * 100:5.1f}" for f in EVAL_FILES if v[f] and v[f]["accuracy"] is not None))
    if out["training"]:
        for n in names:
            for label, v in out["training"][n].items():
                if v:
                    print(f"  {label:34s} steps {v['logged_steps']:3d} reward {v['score_first']:.2f} -> {v['score_last']:.2f} (last10 {v['score_last10_mean']:.2f}) clip {v['clip_mean']:.3f} len {v['response_length_first']:.0f} -> {v['response_length_last']:.0f}")


def selftest() -> int:
    def obs(mech, seed, dose, p, rl, fold="F1"):
        return {"fold": fold, "seed": seed, "mechanism": mech, "dose": dose, "warm_p": p, "warm_pass64": min(1.0, p * 64), "warm": p, "rl": rl, "rows": 20}
    pa = {"observations": [obs("m1", 42, 1, 0.5, 1.0), obs("m1", 43, 1, 0.001, 0.0), obs("m2", 42, 0, 0.0, 0.0), obs("m2", 43, 0, 0.0, 0.0),
                           obs("m3", 42, 2, 0.01, 1.0), obs("m3", 43, 4, 0.03, 1.0), obs("m4", 42, 2, 0.003, 0.0), obs("m4", 43, 2, 0.004, 0.0)],
          "dosed_by_warm_p": [{"p_from": 0.0, "p_to": 0.005, "n": 3, "learned": 0}, {"p_from": 0.005, "p_to": 0.02, "n": 1, "learned": 1}, {"p_from": 0.02, "p_to": 2.0, "n": 2, "learned": 2}],
          "dose0_by_warm_p": [{"p_from": 0.0, "p_to": 0.005, "n": 2, "learned": 0}, {"p_from": 0.005, "p_to": 0.02, "n": 0, "learned": 0}, {"p_from": 0.02, "p_to": 2.0, "n": 0, "learned": 0}],
          "rl_failures": {"n": 4, "dosed": 2, "shortcut_rl_all_rows": 4, "shortcut_warm": [0, 1, 2, 3]}}
    pb = {"observations": [obs("m1", 42, 2, 0.2, 1.0), obs("m1", 43, 2, 0.3, 1.0), obs("m2", 42, 0, 0.01, 1.0), obs("m2", 43, 0, 0.0, 0.0),
                           obs("m3", 42, 1, 0.001, 0.0), obs("m3", 43, 1, 0.002, 0.0), obs("m5", 42, 4, 0.1, 1.0), obs("m5", 43, 4, 0.1, 1.0)],
          "dosed_by_warm_p": pa["dosed_by_warm_p"], "dose0_by_warm_p": pa["dose0_by_warm_p"], "rl_failures": pa["rl_failures"]}
    a, b = by_mechanism(pa), by_mechanism(pb)
    s = paired(a, b, "A", "B")["summary"]
    assert s["common"] == 3 and s["only_a"] == ["m4"] and s["only_b"] == ["m5"], s
    assert s["both_seeds_dosed_in_both_families"]["n"] == 2 and s["both_seeds_dosed_in_both_families"]["learned_both_seeds_in_both"] == [] \
        and s["both_seeds_dosed_in_both_families"]["mixed"] == ["m1", "m3"], s
    assert s["dosed_in_any_seed_in_both_families"]["learned_in_some_seed_in_both"] == ["m1"] and s["dosed_in_any_seed_in_both_families"]["learned_in_one_family_only"] == ["m3"]
    assert s["dose_0_in_both_seeds_of_both_families"] == ["m2"] and s["dose_0_observations"]["B"] == {"n": 2, "learned": 1, "mechanisms_learned": ["m2"]}
    c = same_dose_seeds(a)
    assert [x["mechanism"] for x in c] == ["m1", "m4"] and round(c[0]["p_ratio"]) == 500 and c[0]["learned"] == [True, False], c
    pb_ = p_bins({"A": pa, "B": pb})
    assert pb_["pooled_dosed"]["[0.0, 0.005)"] == {"n": 6, "learned": 0} and pb_["A"]["lowest_bin"]["not_learned"][0][1] == "m1"
    t = parse_training("\x1b[36mstep:1 critic/score/mean:0.5 response_length/clip_ratio:0.02 response_length/mean:600\n"
                       "step:2 critic/score/mean:0.7 response_length/clip_ratio:0.04 response_length/mean:800\n")
    assert t["logged_steps"] == 2 and t["max_step"] == 2 and t["score_last10_mean"] == 0.6 and t["clip_mean"] == 0.03 and t["response_length_last"] == 800, t
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in "ab":
        ap.add_argument(f"--name-{x}"); ap.add_argument(f"--report-{x}", type=Path); ap.add_argument(f"--posthoc-{x}", type=Path)
    ap.add_argument("--eval-dirs", type=Path, nargs="*", default=[])
    ap.add_argument("--train-log-dirs", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    for k in ("name_a", "report_a", "posthoc_a", "name_b", "report_b", "posthoc_b", "out"):
        if getattr(args, k) is None:
            ap.error(f"--{k.replace('_', '-')} is required")
    out = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    show(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
