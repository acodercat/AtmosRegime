"""PREREG_X X-40 (exploratory, post hoc; numbers only): does the knowledge the warm start injects let answer-reward RL
succeed on mechanisms the base never solves? Written and committed before any of its numbers were computed.

    python -m explore.x_warm_knowledge --name qwen3-8b --seeds 42 43 \
        --passk-wf F1=$E/_x/qwen3-8b/passk_final_outside.wf-F1.jsonl F2=$E/_x/qwen3-8b/passk_final_outside.wf-F2.jsonl \
        --warm-summary F1=$R/data/sft_x-qwen3-8b-F1.summary.json F2=$R/data/sft_x-qwen3-8b-F2.summary.json \
        --eval-dir $E/_x_eval --final $E/_inputs/kc/final.json --inputs $E/_inputs/x_eval --out warm_knowledge_qwen3-8b.json

Units, accuracy and resampling are P1's (explore/x_report.py, reused unchanged): training-fold mechanisms (templates) without
plume_mixing_lid, each scored by the arm trained on its fold, regraded final.json outside accuracy, seeds averaged per
mechanism, mechanisms resampled 10,000 times with random.Random("x:20261012"). Base pass@64 is the X-18 sample; W_f pass@64
is the §4 sample of the fold's W_f. "Injected" = the fold's warm-build kept at least one verified pair for the template
(warm-build summary `pairs_per_template`: given rows of the template whose 8 base samples gave a two-sided-correct program).
1. P1 with the warm start as the start: Z_wf (W_f pass@64 = 0), its mechanisms at >= 50 % on WA / WD (any seed, every seed),
   Spearman(W_f pass@64, WA / WD outside accuracy);
2. Z (base pass@64 = 0) split into injected / not: per group the outside accuracy of base, W_f, A, D, WA, WD and the count of
   mechanisms at >= 50 %;
3. paired differences over mechanisms, on Z and on the rest: W_f - base, WA - W_f, WD - W_f, A - base, WA - A.
No thresholds and no verdicts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore.x_report import FOLDS, P1_EXCLUDED, Family, avg, ci, diff, passk_table, spearman_ci

LABELS = ("base", "W_f", "A", "D", "WA", "WD")
ARMS = ("A", "D", "WA", "WD")
DIFFS = (("W_f", "base"), ("WA", "W_f"), ("WD", "W_f"), ("A", "base"), ("WA", "A"))


def folds_arg(items: list[str]) -> dict[str, Path]:
    out = dict(i.split("=", 1) for i in items)
    if sorted(out) != list(FOLDS):
        raise SystemExit(f"need F1=... F2=..., got {items}")
    return {f: Path(p) for f, p in out.items()}


def table(fam: Family, passk_base: dict, passk_wf: dict, pairs: dict) -> dict:
    """mechanism -> fold, pass@64 of base and W_f, injected pairs, outside accuracy per label (arms per seed and seed mean)."""
    out = {}
    base_acc = fam.accuracy(fam.base(), "outside")
    for fold in FOLDS:
        wf_acc = fam.accuracy(fam.warm(fold), "outside")
        arm_acc = {a: {s: fam.accuracy(fam.arm(a, fold, s), "outside") for s in fam.seeds} for a in ARMS}
        for t in fam.templates(fold, "train"):
            if t in P1_EXCLUDED:
                continue
            row = {"fold": fold, "base_pass64": passk_base.get(t, {}).get(64), "wf_pass64": passk_wf[fold].get(t, {}).get(64),
                   "pairs": pairs[fold].get(t, 0), "base": base_acc.get(t), "W_f": wf_acc.get(t)}
            for a in ARMS:
                seeds = {s: arm_acc[a][s][t] for s in fam.seeds if t in arm_acc[a][s]}
                row.update({f"{a}_s{s}": v for s, v in seeds.items()})
                row[a] = avg([{0: v} for v in seeds.values()]).get(0)
            out[t] = row
    return out


def column(rows: dict, label: str, keep) -> dict:
    return {t: r[label] for t, r in rows.items() if keep(t, r) and r[label] is not None}


def analyse(rows: dict, seeds: list[int]) -> dict:
    with_wf = {t: r for t, r in rows.items() if r["wf_pass64"] is not None}
    z_wf = sorted(t for t, r in with_wf.items() if r["wf_pass64"] == 0)
    p1w = {"Z_wf": z_wf, "with_wf_pass64": len(with_wf)}
    for a in ("WA", "WD"):
        p1w[f"Z_wf_{a}_at_or_above_50_any_seed"] = [t for t in z_wf if any((rows[t].get(f"{a}_s{s}") or 0) >= .5 for s in seeds)]
        p1w[f"Z_wf_{a}_at_or_above_50_every_seed"] = [t for t in z_wf if all((rows[t].get(f"{a}_s{s}") or 0) >= .5 for s in seeds)]
        p1w[f"spearman_wf_pass64_vs_{a}"] = spearman_ci([(r["wf_pass64"], r[a]) for r in with_wf.values() if r[a] is not None])

    z = {t for t, r in rows.items() if r["base_pass64"] == 0}
    groups = {"Z_injected": lambda t, r: t in z and r["pairs"] >= 1,
              "Z_not_injected": lambda t, r: t in z and r["pairs"] == 0,
              "Z": lambda t, r: t in z,
              "rest": lambda t, r: t not in z and r["base_pass64"] is not None}
    out = {"p1_warm_start": p1w, "Z": sorted(z), "groups": {}}
    for g, keep in groups.items():
        members = sorted(t for t, r in rows.items() if keep(t, r))
        acc = {lab: ci(list(column(rows, lab, keep).values())) for lab in LABELS}
        at50 = {lab: sorted(t for t, v in column(rows, lab, keep).items() if v >= .5) for lab in LABELS}
        dif = {f"{x}-{y}": diff(column(rows, x, keep), column(rows, y, keep)) for x, y in DIFFS}
        out["groups"][g] = {"mechanisms": members, "pairs": {t: rows[t]["pairs"] for t in members},
                            "accuracy": acc, "at_or_above_50": at50, "differences": dif}
    return out


def report(args) -> dict:
    fam = Family(args)
    passk_wf = {f: passk_table(p) for f, p in args.passk_wf.items()}
    pairs = {f: json.loads(p.read_text())["pairs_per_template"] for f, p in args.warm_summary.items()}
    rows = table(fam, passk_table(args.passk), passk_wf, pairs)
    return {"family": args.name, "seeds": fam.seeds,
            "note": "X-40：探索性、事后（写于计算之前）；注入非随机，不作因果解读；只报数字",
            "inputs": {"passk": str(args.passk), **{f"passk_wf_{f}": str(p) for f, p in args.passk_wf.items()},
                       **{f"warm_summary_{f}": str(p) for f, p in args.warm_summary.items()}},
            "analysis": analyse(rows, fam.seeds), "per_mechanism": rows}


def pts(x) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def show(r: dict) -> None:
    a = r["analysis"]
    print(f"== X-40 warm-start knowledge: {r['family']}, seeds {r['seeds']}")
    p = a["p1_warm_start"]
    print(f"P1 with W_f as start: |Z_wf| = {len(p['Z_wf'])} of {p['with_wf_pass64']}")
    for arm in ("WA", "WD"):
        s = p[f"spearman_wf_pass64_vs_{arm}"]
        lo_hi = "" if not s["ci95"] else f" [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]"
        print(f"  {arm}: Spearman(W_f pass@64, {arm}) = {s['rho'] if s['rho'] is None else round(s['rho'], 3)}{lo_hi}; "
              f"Z_wf >= 50 % any seed {p[f'Z_wf_{arm}_at_or_above_50_any_seed']}, every seed {p[f'Z_wf_{arm}_at_or_above_50_every_seed']}")
    for g, d in a["groups"].items():
        print(f"{g} ({len(d['mechanisms'])} mechanisms)")
        print("  accuracy " + "  ".join(f"{lab} {pts(d['accuracy'][lab]['mean'])}" for lab in LABELS))
        print("  >= 50 %  " + "  ".join(f"{lab} {len(d['at_or_above_50'][lab])}" for lab in LABELS))
        for k, v in d["differences"].items():
            c = v["ci95"]
            print(f"  {k:9s} {pts(v['mean'])}" + ("" if not c else f" [{pts(c[0])}, {pts(c[1])}]") + f"  n {v['units']}")


def main() -> int:
    from atmoscoder_rl.paths import EXPERIMENTS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--passk", type=Path, help="base pass@64 (default: the X-18 sample)")
    ap.add_argument("--passk-wf", nargs=2, required=True, metavar="FOLD=PATH")
    ap.add_argument("--warm-summary", nargs=2, required=True, metavar="FOLD=PATH")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.passk = args.passk or EXPERIMENTS_DIR / "_x" / args.name / "passk_final_outside.x18.jsonl"
    args.passk_wf, args.warm_summary = folds_arg(args.passk_wf), folds_arg(args.warm_summary)
    r = report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    show(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
