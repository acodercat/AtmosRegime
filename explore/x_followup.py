"""PREREG_X X-29: the batch check of the re-evaluated starts (section 1) and the random-reward arm R (section 2).

    python -m explore.x_followup --name qwen3-8b --report results/x/report_qwen3-8b.json --out results/x/followup_qwen3-8b.json

Everything goes through `x_report`'s own measures: template mean -> cluster mean, seeds averaged within a cluster, 10,000
draws, `random.Random("x:20261012")` (`diff`, `ci`). Starts re-evaluated in the new batch carry SUFFIX ("-b2").

1. batch: for each start that has a "-b2" twin, new - old per variant (outside, inside, given, flag) over all templates.
   "no batch effect" iff every |mean| <= 3 points and every interval contains 0 (X-29 section 1).
2. R: R - base, A - R on the held-out and the training fold, outside and inside. R's start is the new-batch base. A keeps its
   own seeds (all of them, as `x_report`); A s42 - R s42 is reported beside it. Without a batch effect A - R is the direct
   cluster difference; with one, it is (A - old base) - (R - new base), cluster by cluster. Readings on the held-out fold
   (X-29 section 2, several can apply): R - base < 5 points with the interval containing 0 on both inside and outside ->
   "no detectable effect of the random reward"; A - R on inside >= 10 points with a lower bound > 0 -> "the held-out inside gain comes from the correctness reward, not from the random reward";
   R - base >= 10 points with a lower bound > 0 on inside or outside -> "the random reward also adds R - base points"; none -> "numbers only".
   Training fold: per P1 zero-hit mechanism, A per seed and R outside accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl.paths import EXPERIMENTS_DIR
from explore.x_report import FOLDS, Family, acc_measure, avg, ci, diff

SUFFIX = "-b2"
VARIANTS = ("outside", "inside", "given", "flag")
SIDES = ("held", "train")
NUMBERS = "只报数字"


def exists(fam: Family, label: str) -> bool:
    return (fam.dir / label / "final.json").exists()


def batch_check(fam: Family) -> dict:
    tpls = sorted(fam.cluster)
    starts = [fam.base()] + [fam.warm(f) for f in FOLDS]
    out, ok = {}, True
    for old in starts:
        new = old + SUFFIX
        if not exists(fam, new):
            continue
        part = {}
        for v in VARIANTS:
            m = acc_measure(fam, v)
            d = diff(m(new, tpls), m(old, tpls))
            part[v] = d
            ok &= d["mean"] is not None and abs(d["mean"]) <= .03 and d["ci95"][0] <= 0 <= d["ci95"][1]
        out[old] = part
    return {"starts": out, "verdict": ("未评" if not out else "无批次效应" if ok else "有批次效应")}


def side_of(fam: Family, arm: str, seeds: list[int], side: str, measure) -> dict:
    out = {}
    for fold in FOLDS:
        out.update(avg([measure(fam.arm(arm, fold, s), fam.templates(fold, side)) for s in seeds]))
    return out


def start_of(fam: Family, label: str, side: str, measure) -> dict:
    out = {}
    for fold in FOLDS:
        out.update(measure(label, fam.templates(fold, side)))
    return out


def gain_diff(a: dict, a0: dict, r: dict, r0: dict) -> dict:
    """(A - A's start) - (R - R's start), cluster by cluster."""
    units = sorted(set(a) & set(a0) & set(r) & set(r0))
    return ci([(a[k] - a0[k]) - (r[k] - r0[k]) for k in units])


def r_arm(fam: Family, r_seeds: list[int], batch_ok: bool, report: dict) -> dict:
    base_old, base_new = fam.base(), fam.base() + SUFFIX
    if not exists(fam, base_new) or not all(exists(fam, fam.arm("R", f, s)) for f in FOLDS for s in r_seeds):
        return {"verdict": "不完整：只报已有数字"}
    out = {"batch_ok": batch_ok, "A_seeds": fam.seeds, "R_seeds": r_seeds}
    for side in SIDES:
        for v in ("outside", "inside"):
            m = acc_measure(fam, v)
            r, r0 = side_of(fam, "R", r_seeds, side, m), start_of(fam, base_new, side, m)
            a, a0 = side_of(fam, "A", fam.seeds, side, m), start_of(fam, base_old, side, m)
            a42 = side_of(fam, "A", [s for s in fam.seeds if s in r_seeds], side, m)
            out[f"{side}_{v}"] = {
                "R-base": diff(r, r0),
                "A-R": diff(a, r) if batch_ok else gain_diff(a, a0, r, r0),
                "A_matched_seed-R": diff(a42, r) if batch_ok else gain_diff(a42, a0, r, r0),
                "A-R_is": "direct" if batch_ok else "gain over each arm's own-batch start"}
    hi, ho = out["held_inside"], out["held_outside"]
    small = lambda d: d["mean"] is not None and d["mean"] < .05 and d["ci95"][0] <= 0 <= d["ci95"][1]
    big = lambda d: d["mean"] is not None and d["mean"] >= .10 and d["ci95"][0] > 0
    readings = []
    if small(hi["R-base"]) and small(ho["R-base"]):
        readings.append("随机奖励无可检出效应")
    if big(hi["A-R"]):
        readings.append("留出折原题的提高来自正确性奖励，不是随机奖励的效应")
    for v, d in (("原题", hi["R-base"]), ("越界", ho["R-base"])):
        if big(d):
            readings.append(f"随机奖励也带来 R − 基座 {100 * d['mean']:.1f} 点（留出折{v}）")
    out["readings"] = readings or [NUMBERS]
    zero = report["P1"]["Z"] if report else []
    per = {}
    for t in zero:
        f = fam.fold_of[t]
        row = {f"A_s{s}": fam.accuracy(fam.arm("A", f, s), "outside").get(t) for s in fam.seeds}
        row.update({f"R_s{s}": fam.accuracy(fam.arm("R", f, s), "outside").get(t) for s in r_seeds})
        per[t] = row
    out["zero_hit_training_fold"] = per
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", help="A's seeds (default as x_report)")
    ap.add_argument("--r-seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--report", type=Path, help="the family's x_report output, for P1's zero-hit mechanisms")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.seeds = args.seeds or ([42, 43] if args.name == "qwen3-8b" else [42])
    fam = Family(args)
    report = json.loads(args.report.read_text()) if args.report else None
    if report and report.get("family") != args.name:
        raise SystemExit(f"--report is family {report.get('family')!r}, not {args.name!r}")
    b = batch_check(fam)
    r = {"family": args.name, "note": "X-29：读结果后写定的补充分析，不改第 6 节判定", "batch": b,
         "R": r_arm(fam, args.r_seeds, b["verdict"] == "无批次效应", report)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    pct = lambda d: "-" if d.get("mean") is None else f"{100 * d['mean']:.1f} [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}]"
    print("batch:", b["verdict"])
    for s, part in b["starts"].items():
        print(f"  {s}", "  ".join(f"{v} {pct(d)}" for v, d in part.items()))
    for k, v in r["R"].items():
        if isinstance(v, dict) and "R-base" in v:
            print(f"R {k:14s} R-base {pct(v['R-base'])}  A-R {pct(v['A-R'])}  A(matched)-R {pct(v['A_matched_seed-R'])}")
    print("R readings:", r["R"].get("readings", r["R"].get("verdict")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
