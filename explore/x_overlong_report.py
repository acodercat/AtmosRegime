"""PREREG_X X-29 section 3 as read by X-31: Qwen3.5 F2-F with DAPO's overlong penalty (FOL) against F2-F, seeds 42 and 43.

    python -m explore.x_overlong_report --logs $ATMOS_ROOT/logs --out overlong_report.json

Everything goes through `x_report`'s measures and resampling (`diff`, 10,000 draws, `x:20261012`). The batch verdict is
`x_followup.batch_check` (X-29 section 1). Without a batch effect FOL - F is the direct cluster difference, each arm's
seeds averaged within a cluster first; with one, each seed's endpoint minus its own-batch start (FOL both seeds and F
seed 43: the -b2 base; F seed 42: the old base), seeds averaged, then FOL - F. FOL s43 - F s43 (one batch) is always
reported beside it. Row-pooled rates (parse, all-zero, format failure) and truncation are listed per arm, not differenced.
No threshold: the only sentence is whether both FOL seeds stay under both X-29 section 3 truncation thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.paths import EXPERIMENTS_DIR
from explore import x_overlong_check
from explore.x_followup import SUFFIX, batch_check, exists
from explore.x_report import Family, acc_measure, avg, diff, summary, truncation

NAME, FOLD, SEEDS, OLD_BATCH = "qwen35-9b", "F2", (42, 43), {42}   # F2-F seed 42 was evaluated in the X-26 batch
ARMS = ("FOL", "F")
SIDES = ("held", "train")
UNDER = "开超长惩罚后未再退化"
STILL = "开超长惩罚后仍退化"


def measures(fam: Family) -> dict:
    return {"H": lambda lab, t: fam.clusters(fam.flag0(lab, "outside"), t),
            "F": lambda lab, t: fam.clusters(fam.flag0(lab, "inside"), t),
            "H-F": fam.hf,
            "outside_acc": acc_measure(fam, "outside"),
            "inside_acc": acc_measure(fam, "inside")}


def start(fam: Family, arm: str, seed: int) -> str:
    return fam.base() if arm == "F" and seed in OLD_BATCH else fam.base() + SUFFIX


def compare(fam: Family, measure, tpls: list[str], batch_ok: bool) -> dict:
    per = {a: {s: measure(fam.arm(a, FOLD, s), tpls) for s in SEEDS} for a in ARMS}
    if batch_ok:
        d = diff(avg(list(per["FOL"].values())), avg(list(per["F"].values())))
    else:
        gain = lambda a: avg([{k: v - st[k] for k, v in per[a][s].items() if k in st}
                              for s in SEEDS for st in [measure(start(fam, a, s), tpls)]])
        d = diff(gain("FOL"), gain("F"))
    return {"FOL": summary(avg(list(per["FOL"].values()))), "F": summary(avg(list(per["F"].values()))),
            "FOL-F": d, "FOL-F_is": "direct" if batch_ok else "gain over each seed's own-batch start",
            "FOL_s43-F_s43": diff(per["FOL"][43], per["F"][43])}


def rates(fam: Family, arm: str, tpls: list[str]) -> dict:
    keep = set(tpls)
    scored = [x for s in SEEDS for x in fam.judged(fam.arm(arm, FOLD, s)) or [] if x["row"]["template"] in keep]
    parsed = [x for x in scored if x["judgement"] is not None]
    return {"rows": len(scored), "parse_rate": len(parsed) / len(scored) if scored else None,
            "all_zero_rate": fmean(float(x["judgement"] == 0) for x in parsed) if parsed else None,
            "format_failure_rate": 1 - len(parsed) / len(scored) if scored else None}


def report(fam: Family, logs: Path | None) -> dict:
    labels = [fam.arm(a, FOLD, s) for a in ARMS for s in SEEDS]
    needed = labels + [fam.base(), fam.base() + SUFFIX]
    missing = [lab for lab in needed if not exists(fam, lab) or fam.records(lab, "judge") is None]
    b = batch_check(fam)
    batch_ok = b["verdict"] == "无批次效应"
    out = {"family": fam.name, "note": "X-31：X-29 §3 的读法，不设门槛", "missing": missing, "batch": b["verdict"],
           "sides": {}}
    if missing:
        out["verdict"] = "不完整：只报已有数字"
    for side in SIDES:
        tpls = fam.templates(FOLD, side)
        part = {}
        if not missing:
            part = {k: compare(fam, m, tpls, batch_ok) for k, m in measures(fam).items()}
        part["rates"] = {a: rates(fam, a, tpls) for a in ARMS}
        out["sides"][side] = part
    tr = truncation(fam, [lab for lab in labels if exists(fam, lab)])["per_label"]
    per = {}
    for lab in labels:
        n, m = tr.get(lab, {}).get("final", [None, 0])
        per[lab] = {"final_truncated": [n, m], "final_share": n / m if m else None,
                    "clip_last20": x_overlong_check.training(logs, lab)["clip_last20"] if logs else None}
    out["truncation"] = per
    fol = [per[fam.arm("FOL", FOLD, s)] for s in SEEDS]
    if "verdict" not in out:
        known = all(r["final_share"] is not None and r["clip_last20"] is not None for r in fol)
        under = known and all(r["final_share"] < x_overlong_check.TRUNC_MIN and r["clip_last20"] < x_overlong_check.CLIP_MIN
                              for r in fol)
        out["verdict"] = UNDER if under else STILL if known else "不完整：训练日志缺失"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--logs", type=Path, help="$ATMOS_ROOT/logs, for the training-side truncation (X-29 section 3)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.name, args.seeds = NAME, list(SEEDS)
    fam = Family(args)
    r = report(fam, args.logs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    pct = lambda d: "-" if d.get("mean") is None else f"{100 * d['mean']:.1f} [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}]"
    print(f"batch {r['batch']}; missing {r['missing'] or 'none'}")
    for side, part in r["sides"].items():
        for k, v in part.items():
            if k != "rates":
                print(f"{side:5s} {k:11s} FOL {pct(v['FOL'])}  F {pct(v['F'])}  FOL-F {pct(v['FOL-F'])}  s43 {pct(v['FOL_s43-F_s43'])}")
        for a, x in part["rates"].items():
            print(f"{side:5s} rates {a:3s} parse {x['parse_rate']}  all-zero {x['all_zero_rate']}")
    for lab, x in r["truncation"].items():
        print(f"{lab:26s} final truncated {x['final_truncated']}  clip last 20 {x['clip_last20']}")
    print("->", r["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
