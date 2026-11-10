"""E7 addendum (docs/PREREG_E7.md, 2026-10-04): both controls with every arm from the same pass.

    python -m explore.e7_addendum_report [--experiments DIR]

E7s (`e7_single`): template outside accuracy and recognition under none / 60-entry handbook (`hb:`) / the one
relevant entry (`hb1:`) / given, inside accuracy per arm, a bootstrap 95 % interval over templates for one entry
minus none, and the pre-registered reading.
E7p (`e7_handbook_traps_paired`): textbook trap right | parent right per family under none and handbook, the
same rep, parent accuracy, a bootstrap 95 % interval over traps for the regime:recognise difference, and the
verdict from the PREREG_E7 table (textbook part).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from atmoscoder_rl.datasets import load_traps
from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.runs import load_numbered_reps, load_runs
from explore.e7_report import B, pooled, share, template_cells, trap_cells

ARMS = (("none", ""), ("hb60", "hb:"), ("hb1", "hb1:"))


def single(experiments: Path) -> None:
    runs = load_runs(experiments, "e7_single")
    rng = random.Random(2)
    print("E7s TEMPLATES (one pass; none / 60 entries / one entry / given)")
    for cfg in sorted(runs):
        rep = runs[cfg][0]
        c = {arm: template_cells(rep, pre) for arm, pre in ARMS}
        names = sorted(set.intersection(*(set(v) for v in c.values())))
        out = {arm: pooled(c[arm], names, "outside") for arm, _ in ARMS}
        rec = {arm: pooled(c[arm], names, "recog") for arm, _ in ARMS}
        ins = {arm: pooled(c[arm], names, "inside") for arm, _ in ARMS}
        g = pooled(c["none"], names, "given")
        diffs = []
        for _ in range(B):
            s = [rng.choice(names) for _ in names]
            diffs.append(share(*pooled(c["hb1"], s, "outside")) - share(*pooled(c["none"], s, "outside")))
        diffs.sort()
        o = {k: share(*v) for k, v in out.items()}; gv = share(*g)
        if o["hb1"] >= gv - 0.15 and o["hb1"] - o["hb60"] >= 0.15:
            reading = "SELECTION / LONG PROMPT: one entry works, 60 do not"
        elif abs(o["hb1"] - o["hb60"]) < 0.10 and max(o["hb1"], o["hb60"]) < gv - 0.15:
            reading = "CANNOT APPLY THE GENERIC ENTRY: one entry no better than 60, both far below given"
        else:
            reading = "other (report as is)"
        print(f"\n== {cfg}  ({len(names)} templates)")
        print("  outside acc  " + "  ".join(f"{a} {out[a][0]}/{out[a][1]}={o[a]:.0%}" for a, _ in ARMS)
              + f"  given {g[0]}/{g[1]}={gv:.0%}")
        print(f"  one entry - none {o['hb1'] - o['none']:+.1%} [{diffs[int(.025 * B)]:+.1%}, {diffs[int(.975 * B)]:+.1%}]")
        print("  recognition  " + "  ".join(f"{a} {rec[a][0]}/{rec[a][1]}={share(*rec[a]):.0%}" for a, _ in ARMS))
        print("  inside acc   " + "  ".join(f"{a} {share(*ins[a]):.0%}" for a, _ in ARMS))
        print(f"  -> {reading}")
        print("  per template (outside right: none / 60 / one / given)")
        for n in names:
            f = lambda arm, k="outside": f"{c[arm][n][k][0]:2d}/{c[arm][n][k][1]:<2d}"
            print(f"    {n:34s} {f('none')} {f('hb60')} {f('hb1')} {f('none', 'given')}")


def paired(experiments: Path) -> None:
    traps = {t["id"]: t for t in load_traps()}
    runs = load_numbered_reps(experiments, "e7_handbook_traps_paired")
    rng = random.Random(3)
    print("\nE7p TEXTBOOK TRAPS (none and handbook in the same pass; trap right | parent right)")
    for cfg in sorted(runs):
        reps = runs[cfg]
        none_reps = {i: {k: v for k, v in rep.items() if not k.startswith("hb:")} for i, rep in reps.items()}
        hb_reps = {i: {k: v for k, v in rep.items() if k.startswith("hb:")} for i, rep in reps.items()}
        cells = trap_cells(none_reps, hb_reps, traps)
        fams: dict[str, list] = {}
        for (f, tid) in cells:
            fams.setdefault(f, []).append(tid)
        par = {a: [sum(cells[k][a][2] for k in cells), sum(cells[k][a][3] for k in cells)] for a in ("none", "hb")}
        print(f"\n== {cfg}  (reps {sorted(reps)})")
        print(f"  parent acc   none {share(*par['none']):.0%}  handbook {share(*par['hb']):.0%}"
              f"  ({share(*par['hb']) - share(*par['none']):+.1%})")
        for f in sorted(fams):
            k = {a: [sum(cells[(f, t)][a][j] for t in fams[f]) for j in (0, 1)] for a in ("none", "hb")}
            d = share(*k["hb"]) - share(*k["none"])
            line = (f"  {f:22s} none {k['none'][0]:3d}/{k['none'][1]:<3d}={share(*k['none']):.0%}  "
                    f"handbook {k['hb'][0]:3d}/{k['hb'][1]:<3d}={share(*k['hb']):.0%}  ({d:+.1%})")
            if f == "regime:recognise":
                diffs = []
                for _ in range(B):
                    s = [rng.choice(fams[f]) for _ in fams[f]]
                    agg = {a: [sum(cells[(f, t)][a][j] for t in s) for j in (0, 1)] for a in ("none", "hb")}
                    diffs.append(share(*agg["hb"]) - share(*agg["none"]))
                diffs.sort()
                harm = share(*par["hb"]) - share(*par["none"]) < -0.05
                verdict = ("HARMFUL (parent accuracy down > 5 points)" if harm else
                           "fixes textbook part (>= +15)" if d >= 0.15 else
                           "partial (+5 to +15)" if d >= 0.05 else "no textbook effect (< +5)")
                line += f" [{diffs[int(.025 * B)]:+.1%}, {diffs[int(.975 * B)]:+.1%}]  -> {verdict}"
            print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    args = ap.parse_args()
    single(args.experiments)
    paired(args.experiments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
