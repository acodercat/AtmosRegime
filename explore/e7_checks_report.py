"""E7 checks 1-2 (docs/PREREG_E7.md, second addendum): position, selection vs length, BM25 retrieval — one pass.

    python -m explore.e7_checks_report [--experiments DIR]

Template outside accuracy per arm (none, 60 entries, hbL, hbP, one entry, BM25 top-1, top-3, given), inside accuracy,
the four contrasts with 95 % bootstrap intervals over templates, and the pre-registered outcome A / B / C / D.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.runs import load_runs
from explore.e7_report import B, pooled, share, template_cells

ARMS = (("none", ""), ("hb60", "hb:"), ("hbL", "hbL:"), ("hbP", "hbP:"), ("one", "hb1:"), ("bm1", "bm1:"), ("bm3", "bm3:"))
CONTRASTS = (("position: hbL - hb60", "hbL", "hb60"), ("SELECTION: hbP - hbL", "hbP", "hbL"),
             ("length: one - hbP", "one", "hbP"), ("retrieval: bm3 - one", "bm3", "one"))


def holds(d: float, lo: float) -> bool:
    return d >= 0.15 and lo > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    args = ap.parse_args()
    rng = random.Random(6)
    for cfg, reps in sorted(load_runs(args.experiments, "e7_checks").items()):
        rep = reps[0]
        c = {arm: template_cells(rep, pre) for arm, pre in ARMS}
        names = sorted(set.intersection(*(set(v) for v in c.values())))
        acc = {a: share(*pooled(c[a], names, "outside")) for a, _ in ARMS}
        given = share(*pooled(c["none"], names, "given"))
        print(f"\n== {cfg}  ({len(names)} templates)")
        print("  outside acc  " + "  ".join(f"{a} {acc[a]:.0%}" for a, _ in ARMS) + f"  given {given:.0%}")
        print("  inside acc   " + "  ".join(f"{a} {share(*pooled(c[a], names, 'inside')):.0%}" for a, _ in ARMS))
        res = {}
        for label, x, y in CONTRASTS:
            diffs = []
            for _ in range(B):
                s = [rng.choice(names) for _ in names]
                diffs.append(share(*pooled(c[x], s, "outside")) - share(*pooled(c[y], s, "outside")))
            diffs.sort()
            d, lo, hi = acc[x] - acc[y], diffs[int(.025 * B)], diffs[int(.975 * B)]
            res[label.split(":")[0].lower()] = (d, lo, hi)
            print(f"  {label:24s} {d:+.1%} [{lo:+.1%}, {hi:+.1%}]")
        sel, length = res["selection"], res["length"]
        retrieval_closes = acc["bm3"] >= acc["one"] - 0.10
        if holds(sel[0], sel[1]) and not retrieval_closes:
            out = "A: selection holds, retrieval does not close it -> train with the handbook, reward picking the entry"
        elif holds(sel[0], sel[1]) and retrieval_closes:
            out = "B: selection holds, retrieval closes it -> retrieval does the selection; train application"
        elif sel[0] < 0.15 and holds(length[0], length[1]):
            out = "C: selection does not hold, length does -> long-prompt problem; no selection training"
        else:
            out = "D: other combination -> report as is, decide with the user"
        print(f"  -> {out}")
        print("  per template (outside right): " + " / ".join(a for a, _ in ARMS) + " / given")
        for n in names:
            print(f"    {n:34s} " + " ".join(f"{c[a][n]['outside'][0]:2d}" for a, _ in ARMS) + f"  {c['none'][n]['given'][0]:2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
