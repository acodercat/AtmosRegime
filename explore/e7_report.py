"""E7 reference-handbook control (docs/PREREG_E7.md): the pre-registered measures and verdicts.

    python -m explore.e7_report [--experiments DIR]

Templates (E0 confirmation items, seed 20261002, one pass): per configuration, outside accuracy under none /
handbook / given, recognition (outside right | inside right, same pair and arm), inside accuracy, and the
recovery share (handbook - none) / (given - none) on outside accuracy, with a bootstrap 95 % interval over
templates for handbook - none. Qwen's none and given rows are in `e7_handbook`; deepseek's are in `e0_confirm`.
Textbook traps (three reps): per family, trap right | parent right under none (`pilot_eval`) and handbook
(`e7_handbook_traps`), same rep index; parent accuracy; bootstrap 95 % interval over traps for the
regime:recognise difference. Verdicts follow the table in the pre-registration.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from atmoscoder_rl.datasets import load_traps
from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.runs import is_gradable, load_numbered_reps, load_runs
from atmoscoder_rl.traps import family

SEED = 20261002
HB = "hb:"
B = 2000


def _right(r) -> bool:
    return bool(is_gradable(r) and r["passed"])


def template_cells(rep: dict, prefix: str) -> dict[str, dict[str, list[int]]]:
    """Per template: inside [right, n], outside [right, n], given [right, n], recog [right, n]."""
    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for rid, r in rep.items():
        if not rid.startswith(prefix + f"tpl_{SEED}_") or not is_gradable(r):
            continue
        stem, variant = rid[len(prefix):].rsplit("_", 1)
        name = stem.split("_", 2)[2].rsplit("_", 1)[0]
        c = out[name][variant]; c[1] += 1; c[0] += bool(r["passed"])
        if variant == "inside" and r["passed"]:
            o = rep.get(prefix + stem + "_outside")
            if is_gradable(o):
                c = out[name]["recog"]; c[1] += 1; c[0] += bool(o["passed"])
    return out


def pooled(cells: dict, names, key: str) -> tuple[int, int]:
    return sum(cells[n][key][0] for n in names), sum(cells[n][key][1] for n in names)


def share(k: int, n: int) -> float:
    return k / n if n else float("nan")


def templates(experiments: Path) -> None:
    e7 = load_runs(experiments, "e7_handbook")
    e0 = load_runs(experiments, "e0_confirm")
    rng = random.Random(0)
    print("TEMPLATES (seed 20261002, one pass)")
    for cfg in sorted(e7):
        hb_rep = e7[cfg][0]
        none_rep = hb_rep if cfg.startswith(("qwen", "rl-")) else (e0.get(cfg) or [{}])[0]
        none, hb = template_cells(none_rep, ""), template_cells(hb_rep, HB)
        names = sorted(set(hb) & set(none))
        o_none, o_hb, g = (pooled(none, names, "outside"), pooled(hb, names, "outside"), pooled(none, names, "given"))
        i_none, i_hb = pooled(none, names, "inside"), pooled(hb, names, "inside")
        r_none, r_hb = pooled(none, names, "recog"), pooled(hb, names, "recog")
        rec = (share(*o_hb) - share(*o_none)) / (share(*g) - share(*o_none)) if share(*g) > share(*o_none) else float("nan")
        diffs = []
        for _ in range(B):
            s = [rng.choice(names) for _ in names]
            diffs.append(share(*pooled(hb, s, "outside")) - share(*pooled(none, s, "outside")))
        diffs.sort()
        print(f"\n== {cfg}  ({len(names)} templates)")
        print(f"  outside acc  none {o_none[0]}/{o_none[1]}={share(*o_none):.0%}  handbook {o_hb[0]}/{o_hb[1]}={share(*o_hb):.0%}"
              f"  given {g[0]}/{g[1]}={share(*g):.0%}   handbook-none {share(*o_hb) - share(*o_none):+.1%}"
              f" [{diffs[int(.025 * B)]:+.1%}, {diffs[int(.975 * B)]:+.1%}]")
        print(f"  recovery share {rec:.0%}")
        print(f"  recognition  none {r_none[0]}/{r_none[1]}={share(*r_none):.0%}  handbook {r_hb[0]}/{r_hb[1]}={share(*r_hb):.0%}")
        print(f"  inside acc   none {share(*i_none):.0%}  handbook {share(*i_hb):.0%}  ({share(*i_hb) - share(*i_none):+.1%})")
        print("  per template (outside right: none / handbook / given; recog none -> handbook)")
        for n in names:
            f = lambda c, k: f"{c[n][k][0]:2d}/{c[n][k][1]:<2d}"
            print(f"    {n:34s} {f(none, 'outside')} {f(hb, 'outside')} {f(none, 'given')}   "
                  f"{none[n]['recog'][0]}/{none[n]['recog'][1]} -> {hb[n]['recog'][0]}/{hb[n]['recog'][1]}")


def trap_cells(none_reps: dict[int, dict], hb_reps: dict[int, dict], traps: dict) -> dict:
    """{(family, trap id): {'none': [rec, n, parent_right, parent_n], 'hb': [...]}} over paired reps."""
    out: dict = defaultdict(lambda: {"none": [0, 0, 0, 0], "hb": [0, 0, 0, 0]})
    for i in sorted(set(none_reps) & set(hb_reps)):
        for arm, rep, pre in (("none", none_reps[i], ""), ("hb", hb_reps[i], HB)):
            for rid, r in rep.items():
                if not rid.startswith(pre) or rid[len(pre):] not in traps:
                    continue
                t = traps[rid[len(pre):]]
                p = rep.get(pre + t["parent_id"])
                c = out[(family(t), t["id"])][arm]
                if is_gradable(p):
                    c[3] += 1; c[2] += bool(p["passed"])
                if _right(p) and is_gradable(r):
                    c[1] += 1; c[0] += bool(r["passed"])
    return out


def textbook(experiments: Path) -> None:
    traps = {t["id"]: t for t in load_traps()}
    none_all = load_numbered_reps(experiments, "pilot_eval")
    hb_all = load_numbered_reps(experiments, "e7_handbook_traps")
    rng = random.Random(1)
    print("\nTEXTBOOK TRAPS (trap right | parent right; reps paired by index)")
    for cfg in sorted(hb_all):
        hb_reps = {i: {k: v for k, v in rep.items() if k.startswith(HB)} for i, rep in hb_all[cfg].items()}
        ids = {k[len(HB):] for rep in hb_reps.values() for k in rep}
        none_reps = {i: {k: v for k, v in rep.items() if k in ids} for i, rep in none_all.get(cfg, {}).items()}
        cells = trap_cells(none_reps, hb_reps, traps)
        fams = defaultdict(list)
        for (f, tid) in cells:
            fams[f].append(tid)
        print(f"\n== {cfg}  (reps {sorted(set(none_reps) & set(hb_reps))})")
        par = {a: [sum(cells[k][a][2] for k in cells), sum(cells[k][a][3] for k in cells)] for a in ("none", "hb")}
        print(f"  parent acc   none {share(*par['none']):.0%}  handbook {share(*par['hb']):.0%}"
              f"  ({share(*par['hb']) - share(*par['none']):+.1%})")
        for f in sorted(fams):
            k = {a: [sum(cells[(f, t)][a][j] for t in fams[f]) for j in (0, 1)] for a in ("none", "hb")}
            line = (f"  {f:22s} none {k['none'][0]:3d}/{k['none'][1]:<3d}={share(*k['none']):.0%}  "
                    f"handbook {k['hb'][0]:3d}/{k['hb'][1]:<3d}={share(*k['hb']):.0%}  "
                    f"({share(*k['hb']) - share(*k['none']):+.1%})")
            if f == "regime:recognise":
                diffs = []
                for _ in range(B):
                    s = [rng.choice(fams[f]) for _ in fams[f]]
                    agg = {a: [sum(cells[(f, t)][a][j] for t in s) for j in (0, 1)] for a in ("none", "hb")}
                    diffs.append(share(*agg["hb"]) - share(*agg["none"]))
                diffs.sort()
                line += f" [{diffs[int(.025 * B)]:+.1%}, {diffs[int(.975 * B)]:+.1%}]"
            print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    args = ap.parse_args()
    templates(args.experiments)
    textbook(args.experiments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
