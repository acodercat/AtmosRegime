"""Parent-vs-trap gap for every trap family, on the stored API-model runs.

    python -m explore.family_report [--experiments <dir>] [--originals]

Reads the traps of `data/traps.json`, the runs under `experiments/trap/` and the matching
repetition under `experiments/core_code/` (the parent's control). Per configuration and
family: paired parent/trap accuracy, discordant pairs, and shortcut capture among trap
failures. Then the item-level sign test pooled over the configurations that did not witness
the traps. The regime family is split by `traps.family`, the same predicates the arms use. Pairing is per repetition (run k of the traps against run k of
the parents), not a majority over repetitions. Exploratory (explore/GUIDE.md section 6).

Traps with a `magnitude_defect` are left out, as the arms leave them out. `--originals` keeps
only the 67 pre-expansion traps: every configuration measured those, while the expansion was
back-filled for three configurations only, so this is the comparison with equal coverage
across families — and the originals were not admitted through a reasoning-model witness.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from atmoscoder_rl.capture import shortcut_match
from atmoscoder_rl.datasets import index_by_id, load_traps
from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.runs import is_gradable, load_numbered_reps
from atmoscoder_rl.stats import exact_binomial_p
from atmoscoder_rl.traps import family, sound

#: The expansion's witnesses: a trap entered the set only if they solved it, so their gap
#: is biased toward zero and they are kept out of the pooled item test.
WITNESSES = {"gpt55-reasoning", "deepseek-v4-pro-reasoning"}
FAMILIES = ("regime:recognise", "regime:given", "regime:unjudged", "geometry_detail",
            "sign_direction", "formula_selection", "definition_confusion", "averaging_space")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--originals", action="store_true",
                        help="only the 67 pre-expansion traps (equal coverage across configurations)")
    args = parser.parse_args()

    traps = index_by_id(t for t in load_traps()
                        if sound(t) and not (args.originals and t.get("expansion")))
    trap_runs = load_numbered_reps(args.experiments, "trap")
    core_runs = load_numbered_reps(args.experiments, "core_code")
    item = defaultdict(lambda: [0, 0])          # trap id -> [parent-only, trap-only] over non-witnesses
    rows: dict[str, dict[str, list[int]]] = {}
    for config in sorted(trap_runs):
        agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # pairs, P, T, P>T, T>P, captured of trap fails
        parents = core_runs.get(config, {})
        for k in sorted(set(trap_runs[config]) & set(parents)):     # repetition k with repetition k
            trap_rep, core_rep = trap_runs[config][k], parents[k]
            for i, r in trap_rep.items():
                t = traps.get(i)
                if not t:
                    continue
                p = core_rep.get(t["parent_id"])
                if not (is_gradable(r) and is_gradable(p)):
                    continue
                g = agg[family(t)]
                parent_ok, trap_ok = bool(p["passed"]), bool(r["passed"])
                g[0] += 1; g[1] += parent_ok; g[2] += trap_ok
                g[3] += parent_ok and not trap_ok; g[4] += trap_ok and not parent_ok
                if not trap_ok:
                    g[5] += shortcut_match(r, t)
                if config not in WITNESSES:
                    item[i][0] += parent_ok and not trap_ok
                    item[i][1] += trap_ok and not parent_ok
        rows[config] = agg

    print(f"{'config':<30}{'family':<22}{'pairs':>6}{'par%':>6}{'trap%':>6}  P>T:T>P    p      capt/fails")
    for config, agg in rows.items():
        for f in FAMILIES:
            g = agg.get(f)
            if not g or not g[0]:
                continue
            n = g[0]
            label = config + (" [W]" if config in WITNESSES else "")
            print(f"{label:<30}{f:<22}{n:>6}{100 * g[1] / n:>6.0f}{100 * g[2] / n:>6.0f}  "
                  f"{g[3]:>4}:{g[4]:<4}{exact_binomial_p(g[3], g[4]):>8.2g}   {g[5]}/{n - g[2]}")
        print()
    print("ITEM level, non-witness configs pooled: items leaning parent-better vs trap-better")
    for f in FAMILIES:
        ids = [i for i in item if family(traps[i]) == f]
        pos = sum(item[i][0] > item[i][1] for i in ids)
        neg = sum(item[i][0] < item[i][1] for i in ids)
        print(f"  {f:<22} items {len(ids):>4}  parent>trap {pos:>3}  trap>parent {neg:>3}  "
              f"tie {len(ids) - pos - neg:>3}  p={exact_binomial_p(pos, neg):.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
