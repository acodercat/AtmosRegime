"""Shortcut capture on E1: when a model misses a trap, did it give the parent method's answer?

    python -m explore.capture_report [--experiments <dir>] [--exp-id explore_e1]

The regime claim predicts a specific error, not just an error: the model applies the parent's
relation where it has failed, so a wrong trap answer should equal the trap's `shortcut_values`.
A capture is a failed record that reproduces the shortcut (`atmoscoder_rl.capture`, the one
definition every report uses). Exploratory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from atmoscoder_rl.traps import ARMS, arm_ids, trap_index
from atmoscoder_rl.capture import shortcut_match
from atmoscoder_rl.datasets import split_ids
from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.runs import is_gradable, load_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--exp-id", default="explore_e1")
    args = parser.parse_args()
    index = trap_index()
    train = set(split_ids("train"))
    arm_of = {i: a for a in ARMS for i in arm_ids(a, index, train)}
    runs = load_runs(args.experiments, args.exp_id)
    if not runs:
        print(f"no run files under {args.experiments / args.exp_id}")
        return 1
    print(f"{'config':<28}" + "".join(f"{a + ': wrong, of them shortcut':>34}" for a in ARMS))
    for config, reps in sorted(runs.items()):
        cells = []
        for arm in ARMS:
            wrong = cap = 0
            for rep in reps:
                for r in rep.values():
                    if arm_of.get(r["id"]) != arm or not is_gradable(r) or r["passed"] is not False:
                        continue
                    wrong += 1
                    cap += shortcut_match(r, index[r["id"]][0])
            cells.append(f"{wrong:>20} {cap:>5} ({100 * cap / wrong if wrong else 0:>4.0f}%)")
        print(f"{config:<28}" + "".join(f"{c:>34}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
