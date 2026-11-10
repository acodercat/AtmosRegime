"""E1 input: every trap in the three analysis arms, plus each one's parent, in one file.

    python -m explore.build_e1_input --out <path>.json
    python -m explore.build_e1_input --out <path>.json --hint    # E2: same items, one added sentence
    python -m explore.build_e1_input --out <path>.json --families  # every trap family (PREREG Part 3)

Traps and parents go into the same file so that a model sees both under one server, one
harness version and one sampling seed -- the pairing the trap-vs-parent test relies on.
Exploratory (TODO.md, stage 1): arms are taken from the current split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl.datasets import index_by_id, load_core, load_traps, split_ids
from atmoscoder_rl.traps import ARMS, arm_ids, sound, trap_index


#: E2. Generic on purpose: it names no relation and no quantity, so it cannot hand over a
#: replacement. It is added to traps and parents alike, because a warning that costs the
#: parents would otherwise read as a gain on the traps.
HINT = ("Before answering, check whether each relation you intend to use is valid for the "
        "values given in this problem.")


def build(hint: bool = False, families: bool = False) -> list[dict]:
    """The arms' traps (E1), or with `families` every trap in `traps.json` whose parent is outside
    `train` and that has no `magnitude_defect` — all six families (docs/PREREG_TEMPLATE_SCREENING.md
    Part 3) — each with its parent."""
    index = trap_index()
    train = set(split_ids("train"))
    core = index_by_id(load_core())
    if families:
        trap_ids = sorted(t["id"] for t in load_traps() if t["parent_id"] not in train and sound(t))
    else:
        trap_ids = sorted({i for arm in ARMS for i in arm_ids(arm, index, train)})
    parent_ids = sorted({index[i][0]["parent_id"] for i in trap_ids})
    rows = [index[i][0] for i in trap_ids] + [core[p] for p in parent_ids]
    return [{**r, "problem": f"{HINT}\n\n{r['problem']}"} for r in rows] if hint else rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hint", action="store_true", help="E2: prepend the validity hint to every problem")
    parser.add_argument("--families", action="store_true",
                        help="every held-out, sound trap of traps.json (all families), not only the arms")
    args = parser.parse_args()
    rows = build(hint=args.hint, families=args.families)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_traps = sum(1 for r in rows if "parent_id" in r)
    print(f"wrote {args.out}: {n_traps} traps + {len(rows) - n_traps} parents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
