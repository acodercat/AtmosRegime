"""D-5 input: textbook regime:recognise and geometry_detail traps that D did not use, plus their parents.

    python -m explore.build_d5_input --out <path>.json

`docs/PREREG_MAIN.md` amendment D-5. D used every sound trap whose parent is outside `train`
(`build_e1_input --families`); this takes the disjoint rest — sound traps of `traps.json` whose
parent is inside `train`, plus the sound regime twins — restricted to the two families of the
main contrast. Parents go into the same file so traps and parents share one server and one seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from atmoscoder_rl.datasets import index_by_id, load_core, load_regime_twins, load_traps, split_ids
from atmoscoder_rl.traps import family, sound

FAMILIES = ("regime:recognise", "geometry_detail")


def build() -> list[dict]:
    train = set(split_ids("train"))
    candidates = [t for t in load_traps() if t["parent_id"] in train] + list(load_regime_twins())
    traps = sorted((t for t in candidates if sound(t) and family(t) in FAMILIES), key=lambda t: t["id"])
    if len({t["id"] for t in traps}) != len(traps):
        raise SystemExit("duplicate trap id")
    core = index_by_id(load_core(include_unsplit=True))
    parents = [core[p] for p in sorted({t["parent_id"] for t in traps})]
    return traps + parents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = build()
    text = json.dumps(rows, ensure_ascii=False, indent=1) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    counts = Counter(family(r) for r in rows if "parent_id" in r)
    n_traps = sum(counts.values())
    print(f"wrote {args.out}: {n_traps} traps {dict(sorted(counts.items()))} + {len(rows) - n_traps} parents; "
          f"sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
