"""Check the corpus `datasets.load_problems()` assembles, and report what it is made of.

    uv run python -m atmoscoder_rl.check_corpus

There is no built file to keep in step: the corpus is assembled in memory from `core.json`,
`unseen_book.json` and `traps.json` every time it is loaded (`datasets.CORPUS_SOURCES`), so a
record exists in exactly one place on disk. What used to be `build_corpus --apply` is
therefore not needed. What was worth keeping is this report; the invariants it used to
enforce now live in `datasets._assemble_corpus` and raise `CorpusError` on every load:

- unique ids across the three sources,
- every trap's `parent_id` is one of the 436 frozen core problems,
- at most one trap per parent — two would double-count a parent in any per-family rate,
- no source record already carries `role` or `frozen_split_member`.

Loading the corpus here therefore exercises them, but it is not the only thing that does.

Re-vendoring upstream is now one step: replace the file under `data/`, run this, record the
new sha256 in `data/PROVENANCE.md`. `splits --check` must still pass afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .datasets import CORPUS_SOURCES, CorpusError, load_problems, split_ids
from .paths import DATA_DIR


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def contamination(records: list[dict]) -> dict[str, int]:
    """How many traps have their parent inside the frozen `train` split (trained-on parents)."""
    train = set(split_ids("train"))
    traps = [r for r in records if r["role"] == "trap"]
    inside = sum(r["parent_id"] in train for r in traps)
    return {"traps": len(traps), "parent_in_train": inside, "parent_held_out": len(traps) - inside}


def main() -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    records = load_problems()          # raises CorpusError if an invariant is broken

    print(f"corpus: {len(records)} records, assembled from")
    total = 0
    for source in CORPUS_SOURCES:
        path = DATA_DIR / source.file
        count = len(json.loads(path.read_text(encoding="utf-8")))
        total += count
        print(f"  {source.file:18s} {count:4d}  role={source.role:<5s} "
              f"frozen_split_member={str(source.frozen_split_member):<5s} sha256 {sha16(path)}")
    if total != len(records):
        raise CorpusError(f"sources hold {total} records but the corpus has {len(records)}")

    print(f"trap parents: {contamination(records)}")
    print("invariants hold: unique ids, every trap parent a frozen core problem, one trap per parent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
