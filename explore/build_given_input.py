"""Input for the replacement-given versions (docs/PREREG_GIVEN_VERSIONS.md): the 46 regime traps,
their replacement-given versions (`data/authored/regime_given_versions.json`) and their parents,
in one file, so all three are answered under one server, one harness and one sampling seed.

    python -m explore.build_given_input --out <path>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl.datasets import index_by_id, load_core, split_ids
from atmoscoder_rl.paths import DATA_DIR
from atmoscoder_rl.traps import arm_ids, trap_index

GIVEN_FILE = DATA_DIR / "authored" / "regime_given_versions.json"


def build() -> list[dict]:
    index = trap_index()
    trap_ids = sorted(arm_ids("regime", index, set(split_ids("train"))))
    given = [g for g in json.loads(GIVEN_FILE.read_text()) if g["trap_id"] in set(trap_ids)]
    assert {g["trap_id"] for g in given} == set(trap_ids), "a regime trap has no given version"
    core = index_by_id(load_core())
    parents = sorted({index[i][0]["parent_id"] for i in trap_ids})
    return [index[i][0] for i in trap_ids] + given + [core[p] for p in parents]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
