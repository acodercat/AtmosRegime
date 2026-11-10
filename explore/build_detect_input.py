"""D-3 detection-probe inputs (docs/PREREG_MAIN.md, revision D-3): the judgement sentence appended to

- templates: the inside / outside rows of the D template input (seed 20261008), label 1 / 0;
- textbook:  every parent (label 1), regime:recognise trap (label 0) and geometry_detail trap (no label).

    python -m explore.build_detect_input --input-dir experiments/_inputs/families_v4 --out-dir experiments/_inputs/d3_detect
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl import pilot
from atmoscoder_rl.traps import family

PREFIX = "judge:"
TEXTBOOK_FAMILIES = ("regime:recognise", "geometry_detail")


def judged(item: dict) -> dict:
    return {**item, "id": PREFIX + item["id"], "problem": pilot.judge_text(item["problem"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    tpl = json.loads((args.input_dir / "templates.json").read_text())
    tb = json.loads((args.input_dir / "textbook.json").read_text())
    tpl_rows = [judged(i) for i in tpl if i["variant"] in pilot.JUDGE_LABEL]
    tb_rows = [judged(i) for i in tb if "parent_id" not in i or family(i) in TEXTBOOK_FAMILIES]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("templates", tpl_rows), ("textbook", tb_rows)):
        (args.out_dir / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
        print(name, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
