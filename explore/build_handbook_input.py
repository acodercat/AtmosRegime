"""E7 (reference-sheet control): prepend the handbook `explore/handbook.md` to every problem of an input.

    python -m explore.build_handbook_input --in <input>.json --out <output>.json

Each row is copied unchanged except two fields: `id` gets the prefix `hb:` (so a handbook row never
collides with, or resumes from, the bare row it was made from), and `problem` becomes

    <handbook>
    ---
    Use the reference notes above only where they apply.

    <original problem>

Rows that already carry help are skipped: `flag:` rows and template `given` rows (two kinds of help
together could not be told apart; the bare given rows already exist). Every other row — traps, parents,
template inside and outside — gets the same handbook, so any cost it has on problems where the standard
relation is right shows up in the same run. Which entry
covers which template or trap is in `explore/handbook_map.json`, for analysis only; nothing from it
goes into the prompt. Design and arms: docs/E7_HANDBOOK.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HANDBOOK = Path(__file__).with_name("handbook.md")
PREFIX = "hb:"
INSTRUCTION = "Use the reference notes above only where they apply."
SEPARATOR = "---"
FLAG_PREFIX = "flag:"


def handbook_text() -> str:
    return HANDBOOK.read_text(encoding="utf-8").strip()


def wrap(problem: str, handbook: str) -> str:
    return f"{handbook}\n\n{SEPARATOR}\n{INSTRUCTION}\n\n{problem}"


def build(rows: list[dict], handbook: str) -> list[dict]:
    out = []
    for row in rows:
        if row["id"].startswith(FLAG_PREFIX) or row.get("variant") == "given":
            continue
        if row["id"].startswith(PREFIX):
            raise ValueError(f"{row['id']} already carries the handbook prefix")
        out.append({**row, "id": PREFIX + row["id"], "problem": wrap(row["problem"], handbook)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inp", type=Path, required=True, help="input json: a list of rows with id, problem")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    handbook = handbook_text()
    rows = json.loads(args.inp.read_text(encoding="utf-8"))
    out = build(rows, handbook)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    entries = sum(1 for line in handbook.splitlines() if line.startswith("### "))
    print(f"handbook: {entries} entries, {len(handbook)} chars, ~{len(handbook) // 4} tokens (chars/4)")
    print(f"wrote {args.out}: {len(out)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
