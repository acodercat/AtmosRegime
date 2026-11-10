"""E7 checks (docs/PREREG_E7.md, second addendum): position, selection vs length, and a retrieval baseline, one pass.

    python -m explore.build_e7_checks --templates --out <path>.json
    python -m explore.build_e7_checks --variants --out <path>.json

`--templates`: the 17 E0-confirmation templates (seed 20261002), inside and outside rows under every arm, plus the
bare given rows:
  (bare)  none                     hb:   the 60-entry handbook, original order
  hb1:    the relevant entry only  hbL:  the other 59 entries, then the relevant one last
  hbP:    59 filler entries (explore/handbook_filler.md, same format and length, no physics), then the relevant one
  bm1:    BM25 top-1 entry         bm3:  BM25 top-3 entries (explore.handbook_retrieval)
hbL vs hbP differ only in whether the other 59 entries are physics (same length, same position): the selection
effect. hbP vs hb1: the length effect. hb vs hbL: position.
`--variants`: E4b's final variants (`trap_variants_wl_final.json`) with their original traps and parents, bare, to
measure the test set's real precision (cluster by trap).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl import pilot
from atmoscoder_rl.build_templates import build
from atmoscoder_rl.datasets import index_by_id, load_core, load_traps
from explore.build_e7_input import NAMES, SEED, handbook_entries, handbook_text
from explore.build_handbook_input import INSTRUCTION, SEPARATOR, wrap
from explore.handbook_retrieval import relevant_titles, retrieve

FILLER = Path(__file__).with_name("handbook_filler.md")
VARIANTS = Path("experiments/_inputs/trap_variants_wl_final.json")


def _sheet(blocks: list[str]) -> str:
    return "# Reference notes\n\n" + "\n\n".join(blocks)


def _filler_blocks() -> list[str]:
    out, cur = [], []
    for line in FILLER.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            if cur:
                out.append("\n".join(cur).strip())
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        out.append("\n".join(cur).strip())
    if len(out) != 59:
        raise ValueError(f"filler has {len(out)} entries, expected 59")
    return out


def template_rows() -> list[dict]:
    entries = handbook_entries()
    tpl, _ = relevant_titles()
    full = handbook_text()
    filler = _filler_blocks()
    items = build(pilot.PAIRS, SEED, set(NAMES))
    rows = list(items)                                        # none, given (and inside/outside bare)
    for i in items:
        if i["variant"] not in ("inside", "outside"):
            continue
        rel = tpl[i["template"]]
        others = [b for t, b in entries.items() if t != rel]
        arms = {
            "hb:": full,
            "hb1:": _sheet([entries[rel]]),
            "hbL:": _sheet(others + [entries[rel]]),
            "hbP:": _sheet(filler + [entries[rel]]),
            "bm1:": _sheet(retrieve(i["problem"], 1)),
            "bm3:": _sheet(retrieve(i["problem"], 3)),
        }
        for pre, sheet in arms.items():
            rows.append({**i, "id": pre + i["id"], "problem": wrap(i["problem"], sheet)})
    return rows


def variant_rows() -> list[dict]:
    variants = json.loads(VARIANTS.read_text(encoding="utf-8"))
    traps = {t["id"]: t for t in load_traps()}
    core = index_by_id(load_core())
    ids = sorted({v["variant_of"] for v in variants})
    return variants + [traps[t] for t in ids] + [core[traps[t]["parent_id"]] for t in ids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--templates", action="store_true")
    g.add_argument("--variants", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = template_rows() if args.templates else variant_rows()
    seen = {}
    for r in rows:                                            # a parent may serve two traps: keep one copy
        if r["id"] in seen and seen[r["id"]]["problem"] != r["problem"]:
            raise ValueError(f"id {r['id']} has two texts")
        seen.setdefault(r["id"], r)
    rows = list(seen.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
