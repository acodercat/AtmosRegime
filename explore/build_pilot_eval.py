"""Training pilot, evaluation inputs (docs/PILOT_DESIGN.md §4). Every model is measured on the same files.

    python -m explore.build_pilot_eval --out-dir $ATMOS_ROOT/data/explore

`pilot_eval.json` (standard prompt, 3 repetitions per model), one file so one server serves it all:
  - held-out templates, seed pilot.EVAL_SEED, inside / outside / given       (primary)
  - every sound trap outside `train` with its parent, all families          (secondary: the 398 of
    prereg_families, rebuilt by explore.build_e1_input --families)
  - test_id, test_core_only, test_book, probe_scaffolding (original and stripped), cross-domain (safety)
`pilot_trained.json` (1 repetition, post hoc): the trained templates at the evaluation seed — does
training on a mechanism teach its outside version (within-mechanism transfer)?
`pilot_flag.json`: every template's outside row (evaluation seed) with `pilot.FLAG_INSTRUCTION` — told the
standard treatment fails, not told what to use (explore.flag_report).
`pilot_flag_traps.json`: the 199 textbook traps with the flag sentence (explore.flag_report --traps).
`pilot_judge.json` (1 repetition): the held-out inside / outside rows with the judgement instruction;
the judgement is read back by re-running the stored solver (explore.pilot_report).

Group membership is recomputed from the loaders at analysis time, so the files carry no labels
beyond what the runner stores. The corpus text of a probe_scaffolding problem *is* its stripped
version; the original (with the textbook's scaffolding) shares the id and is stored as
`scaffold_original:<id>`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl import pilot
from atmoscoder_rl.datasets import index_by_id, load_core, load_cross_domain, load_scaffolding, split_ids
from explore.build_e1_input import build as build_families

SAFETY_SPLITS = ("test_id", "test_core_only", "test_book", "probe_scaffolding")
ORIGINAL_PREFIX = "scaffold_original:"


def eval_rows() -> list[dict]:
    split = pilot.load_split()
    rows = list(pilot.template_items(split["held_out_templates"], pilot.EVAL_SEED))
    rows += build_families(families=True)
    core = index_by_id(load_core())
    rows += [core[i] for s in SAFETY_SPLITS for i in split_ids(s)]
    scaffold = set(split_ids("probe_scaffolding"))
    original, _ = load_scaffolding()
    rows += [{**p, "id": ORIGINAL_PREFIX + p["id"]} for p in original if p["id"] in scaffold]
    rows += load_cross_domain()
    seen: dict[str, dict] = {}
    for r in rows:                       # a trap parent may also sit in a safety split: keep one copy
        if r["id"] in seen and seen[r["id"]]["problem"] != r["problem"]:
            raise ValueError(f"id {r['id']} has two different texts")
        seen.setdefault(r["id"], r)
    return list(seen.values())


def judge_rows() -> list[dict]:
    split = pilot.load_split()
    items = pilot.template_items(split["held_out_templates"], pilot.EVAL_SEED)
    return [{**i, "id": "judge:" + i["id"], "problem": pilot.judge_text(i["problem"])}
            for i in items if i["variant"] in pilot.JUDGE_LABEL]


FLAG_PREFIX = "flag:"


def flag_rows() -> list[dict]:
    """Outside rows of every template (held-out and trained, evaluation seed) with the flag sentence."""
    split = pilot.load_split()
    items = pilot.template_items(split["held_out_templates"] + split["trained_templates"], pilot.EVAL_SEED)
    return [{**i, "id": FLAG_PREFIX + i["id"], "problem": f"{i['problem']}\n\n{pilot.FLAG_INSTRUCTION}"}
            for i in items if i["variant"] == "outside"]


def flag_trap_rows() -> list[dict]:
    """The textbook traps of pilot_eval (every sound trap outside train) with the flag sentence."""
    traps = [r for r in build_families(families=True) if "parent_id" in r]
    return [{**t, "id": FLAG_PREFIX + t["id"], "problem": f"{t['problem']}\n\n{pilot.FLAG_INSTRUCTION}"} for t in traps]


def e0_rows() -> list[dict]:
    """E0 (PROPOSAL.md): every template at the evaluation seed — inside, outside, given — plus the flag rows,
    one file for an API model."""
    split = pilot.load_split()
    return pilot.template_items(split["held_out_templates"] + split["trained_templates"], pilot.EVAL_SEED) + flag_rows()


def trained_rows() -> list[dict]:
    """The trained templates at the evaluation seed: new numbers, mechanisms the arms trained on."""
    return pilot.template_items(pilot.load_split()["trained_templates"], pilot.EVAL_SEED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pilot_eval", eval_rows()), ("pilot_judge", judge_rows()), ("pilot_trained", trained_rows()),
                       ("pilot_flag", flag_rows()), ("pilot_flag_traps", flag_trap_rows()),
                       ("e0_templates", e0_rows())):
        path = args.out_dir / f"{name}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}: {len(rows)} problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
