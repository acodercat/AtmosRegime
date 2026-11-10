"""Export the frozen splits as verl parquet datasets.

    uv run python -m atmoscoder_rl.export_verl --out-dir $ATMOS_ROOT/data [--targeted-regime]

One file per split, in the schema verl's own preprocessors produce (`data_source`,
`prompt` as chat messages, `ability`, `reward_model.ground_truth`, `extra_info`). The
prompt is the benchmark harness's, verbatim (`prompts.py`); `ground_truth` carries the
problem's `sub_answers` as JSON, which `verl_reward.compute_score` decodes.

What each split contains:

- **train** — every training parent *and* its five stored numeric variants: six verified
  instances per parent, 1,140 rows, each with a GT computed by the parent's faithful
  solver. This is the dataset until on-the-fly re-sampling lands; it already prevents a
  policy from matching a single set of numbers to an answer.
- **the held-out and probe splits** — parent text only, one row per problem, for verl's
  periodic validation. Traps are exported as their own file (`probe_trap_traps`) next to
  their parents (`probe_trap`), so the Trap Gap can be read off two validation curves.

`--targeted-regime` adds the regime traps whose parent is already a training parent to
`train`, and removes those same ids from `probe_trap_traps` so the run cannot validate on its
own training data. The frozen split is not touched: these parents are training parents
already. It implements a pre-registered run that was superseded and never made (the document
is retired; `docs/README.md`); without the flag the export is byte-identical to the frozen run's.

`--pilot-arm D` (docs/REWARD_V2.md) has arm A's rows and texts, but every template row's truth carries its
twin, so `verl_reward_regime.py` scores it with the counterfactual-execution reward.

`--pilot-arm A|B` builds an arm of the training pilot (docs/PILOT_DESIGN.md, `pilot.py`): `train` gains
the trained templates' rows (seed `pilot.TRAIN_SEED`, inside / outside / given), and two validation
files hold the held-out templates (seed `pilot.EVAL_SEED`): `probe_tpl_heldout` with the standard
prompt, `probe_tpl_heldout_judge` (inside / outside) with the judgement instruction and its label. In
arm B the trained inside / outside rows carry the instruction and the label; in arm A they do not. Both
arms train with `verl_reward_regime.py`, which on an unlabelled row is the answer reward exactly.

Needs pyarrow; run it inside the verl environment or with `uv run --with pyarrow`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .traps import targeted_regime_ids
from .datasets import (index_by_id, load_core, load_cross_domain, load_regime_twins, load_splits,
                       load_traps, train_families)
from . import pilot
from .prompts import chat

DATA_SOURCE = "atmoscoder"


def _row(problem: dict[str, Any], split: str, index: int, parent: dict[str, Any] | None = None,
         **extra: Any) -> dict[str, Any]:
    """One verl record. Variants carry no book/category/difficulty of their own — pass `parent`."""
    meta = parent or problem
    return {
        "data_source": f"{DATA_SOURCE}/{split}",
        "prompt": chat(problem["problem"]),
        "ability": "atmospheric_science",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(problem["sub_answers"])},
        "extra_info": {
            "split": split, "index": index, "id": problem["id"],
            "parent_id": problem.get("parent_id", problem["id"]),
            "book": meta.get("book", ""), "category": meta.get("category", ""),
            "difficulty": meta.get("difficulty", ""), **extra,
        },
    }


def _template_row(item: dict[str, Any], split: str, index: int, judge: bool,
                  twin: dict[str, Any] | None = None) -> dict[str, Any]:
    """A template item; with `judge`, the instruction is appended and the label travels in the truth;
    with `twin`, the truth carries the twin for the counterfactual reward (the text is unchanged)."""
    label = pilot.JUDGE_LABEL.get(item["variant"]) if judge else None
    text = pilot.judge_text(item["problem"]) if label is not None else item["problem"]
    row = _row({**item, "problem": text}, split, index, kind="template", template=item["template"],
               variant=item["variant"], pair=item["pair"])
    row["reward_model"]["ground_truth"] = pilot.encode_truth(
        item["sub_answers"], label, pilot.twin_for(item, twin) if twin is not None else None)
    return row


def pilot_tables(arm: str, split: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The pilot's template rows: extra `train` rows, and the two held-out validation files."""
    trained = pilot.template_items(split["trained_templates"], pilot.TRAIN_SEED)
    held = pilot.template_items(split["held_out_templates"], pilot.EVAL_SEED)
    judged = [i for i in held if i["variant"] in pilot.JUDGE_LABEL]
    by_id = {i["id"]: i for i in trained}

    def twin(i: dict[str, Any]) -> dict[str, Any] | None:
        """Arm D: every template row is scored against its twin (docs/REWARD_V2.md)."""
        if arm != "D":
            return None
        return by_id[i["id"].rsplit("_", 1)[0] + "_" + pilot.TWIN_OF[i["variant"]]]

    return {
        "train": [_template_row(i, "train", n, judge=arm == "B", twin=twin(i)) for n, i in enumerate(trained)],
        "probe_tpl_heldout": [_template_row(i, "probe_tpl_heldout", n, judge=False) for n, i in enumerate(held)],
        "probe_tpl_heldout_judge": [_template_row(i, "probe_tpl_heldout_judge", n, judge=True)
                                    for n, i in enumerate(judged)],
    }


def build_tables(targeted_regime: bool = False) -> dict[str, list[dict[str, Any]]]:
    splits = load_splits()["splits"]
    core = index_by_id(load_core())
    tables: dict[str, list[dict[str, Any]]] = {}

    train_rows: list[dict[str, Any]] = []
    for parent, variants in train_families():
        train_rows.append(_row(parent, "train", len(train_rows), kind="parent"))
        for variant in variants:
            train_rows.append(_row(variant, "train", len(train_rows), parent=parent,
                                   kind="numeric_variant"))
    trained = targeted_regime_ids() if targeted_regime else []
    if trained:
        by_id = index_by_id(load_traps() + load_regime_twins())
        for trap_id in trained:
            trap = by_id[trap_id]
            train_rows.append(_row(trap, "train", len(train_rows), parent=core[trap["parent_id"]],
                                   kind="regime_trap", trap_type=trap["trap_type"]))
    tables["train"] = train_rows

    for name in ("test_id", "test_core_only", "test_book", "probe_trap", "probe_scaffolding"):
        tables[name] = [_row(core[pid], name, i) for i, pid in enumerate(splits[name]["ids"])]

    # A trained trap must not also be a validation row.
    held_out = [t for t in load_traps() if t["id"] not in set(trained)]
    tables["probe_trap_traps"] = [_row(trap, "probe_trap_traps", i, trap_type=trap["trap_type"])
                                  for i, trap in enumerate(held_out)]
    tables["probe_cross_domain"] = [_row(p, "probe_cross_domain", i, domain=p["domain"])
                                    for i, p in enumerate(load_cross_domain())]
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for the parquet files")
    parser.add_argument("--targeted-regime", action="store_true",
                        help="train on the regime traps whose parents are already training parents"
                             " (a superseded, never-run pre-registration; docs/README.md)")
    parser.add_argument("--pilot-arm", choices=("A", "B", "D"),
                        help="an arm of the training pilot (docs/PILOT_DESIGN.md; D: docs/REWARD_V2.md)")
    parser.add_argument("--pilot-split", type=Path, default=pilot.SPLIT_PATH, help="the held-out split (pilot.py split)")
    args = parser.parse_args()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required: run inside the verl env, or `uv run --with pyarrow ...`")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(targeted_regime=args.targeted_regime)
    if args.pilot_arm:
        extra = pilot_tables(args.pilot_arm, pilot.load_split(args.pilot_split))
        offset = len(tables["train"])
        for row in extra.pop("train"):
            row["extra_info"]["index"] += offset
            tables["train"].append(row)
        tables.update(extra)
    for name, rows in tables.items():
        path = args.out_dir / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        print(f"  {len(rows):>5} rows  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
