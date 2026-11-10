"""Export the training split as an SFT dataset, in verl's multi-turn `messages` format.

    uv run python -m atmoscoder_rl.export_sft --out-dir $ATMOS_ROOT/data \
        --source gold

Why this exists: every result in `results/FINAL.md` comes from RL, and a reviewer will ask
whether supervised training on the same problems produces the same split — large gains on
canonical problems, none on the ones that need a relation recognised as inapplicable. Two
sources answer two different questions, so both are supported:

- **`gold`** — the certified `solve()` that ships with each problem. Free (it is part of the
  dataset) and it answers "does teaching the textbook solution do this too". But it changes
  *two* things at once relative to RL: the algorithm (imitation rather than policy gradient)
  **and** where the supervision comes from (a pipeline-authored solver rather than the
  model's own output), so a split here cannot be attributed to either alone.
- **`raft`** — the base model's own rollouts, kept only where they passed verification
  (written by `sample_raft.py`). Same supervision *content* as RL, so the only difference is
  the algorithm. This is the clean control; it costs a sampling pass.

The target string is `” ```python\\n{code}\\n``` ”` because that is what `engine.extract_code`
reads and therefore what the reward scores. Training on bare code would teach a format the
grader does not accept, and the two arms would stop being comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import train_families
from .prompts import chat


def _messages(problem_text: str, code: str) -> list[dict[str, str]]:
    """The prompt the harness sends, plus the fenced answer the grader expects."""
    return chat(problem_text) + [{"role": "assistant", "content": f"```python\n{code.strip()}\n```"}]


def gold_rows() -> list[dict]:
    """One row per training parent and per stored numeric variant: 190 + 950."""
    rows = []
    for parent, variants in sorted(train_families(), key=lambda family: family[0]["id"]):
        rows.append({"messages": _messages(parent["problem"], parent["code"]),
                     "id": parent["id"], "kind": "parent"})
        for variant in variants:
            rows.append({"messages": _messages(variant["problem"], variant["code"]),
                         "id": variant["id"], "kind": "numeric_variant"})
    return rows


def raft_rows(path: Path) -> list[dict]:
    """Verified base-model rollouts, as written by `sample_raft.py`.

    Each record is `{id, problem, completions: [str, ...]}` holding only completions that
    passed. One row per (problem, completion) so a problem with several accepted solutions
    contributes each of them, which is what rejection-sampling fine-tuning does.
    """
    rows = []
    for rec in json.loads(path.read_text()):
        for completion in rec["completions"]:
            rows.append({"messages": chat(rec["problem"]) + [{"role": "assistant", "content": completion}],
                         "id": rec["id"], "kind": "raft"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for the parquet file")
    parser.add_argument("--source", choices=("gold", "raft"), default="gold")
    parser.add_argument("--raft-input", type=Path,
                        help="verified rollouts from sample_raft.py (required for --source raft)")
    args = parser.parse_args()

    if args.source == "raft":
        if not args.raft_input:
            parser.error("--source raft needs --raft-input")
        rows = raft_rows(args.raft_input)
    else:
        rows = gold_rows()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required: run inside the verl env, or `uv run --with pyarrow ...`")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"sft_{args.source}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(f"{path}: {len(rows)} rows  {kinds}")
    print(f"  distinct problems: {len({r['id'] for r in rows})}")
    sample = rows[0]["messages"][-1]["content"]
    print(f"  target starts: {sample[:40]!r}")
    print(f"  target ends:   {sample[-20:]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
