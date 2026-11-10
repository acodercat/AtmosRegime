"""PREREG_X data: the RL pools per fold and arm, and the warm start W_f per fold.

    # RL pool (needs pyarrow: run in the verl venv). DATA dir gets train.parquet + the standard validation parquets.
    python -m explore.x_data export --fold F1 --arm A --kc experiments/_inputs/kc --out-dir $ATMOS_ROOT/data-x/F1-A
    # warm start, step 1: sample the fold's given rows from a served base model (8 per row, T = 1, seed 20261012)
    python -m explore.x_data warm-sample --fold F1 --kc ... --base-url http://127.0.0.1:8241/v1 --model NAME --out samples_F1.jsonl
    # warm start, step 2: verified programs -> SFT rows (+ RAFT background, <= 2 per template row), parquet for train/sft_qwen3_8b.sh
    python -m explore.x_data warm-build --fold F1 --kc ... --samples samples_F1.jsonl --raft $ATMOS_ROOT/data/sft_raft.parquet --out $ATMOS_ROOT/data/sft_x-F1.parquet

Arms (PREREG_X section 3, amendment X-1): A and D carry the standard text and the answer / counterfactual reward (as the
pilot's A / D); F carries the judgement instruction on inside / outside rows with answer_weight 0.8 (answer + 0.25 x judgement).
The warm start follows pilot.warmstart_rows exactly (first two-sided-correct program per given row, hint references dropped,
comments kept, inside twin by moving the trigger default), over the fold's kc rows instead of the pilot split.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atmoscoder_rl import pilot
from atmoscoder_rl.engine import extract_code
from atmoscoder_rl.export_verl import _template_row, build_tables
from atmoscoder_rl.prompts import chat
from atmoscoder_rl.reward import grade

ANSWER_WEIGHT_F = 0.8
WARM_SEED = 20261012
PROBE_ROWS = 256      # amendment X-5: training-side probe, same template rows for every arm of a fold
RAFT_SEED = 20260930
VAL_FILES = ("test_id", "test_core_only", "test_book", "probe_trap", "probe_trap_traps", "probe_scaffolding", "probe_cross_domain")


def fold_rows(kc: Path, fold: str) -> list[dict]:
    rows = json.loads((kc / f"train_{fold}.json").read_text())
    assert all(r["cluster_fold"] == fold and r["purpose"] == "train" for r in rows)
    return rows


def export(args) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = fold_rows(args.kc, args.fold)
    by_id = {r["id"]: r for r in rows}
    standard = build_tables()["train"]
    assert len(standard) == 1140, len(standard)
    template_rows = []
    for n, item in enumerate(rows, len(standard)):
        twin = by_id[item["pair"] + "_" + pilot.TWIN_OF[item["variant"]]] if args.arm == "D" else None
        row = _template_row(item, "train", n, judge=args.arm == "F", twin=twin)
        if args.arm == "F" and item["variant"] in pilot.JUDGE_LABEL:
            row["reward_model"]["ground_truth"] = pilot.encode_truth(
                item["sub_answers"], pilot.JUDGE_LABEL[item["variant"]], answer_weight=ANSWER_WEIGHT_F)
        template_rows.append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(standard + template_rows), args.out_dir / "train.parquet")
    eligible = sorted(r["extra_info"]["id"] for r in template_rows if r["extra_info"]["variant"] in ("inside", "outside"))
    chosen = set(random.Random(f"x-probe:{args.fold}").sample(eligible, PROBE_ROWS))
    probe = [{**r, "data_source": "atmoscoder/train_probe"} for r in template_rows if r["extra_info"]["id"] in chosen]
    pq.write_table(pa.Table.from_pylist(probe), args.out_dir / "train_probe.parquet")
    for name in VAL_FILES:
        src = args.data / f"{name}.parquet"
        if src.exists():
            shutil.copyfile(src, args.out_dir / f"{name}.parquet")
    print(f"{args.fold}-{args.arm}: {len(standard)} standard + {len(template_rows)} template rows -> {args.out_dir}")
    return 0


def warm_sample(args) -> int:
    import openai
    rows = [r for r in fold_rows(args.kc, args.fold) if r["variant"] == "given"]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} given rows to sample ({len(done)} done)", flush=True)
    client = openai.OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=3600.0, max_retries=0)
    extra = {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    if os.environ.get("ATMOS_NO_CHAT_TEMPLATE_KWARGS") == "1":   # Mistral-tokenizer servers reject chat_template_kwargs (Ministral 3; D-5 sent none)
        extra.pop("chat_template_kwargs")

    def one(row):
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(model=args.model, messages=chat(row["problem"]), temperature=1.0, top_p=1.0,
                                                      max_tokens=3072, n=args.n, seed=WARM_SEED, extra_body=extra)
                return {"id": row["id"], "arm": "G", "completions": [c.message.content or "" for c in resp.choices]}
            except openai.OpenAIError as error:
                if attempt == 4:
                    return {"id": row["id"], "arm": "G", "completions": [], "error": str(error)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh, ThreadPoolExecutor(8) as pool:
        for i, out in enumerate(pool.map(one, todo), 1):
            fh.write(json.dumps(out) + "\n"); fh.flush()
            if i % 50 == 0:
                print(i, flush=True)
    return 0


def warm_rows(rows: list[dict], samples: Path) -> tuple[list[dict], dict[str, int]]:
    """pilot.warmstart_rows over kc rows (comments kept)."""
    items = {r["id"]: r for r in rows}
    out, counts = [], defaultdict(int)
    for line in samples.open():
        rec = json.loads(line)
        given = items[rec["id"]]
        inside, outside = items[given["pair"] + "_inside"], items[given["pair"] + "_outside"]
        own, other = pilot.twin_for(given, inside)["trigger"]
        for completion in rec["completions"]:
            code = extract_code(completion or "")
            if not code.strip() or pilot.HINT_REFERENCE.search(code) \
                    or pilot.grade_cf(pilot.fence(code), given["sub_answers"], pilot.twin_for(given, inside))["score"] != 1.0:
                continue
            hit = pilot.trigger_parameter(code, own, other)
            if not hit:
                continue
            inside_code = pilot.set_default_text(code, hit[0], hit[1])
            if inside_code is None or not grade(pilot.fence(code), outside["sub_answers"]).passed \
                    or not grade(pilot.fence(inside_code), inside["sub_answers"]).passed:
                continue
            for item, prog in ((outside, code), (inside, inside_code)):
                out.append({"messages": chat(item["problem"]) + [{"role": "assistant", "content": pilot.fence(prog)}],
                            "id": item["id"], "kind": "warmstart", "template": item["template"]})
            counts[given["template"]] += 1
            break
    return out, dict(counts)


def warm_build(args) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = fold_rows(args.kc, args.fold)
    sft, counts = warm_rows(rows, args.samples)
    raft = pq.read_table(args.raft).to_pylist()
    k = min(int(len(sft) * args.raft_ratio), len(raft))
    extra = [{"messages": r["messages"], "id": r.get("id", ""), "kind": "raft", "template": ""}
             for r in random.Random(RAFT_SEED).sample(raft, k)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sft + extra), args.out)
    templates = sorted({r["template"] for r in rows})
    missing = [t for t in templates if t not in counts]
    summary = {"fold": args.fold, "templates": len(templates), "pairs_per_template": counts, "templates_without_pair": missing,
               "template_rows": len(sft), "raft_rows": len(extra), "raft_ratio": args.raft_ratio, "out": str(args.out)}
    print(json.dumps(summary, indent=1))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--fold", required=True, choices=("F1", "F2")); e.add_argument("--arm", required=True, choices=("A", "D", "F"))
    e.add_argument("--kc", type=Path, required=True); e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--data", type=Path, required=True, help="the standard export dir holding the validation parquets"); e.set_defaults(fn=export)
    s = sub.add_parser("warm-sample"); s.add_argument("--fold", required=True); s.add_argument("--kc", type=Path, required=True)
    s.add_argument("--base-url", required=True); s.add_argument("--model", required=True); s.add_argument("--n", type=int, default=8)
    s.add_argument("--out", type=Path, required=True); s.set_defaults(fn=warm_sample)
    b = sub.add_parser("warm-build"); b.add_argument("--fold", required=True); b.add_argument("--kc", type=Path, required=True)
    b.add_argument("--samples", type=Path, required=True); b.add_argument("--raft", type=Path, required=True)
    b.add_argument("--raft-ratio", type=float, default=2.0); b.add_argument("--out", type=Path, required=True); b.set_defaults(fn=warm_build)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
