"""PREREG_OPD §4.1 gate: the same response must get a different teacher log-probability under the given-twin prompt than under the
student's prompt — otherwise the privilege is not reaching the teacher. Served model = the teacher (Qwen3-8B) behind vLLM's
OpenAI-compatible completions endpoint (echo + logprobs give prompt log-probabilities).

    python train/opd/check_privilege.py --data $R/data-opd/F1-priv/train.parquet --model /path/to/shared/Qwen3-8B \
        --base-url http://127.0.0.1:8271/v1 --served x-teacher --warm $R/experiments/_x/qwen3-8b/warm_samples_F1.jsonl --n 5

For n privileged rows: student prompt and teacher prompt rendered with the tokenizer's chat template (add_generation_prompt), each
followed by one response — the base model's first stored warm-sample completion for that pair's given row (written with the
replacement relation stated); rows without one are skipped. Reports, per row, the summed
response log-prob under both prompts and their difference; PASS if every row differs by more than 1e-3 nats in total and the
median difference favours the given prompt (the replacement-using response should be more likely when the relation is stated).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
from openai import OpenAI
from transformers import AutoTokenizer


def response_logprob(client, served, text, prefix_len):
    r = client.completions.create(model=served, prompt=text, max_tokens=1, temperature=0, echo=True, logprobs=0)
    lp = r.choices[0].logprobs
    offs, vals = lp.text_offset, lp.token_logprobs
    return sum(v for o, v in zip(offs, vals) if v is not None and o >= prefix_len), sum(1 for o in offs if o >= prefix_len)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--warm", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    programs = {}
    for line in args.warm.read_text().splitlines():
        rec = json.loads(line)       # {"id": "<pair>_given", "arm": "G", "completions": [8 texts]} (explore/x_data.py warm-sample)
        if rec.get("completions"):
            programs[rec["id"]] = rec["completions"][0]
    df = pd.read_parquet(args.data)
    rows = []
    for i, e in enumerate(df["extra_info"]):
        if e.get("teacher_prompt") is None:
            continue
        given_id = e["pair"] + "_given"
        if given_id not in programs or not programs[given_id]:
            continue
        rows.append((i, e, programs[given_id]))
        if len(rows) >= args.n:
            break
    if not rows:
        print("no privileged row with a stored given-row program"); return 1
    render = lambda m: tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    results = []
    for i, e, prog in rows:
        sp, tp = render(df.iloc[i]["prompt"]), render(e["teacher_prompt"])
        ls, ns = response_logprob(client, args.served, sp + prog, len(sp))
        lt, nt = response_logprob(client, args.served, tp + prog, len(tp))
        results.append({"id": e["id"], "student_logprob": ls, "teacher_logprob": lt, "diff_given_minus_student": lt - ls, "response_tokens": [ns, nt]})
        print(f"{e['id']:60s} student {ls:9.2f} given {lt:9.2f} diff {lt - ls:+8.3f} tokens {ns}/{nt}")
    diffs = [r["diff_given_minus_student"] for r in results]
    verdict = all(abs(d) > 1e-3 for d in diffs) and statistics.median(diffs) > 0
    print("median diff", round(statistics.median(diffs), 3), "| PRIVILEGE CHECK", "PASS" if verdict else "FAIL")
    if args.out:
        args.out.write_text(json.dumps({"rows": results, "median_diff": statistics.median(diffs), "pass": verdict}, indent=1) + "\n")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
