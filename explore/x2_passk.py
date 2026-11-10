"""X2, retrospective part (docs/PROPOSAL.md section 5.3): base-model pass@k on the trained templates' out-of-range rows.

    # sample (resume-safe; one jsonl line per row with n graded samples)
    python -m explore.x2_passk sample --base-url http://127.0.0.1:8231/v1 --model NAME --input pilot_trained.json \
        --variants outside inside --n 64 --out samples_base.jsonl
    # per-template pass@k table against the pilot arms' accuracy on the same rows
    python -m explore.x2_passk report --samples base=samples_base.jsonl warm=samples_warm.jsonl \
        --arms A=.../pilot-A-s120.json D=.../pilot-D-s80.json ... --out results/x2/retro.json

Sampling: the pilot prompt (`atmoscoder_rl.prompts.chat`), thinking off, temperature 1.0, top_p 1.0, n independent
samples per row, max_tokens 8192, no self-repair; each sample graded once with `reward.grade` (5 % tolerance, strict units).
The sampled texts go to <out stem>.texts.jsonl (PREREG_X X-18), so the verdicts can be re-graded later.
pass@k is the unbiased estimator 1 - C(n-c, k) / C(n, k) averaged over a template's rows.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path
from statistics import fmean

import openai

from atmoscoder_rl.prompts import chat
from atmoscoder_rl.reward import grade
from atmoscoder_rl.sandbox import counts as sandbox_counts

KS = (1, 2, 4, 8, 16, 32, 64)
MAX_TOKENS = 8192


def sample(args) -> int:
    rows = [r for r in json.loads(args.input.read_text()) if r["variant"] in args.variants]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} rows to sample ({len(done)} done), n={args.n}", flush=True)
    client = openai.OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=3600.0, max_retries=0)
    extra = {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    if os.environ.get("ATMOS_NO_CHAT_TEMPLATE_KWARGS") == "1":   # Mistral-tokenizer servers reject chat_template_kwargs (Ministral 3; D-5 sent none)
        extra.pop("chat_template_kwargs")

    def one(row: dict) -> dict:
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(model=args.model, messages=chat(row["problem"]), temperature=args.temperature,
                                                      top_p=1.0, max_tokens=MAX_TOKENS, n=args.n, extra_body=extra)
                choices = resp.choices
                break
            except openai.OpenAIError as error:
                if attempt == 4:
                    return {"id": row["id"], "template": row["template"], "variant": row["variant"], "error": str(error)}
        texts = [c.message.content or "" for c in choices]
        passed = [grade(t, row["sub_answers"]).reward == 1.0 for t in texts]
        finish = [c.finish_reason for c in choices]
        return {"id": row["id"], "template": row["template"], "variant": row["variant"], "n": len(passed),
                "c": sum(passed), "passed": passed, "truncated": sum(f == "length" for f in finish), "texts": texts}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    texts_path = args.out.with_suffix(".texts.jsonl")
    with args.out.open("a") as fh, texts_path.open("a") as th, ThreadPoolExecutor(args.workers) as pool:
        for i, out in enumerate(pool.map(one, todo), 1):
            texts = out.pop("texts", None)
            if texts is not None:   # texts first: a row in `out` always has its texts
                th.write(json.dumps({"id": out["id"], "texts": texts}) + "\n"); th.flush()
            fh.write(json.dumps(out) + "\n"); fh.flush()
            print(f"[{i}/{len(todo)}] {out['id']} c={out.get('c')}/{out.get('n')}", flush=True)
    print(f"sandbox: {sandbox_counts()}", flush=True)
    return 0


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def arm_accuracy(path: Path) -> dict[tuple[str, str], float]:
    """Per (template, variant) accuracy of one stored evaluation run (temperature 0.3, self-repair)."""
    data = json.loads(path.read_text())["results"]
    recs = data if isinstance(data, list) else list(data.values())
    acc = defaultdict(list)
    for r in recs:
        if r.get("passed") is None:
            continue
        parts = r["id"].split("_")  # tpl_<seed>_<template...>_<nn>_<variant>
        variant = parts[-1]
        template = "_".join(parts[2:-2])
        acc[(template, variant)].append(float(r["passed"]))
    return {k: fmean(v) for k, v in acc.items()}


def report(args) -> int:
    out = {"pass_at_k": {}, "arms": {}, "ks": list(KS)}
    for spec in args.samples:
        name, path = spec.split("=", 1)
        per = defaultdict(lambda: defaultdict(list))
        for s in load_jsonl(Path(path)):
            if "error" in s:
                continue
            for k in KS:
                if k <= s["n"]:
                    per[(s["template"], s["variant"])][k].append(pass_at_k(s["n"], s["c"], k))
        out["pass_at_k"][name] = {f"{t}|{v}": {str(k): fmean(x) for k, x in d.items()} for (t, v), d in per.items()}
    for spec in args.arms or []:
        name, path = spec.split("=", 1)
        out["arms"][name] = {f"{t}|{v}": a for (t, v), a in arm_accuracy(Path(path)).items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    # console table: outside rows, pass@1/8/64 per sampled model, arms' accuracy
    keys = sorted({k for m in out["pass_at_k"].values() for k in m if k.endswith("|outside")})
    names = list(out["pass_at_k"])
    arms = list(out["arms"])
    print("template".ljust(28), *[f"{n}@1 @8 @64".ljust(20) for n in names], *[a.rjust(6) for a in arms])
    for key in keys:
        cells = []
        for n in names:
            d = out["pass_at_k"][n].get(key, {})
            cells.append(" ".join(f"{100*d.get(str(k), float('nan')):4.0f}" for k in (1, 8, 64)).ljust(20))
        cells += [f"{100*out['arms'][a].get(key, float('nan')):6.0f}" for a in arms]
        print(key.split("|")[0].ljust(28), *cells)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sample")
    a.add_argument("--base-url", required=True); a.add_argument("--model", required=True)
    a.add_argument("--input", type=Path, required=True); a.add_argument("--variants", nargs="+", default=["outside"])
    a.add_argument("--n", type=int, default=64); a.add_argument("--temperature", type=float, default=1.0)
    a.add_argument("--workers", type=int, default=4); a.add_argument("--out", type=Path, required=True)
    a.set_defaults(fn=sample)
    b = sub.add_parser("report")
    b.add_argument("--samples", nargs="+", required=True); b.add_argument("--arms", nargs="*")
    b.add_argument("--out", type=Path, required=True); b.set_defaults(fn=report)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
