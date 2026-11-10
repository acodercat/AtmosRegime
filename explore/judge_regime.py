"""Classify HOW a solver approached a regime trap: kept the parent method, or accounted for the change.

    GATEWAY_A_API_KEY=... uv run --extra eval python -m explore.judge_regime --items items.json --out labels.json

The judge is the `judge` section of `models.toml` (deepseek-v4-flash with reasoning, T = 0).
On the training box, run it in the eval venv (README, "Model calls on the server").

A wrong answer can mean two different failures: the solver never noticed that the parent's
relation fails (A), or it noticed and could not produce the replacement (B). A numerical match
to the shortcut only catches the clean cases of A. The judge reads the solver's code with an
expert note on what changed, and is never told whether the answer was right. Validation: right
answers should come out B, numerical shortcut captures A. Resume-safe. Exploratory.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atmoscoder_rl.traps import trap_index
from atmoscoder_rl.evaluation.client import ChatModel, ModelError, build_model

TRIES = 3   # re-asks when the reply has no parsable label; failed calls are the client's to retry

PROMPT = """You are grading HOW a solution approached a physics problem, not whether its number is right.

PROBLEM:
{problem}

EXPERT NOTE (the solver did not see this): the standard textbook method for this kind of problem does not hold here. {shortcut} (Mechanism: {mechanism}.)

SOLVER'S CODE:
```
{code}
```

Classify the solver's approach:
A - applied the standard textbook method as if it still held; no step accounts for the change described in the expert note.
B - recognised that the standard method fails here and tried to account for it, even if the attempt is incomplete or numerically wrong.
C - neither: the code is broken, unrelated to the problem, or its approach cannot be determined.

Reply with JSON only: {{"label": "A" | "B" | "C", "evidence": "<the line(s) that decide it, at most 30 words>"}}"""


def ask(judge: ChatModel, prompt: str) -> dict:
    err = "no attempt"
    for _ in range(TRIES):
        try:
            text = judge.generate(prompt).content
        except ModelError as error:     # the gateway's intermittent 404s are retried by the client
            return {"label": None, "error": str(error)}
        match = re.search(r"\{.*\}", text, re.S)
        try:
            out = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            out = {}
        if out.get("label") in ("A", "B", "C"):
            return out
        err = f"unparsed: {text[:120]}"
    return {"label": None, "error": err}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    judge = build_model("judge")
    index = trap_index()
    items = json.loads(args.items.read_text())
    done = json.loads(args.out.read_text()) if args.out.exists() else {}
    key = lambda it: f"{it['config']}|{it['run']}|{it['id']}"
    todo = [it for it in items if not (done.get(key(it)) or {}).get("label")]

    def one(it):
        t = index[it["id"]][0]
        return it, ask(judge, PROMPT.format(problem=t["problem"], shortcut=t.get("shortcut") or "",
                                     mechanism=t.get("mechanism") or "see note", code=it["code"]))

    with ThreadPoolExecutor(args.workers) as ex:
        for n, (it, res) in enumerate(ex.map(one, todo), 1):
            done[key(it)] = {**{k: it[k] for k in ("config", "run", "id", "passed", "captured")}, **res}
            if n % 50 == 0 or n == len(todo):
                args.out.write_text(json.dumps(done, ensure_ascii=False))
                print(f"  {n}/{len(todo)}", flush=True)
    args.out.write_text(json.dumps(done, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
