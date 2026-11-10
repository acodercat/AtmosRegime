"""E3 knowledge probe: does the model know the parent's method fails, when asked directly?

    python -m explore.e3_probe --tag qwen3-8b.think-off --thinking off --out-dir <dir> [--reps 3]

For each regime trap, the same question is asked about the trap's text and its parent's text:
"for these conditions, is <the parent's own method> valid?" The parent should get VALID and
the trap INVALID. Asking one sentence of both removes a bias toward either answer.
Talks to the model served as "local" on :8077 (the server explore/e1_run.sh starts) with the
`local-e3-off` / `local-e3-on` sections of `models.toml`. Not the evaluation runner: that grades
numbers, and a parent's "express your answer in ..." line can make a model box a number that
happens to land on a numeric verdict code.
Resume-safe. Exploratory (TODO.md, stage 1); null approaches are dropped by the probe review.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atmoscoder_rl.traps import trap_index
from atmoscoder_rl.datasets import index_by_id, load_core
from atmoscoder_rl.evaluation.client import ChatModel, ModelError, build_model

PROBE_FILE = Path(__file__).with_name("e3_probes.json")
VERDICT = re.compile(r"VERDICT:\s*\**\s*(VALID|INVALID)\b", re.IGNORECASE)
QUESTION = ("{problem}\n\n---\nDo not solve the problem above. Answer only this question: for the "
            "conditions stated in the problem, is the following approach valid?\n\nApproach: {approach}.\n\n"
            "End your answer with a final line that is exactly VERDICT: VALID or VERDICT: INVALID.")
MODELS = {"off": "local-e3-off", "on": "local-e3-on"}      # models.toml sections


def items() -> list[dict]:
    index, core = trap_index(), index_by_id(load_core())
    out = []
    for p in json.loads(PROBE_FILE.read_text())["probes"]:
        if p["approach"] is None:
            continue
        trap = index[p["trap"]][0]
        for role, rec, expect in (("trap", trap, "INVALID"), ("parent", core[trap["parent_id"]], "VALID")):
            out.append({"trap": p["trap"], "role": role, "expect": expect,
                        "prompt": QUESTION.format(problem=rec["problem"], approach=p["approach"])})
    return out


def ask(model: ChatModel, prompt: str) -> dict:
    """One verdict. Transport failures are retried by the client; an unparsed verdict is a
    result, not a retry."""
    try:
        completion = model.generate(prompt)
    except ModelError as error:
        return {"verdict": None, "error": str(error)}
    found = VERDICT.findall(completion.content)
    return {"verdict": found[-1].upper() if found else None, "content": completion.content,
            "reasoning": completion.reasoning}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--thinking", choices=MODELS, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    path = args.out_dir / f"{args.tag}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    todo = [(f"{it['trap']}|{it['role']}|{r}", it) for it in items() for r in range(1, args.reps + 1)]
    todo = [(k, it) for k, it in todo if k not in done or done[k].get("error")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(MODELS[args.thinking])
    with ThreadPoolExecutor(args.workers) as ex:
        for (key, it), res in zip(todo, ex.map(lambda t: ask(model, t[1]["prompt"]), todo)):
            done[key] = {"trap": it["trap"], "role": it["role"], "expect": it["expect"], **res}
            path.write_text(json.dumps(done, ensure_ascii=False))
    right = sum(1 for v in done.values() if v.get("verdict") == v["expect"])
    parsed = sum(1 for v in done.values() if v.get("verdict"))
    print(f"{args.tag}: {len(done)} answers, {parsed} parsed, {right} as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
