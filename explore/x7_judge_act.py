"""X7 judge-then-act (docs/PREREG_X.md amendment X-7, section 4) and its report.

    python -m explore.x7_judge_act build --atmos <families_v4>/templates.json --x5 <x5>/templates.json --out x7_items.json
    python -m explore.x7_judge_act run --input x7_items.json --base-url URL --model NAME [--api-key-env VAR] [--thinking-off | --extra-body JSON] --out x7_<cfg>.jsonl
    python -m explore.x7_judge_act report --input x7_items.json --runs <cfg>=x7_<cfg>.jsonl ... --k2 k2_table.json --out x7_report.json

Turn 1: the code-protocol prompt for the outside problem, followed by JUDGE_Q (answer yes / no only). Turn 2, sharing turn 1:
C (neutral) "Now solve the problem." ; T (act on own judgement) when turn 1 said no. Graded once with reward.grade (5 %, strict
units), no self-repair. When turn 1 says yes (or is unparsable) T is C by construction and only C is generated.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.prompts import chat
from atmoscoder_rl.reward import grade

PER_TEMPLATE = 5
JUDGE_Q = ("\n\nBefore solving: does the standard textbook treatment of this problem remain valid for the values given here? "
           "Answer only \"yes\" or \"no\", followed by one sentence. Do not solve the problem yet.")
TURN_C = "Now solve the problem, following the instructions above."
TURN_T = ("You judged that the standard textbook treatment is not valid for these values. "
          "Now solve the problem accordingly, following the instructions above.")
DRAWS, SEED = 10_000, "x7:20261012"

CLASSIFY = """Below is the first reply a model gave when asked, before solving a physics problem, whether "the standard textbook
treatment of this problem remains valid for the values given here". Decide what the MODEL ITSELF judged, in its own words
(whatever it means by "standard"; ignore any code or solution that follows).

REPLY:
<<<
{reply}
>>>

valid     - it judged that the standard treatment remains valid / applies for these values
not_valid - it judged that the standard treatment does not remain valid / must be replaced or corrected for these values
unclear   - it gives no judgement, or contradicts itself

Reply with JSON only: {{"judgement": "valid" | "not_valid" | "unclear"}}"""


def build(args) -> int:
    rows = []
    for source, path in (("atmos", args.atmos), ("x5", args.x5)):
        seen = defaultdict(int)
        for it in json.loads(path.read_text()):
            if it["variant"] != "outside" or seen[it["template"]] >= PER_TEMPLATE:
                continue
            seen[it["template"]] += 1
            rows.append({"id": "x7:" + it["id"], "source": source, "template": it["template"],
                         "problem": it["problem"], "sub_answers": it["sub_answers"]})
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(len(rows), "rows")
    return 0


def _judgement(text: str) -> str | None:
    m = re.search(r"\b(yes|no)\b", (text or "").strip().lower()[:200])
    return m.group(1) if m else None


def run(args) -> int:
    import openai
    rows = json.loads(args.input.read_text())
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines() if l.strip()}
    key = os.environ[args.api_key_env] if args.api_key_env else "EMPTY"
    client = openai.OpenAI(api_key=key, base_url=args.base_url, timeout=1800.0, max_retries=3)
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.thinking_off else {}
    if args.extra_body:
        extra.update(json.loads(args.extra_body))

    def call(messages, max_tokens):
        for attempt in range(5):
            try:
                r = client.chat.completions.create(model=args.model, messages=messages, temperature=0.3,
                                                   max_tokens=max_tokens, extra_body=extra or None)
                return r.choices[0].message.content or ""
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    return None

    def one(row):
        msgs = chat(row["problem"] + JUDGE_Q)
        first = call(msgs, 512)
        if first is None:
            return {"id": row["id"], "error": "turn 1 failed"}
        j = _judgement(first)
        base = msgs + [{"role": "assistant", "content": first}]
        c_text = call(base + [{"role": "user", "content": TURN_C}], 8192)
        t_text = call(base + [{"role": "user", "content": TURN_T}], 8192) if j == "no" else c_text
        if c_text is None or t_text is None:
            return {"id": row["id"], "error": "turn 2 failed"}
        return {"id": row["id"], "judgement": j, "turn1": first,
                "c_pass": grade(c_text, row["sub_answers"]).reward == 1.0,
                "t_pass": grade(t_text, row["sub_answers"]).reward == 1.0, "c_text": c_text, "t_text": t_text}
    with args.out.open("a") as fh, ThreadPoolExecutor(args.workers) as pool:
        for n, out in enumerate(pool.map(one, [r for r in rows if r["id"] not in done]), 1):
            fh.write(json.dumps(out, ensure_ascii=False) + "\n"); fh.flush()
            if n % 25 == 0:
                print(n, flush=True)
    return 0


def reclassify(args) -> int:
    """X-9: semantic judgement of every turn-1 reply (judgement2: valid / not_valid / unclear), written back in place."""
    from atmoscoder_rl.evaluation.client import ModelError, build_model
    recs = [json.loads(l) for l in args.run.read_text().splitlines() if l.strip()]
    judge = build_model("judge-oneapi")

    def one(r):
        if "judgement2" in r or "error" in r:
            return r
        for _ in range(3):
            try:
                text = judge.generate(CLASSIFY.format(reply=(r.get("turn1") or "")[:3000])).content
            except ModelError:
                break
            m = re.search(r"\{.*\}", text or "", re.S)
            try:
                j = json.loads(m.group(0)).get("judgement") if m else None
            except json.JSONDecodeError:
                j = None
            if j in ("valid", "not_valid", "unclear"):
                return {**r, "judgement2": j}
        return {**r, "judgement2": None}
    with ThreadPoolExecutor(5) as pool:
        recs = list(pool.map(one, recs))
    args.run.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    from collections import Counter
    print(Counter(r.get("judgement2") for r in recs))
    return 0


def topup(args) -> int:
    """X-9: generate the T arm for turn-1 replies now classified not_valid that have none (same turn 1, same settings)."""
    import openai
    rows = {r["id"]: r for r in json.loads(args.input.read_text())}
    recs = [json.loads(l) for l in args.run.read_text().splitlines() if l.strip()]
    key = os.environ[args.api_key_env] if args.api_key_env else "EMPTY"
    client = openai.OpenAI(api_key=key, base_url=args.base_url, timeout=1800.0, max_retries=3)
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.thinking_off else {}
    if args.extra_body:
        extra.update(json.loads(args.extra_body))

    def one(r):
        if r.get("judgement2") != "not_valid" or r.get("judgement") == "no" or "error" in r:
            return r
        row = rows[r["id"]]
        msgs = chat(row["problem"] + JUDGE_Q) + [{"role": "assistant", "content": r["turn1"]}, {"role": "user", "content": TURN_T}]
        for attempt in range(5):
            try:
                t = client.chat.completions.create(model=args.model, messages=msgs, temperature=0.3, max_tokens=8192,
                                                   extra_body=extra or None).choices[0].message.content or ""
                return {**r, "t_text": t, "t_pass": grade(t, row["sub_answers"]).reward == 1.0, "t_topup": True}
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    return {**r, "t_topup_error": True}
    with ThreadPoolExecutor(args.workers) as pool:
        recs = list(pool.map(one, recs))
    args.run.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    print(sum(1 for r in recs if r.get("t_topup")), "T replies added")
    return 0


def _unit(template: str, source: str) -> str:
    if source == "x5":
        return "x5:" + template
    from atmoscoder_rl.coverage_clusters import unit_of
    return unit_of(template)


def _ci(values):
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(SEED)
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": len(values)}


def report(args) -> int:
    rows = {r["id"]: r for r in json.loads(args.input.read_text())}
    k2 = json.loads(args.k2.read_text()) if args.k2 else {}
    out = {}
    for spec in args.runs:
        cfg, path = spec.split("=", 1)
        recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        ok = [r for r in recs if "error" not in r]
        key = "judgement2" if any("judgement2" in r for r in ok) else "judgement"
        said_no = [r for r in ok if r.get(key) in ("no", "not_valid") and not r.get("t_topup_error")]
        tab = (k2.get(cfg) or {}).get("per_template", {})

        def diff_by_cluster(pick):
            per_t = defaultdict(list)
            for r in said_no:
                row = rows[r["id"]]
                if pick(row["template"]):
                    per_t[(row["template"], row["source"])].append(int(r["t_pass"]) - int(r["c_pass"]))
            per_u = defaultdict(list)
            for (t, s), v in per_t.items():
                per_u[_unit(t, s)].append(fmean(v))
            return _ci([fmean(v) for v in per_u.values()])
        part = {"rows": len(recs), "errors": len(recs) - len(ok), "said_no_rate": len(said_no) / len(ok) if ok else None,
                "c_accuracy": fmean(r["c_pass"] for r in ok),
                # T equals C wherever the (re)classified turn 1 did not say "not valid" (X-9: some old "no" are now valid)
                "t_accuracy": fmean(r["t_pass"] if r.get(key) in ("no", "not_valid") else r["c_pass"] for r in ok),
                "T_minus_C_said_no": diff_by_cluster(lambda t: True)}
        if tab:
            part["T_minus_C_K_high"] = diff_by_cluster(lambda t: t in tab and tab[t]["K2"] >= 2 / 3 - 1e-9)
            part["T_minus_C_K_low"] = diff_by_cluster(lambda t: t in tab and tab[t]["K2"] <= 1 / 3 + 1e-9)
            hi = part["T_minus_C_K_high"]
            part["reading"] = ("acting on its own judgement recovers answers where knowledge exists"
                               if hi["mean"] is not None and hi["mean"] >= .10 and hi["ci95"][0] > 0 else
                               "judgement does not unlock the knowledge" if hi["mean"] is not None and hi["mean"] < .05 else "numbers only")
        f = lambda d: None if d["mean"] is None else (round(100 * d["mean"], 1), [round(100 * x, 1) for x in d["ci95"]], d["units"])
        print(cfg, "said no", round(100 * part["said_no_rate"], 1), "C", round(100 * part["c_accuracy"], 1), "T", round(100 * part["t_accuracy"], 1),
              "T-C|no", f(part["T_minus_C_said_no"]), "K-high", f(part.get("T_minus_C_K_high", {"mean": None})),
              "K-low", f(part.get("T_minus_C_K_low", {"mean": None})), part.get("reading"))
        out[cfg] = part
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--atmos", type=Path, required=True); b.add_argument("--x5", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True); b.set_defaults(fn=build)
    r = sub.add_parser("run"); r.add_argument("--input", type=Path, required=True); r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True); r.add_argument("--api-key-env"); r.add_argument("--thinking-off", action="store_true")
    r.add_argument("--extra-body"); r.add_argument("--workers", type=int, default=5); r.add_argument("--out", type=Path, required=True); r.set_defaults(fn=run)
    p = sub.add_parser("report"); p.add_argument("--input", type=Path, required=True); p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--k2", type=Path); p.add_argument("--out", type=Path, required=True); p.set_defaults(fn=report)
    c = sub.add_parser("reclassify"); c.add_argument("--run", type=Path, required=True); c.set_defaults(fn=reclassify)
    t = sub.add_parser("topup"); t.add_argument("--input", type=Path, required=True); t.add_argument("--run", type=Path, required=True)
    t.add_argument("--base-url", required=True); t.add_argument("--model", required=True); t.add_argument("--api-key-env")
    t.add_argument("--thinking-off", action="store_true"); t.add_argument("--extra-body"); t.add_argument("--workers", type=int, default=5)
    t.set_defaults(fn=topup)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
