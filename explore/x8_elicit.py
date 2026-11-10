"""X8 elicit-then-solve (docs/PREREG_X.md amendment X-20, section 2): does knowledge the model writes out itself recover
out-of-range failures?

    python -m explore.x8_elicit build --atmos <families_v4>/templates.json --x5 <x5>/templates.json --out x8_items.json
    python -m explore.x8_elicit run --input x8_items.json --base-url URL --model NAME [--api-key-env VAR] \
        [--thinking-off | --extra-body JSON] --out x8_<cfg>.jsonl
    python -m explore.x8_elicit report --input x8_items.json --runs <cfg>=x8_<cfg>.jsonl ... [--kany results/x6/report_kany.json] \
        --out results/x8/report.json

Arms on every row (outside and inside), same model and batch, temperature 0.3, max_tokens 8192, code protocol (`prompts.chat`,
as X7), no self-repair, graded once with `reward.grade`: C direct; H the problem + `pilot.FLAG_INSTRUCTION`, one turn; E two turns,
ELICIT_Q first (no solving), then TURN_SOLVE. Every reply is stored.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl import pilot
from atmoscoder_rl.prompts import chat
from atmoscoder_rl.reward import grade
from explore.x7_judge_act import _unit

PER_TEMPLATE = 5
ELICIT_Q = ("\n\nBefore solving, do not solve the problem yet. Write out: (1) the standard textbook relation used for this "
            "kind of problem; (2) the conditions under which it holds, and whether the values given here satisfy them; "
            "(3) if they do not, the relation (equation) that should be used instead.")
TURN_SOLVE = "Now solve the problem, following the instructions above and using your analysis."
ARMS = ("C", "H", "E")
DRAWS, SEED = 10_000, "x8:20261018"


def build(args) -> int:
    rows = []
    for source, path in (("atmos", args.atmos), ("x5", args.x5)):
        items = json.loads(path.read_text())
        by_id = {it["id"]: it for it in items}
        seen = defaultdict(int)
        for it in items:
            if it["variant"] != "outside" or seen[it["template"]] >= PER_TEMPLATE:
                continue
            seen[it["template"]] += 1
            assert it["id"].endswith("_outside")
            inside = by_id[it["id"][:-len("_outside")] + "_inside"]
            assert inside["template"] == it["template"] and inside["variant"] == "inside"
            for row in (it, inside):
                rows.append({"id": "x8:" + row["id"], "source": source, "template": row["template"], "variant": row["variant"],
                             "problem": row["problem"], "sub_answers": row["sub_answers"]})
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(len(rows), "rows,", len({(r["source"], r["template"]) for r in rows}), "templates")
    return 0


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

    def call(messages, max_tokens=8192):
        for attempt in range(5):
            try:
                r = client.chat.completions.create(model=args.model, messages=messages, temperature=0.3,
                                                   max_tokens=max_tokens, extra_body=extra or None)
                return r.choices[0].message.content or ""
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    return None

    def one(row):
        c = call(chat(row["problem"]))
        h = call(chat(row["problem"] + "\n\n" + pilot.FLAG_INSTRUCTION))
        first_msgs = chat(row["problem"] + ELICIT_Q)
        first = call(first_msgs)
        e = None if first is None else call(first_msgs + [{"role": "assistant", "content": first},
                                                          {"role": "user", "content": TURN_SOLVE}])
        if None in (c, h, first, e):   # an API failure is not a wrong answer: the row is retried on resume
            return {"id": row["id"], "error": "call failed"}
        return {"id": row["id"], "elicit": first, **{f"{a}_text": t for a, t in zip(ARMS, (c, h, e))},
                **{f"{a}_pass": grade(t, row["sub_answers"]).reward == 1.0 for a, t in zip(ARMS, (c, h, e))}}

    with args.out.open("a") as fh, ThreadPoolExecutor(args.workers) as pool:
        for n, out in enumerate(pool.map(one, [r for r in rows if r["id"] not in done]), 1):
            if "error" in out:
                print("error", out["id"], flush=True)
                continue
            fh.write(json.dumps(out, ensure_ascii=False) + "\n"); fh.flush()
            if n % 25 == 0:
                print(n, flush=True)
    return 0


def _ci(values):
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(SEED)
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": len(values)}


def contrast(recs: list[dict], rows: dict, variant: str, a: str, b: str, pick=lambda t: True) -> dict:
    """a - b on `variant` rows: per template, then the equal-weight mean of a cluster's templates, clusters resampled."""
    per_t = defaultdict(list)
    for r in recs:
        row = rows[r["id"]]
        if row["variant"] == variant and pick(row["template"]):
            per_t[(row["template"], row["source"])].append(int(r[a + "_pass"]) - int(r[b + "_pass"]))
    per_u = defaultdict(list)
    for (t, s), v in per_t.items():
        per_u[_unit(t, s)].append(fmean(v))
    return _ci([fmean(v) for v in per_u.values()])


def reading(ec_out: dict, ec_in: dict, eh_out: dict) -> str:
    if ec_out["mean"] is None:
        return "只报数字"
    if ec_out["mean"] >= .10 and ec_out["ci95"][0] > 0 and ec_in["mean"] is not None and ec_in["mean"] >= -.05:
        text = "模型自己调出的知识能补回越界失败——缺的是调用，不只是知识"
    elif ec_out["mean"] < .05 and ec_out["ci95"][0] <= 0 <= ec_out["ci95"][1]:
        text = "自己调出也补不回——缺的是知识本身"
    else:
        text = "只报数字"
    if eh_out["mean"] is not None and eh_out["mean"] >= .10 and eh_out["ci95"][0] > 0:
        text += "；具体调出强于泛泛提示"
    return text


def report(args) -> int:
    rows = {r["id"]: r for r in json.loads(args.input.read_text())}
    kany = json.loads(args.kany.read_text()) if args.kany else {}
    out = {}
    for spec in args.runs:
        cfg, path = spec.split("=", 1)
        recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        recs = [r for r in recs if "error" not in r and r["id"] in rows]
        acc = {v: {a: fmean(r[a + "_pass"] for r in recs if rows[r["id"]]["variant"] == v) if recs else None for a in ARMS}
               for v in ("outside", "inside")}
        part = {"rows": len(recs), "of": len(rows), "accuracy": acc,
                "E-C_outside": contrast(recs, rows, "outside", "E", "C"), "E-H_outside": contrast(recs, rows, "outside", "E", "H"),
                "H-C_outside": contrast(recs, rows, "outside", "H", "C"), "E-C_inside": contrast(recs, rows, "inside", "E", "C")}
        tab = (kany.get(args.kany_key.get(cfg, cfg)) or {}).get("per_template", {})
        if tab:   # descriptive
            part["E-C_outside_Kany_high"] = contrast(recs, rows, "outside", "E", "C", lambda t: t in tab and tab[t]["K"] >= 2 / 3 - 1e-9)
            part["E-C_outside_Kany_low"] = contrast(recs, rows, "outside", "E", "C", lambda t: t in tab and tab[t]["K"] <= 1 / 3 + 1e-9)
        part["complete"] = len(recs) == len(rows)
        part["reading"] = reading(part["E-C_outside"], part["E-C_inside"], part["E-H_outside"]) if part["complete"] else "不完整：只报数字"
        f = lambda d: None if d["mean"] is None else (round(100 * d["mean"], 1), [round(100 * x, 1) for x in d["ci95"]], d["units"])
        print(cfg, f"{len(recs)}/{len(rows)}", {v: {a: None if x is None else round(100 * x, 1) for a, x in d.items()} for v, d in acc.items()},
              "E-C out", f(part["E-C_outside"]), "E-H out", f(part["E-H_outside"]), "E-C in", f(part["E-C_inside"]), "->", part["reading"])
        out[cfg] = part
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--atmos", type=Path, required=True); b.add_argument("--x5", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True); b.set_defaults(fn=build)
    r = sub.add_parser("run"); r.add_argument("--input", type=Path, required=True); r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True); r.add_argument("--api-key-env"); r.add_argument("--thinking-off", action="store_true")
    r.add_argument("--extra-body"); r.add_argument("--workers", type=int, default=5); r.add_argument("--out", type=Path, required=True)
    r.set_defaults(fn=run)
    p = sub.add_parser("report"); p.add_argument("--input", type=Path, required=True); p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--kany", type=Path, help="results/x6/report_kany.json, for the descriptive K_any strata")
    p.add_argument("--out", type=Path, required=True); p.set_defaults(fn=report, kany_key={"deepseek-off": "deepseek", "qwen3-8b-off": "qwen3-8b",
                                                                    "qwen3.5-9b-off": "qwen35-9b"})
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
