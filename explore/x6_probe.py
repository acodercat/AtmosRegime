"""X6 knowledge probe (docs/PREREG_X.md amendment X-6).

    python -m explore.x6_probe build --atmos <families_v4>/templates.json --x5 <x5>/templates.json --out probe.json
    python -m explore.x6_probe ask --input probe.json --base-url URL --model NAME [--api-key-env VAR] [--thinking-off] --out answers_<cfg>.jsonl
    python -m explore.x6_probe grade --input probe.json --answers answers_<cfg>.jsonl --out graded_<cfg>.jsonl
    python -m explore.x6_probe audit --input probe.json --graded <cfg>=graded.jsonl ... --out audit.json
    python -m explore.x6_probe report --input probe.json --graded <cfg>=graded.jsonl ... \
        --outside <cfg>=atmos_run.json,x5_run.json ... --atmos-items ... --x5-items ... --out report.json

The rubric of each mechanism is its template module's own docstring, from "Parent's method" to "Fair from the text".
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

QUESTION = """

Do not solve this problem numerically. Instead, answer these three questions:
(1) Which standard textbook relation would normally be used to solve a problem of this kind? Write it as an equation.
(2) For the values given in this problem, does that standard relation remain accurate? Answer "yes" or "no" first, then explain why in one or two sentences.
(3) If it does not remain accurate, write the relation that should be used instead, as an equation (or "not applicable" if you answered yes)."""

JUDGE = """You are grading a model's answer about the physics of a problem. Use ONLY the rubric to decide.

PROBLEM:
{problem}

RUBRIC (written by the problem's author; the model did not see it):
{rubric}

MODEL'S ANSWER:
<<<
{answer}
>>>

Decide two things:
R - In its answer to question (2), does the model say the standard relation does NOT remain accurate for these values, for a reason that points to the condition in the rubric's "Where it fails" (the same physical quantity or effect)? A "no" for a different reason does not count.
K - In its answer to question (3), does the model write a relation equivalent to the replacement in the rubric's "Where it fails" (equivalent algebra, or an approximation the rubric itself lists as acceptable)? Naming it without writing the relation, or writing a different correction, is "no". If the model answered "yes" in (2) and gave no replacement, K is "no".

Reply with JSON only: {{"R": "yes" | "no", "K": "yes" | "no", "evidence": "<at most 30 words>"}}"""

JUDGE_ANY = """You are grading whether a model's answer shows it KNOWS a specific physics relation. Use ONLY the rubric.

PROBLEM:
{problem}

RUBRIC (written by the problem's author; the model did not see it):
{rubric}

MODEL'S ANSWER:
<<<
{answer}
>>>

KA - Anywhere in the answer (in (1), (2) or (3), whatever the model calls it), does the model write a relation equivalent to the
replacement in the rubric's "Where it fails" (equivalent algebra, or an approximation the rubric itself lists as acceptable)?
It does not matter whether the model calls it "the standard relation" or "the replacement", or whether it answered yes or no in (2).
Naming it without writing the relation, or writing a different relation, is "no".

Reply with JSON only: {{"KA": "yes" | "no", "evidence": "<at most 30 words>"}}"""

DRAWS, SEED, PAIRS = 10_000, "x6:20261012", 3


def rubric_of(template: str) -> str:
    from atmoscoder_rl.templates import EXTENDED_MODULES, MODULES
    from atmoscoder_rl.templates_x5 import MODULES as X5
    doc = ({**MODULES, **EXTENDED_MODULES, **X5}[template].__doc__ or "")
    start = doc.find("Parent's method")
    end = doc.find("Fair from the text")
    return doc[start:end if end > start else start + 1500].strip()


def build(args) -> int:
    rows = []
    for source, path in (("atmos", args.atmos), ("x5", args.x5)):
        items = json.loads(path.read_text())
        seen = defaultdict(int)
        for it in items:
            if it["variant"] != "outside":
                continue
            if seen[it["template"]] >= PAIRS:
                continue
            seen[it["template"]] += 1
            rows.append({"id": "x6:" + it["id"], "source": source, "template": it["template"],
                         "problem": it["problem"] + QUESTION, "item_problem": it["problem"], "rubric": rubric_of(it["template"])})
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(len(rows), "rows,", len({r["template"] for r in rows}), "templates")
    return 0


def _client(base_url: str, key_env: str | None):
    import openai
    key = os.environ[key_env] if key_env else "EMPTY"
    return openai.OpenAI(api_key=key, base_url=base_url, timeout=1800.0, max_retries=3)


def ask(args) -> int:
    rows = json.loads(args.input.read_text())
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    client = _client(args.base_url, args.api_key_env)
    extra = {}
    if args.thinking_off:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if args.extra_body:
        extra.update(json.loads(args.extra_body))

    def one(r):
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": r["problem"]}],
                                                      temperature=0.3, max_tokens=4096, extra_body=extra or None)
                return {"id": r["id"], "answer": resp.choices[0].message.content or ""}
            except Exception as error:  # noqa: BLE001
                if attempt == 4:
                    return {"id": r["id"], "answer": None, "error": str(error)[:300]}
    with args.out.open("a") as fh, ThreadPoolExecutor(args.workers) as pool:
        for n, out in enumerate(pool.map(one, todo), 1):
            fh.write(json.dumps(out, ensure_ascii=False) + "\n"); fh.flush()
            if n % 25 == 0:
                print(n, flush=True)
    return 0


def grade(args) -> int:
    """K and R (amendment X-6); with --any, the exploratory K_any (replacement written anywhere, any framing)."""
    from atmoscoder_rl.evaluation.client import ModelError, build_model
    rows = {r["id"]: r for r in json.loads(args.input.read_text())}
    answers = [json.loads(l) for l in args.answers.read_text().splitlines() if l.strip()]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines() if l.strip()}
    judge = build_model("judge-oneapi")

    def one(a):
        r = rows[a["id"]]
        if not a.get("answer"):
            return {"id": a["id"], "R": None, "K": None, "error": "no answer"}
        prompt = (JUDGE_ANY if args.any else JUDGE).format(problem=r["item_problem"], rubric=r["rubric"], answer=a["answer"][-12000:])
        for _ in range(3):
            try:
                text = judge.generate(prompt).content
            except ModelError as error:
                return {"id": a["id"], "R": None, "K": None, "error": str(error)[:200]}
            m = re.search(r"\{.*\}", text or "", re.S)
            try:
                out = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                out = {}
            if args.any and out.get("KA") in ("yes", "no"):
                return {"id": a["id"], "R": out["KA"], "K": out["KA"], "evidence": str(out.get("evidence", ""))[:300]}
            if not args.any and out.get("R") in ("yes", "no") and out.get("K") in ("yes", "no"):
                return {"id": a["id"], "R": out["R"], "K": out["K"], "evidence": str(out.get("evidence", ""))[:300]}
        return {"id": a["id"], "R": None, "K": None, "error": "unparsable"}
    with args.out.open("a") as fh, ThreadPoolExecutor(5) as pool:
        for out in pool.map(one, [a for a in answers if a["id"] not in done]):
            fh.write(json.dumps(out, ensure_ascii=False) + "\n"); fh.flush()
    return 0


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def audit(args) -> int:
    rows = {r["id"]: r for r in json.loads(args.input.read_text())}
    rng = random.Random(SEED + ":audit")
    sample = []
    for spec in args.graded:
        cfg, graded_path, answers_path = spec.split("=", 1)[0], *spec.split("=", 1)[1].split(",")
        graded = {g["id"]: g for g in _jsonl(Path(graded_path)) if g.get("R")}
        answers = {a["id"]: a for a in _jsonl(Path(answers_path))}
        for gid in rng.sample(sorted(graded), 20):
            r = rows[gid]
            sample.append({"config": cfg, "id": gid, "problem": r["item_problem"], "rubric": r["rubric"],
                           "answer": answers[gid]["answer"], "judge_R": graded[gid]["R"], "judge_K": graded[gid]["K"]})
    args.out.write_text(json.dumps(sample, indent=1, ensure_ascii=False) + "\n")
    print(len(sample), "rows for the audit")
    return 0


def _outside_acc(items_path: Path, run_path: Path) -> dict[str, float]:
    from explore.d2_report import load
    items = json.loads(items_path.read_text())
    rep = load(run_path)
    acc = defaultdict(list)
    for it in items:
        if it["variant"] == "outside":
            acc[it["template"]].append(int(bool(rep[it["id"]].get("passed"))))
    return {t: fmean(v) for t, v in acc.items()}


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    mx, my = fmean(rx), fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def report(args) -> int:
    rows = {r["id"]: r for r in json.loads(args.input.read_text())}
    outside = {}
    for spec in args.outside:
        cfg, paths = spec.split("=", 1)
        atmos_run, x5_run = paths.split(",")
        outside[cfg] = {**_outside_acc(args.atmos_items, Path(atmos_run)), **_outside_acc(args.x5_items, Path(x5_run))}
    out = {}
    for spec in args.graded:
        cfg, path = spec.split("=", 1)
        graded = _jsonl(Path(path))
        per = defaultdict(lambda: {"K": [], "R": []})
        source = {}
        unparsed = 0
        for g in graded:
            r = rows[g["id"]]; source[r["template"]] = r["source"]
            if g.get("K") is None:
                unparsed += 1
            per[r["template"]]["K"].append(int(g.get("K") == "yes"))
            per[r["template"]]["R"].append(int(g.get("R") == "yes"))
        tpls = sorted(per)
        k = [fmean(per[t]["K"]) for t in tpls]; rr = [fmean(per[t]["R"]) for t in tpls]
        y = [outside[cfg][t] for t in tpls]
        rng = random.Random(SEED)
        boots = []
        for _ in range(DRAWS):
            idx = [rng.randrange(len(tpls)) for _ in tpls]
            v = _spearman([k[i] for i in idx], [y[i] for i in idx])
            if v == v:
                boots.append(v)
        boots.sort()
        rho = _spearman(k, y)
        sets = {s: [t for t in tpls if source[t] == s] for s in ("atmos", "x5")}
        strata = {}
        for name, pick in (("K_high", lambda v: v >= 2 / 3 - 1e-9), ("K_low", lambda v: v <= 1 / 3 + 1e-9)):
            strata[name] = {s: {"templates": len([t for t in ts if pick(fmean(per[t]["K"]))]),
                                "outside": (fmean(outside[cfg][t] for t in ts if pick(fmean(per[t]["K"])))
                                            if any(pick(fmean(per[t]["K"])) for t in ts) else None)} for s, ts in sets.items()}
        part = {"rows": len(graded), "unparsed": unparsed,
                "spearman_K_outside": {"rho": rho, "ci95": [boots[int(.025 * len(boots))], boots[int(.975 * len(boots)) - 1]], "templates": len(tpls)},
                "spearman_R_outside": _spearman(rr, y),
                "by_set": {s: {"templates": len(ts), "K": fmean(fmean(per[t]["K"]) for t in ts), "R": fmean(fmean(per[t]["R"]) for t in ts),
                               "outside": fmean(outside[cfg][t] for t in ts)} for s, ts in sets.items()},
                "strata": strata,
                "per_template": {t: {"source": source[t], "K": fmean(per[t]["K"]), "R": fmean(per[t]["R"]), "outside": outside[cfg][t]} for t in tpls}}
        part["predicts"] = bool(rho >= .3 and part["spearman_K_outside"]["ci95"][0] > 0)
        print(cfg, "rho(K, outside)", round(rho, 2), [round(x, 2) for x in part["spearman_K_outside"]["ci95"]],
              "rho(R, outside)", round(part["spearman_R_outside"], 2), "unparsed", unparsed)
        for s, v in part["by_set"].items():
            print("  ", s, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
        for name, v in strata.items():
            print("  ", name, {s: (x["templates"], None if x["outside"] is None else round(x["outside"], 2)) for s, x in v.items()})
        out[cfg] = part
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--atmos", type=Path, required=True); b.add_argument("--x5", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True); b.set_defaults(fn=build)
    a = sub.add_parser("ask"); a.add_argument("--input", type=Path, required=True); a.add_argument("--base-url", required=True)
    a.add_argument("--model", required=True); a.add_argument("--api-key-env"); a.add_argument("--thinking-off", action="store_true")
    a.add_argument("--extra-body"); a.add_argument("--workers", type=int, default=5); a.add_argument("--out", type=Path, required=True); a.set_defaults(fn=ask)
    g = sub.add_parser("grade"); g.add_argument("--input", type=Path, required=True); g.add_argument("--answers", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True); g.add_argument("--any", action="store_true",
                   help="exploratory K_any; written into both R and K fields so `report` reads it as K"); g.set_defaults(fn=grade)
    u = sub.add_parser("audit"); u.add_argument("--input", type=Path, required=True)
    u.add_argument("--graded", nargs="+", required=True, help="cfg=graded.jsonl,answers.jsonl"); u.add_argument("--out", type=Path, required=True); u.set_defaults(fn=audit)
    r = sub.add_parser("report"); r.add_argument("--input", type=Path, required=True); r.add_argument("--graded", nargs="+", required=True)
    r.add_argument("--outside", nargs="+", required=True); r.add_argument("--atmos-items", type=Path, required=True)
    r.add_argument("--x5-items", type=Path, required=True); r.add_argument("--out", type=Path, required=True); r.set_defaults(fn=report)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
