"""K2 knowledge probe with the standard method stated (docs/PREREG_X.md amendment X-7, section 2).

    python -m explore.x7_k2 build --split results/kc/split.json --x5 --out k2_probe.json
    python -m explore.x6_probe ask --input k2_probe.json ... --out k2_answers_<cfg>.jsonl        # same asking code as X6
    python -m explore.x7_k2 grade --input k2_probe.json --answers k2_answers_<cfg>.jsonl --out k2_graded_<cfg>.jsonl
    python -m explore.x6_probe audit --input k2_probe.json --graded <cfg>=graded,answers ... --out k2_audit.json   # seed below
    python -m explore.x7_k2 table --input k2_probe.json --graded <cfg>=k2_graded.jsonl ... --out k2_table.json

Templates: the 70 of the T folds (and, with --x5, the 13 X5 templates for X7's stratification); 3 outside items each at seed
20261014. The question states the template's own "Parent's method" text as the standard method; the rubric is "Where it fails".
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.build_templates import build
from atmoscoder_rl.template_wording import text as reviewed_text
from atmoscoder_rl.templates import ALL_TEMPLATES, EXTENDED_MODULES, MODULES
from atmoscoder_rl.templates_x5 import MODULES as X5_MODULES, TEMPLATES as X5_TEMPLATES

SEED, PAIRS = 20261014, 3

QUESTION = """

The standard textbook method for this kind of problem is:
{parent}

Do not solve the problem numerically. Answer two questions:
(a) For the values given in this problem, does that standard method remain accurate? Answer "yes" or "no" first, then explain why in one or two sentences.
(b) If it does not remain accurate, write the relation that should be used instead, as an equation (or "not applicable" if you answered yes)."""

JUDGE = """You are grading a model's answer about the physics of a problem. Use ONLY the rubric to decide.

PROBLEM:
{problem}

RUBRIC (written by the problem's author; the model saw only the "standard method" part):
{rubric}

MODEL'S ANSWER:
<<<
{answer}
>>>

R2 - In (a), does the model say the standard method does NOT remain accurate for these values, for a reason that points to the
condition in the rubric's "Where it fails" (the same physical quantity or effect)?
K2 - Anywhere in the answer, does the model write a relation equivalent to the replacement in the rubric's "Where it fails"
(equivalent algebra, or an approximation the rubric itself lists as acceptable)? It does not matter what it answered in (a).
Naming it without writing the relation, or writing a different relation, is "no".

Reply with JSON only: {{"R2": "yes" | "no", "K2": "yes" | "no", "evidence": "<at most 30 words>"}}"""


def _doc(name: str) -> str:
    return ({**MODULES, **EXTENDED_MODULES, **X5_MODULES}[name].__doc__ or "")


def parent_text(name: str) -> str:
    d = _doc(name)
    a, b = d.find("Parent's method"), d.find("Where it fails")
    return re.sub(r"^Parent's method:\s*", "", d[a:b].strip())


def rubric(name: str) -> str:
    d = _doc(name)
    a, b = d.find("Parent's method"), d.find("Fair from the text")
    return d[a:b if b > a else a + 1500].strip()


def build_cmd(args) -> int:
    split = json.loads(args.split.read_text())
    names = sorted(t for ts in split["clusters"].values() for t in ts)
    if len(names) != 70:
        raise SystemExit(f"expected the 70 T templates, found {len(names)}")
    reworded = set(split["reworded_templates"])
    rows = []
    for source, registry, pick in (("atmos", ALL_TEMPLATES, set(names)), ("x5", X5_TEMPLATES, set(X5_TEMPLATES) if args.x5 else set())):
        if not pick:
            continue
        for it in build(PAIRS, SEED, pick, registry={n: registry[n] for n in pick}):
            if it["variant"] != "outside":
                continue
            problem = reviewed_text(it["template"], it["params"], "outside") if it["template"] in reworded and source == "atmos" else it["problem"]
            rows.append({"id": "k2:" + it["id"], "source": source, "template": it["template"],
                         "problem": problem + QUESTION.format(parent=parent_text(it["template"])),
                         "item_problem": problem, "rubric": rubric(it["template"])})
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(len(rows), "rows,", len({r["template"] for r in rows}), "templates")
    return 0


def grade(args) -> int:
    from atmoscoder_rl.evaluation.client import ModelError, build_model
    rows = {r["id"]: r for r in json.loads(args.input.read_text())}
    answers = [json.loads(l) for l in args.answers.read_text().splitlines() if l.strip()]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines() if l.strip()}
    judge = build_model(args.judge)

    def one(a):
        r = rows[a["id"]]
        if not a.get("answer"):
            return {"id": a["id"], "R": None, "K": None, "error": "no answer"}
        prompt = JUDGE.format(problem=r["item_problem"], rubric=r["rubric"], answer=a["answer"][-12000:])
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
            if out.get("R2") in ("yes", "no") and out.get("K2") in ("yes", "no"):
                return {"id": a["id"], "R": out["R2"], "K": out["K2"], "evidence": str(out.get("evidence", ""))[:300]}
        return {"id": a["id"], "R": None, "K": None, "error": "unparsable"}
    with args.out.open("a") as fh, ThreadPoolExecutor(5) as pool:
        for out in pool.map(one, [a for a in answers if a["id"] not in done]):
            fh.write(json.dumps(out, ensure_ascii=False) + "\n"); fh.flush()
    return 0


def table(args) -> int:
    rows = {r["id"]: r for r in json.loads(args.input.read_text())}
    out = {}
    for spec in args.graded:
        cfg, path = spec.split("=", 1)
        per = defaultdict(lambda: {"K2": [], "R2": []})
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            r = rows[g["id"]]
            per[r["template"]]["K2"].append(int(g.get("K") == "yes"))
            per[r["template"]]["R2"].append(int(g.get("R") == "yes"))
        tab = {t: {"source": next(r["source"] for r in rows.values() if r["template"] == t),
                   "K2": fmean(v["K2"]), "R2": fmean(v["R2"])} for t, v in sorted(per.items())}
        high = [t for t, v in tab.items() if v["K2"] >= 2 / 3 - 1e-9]
        low = [t for t, v in tab.items() if v["K2"] <= 1 / 3 + 1e-9]
        print(cfg, "templates", len(tab), "K-high", len(high), "K-low", len(low),
              "mean K2 atmos", round(fmean(v["K2"] for v in tab.values() if v["source"] == "atmos"), 2),
              "x5", round(fmean(v["K2"] for v in tab.values() if v["source"] == "x5"), 2) if any(v["source"] == "x5" for v in tab.values()) else "")
        out[cfg] = {"per_template": tab, "K_high": high, "K_low": low}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--split", type=Path, required=True); b.add_argument("--x5", action="store_true")
    b.add_argument("--out", type=Path, required=True); b.set_defaults(fn=build_cmd)
    g = sub.add_parser("grade"); g.add_argument("--input", type=Path, required=True); g.add_argument("--answers", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True); g.add_argument("--judge", default="judge-oneapi",
                   help="models.toml entry (X-8: judge-oneapi-think)"); g.set_defaults(fn=grade)
    t = sub.add_parser("table"); t.add_argument("--input", type=Path, required=True); t.add_argument("--graded", nargs="+", required=True)
    t.add_argument("--out", type=Path, required=True); t.set_defaults(fn=table)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
