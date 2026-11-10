"""X1b (docs/PREREG_MAIN.md, revision D-4 section 3): does the stored thinking text say the standard relation does not
hold for the given values? Judge calls, then the per-cluster report.

    # label (resume-safe; appends to --labels as jsonl; deepseek concurrency <= 5)
    python -m explore.x1_trace_judge label --items templates.json --run RUN.json --labels results/x1/traces_<key>.jsonl
    # report over several configurations
    python -m explore.x1_trace_judge report --items templates.json --labels KEY=FILE [KEY=FILE ...] --out results/x1/x1b.json
    # the fixed audit sample
    python -m explore.x1_trace_judge audit --items templates.json --run RUN.json --labels FILE --out audit.tsv

The judge (`judge-oneapi` in models.toml: deepseek-v4-flash, thinking off, T = 0) sees the problem, the template's mechanism note and the thinking text, never
the answer or its verdict. Rows: inside and outside variants whose final attempt stores a non-empty `reasoning`.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.coverage_clusters import unit_of
from atmoscoder_rl.evaluation.client import ModelError, build_model
from atmoscoder_rl.runs import is_gradable
from explore.d3_report import load

DRAWS = 10_000
SEED = "d4:20261012"
TRIES = 3
HEAD = TAIL = 12_000
WORKERS = 5

PROMPT = """You are reading the private reasoning a model wrote before answering a physics calculation. Decide ONE thing:
does the reasoning explicitly state that the standard textbook relation or assumption for this kind of problem does NOT hold
(is invalid, breaks down, must be replaced or corrected) for the values given in this problem?

PROBLEM:
{problem}

NOTE ON THE MECHANISM (the model did not see this): {mechanism}

REASONING TEXT:
<<<
{reasoning}
>>>

Answer "yes" only if the reasoning itself says, in its own words, that the standard relation/assumption fails or must be
replaced for these values. Merely applying a correction without saying the standard form fails, or listing assumptions
without saying they are violated, is "no".

Reply with JSON only: {{"acknowledged": "yes" | "no", "evidence": "<the sentence that decides it, at most 30 words>"}}"""


def clip(text: str) -> str:
    return text if len(text) <= HEAD + TAIL else text[:HEAD] + "\n[... truncated ...]\n" + text[-TAIL:]


def rows_with_reasoning(items: list[dict], records: dict) -> list[tuple[dict, dict]]:
    out = []
    for row in items:
        if row["variant"] not in ("inside", "outside"):
            continue
        rec = records.get(row["id"])
        if not is_gradable(rec) or not rec.get("attempts"):
            continue
        if (rec["attempts"][-1].get("reasoning") or "").strip():
            out.append((row, rec))
    return out


def ask(judge, prompt: str) -> dict:
    for _ in range(TRIES):
        try:
            text = judge.generate(prompt).content
        except ModelError as error:
            return {"acknowledged": None, "error": str(error)}
        match = re.search(r"\{.*\}", text or "", re.S)
        try:
            out = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            out = {}
        if out.get("acknowledged") in ("yes", "no"):
            return {"acknowledged": out["acknowledged"], "evidence": str(out.get("evidence", ""))[:300]}
    return {"acknowledged": None, "error": "unparsable"}


def label(args) -> int:
    items = json.loads(args.items.read_text())
    records = load(args.run)
    done = set()
    if args.labels.exists():
        done = {json.loads(l)["id"] for l in args.labels.read_text().splitlines() if l.strip()}
    todo = [(r, rec) for r, rec in rows_with_reasoning(items, records) if r["id"] not in done]
    print(f"{len(todo)} rows to judge ({len(done)} done)")
    judge = build_model("judge-oneapi")
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    def one(pair):
        row, rec = pair
        prompt = PROMPT.format(problem=row["problem"], mechanism=row["mechanism"], reasoning=clip(rec["attempts"][-1]["reasoning"]))
        return {"id": row["id"], "template": row["template"], "variant": row["variant"],
                "passed": rec.get("passed"), **ask(judge, prompt)}
    with args.labels.open("a") as fh, ThreadPoolExecutor(WORKERS) as pool:
        for n, out in enumerate(pool.map(one, todo), 1):
            fh.write(json.dumps(out, ensure_ascii=False) + "\n"); fh.flush()
            if n % 100 == 0:
                print(n, flush=True)
    return 0


def ci(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(SEED)
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": len(values)}


def by_cluster(labels: list[dict], pick) -> dict[str, float]:
    per_tpl = defaultdict(list)
    for l in labels:
        if pick(l):
            per_tpl[l["template"]].append(float(l["acknowledged"] == "yes"))
    per_cluster = defaultdict(list)
    for tpl, vals in per_tpl.items():
        per_cluster[unit_of(tpl)].append(fmean(vals))
    return {c: fmean(v) for c, v in per_cluster.items()}


def graded_wrong(l: dict) -> bool:
    return l.get("passed") is False


def report(args) -> int:
    out = {}
    for spec in args.labels:
        key, path = spec.split("=", 1)
        labels = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        judged = [l for l in labels if l.get("acknowledged") in ("yes", "no")]
        part = {
            "rows": len(labels), "unjudged": len(labels) - len(judged),
            "yes_among_outside_wrong": ci(list(by_cluster(judged, lambda l: l["variant"] == "outside" and graded_wrong(l)).values())),
            "yes_among_outside_right": ci(list(by_cluster(judged, lambda l: l["variant"] == "outside" and l["passed"] is True).values())),
            "yes_among_inside": ci(list(by_cluster(judged, lambda l: l["variant"] == "inside").values())),
        }
        m, fp = part["yes_among_outside_wrong"], part["yes_among_inside"]
        part["reading"] = ("acknowledged in thinking, still applied" if m["mean"] is not None and fp["mean"] is not None
                           and m["mean"] >= .3 and fp["mean"] <= .1 else "numbers only; audit precision decides whether to read at all")
        print(key, "yes|outside wrong", m["mean"], m["ci95"], "yes|outside right", part["yes_among_outside_right"]["mean"],
              "yes|inside", fp["mean"], "unjudged", part["unjudged"])
        out[key] = part
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return 0


def audit(args) -> int:
    records = load(args.run)
    labels = [json.loads(l) for l in args.labels.read_text().splitlines() if l.strip()]
    pool = [l for l in labels if l["variant"] == "outside" and graded_wrong(l) and l.get("acknowledged") in ("yes", "no")]
    rng = random.Random(SEED + ":audit")
    yes = [l for l in pool if l["acknowledged"] == "yes"]
    no = [l for l in pool if l["acknowledged"] == "no"]
    sample = rng.sample(yes, min(20, len(yes))) + rng.sample(no, min(20, len(no)))
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "judge", "evidence", "is_real_acknowledgement", "reasoning"])
        for l in sample:
            text = records[l["id"]]["attempts"][-1]["reasoning"].replace("\t", " ").replace("\n", " ⏎ ")
            w.writerow([l["id"], l["acknowledged"], l.get("evidence", ""), "", text[:6000]])
    print(len(sample), "rows written")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("label"); a.add_argument("--items", type=Path, required=True); a.add_argument("--run", type=Path, required=True)
    a.add_argument("--labels", type=Path, required=True); a.set_defaults(fn=label)
    b = sub.add_parser("report"); b.add_argument("--items", type=Path, required=True); b.add_argument("--labels", nargs="+", required=True)
    b.add_argument("--out", type=Path, required=True); b.set_defaults(fn=report)
    c = sub.add_parser("audit"); c.add_argument("--items", type=Path, required=True); c.add_argument("--run", type=Path, required=True)
    c.add_argument("--labels", type=Path, required=True); c.add_argument("--out", type=Path, required=True); c.set_defaults(fn=audit)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
