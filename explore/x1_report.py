"""X1a (docs/PREREG_MAIN.md, revision D-4 section 2): among out-of-range rows the model judged "does not hold",
how often is the answer still the shortcut value?

    python -m explore.x1_report --input-dir experiments/_inputs/d3_detect --run-dir experiments/d3_detect \
        --configs deepseek-v4-flash.think-off qwen3-8b.think-off --out results/x1/x1a.json

Each final response is executed once more; the "regime" entry is removed before grading, exactly as D-3 does, and the
graded details are then tested against the template's shortcut vector with the D-2 S1 rule (`capture.shortcut_match`).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.capture import shortcut_match
from atmoscoder_rl.coverage_clusters import unit_of
from atmoscoder_rl.engine import SOLVER_TIMEOUT, extract_code
from atmoscoder_rl.grading import TOLERANCE, verify
from atmoscoder_rl.pilot import JUDGE_KEY, JUDGE_LABEL, parse_judgement
from atmoscoder_rl.reward import DEFAULT_UNIT_RULE
from atmoscoder_rl.runs import is_gradable
from atmoscoder_rl.sandbox import run_solver_sandboxed
from explore.d3_report import load

DRAWS = 10_000
SEED = "d4:20261012"


def ci(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(SEED)
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": len(values)}


def execute(row: dict, record: dict) -> dict | None:
    """Judgement, graded outcome and shortcut flag for one out-of-range row; None when nothing ran."""
    code = extract_code(record["attempts"][-1].get("response") or "")
    if not code.strip():
        return None
    try:
        ok, results, _ = run_solver_sandboxed(code, timeout=SOLVER_TIMEOUT)
    except Exception:  # noqa: BLE001
        return None
    if not ok or not isinstance(results, dict):
        return None
    judgement = parse_judgement(results.pop(JUDGE_KEY, None))
    if not results:
        return None
    passed, details = verify(results, list(row["sub_answers"]), TOLERANCE, DEFAULT_UNIT_RULE)
    graded = any(isinstance(d.get("actual"), (int, float)) for d in details)
    if not graded:
        return None
    return {"template": row["template"], "judgement": judgement, "right": bool(passed),
            "shortcut": shortcut_match({"details": details}, row)}


def by_cluster(items: list[dict], pick, measure) -> dict[str, float]:
    per_tpl = defaultdict(list)
    for s in items:
        if pick(s):
            per_tpl[s["template"]].append(float(measure(s)))
    per_cluster = defaultdict(list)
    for tpl, vals in per_tpl.items():
        per_cluster[unit_of(tpl)].append(fmean(vals))
    return {c: fmean(v) for c, v in per_cluster.items()}


def analyse(rows: list[dict], records: dict) -> dict:
    outside = [r for r in rows if JUDGE_LABEL[r["variant"]] == 0]
    todo = [(r, records[r["id"]]) for r in outside if is_gradable(records.get(r["id"])) and records[r["id"]].get("attempts")]
    with ThreadPoolExecutor(16) as pool:
        executed = [x for x in pool.map(lambda p: execute(*p), todo) if x]
    parsed = [x for x in executed if x["judgement"] is not None]
    wrong = [x for x in parsed if not x["right"]]
    cells = {f"judged{j}_{'shortcut' if s else 'other'}": sum(1 for x in wrong if x["judgement"] == j and x["shortcut"] == s)
             for j in (0, 1) for s in (True, False)}
    return {
        "outside_rows": len(outside), "executed": len(executed), "parsed": len(parsed), "wrong_parsed": len(wrong),
        "shortcut_among_judged0_wrong": ci(list(by_cluster(wrong, lambda x: x["judgement"] == 0, lambda x: x["shortcut"]).values())),
        "shortcut_among_judged1_wrong": ci(list(by_cluster(wrong, lambda x: x["judgement"] == 1, lambda x: x["shortcut"]).values())),
        "judged0_among_shortcut_wrong": ci(list(by_cluster(wrong, lambda x: x["shortcut"], lambda x: x["judgement"] == 0).values())),
        "judged0_among_other_wrong": ci(list(by_cluster(wrong, lambda x: not x["shortcut"], lambda x: x["judgement"] == 0).values())),
        "wrong_cells": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads((args.input_dir / "templates.json").read_text())
    report = {}
    for key in args.configs:
        path = args.run_dir / f"{key}.templates.run1.json"
        if not path.exists():
            report[key] = "not executed"
            continue
        records = load(path)
        if len(records) != len(rows):
            raise SystemExit(f"{path}: {len(records)} of {len(rows)} rows")
        part = analyse(rows, records)
        m = part["shortcut_among_judged0_wrong"]
        part["reading"] = ("said it does not hold, still mostly used the standard relation" if m["mean"] is not None and m["mean"] >= .5
                           else "numbers only")
        print(key, "shortcut|judged0,wrong", m["mean"], m["ci95"], "shortcut|judged1,wrong",
              part["shortcut_among_judged1_wrong"]["mean"], part["wrong_cells"])
        report[key] = part
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
