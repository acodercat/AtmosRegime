"""D-3 detection probe report (docs/PREREG_MAIN.md, revision D-3, section 4).

    python -m explore.d3_report --input-dir experiments/_inputs/d3_detect --run-dir experiments/d3_detect \
        --configs deepseek-v4-flash.think-off ... --out results/d3/report.json

Every final response is executed once more with `pilot.grade_regime`, which removes the "regime" entry before
grading the answer; the runner's own pass / fail is not used. Infrastructure-error rows (no response) are left
out of every denominator and counted.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl import pilot
from atmoscoder_rl.coverage_clusters import unit_of
from atmoscoder_rl.runs import is_gradable
from atmoscoder_rl.traps import family

DRAWS = 10_000
SEED = "d3:20261011"
PREFIX = "judge:"


def load(path: Path) -> dict:
    results = json.loads(path.read_text())["results"]
    return results if isinstance(results, dict) else {r["id"]: r for r in results}


def ci(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(SEED)
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]],
            "units": len(values)}


def score(rows: list[dict], records: dict, label_of) -> list[dict]:
    """One dict per row with a response: label, judgement (None if unparsed), answer right / graded wrong."""
    todo = [(r, records[r["id"]]) for r in rows if is_gradable(records.get(r["id"])) and records[r["id"]].get("attempts")]
    def one(pair):
        row, rec = pair
        label = label_of(row)
        asked = 1 if label is None else label   # an unlabelled row is graded against 1 only to read the judgement back
        s = pilot.grade_regime(rec["attempts"][-1]["response"], row["sub_answers"], asked)
        judgement = None if s["judge/missing"] else (asked if s["judge/correct"] else 1 - asked)
        return {"row": row, "label": label, "judgement": judgement, "right": s["answer"] == 1.0,
                "wrong": bool(s.get("outcome/wrong_answer"))}
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, todo))


def by_cluster(scored: list[dict], pick, measure) -> dict[str, float]:
    """Template-equal within cluster: per template the share of `measure` among rows `pick` selects."""
    per_tpl = defaultdict(list)
    for s in scored:
        if pick(s):
            per_tpl[s["row"]["template"]].append(measure(s))
    per_cluster = defaultdict(list)
    for tpl, vals in per_tpl.items():
        per_cluster[unit_of(tpl)].append(fmean(vals))
    return {c: fmean(v) for c, v in per_cluster.items()}


def templates_part(rows: list[dict], records: dict) -> dict:
    scored = score(rows, records, lambda r: pilot.JUDGE_LABEL[r["variant"]])
    parsed = [s for s in scored if s["judgement"] is not None]
    flag0 = lambda s: float(s["judgement"] == 0)
    hit = by_cluster(parsed, lambda s: s["row"]["variant"] == "outside", flag0)
    fa = by_cluster(parsed, lambda s: s["row"]["variant"] == "inside", flag0)
    both = sorted(set(hit) & set(fa))
    acc = {v: ci(list(by_cluster(scored, lambda s, v=v: s["row"]["variant"] == v, lambda s: float(s["right"])).values()))
           for v in ("inside", "outside")}
    return {
        "rows": len(rows), "infra_errors": len(rows) - len(scored),
        "parse_rate": len(parsed) / len(scored) if scored else None,
        "hit_rate": ci(list(hit.values())), "false_alarm_rate": ci(list(fa.values())),
        "discrimination": ci([hit[c] - fa[c] for c in both]),
        "flag0_among_outside_wrong": ci(list(by_cluster(
            parsed, lambda s: s["row"]["variant"] == "outside" and s["wrong"], flag0).values())),
        "flag0_among_outside_right": ci(list(by_cluster(
            parsed, lambda s: s["row"]["variant"] == "outside" and s["right"], flag0).values())),
        "answer_accuracy_with_judge_prompt": acc,
    }


def textbook_part(rows: list[dict], records: dict) -> dict:
    group = lambda r: "parent" if "parent_id" not in r else family(r)
    label = {"parent": 1, "regime:recognise": 0, "geometry_detail": None}
    scored = score(rows, records, lambda r: label[group(r)])
    out = {"infra_errors": len(rows) - len(scored)}
    for g in label:
        items = [s for s in scored if group(s["row"]) == g]
        parsed = [float(s["judgement"] == 0) for s in items if s["judgement"] is not None]
        out[g] = {"rows": len(items), "parsed": len(parsed), "flag0_rate": ci(parsed)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tpl_rows = json.loads((args.input_dir / "templates.json").read_text())
    tb_rows = json.loads((args.input_dir / "textbook.json").read_text())
    report = {}
    for key in args.configs:
        part = {}
        for name, rows, fn in (("templates", tpl_rows, templates_part), ("textbook", tb_rows, textbook_part)):
            path = args.run_dir / f"{key}.{name}.run1.json"
            if not path.exists():
                part[name] = "not executed"
                continue
            records = load(path)
            if len(records) != len(rows):
                raise SystemExit(f"{path}: {len(records)} of {len(rows)} rows")
            part[name] = fn(rows, records)
        t = part.get("templates")
        if isinstance(t, dict):
            d = t["discrimination"]
            part["can_detect"] = bool(t["parse_rate"] >= .8 and d["mean"] >= .20 and d["ci95"][0] > 0)
            print(key, "parse", round(t["parse_rate"], 3), "H", round(t["hit_rate"]["mean"], 3),
                  "F", round(t["false_alarm_rate"]["mean"], 3), "H-F", round(d["mean"], 3), d["ci95"],
                  "flag0|wrong", t["flag0_among_outside_wrong"]["mean"])
        report[key] = part
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
