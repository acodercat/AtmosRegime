"""Descriptive analyses added after the PREREG_X results were read (docs/PREREG_X.md X-27). None changes a verdict.

    python -m explore.x_posthoc --name qwen3-8b --report results/x/report_qwen3-8b.json --out results/x/posthoc_qwen3-8b.json

1. held_out_by_variant: held-out endpoint - start per arm for outside, inside and given rows, with the same weighting and
   resampling as `x_report` transfer (template mean -> cluster mean, seeds averaged within a cluster, 10,000 draws,
   `random.Random("x:20261012")`). The outside line equals `x_report` transfer; inside and given show how much of it is a
   general gain on rows where the standard treatment holds.
2. zero_hit_exceptions: P1's Z mechanisms (base pass@64 = 0) that an arm without warm start brought to >= 50 % on some seed.
3. truncation_split: per arm label, held-out outside / inside accuracy of rows whose graded (last) call did or did not use
   all 8,192 tokens (`x_eval` max_tokens), from the regraded verdicts.
4. answer_after_judgement: F and A arms, held-out outside judge rows. Each final response is executed once (sandbox, as
   `x_report`); the "regime" entry is read and removed, the answer graded against the right answer and against the
   standard treatment's value (`shortcut_values`, same units), both under the runner's rule (5 %, benchmark units).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl import pilot
from atmoscoder_rl.engine import extract_code
from atmoscoder_rl.grading import TOLERANCE, verify
from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.sandbox import run_solver_sandboxed
from explore.build_detect_input import PREFIX
from explore.x_report import ARMS, FOLDS, MAX_TOKENS, NO_WARM, OTHER, Family, acc_measure, diff

VARIANTS = ("outside", "inside", "given")


def held_out_by_variant(fam: Family) -> dict:
    """`x_report.transfer` exactly (same measure, unit order and resampling), for each variant."""
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for variant in VARIANTS:
            m = acc_measure(fam, variant)
            out[arm][variant] = diff(fam.arm_side(arm, "held", m), fam.start_side(arm, "held", m))
    return out


def zero_hit_exceptions(report: dict) -> list[dict]:
    p1 = report["P1"]
    out = []
    for m in p1["Z"]:
        row = p1["per_mechanism"][m]
        hits = {k: v for k, v in row.items() if k.split("_")[0] in NO_WARM and v is not None and v >= .5}
        if hits:
            out.append({"mechanism": m, "fold": row["fold"], "at_or_above_50": hits,
                        "all_arms": {k: v for k, v in row.items() if k.split("_")[0] in NO_WARM}})
    return out


def truncated(record: dict) -> bool:
    """`x_report.truncation`'s definition: the graded (last) call used all MAX_TOKENS."""
    calls = record.get("attempts") or []
    return bool(calls) and calls[-1].get("usage", {}).get("completion_tokens", 0) >= MAX_TOKENS


def truncation_split(fam: Family) -> dict:
    out = {}
    for arm in ARMS:
        for fold in FOLDS:
            for s in fam.seeds:
                label = fam.arm(arm, fold, s)
                recs, passed = fam.records(label, "final"), fam.final_passed(label)
                part = {}
                for variant in ("outside", "inside"):
                    ids = [i for i, r in fam.rows.items()
                           if r["cluster_fold"] == OTHER[fold] and r["variant"] == variant and i in passed]
                    cut = [i for i in ids if truncated(recs[i])]
                    rest = [i for i in ids if i not in set(cut)]
                    acc = lambda xs: fmean(float(passed[i]) for i in xs) if xs else None
                    part[variant] = {"rows": len(ids), "truncated": len(cut), "accuracy_truncated": acc(cut),
                                     "accuracy_rest": acc(rest), "accuracy_all": acc(ids)}
                out[label] = part
    return out


def classify(text: str, row: dict) -> str:
    code = extract_code(text or "")
    try:
        ok, results, _ = run_solver_sandboxed(code) if code.strip() else (False, {}, "")
    except Exception:  # noqa: BLE001
        ok, results = False, {}
    if not ok or not isinstance(results, dict):
        return "no_result"
    judgement = pilot.parse_judgement(results.pop(pilot.JUDGE_KEY, None))
    j = "none" if judgement is None else str(judgement)
    if not results:
        return f"{j}:no_answer"
    if verify(dict(results), list(row["sub_answers"]), TOLERANCE, units="benchmark")[0]:
        return f"{j}:right"
    short = [{**s, "value": str(row["shortcut_values"][s["sub"]])} for s in row["sub_answers"]
             if s["sub"] in (row.get("shortcut_values") or {})]
    if short and len(short) == len(row["sub_answers"]) and verify(dict(results), short, TOLERANCE, units="benchmark")[0]:
        return f"{j}:shortcut"
    return f"{j}:other"


def answer_after_judgement(fam: Family) -> dict:
    out = {}
    for arm in ("F", "A"):
        todo = []
        for fold in FOLDS:
            for s in fam.seeds:
                recs = fam.records(fam.arm(arm, fold, s), "judge")
                for i, r in recs.items():
                    row = fam.rows.get(i[len(PREFIX):])
                    if row and row["variant"] == "outside" and row["cluster_fold"] == OTHER[fold] and r.get("attempts"):
                        todo.append((r["attempts"][-1]["response"], row))
        with ThreadPoolExecutor(16) as pool:
            counts = Counter(pool.map(lambda p: classify(*p), todo))
        part = {"rows": len(todo), "counts": dict(counts)}
        for j in ("0", "1"):
            n = sum(v for k, v in counts.items() if k.startswith(j + ":"))
            part[f"judged_{j}"] = {"n": n, **{k: (counts[f"{j}:{k}"] / n if n else None)
                                              for k in ("right", "shortcut", "other", "no_answer")}}
        out[arm] = part
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--report", type=Path, required=True, help="the family's explore.x_report output")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.seeds = args.seeds or ([42, 43] if args.name == "qwen3-8b" else [42])
    fam = Family(args)
    r = {"family": args.name, "seeds": args.seeds, "note": "读结果后补充的描述（X-27），不改任何判定",
         "held_out_by_variant": held_out_by_variant(fam),
         "zero_hit_exceptions": zero_hit_exceptions(json.loads(args.report.read_text())),
         "truncation_split": truncation_split(fam),
         "answer_after_judgement": answer_after_judgement(fam)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    pct = lambda x: "-" if x is None else f"{100 * x:5.1f}"
    for arm, v in r["held_out_by_variant"].items():
        print(f"held-out {arm:2s}", "  ".join(f"{k} {pct(d['mean'])} [{pct(d['ci95'][0])}, {pct(d['ci95'][1])}]"
                                              for k, d in v.items()))
    for e in r["zero_hit_exceptions"]:
        print("Z exception", e["mechanism"], e["at_or_above_50"])
    for arm, v in r["answer_after_judgement"].items():
        print(f"answer after judgement {arm}:", "  ".join(
            f"judged {j} n={v[f'judged_{j}']['n']} right {pct(v[f'judged_{j}']['right'])} shortcut "
            f"{pct(v[f'judged_{j}']['shortcut'])}" for j in ("0", "1")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
