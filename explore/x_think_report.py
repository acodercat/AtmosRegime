"""PREREG_X X-34 / X-35: thinking-on evaluation of seed-42 endpoints, read with x_report's own functions. Numbers only.

    python -m explore.x_think_report --name qwen3-8b --eval-dir $ATMOS_ROOT/experiments/_x_eval_think \
        --final $ATMOS_ROOT/experiments/_inputs/kc/final.json --inputs $ATMOS_ROOT/experiments/_inputs/x_eval --out think_qwen3-8b.json

Reads <eval-dir>/<label>/final.json (explore/x_eval.sh THINK=1: the inside / outside rows of final.json) for the base and
the A, F, WA, R endpoints of both folds, seed 42. Every arm, and every difference, is against the thinking-on base:
(1) training-fold outside accuracy; (2) held-out outside and inside accuracy (endpoint - base); (3) Spearman of base
pass@64 (thinking off, the X-18 file) against thinking-on outside accuracy on the training-fold mechanisms of the A and F
arms (x_report.p1_mechanisms). WA is compared with the thinking-on base too, not with its own start W_f (not evaluated
thinking-on), so WA - base is not the pre-registered transfer quantity. Per label: records; rows that ended in "prompt too
long" (a repair prompt that no longer fits the context, X-36: counted as wrong, not excluded); other API errors (excluded,
the label is then missing); rows still ungradable after the attempts; rows graded from the reasoning text because the
answer was empty; rows whose last call used all 28,672 tokens. Clusters, regrade and intervals are x_report's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.runs import is_gradable
from explore import x_report
from explore.x_report import FOLDS, Family, acc_measure, diff, passk_table, spearman_ci, summary

ARMS = ("A", "F", "WA", "R")
SEED = 42
MAX_TOKENS = 28672   # models.toml x-eval-think-*
OVERFLOW = "prompt too long"   # atmoscoder_rl.evaluation.episode: PromptTooLongError on a repair call


def overflow(r: dict) -> bool:
    return str(r.get("error") or "").startswith(OVERFLOW)


class ThinkFamily(Family):
    """x_report.Family whose final verdicts also count overflowed rows as wrong (X-36)."""
    def final_passed(self, label: str) -> dict | None:
        passed = super().final_passed(label)
        if passed is None:
            return None
        over = {i: False for i, r in (self.records(label, "final") or {}).items() if overflow(r)}
        return {**over, **passed}


def labels(fam: Family) -> list[str]:
    return [fam.base()] + [fam.arm(a, f, SEED) for a in ARMS for f in FOLDS]


def missing(fam: Family) -> list[str]:
    want = [i for i, r in fam.rows.items() if r["variant"] in ("inside", "outside")]
    out = []
    for lab in labels(fam):
        done = {i for i, r in (fam.records(lab, "final") or {}).items() if not r.get("error") or overflow(r)}
        if not set(want) <= done:
            out.append(lab)
    return out


def side(fam: Family, label_of, side_: str, measure) -> dict:
    """cluster -> measure, both folds pooled: each fold's templates on `side_` scored by label_of(fold)."""
    out = {}
    for fold in FOLDS:
        out.update(measure(label_of(fold), fam.templates(fold, side_)))
    return out


def report(fam: Family, passk_path: Path) -> dict:
    out = {"labels": labels(fam), "missing": missing(fam) + ([] if passk_path.exists() else [f"pass@64 file {passk_path}"]),
           "numbers_only": x_report.NUMBERS,
           "note": "every difference is against the thinking-on base (WA too); pass@64 is the thinking-off X-18 sample"}
    base = lambda fold: fam.base()
    for name, variant, sd in (("train_outside", "outside", "train"), ("held_outside", "outside", "held"),
                              ("held_inside", "inside", "held")):
        m = acc_measure(fam, variant)
        b = side(fam, base, sd, m)
        block = {"base": summary(b)}
        for a in ARMS:
            arm = side(fam, lambda fold, a=a: fam.arm(a, fold, SEED), sd, m)
            block[a] = summary(arm)
            block[f"{a}-base"] = diff(arm, b)
        out[name] = block
    passk = passk_table(passk_path) if passk_path.exists() else {}
    _, acc = x_report.p1_mechanisms(fam)
    out["p1_spearman_pass64_vs_outside"] = {
        a: spearman_ci([(passk[t][64], acc[a][SEED][t]) for t in sorted(acc[a][SEED]) if 64 in passk.get(t, {})]) for a in ("A", "F")}
    per = {}
    for lab in labels(fam):
        recs = list((fam.records(lab, "final") or {}).values())
        g = [r for r in recs if is_gradable(r) and r.get("attempts")]
        last = [r["attempts"][-1] for r in g]
        per[lab] = {"records": len(recs), "overflow_counted_wrong": sum(map(overflow, recs)),
                    "other_errors": sum(bool(r.get("error")) and not overflow(r) for r in recs),
                    "ungradable_after_attempts": sum(bool(r.get("failed_reason")) for r in g),
                    "graded_from_reasoning": sum(bool(a.get("reasoning")) and a.get("response") == a.get("reasoning") for a in last),
                    "truncated": sum(a.get("usage", {}).get("completion_tokens", 0) >= MAX_TOKENS for a in last)}
    out["rows"] = per
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval_think")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--passk", type=Path, help="default experiments/_x/<name>/passk_final_outside.x18.jsonl, as x_report")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.seeds = [SEED]
    passk = args.passk or EXPERIMENTS_DIR / "_x" / args.name / "passk_final_outside.x18.jsonl"
    r = report(ThinkFamily(args), passk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: r[k] for k in ("missing", "train_outside")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
