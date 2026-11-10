"""PREREG_X X-29 section 3: does a run degenerate the way Qwen3.5 F2-F seed 42 did? Either criterion triggers the
overlong-penalty check (F2-F seeds 42 and 43 again with OVERLONG=1):
  training: mean of response_length/clip_ratio over the run's last 20 logged steps >= 0.3;
  evaluation: share of gradable final.json rows whose last call used all 8,192 tokens >= 5% (x_report's truncation).

    python -m explore.x_overlong_check --logs $ATMOS_ROOT/logs --report r35.json \
        --runs x-qwen35-9b-F2-F-s43 x-qwen35-9b-F2-F-s42

The training lines are the run's own (X-26): its attempt logs written after X-16 started, read the way
explore/x_curve_sources.py chooses them and explore/export_curves.py parses them. A run with no report row or fewer than
20 logged steps is "incomplete", never "not degenerated".
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean

from explore.export_curves import parse
from explore.x_curve_sources import KEPT, X16_START

CLIP, LAST, CLIP_MIN, TRUNC_MIN = "response_length/clip_ratio", 20, 0.3, 0.05


def training(logs: Path, run: str) -> dict:
    attempt = lambda p: int(re.search(r"\.attempt(\d+)\.nohup$", p.name).group(1))
    files = [p for p in sorted((logs / "x").glob(f"{run}.attempt*.nohup"), key=attempt)
             if run in KEPT or p.stat().st_mtime >= X16_START]
    steps: dict[int, dict[str, float]] = {}
    for p in files:
        steps.update(parse(p))
    vals = [steps[s][CLIP] for s in sorted(steps) if CLIP in steps[s]][-LAST:]
    return {"files": [p.name for p in files], "steps": len(vals), "last_step": max(steps) if steps else None,
            "clip_last20": fmean(vals) if len(vals) == LAST else None}


def check(logs: Path, report: dict, run: str) -> dict:
    t = training(logs, run)
    n, m = report.get("truncation", {}).get("per_label", {}).get(run, {}).get("final", [None, 0])
    share = n / m if m else None
    if t["clip_last20"] is None or share is None:
        verdict = "不完整"
    elif t["clip_last20"] >= CLIP_MIN or share >= TRUNC_MIN:
        verdict = "退化：做超长惩罚检验"
    else:
        verdict = "未退化：不做"
    return {**t, "final_truncated": [n, m], "final_share": share, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True, help="x_report output for the family")
    ap.add_argument("--runs", nargs="+", default=["x-qwen35-9b-F2-F-s43"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = json.loads(args.report.read_text())
    out = {run: check(args.logs, report, run) for run in args.runs}
    for run, r in out.items():
        clip = "-" if r["clip_last20"] is None else f"{r['clip_last20']:.3f}"
        share = "-" if r["final_share"] is None else f"{100 * r['final_share']:.1f}%"
        print(f"{run:24s} steps {r['steps']} (last {r['last_step']}) clip last {LAST} {clip} (>= {CLIP_MIN})"
              f"  final truncated {r['final_truncated'][0]}/{r['final_truncated'][1]} {share} (>= {TRUNC_MIN:.0%})"
              f"  -> {r['verdict']}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
