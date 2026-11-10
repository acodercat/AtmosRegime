"""Which log lines are each PREREG_X run's own (X-26), as a manifest for `explore/export_curves.py --manifest`.

    python -m explore.x_curve_sources --logs $ATMOS_ROOT/logs --out x_sources.json [--glob 'x-*-s4[23]']

RL runs: the run's attempt logs (logs/x/<run>.attempt<N>.nohup, in attempt order) written after X-16 started
(2026-10-17 15:00 HKT); the three runs X-16 kept (Qwen3-8B fold F2 seed 42, arms A, D, F) read all of theirs. A redone run's
attempt N overwrote the superseded run's attempt N, and an attempt file the redone run never reached keeps its old mtime,
so the cut-off drops exactly the superseded lines. The shared logs/<run>.log is not read: it holds both runs.
Warm starts, one source each: Qwen3-8B F1 was redone in X-16 (its nohup); Qwen3-8B F2 and Qwen3.5-9B F1 were trained once
(their nohup); Qwen3.5-9B F2's nohup belongs to the 10-16 SFT on 113 that X-16 did not use (X-22), so the used one (114,
10-15) is the first segment of logs/sft-x-qwen35-9b-F2.log. Every file chosen or dropped is printed with its mtime and steps.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

X16_START = datetime(2026, 10, 17, 15, 0, tzinfo=timezone(timedelta(hours=8))).timestamp()
KEPT = {"x-qwen3-8b-F2-A-s42", "x-qwen3-8b-F2-D-s42", "x-qwen3-8b-F2-F-s42"}
WARM = {"sft-x-qwen3-8b-F1": ("x/sft-x-qwen3-8b-F1.nohup", "all"),
        "sft-x-qwen3-8b-F2": ("x/sft-x-qwen3-8b-F2.nohup", "all"),
        "sft-x-qwen35-9b-F1": ("x/sft-x-qwen35-9b-F1.nohup", "all"),
        "sft-x-qwen35-9b-F2": ("sft-x-qwen35-9b-F2.log", "first")}
STEP = re.compile(r"step:(\d+) - ")


def steps(path: Path) -> list[int]:
    return [int(m.group(1)) for line in path.read_text(errors="replace").splitlines() for m in [STEP.search(line)] if m]


def stamp(path: Path) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--glob", default="x-*-s4[23]")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    attempt = lambda p: int(re.search(r"\.attempt(\d+)\.nohup$", p.name).group(1))
    runs = sorted({p.name.split(".attempt")[0] for p in (args.logs / "x").glob(f"{args.glob}.attempt*.nohup")})
    manifest = {}
    for run in runs:
        files = sorted((args.logs / "x").glob(f"{run}.attempt*.nohup"), key=attempt)
        chosen = [p for p in files if run in KEPT or p.stat().st_mtime >= X16_START]
        for p in files:
            s = steps(p)
            print(f"{run:26s} {p.name.split('.')[-2]:9s} {stamp(p)} steps {s[0] if s else '-'}..{s[-1] if s else '-'}"
                  f" {'use' if p in chosen else 'DROP (before X-16)'}")
        if chosen:
            manifest[run] = {"files": [str(p.relative_to(args.logs)) for p in chosen], "segment": "all"}
    for run, (f, seg) in WARM.items():
        p = args.logs / f
        s = steps(p)
        print(f"{run:26s} {f:32s} {stamp(p)} steps {len(s)} lines, segment {seg}")
        manifest[run] = {"files": [f], "segment": seg}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(len(manifest), "runs ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
