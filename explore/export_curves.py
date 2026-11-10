"""Training curves for the paper: every per-step metric line of a verl console log -> one CSV per run.

    python -m explore.export_curves --logs $ATMOS_ROOT/logs --runs x-qwen3-8b-F1-A-s42 ... --out-dir results/curves
    python -m explore.export_curves --logs $ATMOS_ROOT/logs --glob 'x-*' --out-dir results/curves

Columns: step, then every metric the run logged (training reward, response length, entropy, KL, clip fraction, timing,
evicted samples; SFT loss; and the X-5 training-side probe's answer / judge / counterfactual means under val-aux).
A later line for the same step (a resumed run) replaces the earlier one.

PREREG_X runs (X-26: the shared logs/<run>.log of a run redone in X-16 also holds the superseded run's lines) are read from
an explicit manifest instead, written on the server by `explore/x_curve_sources.py`: per run, the files under --logs to read
in order and the segment ("all", or "first" = up to the first step that goes backwards). --val-override takes
`explore/x_val_regrade.py`'s output: those validation points are replaced by their regraded values, marked val_regraded = 1.

    python -m explore.export_curves --logs $ATMOS_ROOT/logs --manifest x_sources.json \
        --val-override results/curves/x_val_regraded.json --out-dir results/curves
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

LINE = re.compile(r"step:(\d+) - (.*)")
PAIR = re.compile(r"([\w/@.\-]+):(?:np\.(?:float|int)64\()?(-?[\d.eE+\-]+|nan|inf)\)?")


def parse(path: Path, first_segment: bool = False) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    last = -1
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        if first_segment and step < last:
            break
        last = step
        vals = {}
        for k, v in PAIR.findall(m.group(2)):
            if not re.search(r"[A-Za-z]", k):   # "19:21" from a timestamp printed onto the same console line, not a metric
                continue
            try:
                vals[k] = float(v)
            except ValueError:
                pass
        if vals:
            rows[step] = vals
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--glob")
    ap.add_argument("--manifest", type=Path, help="explore/x_curve_sources.py output: run -> files and segment")
    ap.add_argument("--val-override", type=Path, help="explore/x_val_regrade.py output")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
    override = json.loads(args.val_override.read_text()) if args.val_override else {}
    names = list(args.runs) + ([p.stem for p in sorted(args.logs.glob(f"{args.glob}.log"))] if args.glob else []) + list(manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name in manifest:
            rows = {}
            for f in manifest[name]["files"]:
                rows.update(parse(args.logs / f, manifest[name].get("segment") == "first"))
        else:
            rows = parse(args.logs / f"{name}.log")
        for o in override.values():
            if o["run"] == name:
                if o["step"] not in rows:
                    raise SystemExit(f"{name}: no logged step {o['step']} to override")
                rows[o["step"]].update(o["values"], val_regraded=1.0)
        if not rows:
            print(f"{name}: no metric lines"); continue
        keys = sorted({k for r in rows.values() for k in r})
        with (args.out_dir / f"{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", *keys])
            for step in sorted(rows):
                w.writerow([step, *(rows[step].get(k, "") for k in keys)])
        print(f"{name}: {len(rows)} steps, {len(keys)} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
