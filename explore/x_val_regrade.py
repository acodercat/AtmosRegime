"""Regraded validation points for the training curves (docs/PREREG_X.md X-24: the points the grading fault hit are drawn
with regraded values). Every row of a point's val_generations file is scored again with the current reward
(`verl_reward_regime.compute_score`, deterministic since X-24) and averaged the way verl logs a validation step
(val-aux/<source>/<key>/mean@1 for every per-row metric, val-core/<source>/reward/mean@1).

    python -m explore.x_val_regrade --points x-qwen3-8b-F2-F-s43:40 x-qwen3-8b-F1-A-s43:40 \
        --controls x-qwen3-8b-F2-F-s43:20 --root $ATMOS_ROOT --out results/curves/x_val_regraded.json

Two checks, both must hold or nothing is written: for every point and control, the mean of the stored per-row values
equals what the run's log recorded at that step (the aggregation is verl's); for every control, the regraded rows equal
the stored ones (the reward is unchanged). The output holds the regraded means of the --points only.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.verl_reward_regime import compute_score

SOURCE = "atmoscoder/train_probe"
SKIP = {"gts", "output", "input", "step", "uid", "score", "reward"}
LINE = re.compile(r"step:(\d+) - (.*)")
PAIR = re.compile(r"([\w/@.\-]+):(?:np\.float64\()?(-?[\d.eE+\-]+|nan|inf)\)?")


def logged(root: Path, run: str, step: int) -> dict[str, float]:
    """The last log line for `step` among the run's attempt logs (a resumed attempt replaces an earlier one)."""
    out = {}
    for f in sorted(glob.glob(str(root / "logs" / "x" / f"{run}.attempt*.nohup"))):
        for line in open(f, errors="replace"):
            m = LINE.search(line)
            if m and int(m.group(1)) == step:
                vals = dict((k, float(v)) for k, v in PAIR.findall(m.group(2)))
                if any(k.startswith("val-") for k in vals):
                    out = vals
    return out


def means(rows: list[dict]) -> dict[str, float]:
    keys = [k for k in rows[0] if k not in SKIP]
    out = {f"val-aux/{SOURCE}/{k}/mean@1": fmean(float(r[k]) for r in rows) for k in keys}
    out[f"val-aux/{SOURCE}/score/mean@1"] = fmean(float(r["score"]) for r in rows)
    out[f"val-core/{SOURCE}/reward/mean@1"] = fmean(float(r["score"]) for r in rows)
    return out


def point(root: Path, spec: str) -> dict:
    run, step = spec.rsplit(":", 1)
    rows = [json.loads(l) for l in open(root / "checkpoints" / run / "val_generations" / f"{step}.jsonl")]
    with ThreadPoolExecutor(16) as pool:
        new = list(pool.map(lambda r: compute_score(SOURCE, r["output"], r["gts"]), rows))
    rescored = [{**r, **{k: v for k, v in n.items()}} for r, n in zip(rows, new)]
    stored, regraded, log = means(rows), means(rescored), logged(root, run, int(step))
    agg_ok = all(abs(stored[k] - log[k]) < 1e-6 for k in stored if k in log) and any(k in log for k in stored)
    changed = sum(any(abs(float(r[k]) - float(n[k])) > 1e-9 for k in n if k in r and k not in SKIP | {"score"})
                  or abs(float(r["score"]) - float(n["score"])) > 1e-9 for r, n in zip(rows, new))
    return {"run": run, "step": int(step), "rows": len(rows), "aggregation_matches_log": agg_ok, "rows_changed": changed,
            "stored": stored, "regraded": regraded, "logged": {k: log[k] for k in stored if k in log}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", required=True, help="run:step to replace")
    ap.add_argument("--controls", nargs="+", required=True, help="run:step expected unchanged")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = {s: point(args.root, s) for s in [*args.points, *args.controls]}
    ok = True
    for s, r in res.items():
        control = s in args.controls
        print(f"{s:32s} rows {r['rows']} aggregation==log {r['aggregation_matches_log']} rows changed {r['rows_changed']}"
              f"{' (control: must be 0)' if control else ''}  score stored {r['stored'][f'val-aux/{SOURCE}/score/mean@1']:.4f}"
              f" -> regraded {r['regraded'][f'val-aux/{SOURCE}/score/mean@1']:.4f}")
        ok &= r["aggregation_matches_log"] and (r["rows_changed"] == 0 if control else True)
    if not ok:
        print("CHECK FAILED: nothing written")
        return 1
    out = {s: {"run": res[s]["run"], "step": res[s]["step"], "values": res[s]["regraded"], "rows_changed": res[s]["rows_changed"]}
           for s in args.points}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
