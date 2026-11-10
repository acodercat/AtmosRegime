"""PREREG_X pre-read checks on training runs (X-25, X-30), as code; they were first run from one-off commands.

    # on the server (reads logs and checkpoints only)
    python -m explore.x_run_checks scan --root $ATMOS_ROOT --glob 'x-*-s4[23]' --dump-val val_exec.jsonl --out scan.json
    # locally, away from the training machines' sandbox (X-25)
    python -m explore.x_run_checks regrade --rows val_exec.jsonl --out regrade.json

scan, per run:
- step signature (X-24/X-25): a step whose evicted samples fall below 0.3x and mean response length below 0.7x the medians
  of its 6 + 6 neighbouring steps, among steps whose neighbourhood evicts > 50 (the run's own attempt logs, chosen as
  explore/x_curve_sources.py chooses them);
- persistent exec_error (X-30): the run's reward-diag rows from its latest "attempt 1" in the chain logs on (a redone run's
  start; an earlier run under the same name is excluded), exec_error over all calls, after the first 200,000 calls, and the
  worst clock hour (>= 5,000 calls) after them; flagged if after-200k > max(2x the family median, 2%) or worst hour > 15%;
- validation rows graded exec_error (every val_generations point), dumped for `regrade`.
A random-reward run (-R-) has no reward diag and evicts nothing, and its gradient never sees correctness, so a grading
fault cannot reach its training: the first two checks are reported as not applicable; its validation rows are regraded.

regrade: every dumped row scored again, twice, with the current reward (`verl_reward_regime.compute_score`); a row that
now runs is a candidate false failure. X-25: only F1-A-s43 and F2-F-s43 step 40 (X-24) had any.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from explore.export_curves import parse
from explore.x_curve_sources import KEPT, X16_START

TRAIN = re.compile(r"^\[(\S+ \S+)\] === train (x-\S+) \(.*attempt 1,")
EVICT, LENGTH = "training/filter_groups/evicted_samples", "response_length/mean"
SOURCE = "atmoscoder/train_probe"


def random_reward(run: str) -> bool:
    return "-R-" in run


def starts(logs: Path) -> dict[str, str]:
    """Latest "attempt 1" of each run in any chain log: the start of the run whose lines count."""
    out = {}
    for f in glob.glob(str(logs / "x" / "chain-*.log")):
        for line in open(f, errors="replace"):
            m = TRAIN.match(line)
            if m:
                out[m.group(2)] = max(out.get(m.group(2), ""), m.group(1))
    return out


def step_signature(logs: Path, run: str) -> dict:
    attempt = lambda p: int(re.search(r"\.attempt(\d+)\.nohup$", p.name).group(1))
    steps: dict[int, dict[str, float]] = {}
    for p in sorted((logs / "x").glob(f"{run}.attempt*.nohup"), key=attempt):
        if run in KEPT or p.stat().st_mtime >= X16_START:
            steps.update(parse(p))
    d = {s: (v[EVICT], v[LENGTH]) for s, v in steps.items() if EVICT in v and LENGTH in v}
    ks = sorted(d)
    if random_reward(run):
        return {"steps": len(ks), "applicable": False}
    bad = []
    for i, s in enumerate(ks):
        nb = [d[k] for k in ks[max(0, i - 6):i] + ks[i + 1:i + 7]]
        if len(nb) < 6:
            continue
        ev, rl = statistics.median(x[0] for x in nb), statistics.median(x[1] for x in nb)
        if ev > 50 and d[s][0] < 0.3 * ev and d[s][1] < 0.7 * rl:
            bad.append(s)
    return {"steps": len(ks), "last": ks[-1] if ks else None, "applicable": True, "signature_steps": bad}


def exec_error(logs: Path, run: str, t0: str | None) -> dict:
    if random_reward(run):
        return {"applicable": False}
    if t0 is None:
        return {"applicable": True, "start": None}
    rows = []
    for f in glob.glob(str(logs / "x" / "reward_diag" / f"{run}.*.jsonl")):
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if str(r.get("t", "")) >= t0:
                rows.append((str(r["t"]), int(r.get("n", 0)), int((r.get("outcome") or {}).get("exec_error", 0))))
    rows.sort()
    seen = calls = errors = late_calls = late_errors = 0
    hours = collections.defaultdict(lambda: [0, 0])
    for t, n, e in rows:
        calls += n; errors += e
        if seen >= 200_000:
            late_calls += n; late_errors += e
            hours[t[:13]][0] += n; hours[t[:13]][1] += e
        seen += n
    worst = max((100 * e / n for n, e in hours.values() if n >= 5000), default=0.0)
    return {"applicable": True, "start": t0, "calls": calls, "all_pct": 100 * errors / max(calls, 1),
            "after_200k_pct": 100 * late_errors / max(late_calls, 1), "worst_hour_pct": worst}


def val_rows(ckpt: Path, run: str) -> tuple[int, list[dict]]:
    files = sorted((ckpt / run / "val_generations").glob("*.jsonl"), key=lambda p: int(p.stem))
    out = []
    for p in files:
        for line in open(p):
            r = json.loads(line)
            if float(r.get("outcome/exec_error", 0)) == 1:
                out.append({"run": run, "step": int(p.stem), "output": r["output"], "gts": r["gts"]})
    return len(files), out


def scan(args) -> int:
    logs, ckpt = args.root / "logs", args.root / "checkpoints"
    t0 = starts(logs)
    runs = sorted(p.name for p in ckpt.glob(args.glob) if (p / "val_generations").is_dir())
    res, dump = {}, []
    for run in runs:
        points, rows = val_rows(ckpt, run)
        dump += rows
        res[run] = {"step_signature": step_signature(logs, run), "exec_error": exec_error(logs, run, t0.get(run)),
                    "val_points": points, "val_exec_error_rows": len(rows)}
    fam = collections.defaultdict(list)
    for run, r in res.items():
        if r["exec_error"].get("calls"):
            fam[run.split("-F")[0]].append(r["exec_error"]["after_200k_pct"])
    for run, r in res.items():
        e = r["exec_error"]
        if e.get("calls"):
            med = statistics.median(fam[run.split("-F")[0]])
            e["flag"] = e["after_200k_pct"] > max(2 * med, 2.0) or e["worst_hour_pct"] > 15
        s = r["step_signature"]
        sig = f"signature {len(s['signature_steps'])} {s['signature_steps'][:8]}" if s["applicable"] else "signature n/a (random reward)"
        ex = ("exec_error n/a (random reward)" if not e["applicable"] else "exec_error NO START LINE" if not e.get("start")
              else "exec_error no diag rows since its start" if not e["calls"]
              else f"exec_error all {e['all_pct']:4.1f}% after-200k {e['after_200k_pct']:4.1f}% worst hour {e['worst_hour_pct']:4.1f}%"
                   f"{'  <-- check' if e.get('flag') else ''}")
        print(f"{run:24s} steps {s['steps']:3d}  {sig:22s}  {ex}  val points {r['val_points']} exec_error rows {r['val_exec_error_rows']}")
    n_sig = sum(len(r["step_signature"].get("signature_steps", [])) for r in res.values())
    n_flag = sum(bool(r["exec_error"].get("flag")) for r in res.values())
    print(f"{len(res)} runs, {sum(r['val_points'] for r in res.values())} val points, {len(dump)} exec_error rows;"
          f" signature steps {n_sig}; persistent exec_error flags {n_flag}")
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    if args.dump_val:
        with open(args.dump_val, "w") as f:
            for r in dump:
                f.write(json.dumps(r) + "\n")
    return 0


def regrade(args) -> int:
    from atmoscoder_rl.verl_reward_regime import compute_score
    rows = [json.loads(line) for line in open(args.rows)]
    outcome = lambda res: next((k.split("/", 1)[1] for k, v in res.items() if k.startswith("outcome/") and v), "none")
    one = lambda r: outcome(compute_score(SOURCE, r["output"], r["gts"]))
    with ThreadPoolExecutor(args.workers) as pool:
        first, second = list(pool.map(one, rows)), list(pool.map(one, rows))
    per = collections.defaultdict(lambda: {"rows": 0, "run_first": 0, "run_second": 0, "steps": set()})
    for r, a, b in zip(rows, first, second):
        p = per[r["run"]]
        p["rows"] += 1
        p["run_first"] += a != "exec_error"
        p["run_second"] += b != "exec_error"
        if a != "exec_error" or b != "exec_error":
            p["steps"].add(r["step"])
    same = sum(a == b for a, b in zip(first, second))
    for run, p in sorted(per.items()):
        print(f"{run:24s} exec_error rows {p['rows']:4d}  run now: first {p['run_first']:3d} second {p['run_second']:3d}"
              f"  steps {sorted(p['steps'])}")
    print(f"{len(rows)} rows; the two regrades agree on {same}")
    out = {"rows": len(rows), "regrades_agree": same,
           "per_run": {k: {**v, "steps": sorted(v["steps"])} for k, v in per.items()}}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--glob", default="x-*-s4[23]")
    s.add_argument("--dump-val", type=Path)
    s.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("regrade")
    g.add_argument("--rows", type=Path, required=True)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    return scan(args) if args.cmd == "scan" else regrade(args)


if __name__ == "__main__":
    raise SystemExit(main())
