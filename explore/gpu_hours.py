"""GPU hours of every RL run whose endpoint the study evaluated (supplementary table; description, no inference).

    python -m explore.gpu_hours [--logs experiments/_gpu_hours] [--out results/paper/gpu_hours.json --md results/paper/gpu_hours.md]
    python -m explore.gpu_hours --selftest

Sources (copied from `$ATMOS_ROOT/logs`, hashes in `experiments/_gpu_hours/MANIFEST.sha256` and `opd_runs/`):
- DAPO runs (X, the R arm, DOSE, DOSE-text, Ministral DOSE): the chains' "attempt exited (rc N) at step S; wall clock M min
  in total" lines. M accumulates over the attempts of one run inside one chain, so a run's wall clock is M of its last exit;
  the one run resumed by a different chain file (Ministral F1 seed 43, DM-21) adds the attempt it resumed from (RESUMED).
  8 GPUs (`train/dapo_qwen3_8b.sh`). This reproduces the 30-row table of PREREG_X (anchor checked below).
- OPD: each `run-*.log`'s "OPD <name>: ... student N GPUs teacher ... (TP k)" start line and "OPD <name> exited" line;
  GPUs = N + k. A run log without an exit line is a discarded attempt.
Not counted: warm-start SFT (under one hour each, PREREG_X), evaluation, pass@64 sampling, smokes, and discarded
attempts other than those listed under `discarded` (a lower bound: chains that stopped before writing an exit line leave none).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXIT = re.compile(r"^\[([\d-]+ [\d:]+)\] (\S+) attempt exited \(rc (\d+)\) at step (\d+); wall clock (\d+) min in total")
OPD_START = re.compile(r"^\[([\d-]+ [\d:]+)\] OPD (\S+): arm .* student (\d+) GPUs teacher .* \(TP (\d+)\)")
OPD_EXIT = re.compile(r"^\[([\d-]+ [\d:]+)\] OPD (\S+) exited (\d+)")
#: run -> (chain file, exit time) of the attempt it resumed from in another chain file
RESUMED = {"dose-ministral3-8b-F1-WA-s43": ("dose/rl-ministral3-8b-F1-s43.sharenet.log", "2026-11-02 18:48:52")}
#: PREREG_X run table (2026-10-22 record): total GPU hours of the 30 X runs per family
ANCHOR_X = {"qwen3-8b": 1147, "qwen35-9b": 1442}
DAPO_GPUS = 8


def study(name: str) -> str:
    if name.startswith("x-") and "-R-" in name:
        return "X R arm"
    if name.startswith("x-qwen35-9b") and name.endswith("-s43"):
        return "X Qwen3.5-9B seed 43"
    if name.startswith("x-"):
        return "X"
    return {"dose-qwen3": "DOSE", "dtext-": "DOSE-text", "dose-minis": "Ministral DOSE", "opd-qwen3": "OPD"}[
        next(p for p in ("dose-qwen3", "dtext-", "dose-minis", "opd-qwen3") if name.startswith(p))]


def dapo_runs(chain_dir: Path) -> tuple[list[dict], list[dict]]:
    exits = []
    for f in sorted(chain_dir.rglob("*.log")):
        rel = str(f.relative_to(chain_dir))
        for line in f.read_text(errors="replace").splitlines():
            if m := EXIT.match(line):
                exits.append({"time": m[1], "run": m[2], "rc": int(m[3]), "step": int(m[4]), "wall_min": int(m[5]), "file": rel})
    by_run = defaultdict(list)
    for e in exits:
        by_run[e["run"]].append(e)
    runs, discarded = [], []
    for name, es in sorted(by_run.items()):
        es.sort(key=lambda e: e["time"])
        last = es[-1]
        counted = [last]
        if name in RESUMED:
            f, t = RESUMED[name]
            counted.insert(0, next(e for e in es if e["file"] == f and e["time"] == t))
        wall = sum(e["wall_min"] for e in counted)
        runs.append({"run": name, "study": study(name), "gpus": DAPO_GPUS, "end_step": last["step"], "end_rc": last["rc"],
                     "wall_h": round(wall / 60, 2), "gpu_h": round(wall / 60 * DAPO_GPUS, 1), "files": sorted({e["file"] for e in counted})})
        # discarded: per (run, chain file) the largest cumulative counter, for files whose attempts were not counted
        for f in sorted({e["file"] for e in es} - {e["file"] for e in counted}):
            m = max(e["wall_min"] for e in es if e["file"] == f)
            discarded.append({"run": name, "file": f, "wall_h": round(m / 60, 2), "gpu_h": round(m / 60 * DAPO_GPUS, 1)})
        if name in RESUMED:
            f = RESUMED[name][0]
            before = [e for e in es if e["file"] == f and e["time"] < RESUMED[name][1]]
            assert not before, f"{name}: attempts before the resumed one in {f}"
    return runs, discarded


def opd_runs(lines_file: Path) -> tuple[list[dict], list[dict]]:
    blocks, cur = {}, None
    for line in lines_file.read_text().splitlines():
        if line.startswith("### "):
            cur = line.split()[1]
            blocks[cur] = {"sha256": line.split()[2], "start": None, "exit": None}
        elif m := OPD_START.match(line):
            blocks[cur]["start"] = (m[1], m[2], int(m[3]) + int(m[4]))
        elif m := OPD_EXIT.match(line):
            blocks[cur]["exit"] = (m[1], int(m[3]))
    runs, discarded = [], []
    for f, b in sorted(blocks.items()):
        t0, name, gpus = b["start"]
        if b["exit"] is None:
            discarded.append({"run": name, "file": f"opd/{f}", "wall_h": None, "gpu_h": None, "note": "no exit line"})
            continue
        h = (datetime.fromisoformat(b["exit"][0]) - datetime.fromisoformat(t0)).total_seconds() / 3600
        runs.append({"run": name, "study": "OPD", "gpus": gpus, "end_step": None, "end_rc": b["exit"][1],
                     "wall_h": round(h, 2), "gpu_h": round(h * gpus, 1), "files": [f"opd/{f}"]})
    return runs, discarded


def summarise(runs: list[dict]) -> dict:
    by = defaultdict(lambda: {"runs": 0, "gpu_h": 0.0})
    for r in runs:
        by[r["study"]]["runs"] += 1
        by[r["study"]]["gpu_h"] += r["gpu_h"]
    return {k: {"runs": v["runs"], "gpu_h": round(v["gpu_h"])} for k, v in sorted(by.items())}


def selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "a.log").write_text("[2026-10-01 00:00:00] x-qwen3-8b-F1-A-s42 attempt exited (rc 1) at step 0; wall clock 4 min in total\n"
                                 "[2026-10-01 06:00:00] x-qwen3-8b-F1-A-s42 attempt exited (rc 0) at step 120; wall clock 360 min in total\n")
        (p / "b.log").write_text("[2026-10-02 06:00:00] x-qwen3-8b-F1-A-s42 attempt exited (rc 0) at step 120; wall clock 300 min in total\n")
        runs, disc = dapo_runs(p)
        assert len(runs) == 1 and runs[0]["gpu_h"] == 40.0 and runs[0]["files"] == ["b.log"]
        assert disc == [{"run": "x-qwen3-8b-F1-A-s42", "file": "a.log", "wall_h": 6.0, "gpu_h": 48.0}]
        (p / "o.txt").write_text("### run-S.log ab\n[2026-10-30 10:00:00] OPD opd-qwen3-8b-F1-S-s42: arm S fold F1 seed 42 student 4 GPUs teacher /m (TP 1) loss k\n"
                                 "[2026-10-30 13:00:00] OPD opd-qwen3-8b-F1-S-s42 exited 0\n"
                                 "### run-S.attempt1.log cd\n[2026-10-29 10:00:00] OPD opd-qwen3-8b-F1-S-s42: arm S fold F1 seed 42 student 4 GPUs teacher /m (TP 1) loss k\n")
        o, od = opd_runs(p / "o.txt")
        assert o[0]["gpu_h"] == 15.0 and len(od) == 1
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=ROOT / "experiments/_gpu_hours")
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper/gpu_hours.json")
    ap.add_argument("--md", type=Path, default=ROOT / "results/paper/gpu_hours.md")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    d_runs, d_disc = dapo_runs(args.logs / "chain")
    o_runs, o_disc = opd_runs(args.logs / "opd_runs/opd_run_lines.txt")
    runs = d_runs + o_runs
    x = [r for r in runs if r["study"] == "X"]
    for fam, want in ANCHOR_X.items():
        got = sum(r["gpu_h"] for r in x if r["run"].startswith(f"x-{fam}-"))
        if abs(got - want) > 2:
            raise SystemExit(f"anchor: X {fam} {got:.0f} GPU h, PREREG_X table {want}")
    summary = summarise(runs)
    total = sum(v["gpu_h"] for v in summary.values())
    out = {"definition": __doc__.split("\n\n")[0], "runs": runs, "by_study": summary, "total_runs": len(runs), "total_gpu_h": total,
           "discarded_logged": d_disc + o_disc}
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    lines = ["# GPU hours per RL run (`explore/gpu_hours.py`)", "",
             "Runs whose endpoint was evaluated. Wall clock summed over the attempts of the run; GPU h = wall × GPUs (DAPO 8; OPD student + teacher).",
             "Warm-start SFT (< 1 h each), evaluation and pass@64 sampling are not included. OPD run logs carry no step at exit; all 12 ended at step 120 (chain gate lines for 10, merge logs of `global_step_120` for P seed 42).", "",
             "| study | runs | GPU h |", "|---|--:|--:|"]
    lines += [f"| {k} | {v['runs']} | {v['gpu_h']:,} |" for k, v in summary.items()] + [f"| **total** | **{len(runs)}** | **{total:,}** |", "",
              "| run | study | GPUs | last step logged | exit rc | wall h | GPU h |", "|---|---|--:|--:|--:|--:|--:|"]
    lines += [f"| `{r['run']}` | {r['study']} | {r['gpus']} | {r['end_step'] if r['end_step'] is not None else '—'} | {r['end_rc']} | {r['wall_h']:.1f} | {r['gpu_h']:.0f} |"
              for r in sorted(runs, key=lambda r: (r["study"], r["run"]))]
    disc_h = sum(r["gpu_h"] or 0 for r in out["discarded_logged"])
    lines += ["", f"Discarded attempts with a logged wall clock (lower bound): {len(out['discarded_logged'])} entries, {disc_h:,.0f} GPU h; listed in the JSON."]
    args.md.write_text("\n".join(lines) + "\n")
    for k, v in summary.items():
        print(f"{k:24s} {v['runs']:3d} runs {v['gpu_h']:7,} GPU h")
    print(f"total {len(runs)} runs {total:,} GPU h; discarded (logged) {disc_h:,.0f} GPU h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
