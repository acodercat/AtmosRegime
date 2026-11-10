"""Pre-training baselines, computed from the stored API-model runs — no model calls.

    uv run python -m atmoscoder_rl.baselines [--experiments DIR] [--out baselines.json]

The runs were produced by AtmosCoder-Bench's sweeps and imported into `experiments/`
(`docs/IMPORTED_RUNS.md`).

Every configuration this project might fine-tune has already been measured by the
benchmark, three runs apiece. Re-slicing those stored runs onto the frozen splits gives
the "before" column for free, and — more importantly — fixes it *before* any training
happens, so the comparison cannot drift.

Reported per model:

- **accuracy per split** (mean ± SD over the available runs) for `train`, `test_id`,
  `test_core_only`, `test_book`. `train` is listed because a policy's starting accuracy on
  its own training problems is the thing an RL curve moves first; it is not a test number.
- **Trap Gap** — control accuracy minus trap accuracy, computed *only over traps whose
  control that model solves* (the "clean" gap the benchmark paper reports). Controls come
  from `core_code`, traps from `trap/`.
- **Scaffolding dependence** — original minus stripped accuracy on the held-out
  `probe_scaffolding` subset (70 of the paper's 169 pairs).
- **Cross-domain accuracy** — the 131 non-atmospheric problems, single run.

Grading vintage: verdicts are read from each record's stored `passed` flag, i.e. the rule
in force when that run was graded. `atmoscoder_rl.check_reward` measures the drift this
introduces (~0.2% of records, offset-scale temperature answers only). Post-training
numbers produced by this project's own reward path will therefore differ from these
baselines by that much before any training effect is counted.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .datasets import load_splits, load_traps
from .paths import EXPERIMENTS_DIR
from .runs import is_gradable, load_runs
from .stats import mean_sd

CORE_SPLITS = ("train", "test_id", "test_core_only", "test_book")


def _stored_verdicts(experiments: Path, exp_id: str) -> dict[str, list[dict[str, bool]]]:
    """Config -> its repetitions, each {problem id: stored passed}, gradable records only."""
    return {config: [{i: bool(r["passed"]) for i, r in rep.items() if is_gradable(r)} for rep in reps]
            for config, reps in load_runs(experiments, exp_id).items()}


def _accuracy(run: dict[str, bool], ids: list[str]) -> float | None:
    graded = [run[i] for i in ids if i in run]
    return sum(graded) / len(graded) if graded else None


def split_accuracies(core_runs: dict[str, list[dict]], splits: dict) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for model, runs in sorted(core_runs.items()):
        entry: dict[str, Any] = {"n_runs": len(runs)}
        for name in CORE_SPLITS:
            ids = splits["splits"][name]["ids"]
            per_run = [a for a in (_accuracy(run, ids) for run in runs) if a is not None]
            mean, sd = mean_sd(per_run)
            entry[name] = {"mean": mean, "sd": sd, "n": len(ids), "runs": len(per_run)}
        table[model] = entry
    return table


def trap_gaps(core_runs: dict[str, list[dict]], trap_runs: dict[str, list[dict]]) -> dict[str, dict]:
    """Clean Trap Gap: control accuracy − trap accuracy, over traps whose control passes.

    Per run pair (control run i, trap run i), restrict to traps whose parent that run
    solved, then the gap is the fraction of those the model loses on the trap. Averaged
    over runs; models absent from either directory are skipped.
    """
    parent_of = {trap["id"]: trap["parent_id"] for trap in load_traps()}
    gaps: dict[str, dict] = {}
    for model, runs in sorted(trap_runs.items()):
        controls = core_runs.get(model)
        if not controls:
            continue
        per_run: list[float] = []
        for index, trap_run in enumerate(runs):
            control_run = controls[min(index, len(controls) - 1)]
            eligible = [tid for tid, pid in parent_of.items()
                        if tid in trap_run and control_run.get(pid)]
            if not eligible:
                continue
            per_run.append(1.0 - sum(trap_run[tid] for tid in eligible) / len(eligible))
        if per_run:
            mean, sd = mean_sd(per_run)
            gaps[model] = {"mean": mean, "sd": sd, "runs": len(per_run)}
    return gaps


def scaffolding_dependence(experiments: Path, ids: list[str]) -> dict[str, dict]:
    """Original − stripped accuracy on the held-out probe subset."""
    original = _stored_verdicts(experiments, "scaffolding_ablation/original_code")
    stripped = _stored_verdicts(experiments, "scaffolding_ablation/stripped_code")
    table: dict[str, dict] = {}
    for model in sorted(set(original) & set(stripped)):
        deltas, originals, strippeds = [], [], []
        for index, run in enumerate(original[model]):
            other = stripped[model][min(index, len(stripped[model]) - 1)]
            before, after = _accuracy(run, ids), _accuracy(other, ids)
            if before is None or after is None:
                continue
            originals.append(before)
            strippeds.append(after)
            deltas.append(before - after)
        if deltas:
            mean, sd = mean_sd(deltas)
            table[model] = {"original": mean_sd(originals)[0], "stripped": mean_sd(strippeds)[0],
                            "delta": mean, "sd": sd, "runs": len(deltas), "n": len(ids)}
    return table


def cross_domain(experiments: Path) -> dict[str, dict]:
    per_model: dict[str, dict[str, float]] = defaultdict(dict)
    for domain_dir in sorted(p for p in (experiments / "cross_domain").glob("*") if p.is_dir()):
        for model, runs in _stored_verdicts(experiments, f"cross_domain/{domain_dir.name}").items():
            if runs:
                per_model[model][domain_dir.name] = sum(runs[0].values()) / len(runs[0])
    return {
        model: {"by_domain": domains,
                "overall": statistics.fmean(domains.values()) if domains else float("nan")}
        for model, domains in sorted(per_model.items())
    }


def _pct(value: float | None) -> str:
    return "  —  " if value is None or value != value else f"{value * 100:5.1f}"


def render(report: dict) -> str:
    lines = ["", "=" * 96, "PRE-TRAINING BASELINES (from AtmosCoder-Bench stored runs)", "=" * 96, ""]
    header = f"{'model':<32}{'runs':>5}" + "".join(f"{name:>16}" for name in CORE_SPLITS)
    lines += ["accuracy by split (%, mean ± SD over runs)", header, "-" * len(header)]
    for model, entry in report["splits"].items():
        row = f"{model:<32}{entry['n_runs']:>5}"
        for name in CORE_SPLITS:
            cell = entry[name]
            row += f"{_pct(cell['mean'])} ±{cell['sd'] * 100:4.1f}"
        lines.append(row)
    counts = report["splits"] and next(iter(report["splits"].values()))
    if counts:
        lines.append(f"{'n problems':<32}{'':>5}" +
                     "".join(f"{counts[name]['n']:>16}" for name in CORE_SPLITS))

    if report["trap_gap"]:
        lines += ["", "clean Trap Gap (%, control−trap over traps whose control passes)",
                  f"{'model':<32}{'gap':>10}{'sd':>8}{'runs':>7}", "-" * 57]
        for model, entry in sorted(report["trap_gap"].items(), key=lambda kv: -kv[1]["mean"]):
            lines.append(f"{model:<32}{_pct(entry['mean']):>10}{entry['sd'] * 100:8.1f}"
                         f"{entry['runs']:>7}")

    if report["scaffolding"]:
        n = next(iter(report["scaffolding"].values()))["n"]
        lines += ["", f"scaffolding dependence on the held-out probe (n={n}, %)",
                  f"{'model':<32}{'original':>10}{'stripped':>10}{'Δ':>8}{'sd':>7}", "-" * 67]
        for model, entry in sorted(report["scaffolding"].items(), key=lambda kv: -kv[1]["delta"]):
            lines.append(f"{model:<32}{_pct(entry['original']):>10}{_pct(entry['stripped']):>10}"
                         f"{_pct(entry['delta']):>8}{entry['sd'] * 100:7.1f}")

    if report["cross_domain"]:
        domains = sorted({d for e in report["cross_domain"].values() for d in e["by_domain"]})
        lines += ["", "cross-domain accuracy (%, single run)",
                  f"{'model':<32}{'overall':>9}" + "".join(f"{d[:12]:>14}" for d in domains),
                  "-" * (41 + 14 * len(domains))]
        for model, entry in sorted(report["cross_domain"].items(),
                                   key=lambda kv: -kv[1]["overall"]):
            row = f"{model:<32}{_pct(entry['overall']):>9}"
            row += "".join(f"{_pct(entry['by_domain'].get(d)):>14}" for d in domains)
            lines.append(row)

    lines += ["", "Grading vintage: stored `passed` flags; see the module docstring.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR,
                        help="the experiments directory (default: this repository's)")
    parser.add_argument("--out", type=Path, help="also write the report as JSON to this file")
    args = parser.parse_args()

    experiments = args.experiments.resolve()
    splits = load_splits()
    core_runs = _stored_verdicts(experiments, "core_code")
    if not core_runs:
        print(f"no core_code runs under {experiments}")
        return 1

    report = {
        "source": str(experiments),
        "splits": split_accuracies(core_runs, splits),
        "trap_gap": trap_gaps(core_runs, _stored_verdicts(experiments, "trap")),
        "scaffolding": scaffolding_dependence(experiments, splits["splits"]["probe_scaffolding"]["ids"]),
        "cross_domain": cross_domain(experiments),
    }
    print(render(report))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
