"""Replacement-given versions of the regime traps — docs/PREREG_GIVEN_VERSIONS.md, in code.

    uv run python -m atmoscoder_rl.given_contrast [--experiments DIR] [--exp-id prereg_given] [--out FILE]

Each regime trap T has a version G = T plus one sentence stating the replacement relation, and a
parent P. All three were answered in the same run files (`explore/build_given_input.py`).
Per item and model group, every quantity is a mean over (configuration, repetition), and the RL
group weights its four checkpoints equally.

- H1: d = [G right, T wrong] − [T right, G wrong]; one-sided exact sign test over items with
  d ≠ 0 that G is solved more often than T.
- H2: Δ = accuracy(P) − accuracy(G) per item; the mean over items and its 90 % bootstrap
  interval (items resampled, 10,000 draws, seed 0). "No gap" iff the interval lies within
  [−0.10, 0.10].
- Descriptive: recognition (trap right given parent right) for T and for G; the same with the
  traps flagged as partly disclosed removed (FLAGGED).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .paths import EXPERIMENTS_DIR
from .runs import is_gradable, load_runs
from .stats import sign_test_greater_p

GROUPS = {"base": ("qwen3-8b.think-off",),
          "RL": ("rl-s120.think-off", "seed1-s120.think-off", "seed2-s120.think-off", "seed3-s120.think-off")}
MARGIN = 0.10
ALPHA = 0.05
#: Traps whose own text, by the authors of the given versions (2026-09-28), already discloses the
#: change wholly or partly. Fixed before the run; used only in the sensitivity analysis.
FLAGGED = ("trap_7.4_gen", "trap_fp_13.2_gen", "trap_dn_15.2_gen", "trap_air_68_gen", "trap_p_96_gen",
           "trap_ry_12.2_gen", "trap_holton_50_gen", "trap_holton_5_gen", "trap_1.1_gen", "trap_2.10_gen",
           "trap_11.1_gen", "trap_6.2_gen", "trap_6.6_gen", "trap_7.5_gen")


def _triples(items: list[dict]) -> dict[str, tuple[str, str]]:
    """trap id -> (given id, parent id)."""
    given = {i["trap_id"]: i["id"] for i in items if i["id"].startswith("given_")}
    return {i["id"]: (given[i["id"]], i["parent_id"]) for i in items
            if "parent_id" in i and not i["id"].startswith("given_")}


def per_item(runs: dict[str, list[dict]], triples: dict[str, tuple[str, str]],
             configs: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """trap id -> group means of: T, G, P accuracy; d (H1); recognition counts."""
    acc: dict[str, dict[str, list[float]]] = {}
    for t, (g, p) in triples.items():
        per_config = []
        for c in configs:
            obs = []
            for rep in runs.get(c, []):
                rt, rg, rp = rep.get(t), rep.get(g), rep.get(p)
                if is_gradable(rt) and is_gradable(rg) and is_gradable(rp):
                    obs.append((bool(rt["passed"]), bool(rg["passed"]), bool(rp["passed"])))
            if obs:
                n = len(obs)
                per_config.append({
                    "T": sum(o[0] for o in obs) / n, "G": sum(o[1] for o in obs) / n,
                    "P": sum(o[2] for o in obs) / n,
                    "d": sum((o[1] and not o[0]) - (o[0] and not o[1]) for o in obs) / n,
                    "T_given_P": [sum(o[0] and o[2] for o in obs), sum(o[2] for o in obs)],
                    "G_given_P": [sum(o[1] and o[2] for o in obs), sum(o[2] for o in obs)]})
        if per_config:
            k = len(per_config)
            acc[t] = {key: sum(x[key] for x in per_config) / k for key in ("T", "G", "P", "d")}
            acc[t]["T_given_P"] = [sum(x["T_given_P"][i] for x in per_config) for i in (0, 1)]
            acc[t]["G_given_P"] = [sum(x["G_given_P"][i] for x in per_config) for i in (0, 1)]
    return acc


def bootstrap_mean_ci(values: list[float], draws: int = 10_000, seed: int = 0, level: float = 0.90) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(draws))
    lo = means[int((1 - level) / 2 * draws)]
    hi = means[int((1 + level) / 2 * draws) - 1]
    return lo, hi


def summarise(items: dict[str, dict[str, float]]) -> dict[str, Any]:
    ids = sorted(items)
    up = sum(items[i]["d"] > 0 for i in ids)
    down = sum(items[i]["d"] < 0 for i in ids)
    delta = [items[i]["P"] - items[i]["G"] for i in ids]
    lo, hi = bootstrap_mean_ci(delta) if delta else (float("nan"), float("nan"))
    rec_t = [sum(items[i]["T_given_P"][k] for i in ids) for k in (0, 1)]
    rec_g = [sum(items[i]["G_given_P"][k] for i in ids) for k in (0, 1)]
    mean = lambda key: sum(items[i][key] for i in ids) / len(ids) if ids else float("nan")
    return {
        "items": len(ids), "accuracy": {"trap": mean("T"), "given": mean("G"), "parent": mean("P")},
        "H1_given_over_trap": {"given_better": up, "trap_better": down, "ties": len(ids) - up - down,
                               "one_sided_p": sign_test_greater_p(up, down) if up + down else None},
        "H2_parent_minus_given": {"mean": sum(delta) / len(delta) if delta else None, "ci90": [lo, hi],
                                  "no_gap": bool(-MARGIN <= lo and hi <= MARGIN)},
        "recognition": {"trap": rec_t, "given": rec_g}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, required=True, help="the input file build_given_input wrote")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--exp-id", default="prereg_given")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    triples = _triples(json.loads(args.items.read_text()))
    runs = load_runs(args.experiments, args.exp_id)
    missing = [c for cs in GROUPS.values() for c in cs if c not in runs]
    if missing:
        print(f"configurations missing: {missing}")
        return 1
    report: dict[str, Any] = {}
    for group, configs in GROUPS.items():
        items = per_item(runs, triples, configs)
        report[group] = summarise(items)
        report[group + " (flagged traps removed)"] = summarise({i: v for i, v in items.items() if i not in FLAGGED})
    rl = report["RL"]
    report["verdict"] = {
        "H1": "supported" if rl["H1_given_over_trap"]["one_sided_p"] is not None
              and rl["H1_given_over_trap"]["one_sided_p"] <= ALPHA else "not supported",
        "H2": "no gap (equivalence shown)" if rl["H2_parent_minus_given"]["no_gap"] else "equivalence not shown"}
    print(json.dumps(report, indent=2, default=float))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
