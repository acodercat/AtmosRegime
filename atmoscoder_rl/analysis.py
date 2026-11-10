"""Every number this project reports, regenerated from the stored runs.

Nothing here calls a model: it reads the run files under `experiments/` (see `runs.py`) and
re-derives the verdicts, so a grading change (units, tolerance) does not need a re-run.

    uv run python -m atmoscoder_rl.analysis [--experiments DIR] paper
    uv run python -m atmoscoder_rl.analysis [--experiments DIR] control
    uv run python -m atmoscoder_rl.analysis [--experiments DIR] traps [--rl <run-file pattern>]
    uv run python -m atmoscoder_rl.analysis [--experiments DIR] pairs [--control]
    uv run python -m atmoscoder_rl.analysis [--experiments DIR] frontier [--models ...] [--all-parents]
    uv run python -m atmoscoder_rl.analysis passk --base a.json --rl-file b.json

Verdicts are re-derived from each record's `details` (expected vs actual **with units**)
rather than trusting the stored `passed`. That matters: the cross-domain sets were
evaluated before their units were restored, and their stored verdicts under-credit a
correct answer given in a commensurate unit.

Multiple repetitions of the same configuration are reduced by **majority of available
reps**, not majority of three — a configuration with one rep would otherwise score zero
everywhere. The base model has 3 reps; RL checkpoints have 1.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from collections.abc import Iterable
from pathlib import Path

from .capture import shortcut_majority
from .datasets import load_core, load_cross_domain, load_scaffolding, load_splits, load_traps
from .paths import EXPERIMENTS_DIR
from .runs import Rep, config_name, is_gradable, load_reps, regrade
from .stats import chi_square_2x2, exact_binomial_p, majority
from .traps import TRAP_EXPERIMENTS, adjudication, arm_ids, trap_index


# --------------------------------------------------------------------------- loading

def _splits() -> dict[str, list[str]]:
    return {name: entry["ids"] for name, entry in load_splits()["splits"].items()}


def _expected(source: str) -> dict[str, list[dict]]:
    """Problem id -> sub_answers, from the vendored corpus (the unit-fixed one)."""
    if source == "core":
        problems = load_core()
    elif source == "traps":
        problems = load_traps()
    elif source == "cross_domain":
        problems = load_cross_domain()
    elif source == "scaffolding":
        original, stripped = load_scaffolding()
        problems = original + stripped
    else:
        raise ValueError(source)
    return {p["id"]: p["sub_answers"] for p in problems}


# --------------------------------------------------------------------------- grading


def _verdicts(runs: list[Rep], expected: dict[str, list[dict]],
              ids: Iterable[str] | None = None) -> dict[str, bool]:
    """Majority over the available reps, per problem.

    Returns {} when no run file matched. Callers that pair two sides must check for that:
    an empty side is indistinguishable downstream from "answered everything wrong", and a
    paired count built on it is not weak evidence, it is fabricated. `_require` below is
    the guard; a filename pattern that matches nothing is the usual cause (cross-domain
    keeps one base file, `qwen3-8b-local.json`, while the trap sets keep three reps
    `qwen3-8b-local.run{1,2,3}.json`, and a pattern written for one silently returns
    nothing for the other).
    """
    if not runs:
        return {}
    keys = set(runs[0])
    union = set(runs[0])
    for r in runs[1:]:
        keys &= set(r)
        union |= set(r)
    if union - keys:
        print(f"  warning: {len(union - keys)} id(s) present in some repetitions but not all; "
              f"dropped from the majority", flush=True)
    if ids is not None:
        keys &= set(ids)
    out = {}
    for i in sorted(keys):
        votes = [regrade(r[i], expected.get(i)) for r in runs if is_gradable(r[i])]
        if votes:
            out[i] = majority(votes)
    return out


def _require(verdicts: dict[str, bool], what: str) -> dict[str, bool]:
    """Fail loudly when a side of a paired comparison has no measurements at all."""
    if not verdicts:
        raise SystemExit(f"analysis: no stored measurements for {what} — "
                         f"check the filename pattern against what the experiment holds")
    return verdicts


def _rate(v: dict[str, bool]) -> float:
    return 100.0 * sum(v.values()) / len(v) if v else float("nan")


# --------------------------------------------------------------------------- reports

#: The untrained base model's run files: three repetitions everywhere but cross-domain.
BASE_RUNS = "qwen3-8b-local*.json"

#: (label, exp_id, corpus, split name or None for "every id in the file")
PROBES = [
    ("test_id",            "core_code",                              "core",         "test_id"),
    ("test_core_only",     "core_code",                              "core",         "test_core_only"),
    ("test_book",          "core_code",                              "core",         "test_book"),
    ("probe_trap (ctrl)",  "core_code",                              "core",         "probe_trap"),
    ("traps 67",           "trap",                                   "traps",        None),
    ("traps exp (clean)",  "trap_expansion",                         "traps",        "CLEAN"),
    ("scaf original",      "scaffolding_ablation/original_code",     "scaffolding",  "probe_scaffolding"),
    ("scaf stripped",      "scaffolding_ablation/stripped_code",     "scaffolding",  "probe_scaffolding"),
    ("cross_domain",       "cross_domain/*",                         "cross_domain", None),
]


def _probe_ids(split: str | None, sp: dict[str, list[str]]) -> list[str] | None:
    if split is None:
        return None
    if split == "CLEAN":
        # Expansion traps whose parent is held out, minus the records upstream adjudicated
        # as asserting an impossible quantity. Leaving those in put 3 of them inside a
        # reported row.
        train = set(sp["train"])
        return [t["id"] for t in load_traps()
                if t.get("expansion") and t["parent_id"] not in train
                and not adjudication(t).get("magnitude_defect")]
    return sp[split]


def _column(experiments: Path, pattern: str, sp: dict[str, list[str]]) -> dict[str, float]:
    """One model's accuracy on every probe."""
    out = {}
    for label, exp_id, corpus, split in PROBES:
        expected = _expected(corpus)
        if "*" in exp_id:  # cross-domain is one exp_id per domain
            runs_by_domain = [load_reps(experiments, str(d.relative_to(experiments)), pattern)
                              for d in sorted(experiments.glob(exp_id)) if d.is_dir()]
            runs_by_domain = [r for r in runs_by_domain if r]
            if not runs_by_domain:
                out[label] = float("nan")
                continue
            merged: dict[str, bool] = {}
            for runs in runs_by_domain:
                part = _verdicts(runs, expected)
                clash = set(part) & set(merged)
                assert not clash, f"{label}: id(s) in two domains, one would overwrite: {sorted(clash)[:3]}"
                merged.update(part)
            out[label] = _rate(merged)
            continue
        runs = load_reps(experiments, exp_id, pattern)
        out[label] = _rate(_verdicts(runs, expected, _probe_ids(split, sp))) if runs else float("nan")
    return out


def _table(experiments: Path, columns: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    sp = _splits()
    table = {name: _column(experiments, pattern, sp) for name, pattern in columns}
    if all(v != v for column in table.values() for v in column.values()):   # every cell NaN
        raise SystemExit(f"analysis: no stored runs match {[p for _, p in columns]} under "
                         f"{experiments} — see docs/IMPORTED_RUNS.md for where they live")
    return table


def _print_table(cols: dict[str, dict[str, float]], baseline: str | None = None) -> None:
    names = list(cols)
    width = max(len(l) for l, *_ in PROBES) + 2
    print(f"{'probe':<{width}}" + "".join(f"{n:>16}" for n in names))
    for label, *_ in PROBES:
        row = f"{label:<{width}}"
        for n in names:
            v = cols[n].get(label, float("nan"))
            if baseline and n != baseline:
                b = cols[baseline].get(label, float("nan"))
                row += f"{v:>10.1f}{v - b:>+6.1f}"
            else:
                row += f"{v:>16.1f}"
        print(row)


def cmd_paper(args: argparse.Namespace) -> None:
    """Base vs each seed's s120, paper protocol, unit-aware."""
    cols = [("base (3 rep)", BASE_RUNS), ("run2 s120", "dapo-qwen3-8b-s120.json")]
    cols += [(f"seed{i} s120", f"seed{i}-s120.json") for i in (1, 2, 3)]
    t = _table(args.experiments, cols)
    _print_table(t, baseline="base (3 rep)")
    print("\nacross the RL runs (Δ vs the shared base):")
    base = t["base (3 rep)"]
    print(f"{'probe':<22}{'mean Δ':>9}{'sd':>7}{'n':>4}")
    for label, *_ in PROBES:
        ds = [t[n][label] - base[label] for n, _ in cols[1:]
              if t[n].get(label) == t[n].get(label) and base.get(label) == base.get(label)]
        if not ds:
            continue
        sd = statistics.stdev(ds) if len(ds) > 1 else float("nan")
        print(f"{label:<22}{statistics.mean(ds):>+9.1f}{sd:>7.1f}{len(ds):>4}")


def cmd_control(args: argparse.Namespace) -> None:
    """Real reward vs the information-free control at the same step."""
    t = _table(args.experiments, [("base (3 rep)", BASE_RUNS),
                                  ("real s120", "dapo-qwen3-8b-s120.json"),
                                  ("random s120", "ctrl-random-s120.json")])
    _print_table(t, baseline="base (3 rep)")


def cmd_traps(args: argparse.Namespace) -> None:
    """Trap families: accuracy, the parent's accuracy (the control), and shortcut capture.

    Spans all four trap experiments. Shortcut capture is the majority over the available
    repetitions, the same reduction every other number here uses -- reading only the first
    repetition made the base column a single sample while its accuracy column was a
    majority of three.
    """
    sp = _splits()
    train = set(sp["train"])
    index = trap_index()
    expected_traps = {i: t["sub_answers"] for i, (t, _) in index.items()}
    expected_core = _expected("core")

    by_exp = collections.defaultdict(list)
    for i, (t, exp) in index.items():
        by_exp[exp].append(i)

    for exp_id in sorted(by_exp):
        ids_here = by_exp[exp_id]
        base = load_reps(args.experiments, exp_id, BASE_RUNS)
        rl = load_reps(args.experiments, exp_id, args.rl)
        if not base or not rl:
            print(f"\n=== {exp_id}: no stored runs ===")
            continue
        vb = _verdicts(base, expected_traps, ids_here)
        vr = _verdicts(rl, expected_traps, ids_here)
        pb = _verdicts(load_reps(args.experiments, "core_code", BASE_RUNS), expected_core)
        pr = _require(_verdicts(load_reps(args.experiments, "core_code", args.rl), expected_core),
                      f"parents under {args.rl}")
        groups = collections.defaultdict(list)
        for i in sorted(set(vb) & set(vr)):
            t = index[i][0]
            arm = "seen_parent" if t["parent_id"] in train else "clean"
            if adjudication(t).get("magnitude_defect"):
                arm = "defect"
            groups[(t["trap_type"], arm)].append(i)
        print(f"\n=== {exp_id}  ({len(set(vb) & set(vr))} measured of {len(ids_here)})  [{args.rl}] ===")
        print(f"{'family / arm':<30}{'n':>4}{'trap s0':>9}{'RL':>7}{'parent s0':>11}{'RL':>7}"
              f"{'gap s0':>8}{'RL':>6}{'shortcut':>11}")
        for (fam, arm), ids in sorted(groups.items()):
            n = len(ids)
            ts = sum(vb[i] for i in ids) / n
            tr = sum(vr[i] for i in ids) / n
            par = [index[i][0]["parent_id"] for i in ids]
            ps = sum(bool(pb.get(x)) for x in par) / n
            pv = sum(bool(pr.get(x)) for x in par) / n
            sb = sum(shortcut_majority([r[i] for r in base if i in r], index[i][0]) for i in ids)
            sr = sum(shortcut_majority([r[i] for r in rl if i in r], index[i][0]) for i in ids)
            print(f"{fam + ' / ' + arm:<30}{n:>4}{ts * 100:>8.1f}%{tr * 100:>6.1f}%"
                  f"{ps * 100:>10.1f}%{pv * 100:>6.1f}%{(ps - ts) * 100:>8.0f}{(pv - tr) * 100:>6.0f}"
                  f"{sb:>7} ->{sr:>3}")


#: The arms of the paired test, in the order they are reported. `regime` is the claim;
#: `regime_given` is the internal control that holds the physics fixed and removes only the
#: recognition step; `geometry` is the family control; `regime_xdom` is supporting evidence
#: and is reported apart because 11 of its 14 parents are soil, so on its own it would be a
#: soil-mechanics result rather than a non-atmospheric one.
PAIR_ARMS = [
    ("regime: must recognise the failure", "regime"),
    ("regime: replacement handed over",    "regime_given"),
    ("geometry: same method, extra step",  "geometry"),
    ("regime: cross-domain (supporting)",  "regime_xdom"),
]

#: Where each arm's parent controls are measured. Cross-domain parents are not core
#: problems, so their control comes from the cross-domain experiment; reading them from
#: `core_code` returns nothing and, before `_require` existed, produced a fabricated count.
PARENT_SOURCE = {"regime_xdom": "cross_domain", "regime": "core", "regime_given": "core",
                 "geometry": "core"}


def _parent_verdicts(experiments: Path, source: str, pattern: str) -> dict[str, bool]:
    if source == "core":
        return _verdicts(load_reps(experiments, "core_code", pattern), _expected("core"))
    merged: dict[str, bool] = {}
    expected = _expected("cross_domain")
    for d in sorted((experiments / "cross_domain").glob("*")):
        if not d.is_dir():
            continue
        runs = load_reps(experiments, f"cross_domain/{d.name}", pattern)
        if runs:
            merged.update(_verdicts(runs, expected))
    return merged


def cmd_pairs(args: argparse.Namespace) -> None:
    """A trap and its untouched parent are one problem under two conditions, so the test is
    McNemar on the discordant pairs, not a comparison of two rates. The base row is the
    same test on the untrained model: McNemar after training says the parent beats the trap
    now, not that it beats it by more than it did before."""
    sp = _splits()
    train = set(sp["train"])
    index = trap_index()
    expected_traps = {i: t["sub_answers"] for i, (t, _) in index.items()}
    runs_model = [("base", BASE_RUNS), ("run2", "dapo-qwen3-8b-s120.json")]
    runs_model += [(f"seed{i}", f"seed{i}-s120.json") for i in (1, 2, 3)]
    if args.control:
        runs_model.append(("random", "ctrl-random-s120.json"))
    summary = {}

    for title, kind in PAIR_ARMS:
        ids = arm_ids(kind, index, train)
        by_exp = collections.defaultdict(list)
        for i in ids:
            by_exp[index[i][1]].append(i)
        parent_of = PARENT_SOURCE[kind]
        print(f"\n{title}   ({len(ids)} pairs from {', '.join(sorted(by_exp))})")
        print(f"    {'run':>8}{'trap':>8}{'parent':>9}{'P+T-':>7}{'P-T+':>7}{'p':>10}")
        pooled_b = pooled_c = 0
        for name, pattern in runs_model:
            tv: dict[str, bool] = {}
            for exp, subset in by_exp.items():
                runs = load_reps(args.experiments, exp, pattern)
                if runs:
                    tv.update(_verdicts(runs, expected_traps, subset))
            if not tv:
                continue
            pv = _require(_parent_verdicts(args.experiments, parent_of, pattern),
                          f"{title} parents ({parent_of}) under {pattern}")
            here = [i for i in ids if i in tv]
            b = sum(1 for i in here if pv.get(index[i][0]["parent_id"]) and not tv[i])
            c = sum(1 for i in here if not pv.get(index[i][0]["parent_id"]) and tv[i])
            print(f"    {name:>8}{100 * sum(tv[i] for i in here) / len(here):>7.1f}%"
                  f"{100 * sum(bool(pv.get(index[i][0]['parent_id'])) for i in here) / len(here):>8.1f}%"
                  f"{b:>7}{c:>7}{exact_binomial_p(b, c):>10.4f}")
            if name not in ("base", "random"):
                pooled_b += b
                pooled_c += c
        print(f"    {'POOLED':>8}{'':>8}{'':>9}{pooled_b:>7}{pooled_c:>7}{exact_binomial_p(pooled_b, pooled_c):>10.7f}"
              f"   (the four real-reward runs)")
        summary[kind] = (pooled_b, pooled_c)

    print("\ncontrasts between arms, on the pooled discordant counts:")
    for a, b_ in (("regime", "regime_given"), ("regime", "geometry"), ("regime_given", "geometry")):
        if a not in summary or b_ not in summary:
            continue
        (w, x), (y, z) = summary[a], summary[b_]
        print(f"  {a:<14} vs {b_:<14} {w}:{x} vs {y}:{z}   chi2(1) = {chi_square_2x2(w, x, y, z):.1f}")


def _reps_anywhere(experiments: Path, dirs: Iterable[str], name: str) -> list[Rep]:
    """Every run file for one configuration, across the directories it might live in."""
    reps = []
    for d in dirs:
        for pattern in (f"{name}.json", f"{name}.run*.json"):
            reps += load_reps(experiments, d, pattern)
    return reps


def _verdicts_partial(reps: list[Rep], expected: dict[str, list[dict]],
                      ids: Iterable[str]) -> tuple[dict[str, bool], dict[str, int]]:
    """Majority over the repetitions that actually measured each id, plus that count.

    `_verdicts` intersects the ids across repetitions, which is right for our own runs: a
    missing id there means a rep did not finish and the safe move is to drop it. The
    benchmark's sweeps are different — the trap expansion is still being back-filled, so
    coverage legitimately differs between reps and between models, and intersecting would
    silently discard most of the arm. Here an id is scored on whatever measured it, and the
    caller prints the repetition count so thin coverage is visible rather than assumed away.
    """
    out: dict[str, bool] = {}
    depth: dict[str, int] = {}
    for i in ids:
        votes = [regrade(r[i], expected.get(i)) for r in reps if i in r and is_gradable(r[i])]
        if votes:
            out[i] = majority(votes)
            depth[i] = len(votes)
    return out, depth


def _frontier_configs(experiments: Path) -> list[str]:
    """Every configuration that has both trap runs and the parent controls they need.

    A name here is a runner `--model` (a run file is `{name}[.run{N}].json`), so this picks
    up whatever the benchmark has evaluated without a list to keep in sync.
    """
    names = {config_name(f) for d in ("trap", "trap_expansion") for f in (experiments / d).glob("*.json")}
    have_parents = {config_name(f) for f in (experiments / "core_code").glob("*.json")}
    return sorted(n for n in names & have_parents if not n.startswith("qwen3-8b-local"))


def cmd_frontier(args: argparse.Namespace) -> None:
    """The same paired test, on models this project never trained.

    The regime deficit is the paper's environmental-science claim, and a deficit found only
    in one 8B model would be a fact about that model. These configurations come from the
    benchmark's own evaluations -- frontier APIs included -- so the question is whether a
    model that answers the parent gets its trap wrong for the same reason ours does.

    The arm definitions are the ones `pairs` uses, unchanged, so the rows are comparable to
    the RL table. The `parent outside train` filter is kept for that comparability although
    these models trained on none of our splits; `--all-parents` drops it and uses every
    adjudicated record instead.
    """
    sp = _splits()
    train = set() if args.all_parents else set(sp["train"])
    index = trap_index()
    expected_traps = {i: t["sub_answers"] for i, (t, _) in index.items()}
    expected_core = _expected("core")
    configs = args.models or _frontier_configs(args.experiments)
    if not configs:
        raise SystemExit("analysis: no configuration has both trap runs and core_code parents")

    for title, kind in PAIR_ARMS:
        if kind == "regime_xdom":
            continue                      # cross-domain parents are not part of the Bench sweep
        ids = arm_ids(kind, index, train)
        print(f"\n{title}   ({len(ids)} pairs defined)")
        print(f"    {'model':>28}{'pairs':>7}{'reps':>6}{'trap':>8}{'parent':>9}"
              f"{'P+T-':>7}{'P-T+':>7}{'p':>10}")
        for name in configs:
            tv, depth = _verdicts_partial(
                _reps_anywhere(args.experiments, TRAP_EXPERIMENTS, name), expected_traps, ids)
            if not tv:
                continue
            parents = {index[i][0]["parent_id"] for i in tv}
            pv, _ = _verdicts_partial(
                _reps_anywhere(args.experiments, ("core_code",), name), expected_core, parents)
            pv = _require(pv, f"{name} parents in core_code")
            here = [i for i in ids if i in tv and index[i][0]["parent_id"] in pv]
            if not here:
                continue
            b = sum(1 for i in here if pv[index[i][0]["parent_id"]] and not tv[i])
            c = sum(1 for i in here if not pv[index[i][0]["parent_id"]] and tv[i])
            reps = statistics.median(depth[i] for i in here)
            print(f"    {name:>28}{len(here):>7}{reps:>6.0f}"
                  f"{100 * sum(tv[i] for i in here) / len(here):>7.1f}%"
                  f"{100 * sum(pv[index[i][0]['parent_id']] for i in here) / len(here):>8.1f}%"
                  f"{b:>7}{c:>7}{exact_binomial_p(b, c):>10.4f}")
    print("\nA pair is counted only where that configuration measured both sides, so `pairs` is")
    print("that model's coverage of the arm and `reps` the median number of repetitions behind")
    print("each verdict. The benchmark's trap back-fill is still in progress.")


def cmd_passk(args: argparse.Namespace) -> None:
    """Coverage at k: does RL reach more problems than the base, or fewer?"""
    B = {r["id"]: r for r in json.loads(args.base.read_text()) if r.get("k") is not None}
    R = {r["id"]: r for r in json.loads(args.rl_file.read_text()) if r.get("k") is not None}
    ids = [i for i in B if i in R]
    print(f"{'split':<22}{'n':>5}{'base pass@k':>13}{'RL':>8}{'lost':>7}{'gained':>8}")
    tot = collections.Counter()
    for spl in sorted({B[i]["split"] for i in ids}):
        g = [i for i in ids if B[i]["split"] == spl]
        cb, cr = sum(B[i]["k"] > 0 for i in g), sum(R[i]["k"] > 0 for i in g)
        lost = sum(B[i]["k"] > 0 and R[i]["k"] == 0 for i in g)
        new = sum(B[i]["k"] == 0 and R[i]["k"] > 0 for i in g)
        tot.update(dict(n=len(g), cb=cb, cr=cr, lost=lost, new=new))
        print(f"{spl:<22}{len(g):>5}{cb / len(g) * 100:>12.1f}%{cr / len(g) * 100:>7.1f}%"
              f"{lost:>7}{new:>8}")
    n = tot["n"]
    print(f"{'all':<22}{n:>5}{tot['cb'] / n * 100:>12.1f}%{tot['cr'] / n * 100:>7.1f}%"
          f"{tot['lost']:>7}{tot['new']:>8}")
    print(f"\n{'base k band':>14}{'n':>5}{'base mean k':>13}{'RL mean k':>11}{'RL k=0':>9}")
    for lo, hi in ((0, 0), (1, 3), (4, 15), (16, 47), (48, 63), (64, 64)):
        g = [i for i in ids if lo <= B[i]["k"] <= hi]
        if not g:
            continue
        print(f"{f'{lo}-{hi}':>14}{len(g):>5}{sum(B[i]['k'] for i in g) / len(g):>13.1f}"
              f"{sum(R[i]['k'] for i in g) / len(g):>11.1f}"
              f"{sum(R[i]['k'] == 0 for i in g) / len(g) * 100:>8.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR,
                        help="the experiments directory (default: this repository's)")
    commands = parser.add_subparsers(dest="cmd", required=True)
    commands.add_parser("paper").set_defaults(fn=cmd_paper)
    commands.add_parser("control").set_defaults(fn=cmd_control)
    traps = commands.add_parser("traps")
    traps.add_argument("--rl", default="dapo-qwen3-8b-s120.json", help="the RL run-file pattern")
    traps.set_defaults(fn=cmd_traps)
    pairs = commands.add_parser("pairs")
    pairs.add_argument("--control", action="store_true", help="also show the random-reward arm")
    pairs.set_defaults(fn=cmd_pairs)
    frontier = commands.add_parser("frontier")
    frontier.add_argument("--models", nargs="*",
                          help="runner --model names; default: every one with both sides")
    frontier.add_argument("--all-parents", action="store_true",
                          help="drop the `parent outside train` filter (these models trained on none of it)")
    frontier.set_defaults(fn=cmd_frontier)
    passk = commands.add_parser("passk")
    passk.add_argument("--base", type=Path, required=True, help="the base model's pass@k file")
    passk.add_argument("--rl-file", type=Path, required=True, help="the RL model's pass@k file")
    passk.set_defaults(fn=cmd_passk)
    args = parser.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
