"""Re-execute every stored `solve()` and check it still reproduces its stored answer.

    uv run python -m atmoscoder_rl.check_solvers                     # the assembled corpus
    uv run python -m atmoscoder_rl.check_solvers --set all           # every problem file
    uv run python -m atmoscoder_rl.check_solvers --input new.json    # a candidate set, before vendoring

This is the first gate for any dataset built or edited here. A problem whose committed
solver no longer reproduces its own answer is not a hard problem, it is a broken record:
it will be scored wrong for every model forever, and — because the reward in this repo is
that same execution rule — it will teach a wrong answer during training.

Two classes of defect are checked, because only the first is self-announcing:

**Reproduction.** The solver runs and its return value matches `sub_answers` within the
grading tolerance, using the same engine the reward uses.

**Structure.** Ids unique, required fields present, `sub_answers` well formed, every
`parent_id` resolvable, and — reported separately — sub-answers with an empty `unit`.
An empty expected unit silently disables unit-aware grading: a correct answer given in a
commensurate unit is then marked wrong, and no solver check can see it, because the
stored solver and the stored answer agree with each other. That defect shipped once in
this corpus (`cross_domain/*.json`, 209 sub-answers, fixed upstream 2026-09-06), so it is
surfaced here by construction rather than left to be noticed.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .datasets import PROBLEM_SETS, Problem, load_problems
from .engine import run_solver
from .grading import TOLERANCE, verify
from .sandbox import isolation, run_solver_sandboxed



def check_structure(problems: list[Problem]) -> tuple[list[str], list[str]]:
    """Defects that make a record unusable, and separately the empty-unit warning."""
    errors: list[str] = []
    seen: set[str] = set()
    ids = {p.get("id") for p in problems}
    empty_units = 0

    for index, problem in enumerate(problems):
        pid = problem.get("id")
        where = pid or f"<record {index}, no id>"
        if not pid:
            errors.append(f"{where}: missing id")
        elif pid in seen:
            errors.append(f"{where}: duplicate id")
        else:
            seen.add(pid)

        for field in ("problem", "code", "sub_answers"):
            if not problem.get(field):
                errors.append(f"{where}: missing or empty {field!r}")

        subs = problem.get("sub_answers")
        if isinstance(subs, list):
            for sub in subs:
                if not isinstance(sub, dict) or "value" not in sub or "sub" not in sub:
                    errors.append(f"{where}: malformed sub_answer {sub!r}")
                    continue
                if str(sub.get("unit", "")).strip() == "":
                    empty_units += 1

        parent = problem.get("parent_id")
        # a parent outside this file is fine (traps in a trap-only view); an unresolvable
        # one inside a file that does contain parents is not
        if parent and parent not in ids and any(p.get("role") == "core" for p in problems):
            errors.append(f"{where}: parent_id {parent!r} not in this set")

    warnings = []
    if empty_units:
        warnings.append(
            f"{empty_units} sub-answer(s) have an empty unit — unit-aware grading is "
            f"disabled for those, so a correct answer in a commensurate unit will be "
            f"marked wrong")
    return errors, warnings


def verify_problems(problems: list[Problem], tolerance: float, workers: int,
                    verbose: bool, sandboxed: bool = False) -> tuple[int, int, list[tuple[str, Any]]]:
    """Returns (passed, errored, failures); failures include both wrong and unrunnable.

    With `sandboxed`, solvers run through `sandbox.run_solver_sandboxed` — the path the
    reward actually uses. Any problem that passes under the engine but fails here is a
    hardening regression: during training it would score 0 for a correct answer.
    """
    runner = run_solver_sandboxed if sandboxed else run_solver

    def one(problem: Problem):
        pid = problem.get("id", "<no id>")
        try:
            ran, results, info = runner(problem["code"])
        except Exception as exc:                       # a malformed record, not a bad solver
            return pid, "error", f"could not execute: {exc!r}"[:120]
        if not ran:
            return pid, "error", f"exec error: {str(info)[:120]}"
        reproduced, details = verify(results, problem["sub_answers"], tolerance, units="benchmark")
        return pid, ("pass" if reproduced else "fail"), details

    passed = errored = 0
    failures: list[tuple[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pid, verdict, details in pool.map(one, problems):
            if verdict == "pass":
                passed += 1
            else:
                if verdict == "error":
                    errored += 1
                failures.append((pid, details))
            if verbose:
                print(f"  {pid}: {verdict.upper()}")
    return passed, errored, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="dataset", default="problems",
                        choices=[*PROBLEM_SETS, "all"], help="which set to verify (default: problems)")
    parser.add_argument("--input", type=Path, default=None,
                        help="verify an explicit json file instead of a --set")
    parser.add_argument("--tolerance", "-t", type=float, default=TOLERANCE,
                        help="relative tolerance (default: grading.TOLERANCE)")
    parser.add_argument("--workers", "-w", type=int, default=8, help="parallel solvers")
    parser.add_argument("--verbose", "-v", action="store_true", help="show every problem")
    parser.add_argument("--sandbox", action="store_true",
                        help="run solvers through the hardened reward sandbox instead of "
                             "the plain engine, to catch hardening regressions")
    args = parser.parse_args()

    if args.sandbox:
        print(f"sandbox isolation: {isolation()}")

    if args.input:
        targets = {str(args.input): json.loads(args.input.read_text(encoding="utf-8"))}
    elif args.dataset == "all":
        targets = {name: loader() for name, loader in PROBLEM_SETS.items() if name != "problems"}
        targets = {"problems": load_problems(), **targets}
    else:
        targets = {args.dataset: PROBLEM_SETS[args.dataset]()}

    total_failures = 0
    for name, problems in targets.items():
        print(f"\n=== {name} ({len(problems)} records, tolerance {args.tolerance:.1%})")
        errors, warnings = check_structure(problems)
        for warning in warnings:
            print(f"  note: {warning}")
        for error in errors[:20]:
            print(f"  STRUCTURE: {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more structural defects")

        passed, errored, failures = verify_problems(problems, args.tolerance,
                                                    args.workers, args.verbose, args.sandbox)
        wrong = len(failures) - errored
        print(f"  passed {passed}/{len(problems)} | wrong answer {wrong} | unrunnable {errored}")
        for pid, details in failures[:10]:
            print(f"    FAIL {pid}: {str(details)[:140]}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
        total_failures += len(failures) + len(errors)

    print(f"\n{'FAILED' if total_failures else 'OK'} — {total_failures} problem(s)")
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
