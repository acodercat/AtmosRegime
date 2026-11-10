"""Validate the reward function before it is ever used to train anything.

    uv run python -m atmoscoder_rl.check_reward
    uv run python -m atmoscoder_rl.check_reward --replay 3

Four checks, in increasing strength:

1. **Gold solvers.** Every stored `solve()` in every vendored problem file must score 1.0
   *through the reward path* (`check_solvers` checks the same solvers through the engine) —
   the data survived the copy and the reward's grading rule is intact.
2. **Adversarial inputs.** Prose, empty strings, malformed returns and infinite loops must
   score 0.0 without raising and without hanging. A reward function that can throw will
   kill a training step; one that can hang will kill a run.
3. **Replay against the vendored rule.** Real model completions from `experiments/` are
   re-graded here under `units="benchmark"` and must agree **exactly** with
   `engine.verify_solver` on the same executed output. This is the check that matters: it
   proves the reward path is the benchmark's own metric rather than a re-implementation
   that merely looks similar. Reads the stored runs under `experiments/core_code/` (imported
   from AtmosCoder-Bench, see docs/IMPORTED_RUNS.md; `--experiments` points elsewhere).

   Two quantities are *reported but not asserted*, because neither is a defect in this code:

   - **Stored-verdict agreement.** Some stored runs predate the harness's unit-aware
     grading (their `details` carry no `actual_unit`), so a stored `passed` can differ from
     what today's rule gives. That is data vintage, not disagreement.
   - **strict − benchmark divergence.** How often the hardened unit rule changes a verdict.
     Expected to be small and confined to offset-scale temperature answers; if it ever grows,
     something about the corpus changed and the rule needs re-examining.

4. **Isolation.** A rollout that tries to read the answer key — by relative path, and by
   the repo's absolute path — must score 0. The relative read is blocked by the sandbox's
   empty working directory on any machine; the absolute read is blocked only when
   bubblewrap can create namespaces (see `sandbox.py`). Both must be blocked for the check
   to pass: a reward through which the key is reachable is not a reward. Also asserts
   that the sandboxed runner and `engine.run_solver` produce identical output on a sample
   of gold solvers, so hardening cannot have changed what is graded, and that the
   concurrency cap holds when more callers arrive than there are slots — an uncapped
   sandbox wedged the training box once already.

Exit code is non-zero if any check fails. `--allow-weak-isolation` downgrades a reachable
absolute path to a warning for development on a machine without namespaces; never train
under it.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import sandbox
from .datasets import PROBLEM_SETS, index_by_id, load_core
from .engine import extract_code, run_solver, verify_solver
from .grading import TOLERANCE, verify
from .reward import grade
from .paths import EXPERIMENTS_DIR, REPO_ROOT
from .sandbox import bwrap_available, isolation, run_solver_sandboxed

ADVERSARIAL = [
    ("empty", "", "no_code"),
    ("whitespace", "   \n\t ", "no_code"),
    ("prose", "The answer is about 42 micrograms per cubic metre.", "exec_error"),
    ("no solve()", "```python\nx = 1 + 1\n```", "exec_error"),
    ("wrong return type", "```python\ndef solve():\n    return [1, 2, 3]\n```", "exec_error"),
    ("empty dict", "```python\ndef solve():\n    return {}\n```", "empty_result"),
    ("raises", "```python\ndef solve():\n    raise ValueError('boom')\n```", "exec_error"),
    ("infinite loop", "```python\ndef solve():\n    while True:\n        pass\n```", "exec_error"),
    ("import blocked lib", "```python\nimport numpy\ndef solve():\n    return {}\n```", "exec_error"),
]


WORKERS = min(16, (os.cpu_count() or 4) * 2)


def check_gold() -> list[str]:
    """Every stored solver reproduces its stored answer at 5%.

    `run_solver` spends its time in a child process, so a thread pool scales this
    nearly linearly despite the GIL.
    """
    failures: list[str] = []
    corpora = {name: load() for name, load in PROBLEM_SETS.items() if name != "problems"}
    def graded(problem: dict) -> tuple[str, bool]:
        return problem["id"], grade(problem["code"], problem["sub_answers"]).passed

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        verdicts = {name: list(pool.map(graded, problems)) for name, problems in corpora.items()}

    for name, problems in corpora.items():
        bad = [pid for pid, ok in verdicts[name] if not ok]
        status = "ok" if not bad else f"{len(bad)} FAILED"
        print(f"  {name:<24} {len(problems):>5} problems  {status}")
        if bad:
            failures.append(f"{name}: gold solver failed for {bad[:5]}")
    return failures


def check_adversarial() -> list[str]:
    """Malformed rollouts score zero, quietly and quickly."""
    failures: list[str] = []
    answers = [{"sub": "1", "value": "1.0", "unit": ""}]
    for label, completion, expected in ADVERSARIAL:
        try:
            result = grade(completion, answers, timeout=3.0)
        except Exception as exc:  # noqa: BLE001 — the point of the check
            failures.append(f"{label}: raised {type(exc).__name__}: {exc}")
            print(f"  {label:<24} RAISED {type(exc).__name__}")
            continue
        ok = result.reward == 0.0 and result.outcome == expected
        print(f"  {label:<24} reward={result.reward}  outcome={result.outcome}"
              f"{'' if ok else f'  EXPECTED {expected}'}")
        if not ok:
            failures.append(f"{label}: reward={result.reward} outcome={result.outcome} "
                            f"(expected 0.0 / {expected})")
    return failures


def check_replay(experiments: Path, n_files: int) -> list[str]:
    """Re-grading real completions must reproduce the vendored rule exactly."""
    exp_dir = experiments / "core_code"
    if not exp_dir.is_dir():
        print(f"  skipped — {exp_dir} not found")
        return []

    core = index_by_id(load_core())
    files = sorted(exp_dir.glob("*.run1.json"))
    if n_files:
        files = files[:n_files]

    failures: list[str] = []
    totals = {"graded": 0, "engine_agree": 0, "stored_agree": 0, "stored_cmp": 0,
              "strict_differs": 0, "no_unit_detail": 0}

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        per = {"graded": 0, "engine_agree": 0, "stored_agree": 0, "strict_differs": 0}

        def execute(record: dict) -> tuple[dict, dict | None]:
            attempts = record.get("attempts") or []
            if record["id"] not in core or not attempts:
                return record, None
            # The harness grades the LAST attempt; earlier ones are repair inputs.
            ok, results, _ = run_solver(extract_code(attempts[-1].get("response", "") or ""))
            return record, (results if ok and results else None)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            executed = list(pool.map(execute, data["results"]))

        for record, results in executed:
            if results is None:
                continue
            problem = core[record["id"]]
            answers = problem["sub_answers"]
            reference, _ = verify_solver(results, list(answers), TOLERANCE)
            as_benchmark, _ = verify(results, answers, TOLERANCE, "benchmark")
            as_strict, _ = verify(results, answers, TOLERANCE, "strict")

            per["graded"] += 1
            if as_benchmark == reference:
                per["engine_agree"] += 1
            elif len(failures) < 10:
                failures.append(f"{path.name}:{record['id']} engine={reference} "
                                f"reward(benchmark)={as_benchmark}")
            if as_strict != as_benchmark:
                per["strict_differs"] += 1
            if as_benchmark == bool(record["passed"]):
                per["stored_agree"] += 1
            totals["stored_cmp"] += 1
            if not any("actual_unit" in d for d in (record.get("details") or [])):
                totals["no_unit_detail"] += 1

        for key in per:
            totals[key] += per[key]
        if per["graded"]:
            print(f"  {path.name:<44} engine {per['engine_agree']:>4}/{per['graded']:<4}"
                  f"  stored {per['stored_agree']:>4}/{per['graded']:<4}"
                  f"  strict\u2260bench {per['strict_differs']:>3}")

    graded = totals["graded"]
    if graded:
        print(f"\n  vendored-rule agreement : {totals['engine_agree']}/{graded} = "
              f"{totals['engine_agree'] / graded:.2%}   (must be 100%)")
        print(f"  stored-verdict agreement: {totals['stored_agree']}/{graded} = "
              f"{totals['stored_agree'] / graded:.2%}   (reported; differs by data vintage)")
        print(f"  strict rule changes     : {totals['strict_differs']}/{graded} = "
              f"{totals['strict_differs'] / graded:.2%}   (reported; offset-scale answers only)")
        print(f"  records predating unit-aware grading: {totals['no_unit_detail']}/"
              f"{totals['stored_cmp']}")
    return failures


def _key_reader(path: str) -> str:
    return (f"```python\nimport json\ndef solve():\n    key = json.load(open({path!r}))\n"
            f"    return {{'1': {{'value': float(key[0]['sub_answers'][0]['value']), 'unit': ''}}}}\n```")


def check_isolation(allow_weak: bool) -> list[str]:
    """The answer key must be unreachable, and hardening must not change what is graded."""
    failures: list[str] = []
    print("  " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in isolation().items()))

    # The target must be a file that really exists and really holds answers, or "blocked"
    # is indistinguishable from "the path was wrong". core.json is both, and its first record
    # is load_core()[0] by construction, so the graded comparison below is exact.
    first = load_core()[0]
    for label, path in (("relative path", "data/core.json"),
                        ("absolute path", str(REPO_ROOT / "data/core.json"))):
        result = grade(_key_reader(path), first["sub_answers"])
        blocked = result.reward == 0.0
        print(f"  read answer key via {label:<14} -> {'blocked' if blocked else 'READABLE  reward=1.0'}")
        if not blocked:
            message = f"answer key readable via {label}"
            if label == "absolute path" and allow_weak and not bwrap_available():
                print(f"  WARNING: {message} — bubblewrap unavailable; do not train on this machine")
            else:
                failures.append(message)

    # Count live sandboxes by wrapping the Popen the sandbox module uses.
    peak, live, lock, orig = 0, 0, threading.Lock(), sandbox.subprocess.Popen

    def counting(*a, **k):
        nonlocal peak, live
        with lock:
            live += 1
            peak = max(peak, live)
        proc = orig(*a, **k)
        done = proc.communicate

        def communicate(*aa, **kk):
            nonlocal live
            try:
                return done(*aa, **kk)
            finally:
                with lock:
                    live -= 1
        proc.communicate = communicate
        return proc

    sandbox.subprocess.Popen = counting
    try:
        gold = load_core()[0]["code"]
        with ThreadPoolExecutor(max_workers=8 * sandbox.MAX_CONCURRENCY) as pool:
            list(pool.map(lambda _: sandbox.run_solver_sandboxed(gold), range(8 * sandbox.MAX_CONCURRENCY)))
    finally:
        sandbox.subprocess.Popen = orig
    capped = peak <= sandbox.MAX_CONCURRENCY
    print(f"  concurrency cap: peak {peak} sandboxes under {8 * sandbox.MAX_CONCURRENCY} callers "
          f"(cap {sandbox.MAX_CONCURRENCY}) -> {'held' if capped else 'BREACHED'}")
    if not capped:
        failures.append(f"sandbox concurrency cap breached: peak {peak} > {sandbox.MAX_CONCURRENCY}")

    sample = load_core()[::40]
    mismatched = [p["id"] for p in sample
                  if run_solver(p["code"])[1] != run_solver_sandboxed(p["code"])[1]]
    print(f"  sandboxed runner == engine runner on {len(sample)} gold solvers: "
          f"{'yes' if not mismatched else f'NO {mismatched}'}")
    if mismatched:
        failures.append(f"sandboxed and plain runner differ on {mismatched}")

    # PREREG_X X-16: a bwrap that fails before the solver runs must not score a correct
    # rollout 0, and the fallback it then takes must still keep the answer key out of reach.
    # X-23: every failed run is counted by kind (last line, numbers and quoted text masked)
    sandbox.counts(reset=True)
    broken = run_solver_sandboxed("def solve(x=1):\n    import numpy\n    return {'a': {'value': 1, 'unit': ''}}")
    looped = run_solver_sandboxed("def solve():\n    while True: pass", timeout=1)
    got = sandbox.counts(reset=True)
    kinds = sorted(k for k in got if k.startswith("fail: "))
    recorded = (not broken[0] and not looped[0] and sum(got[k] for k in kinds) == 2
                and any("ModuleNotFoundError" in k or "ImportError" in k or "blocked" in k.lower() for k in kinds)
                and any("timeout" in k.lower() for k in kinds))
    print(f"  failed runs counted by kind: {kinds} -> {'ok' if recorded else 'FAILED'}")
    if not recorded:
        failures.append("failed runs not counted by kind (X-23)")

    # X-24: a failure that does not repeat is not returned, and is counted by kind; the solver's
    # budget is CPU time, so a run that waits on a slow box without computing still passes
    real_execute, calls = sandbox._execute, []
    def flaky(*a, **kw):
        calls.append(1)
        return "Sandbox failed to start: injected" if len(calls) == 1 else real_execute(*a, **kw)
    sandbox._execute = flaky
    sandbox.counts(reset=True)
    try:
        recovered = grade(first["code"], first["sub_answers"]).reward == 1.0
    finally:
        sandbox._execute = real_execute
    got = sandbox.counts(reset=True)
    recovered = recovered and got.get("recovered: Sandbox failed to start: injected") == 1 \
        and not any(k.startswith("fail: ") for k in got)
    waited = run_solver_sandboxed("def solve():\n    import time\n    time.sleep(3)\n"
                                  "    return {'a': {'value': 1, 'unit': ''}}", timeout=1)[0]
    print(f"  transient failure {'recovered and counted' if recovered else 'NOT RECOVERED'}; a run waiting "
          f"3 s on a 1 s CPU budget {'passes' if waited else 'FAILS'} -> {'ok' if recovered and waited else 'FAILED'}")
    if not (recovered and waited):
        failures.append("a non-repeating failure or a slow box scores a correct solver 0 (X-24)")

    if bwrap_available():
        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "bwrap")
            with open(fake, "w") as f:
                f.write("#!/bin/sh\necho 'bwrap: injected start failure' >&2\nexit 1\n")
            os.chmod(fake, 0o755)
            real, sandbox.BWRAP = sandbox.BWRAP, fake
            sandbox.counts(reset=True)
            try:
                gold_ok = grade(first["code"], first["sub_answers"]).reward == 1.0
                key_blocked = grade(_key_reader(str(REPO_ROOT / "data/core.json")),
                                    first["sub_answers"]).reward == 0.0
            finally:
                sandbox.BWRAP = real
            got = sandbox.counts(reset=True)
        fallbacks = got.get("fallback", 0)
        # X-18: the failure's own message is counted, numbers masked
        logged = got.get("bwrap_error: bwrap: injected start failure", 0) == got.get("bwrap_failed", -1) > 0
        # one run for the gold solver, two for the key reader: its failure is run again (X-24)
        held = gold_ok and key_blocked and fallbacks == 3 and logged
        print(f"  bwrap failing to start: gold {'scores 1.0' if gold_ok else 'SCORES 0'}, answer key "
              f"{'blocked' if key_blocked else 'READABLE'}, {fallbacks} fallbacks, message "
              f"{'logged' if logged else 'NOT LOGGED'} -> {'ok' if held else 'FAILED'}")
        if not held:
            failures.append("bwrap start failure not handled (X-16)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR,
                        help="the experiments directory, for the replay check (default: this repository's)")
    parser.add_argument("--replay", type=int, default=3,
                        help="how many run1 result files to replay (0 = all)")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--allow-weak-isolation", action="store_true",
                        help="warn instead of fail when only layer-1 isolation is available")
    args = parser.parse_args()

    failures: list[str] = []
    print("[1/4] gold solvers")
    failures += check_gold()
    print("\n[2/4] adversarial rollouts")
    failures += check_adversarial()
    print("\n[3/4] replay agreement with the benchmark harness")
    if args.skip_replay:
        print("  skipped (--skip-replay)")
    else:
        failures += check_replay(args.experiments.resolve(), args.replay)
    print("\n[4/4] isolation")
    failures += check_isolation(args.allow_weak_isolation)

    print()
    if failures:
        print(f"FAILED — {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all reward checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
