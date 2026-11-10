"""Execution-verified reward.

A rollout is a model completion that should contain a `solve()` function. Scoring it means
extracting the code, executing it in the isolated subprocess `sandbox.run_solver_sandboxed`
provides, and grading the returned values against the stored answers with `grading.verify`
(per-sub relative tolerance, positional fallback, unit reconciliation — the evaluation
runner's rule, with the one unit correction described below).

Two deliberate differences from the evaluation runner:

- **Single attempt.** The runner feeds an unrunnable output back to the model and lets
  it repair, up to five tries. A reward function must not: the repair loop would score a
  different policy than the one being optimised. Unrunnable code scores 0.
- **No API-error class.** The runner distinguishes model failure from infrastructure
  failure. Here every rollout is local, so every failure is the policy's.

Execution goes through `sandbox.run_solver_sandboxed`, not `engine.run_solver`: the
plain runner lets a rollout read the answer key off disk (confirmed: reward 1.0) and
import whatever the trainer's venv happens to hold. See `sandbox.py` for the two layers
and for what your machine can enforce — log `sandbox.isolation()` at training start.

Unit grading defaults to `units="strict"` (see `grading.py`), which closes an
offset-temperature hole in the vendored rule that would otherwise hand the policy a ~15x
tolerance for the price of declaring an answer in Celsius. Pass `units="benchmark"` to
grade under the published rule bit-for-bit.

Reward shaping is a knob, not a default. `mode="strict"` (default) mirrors the benchmark:
all sub-answers right or nothing, which is the quantity the paper measures. `mode="partial"`
returns the fraction of sub-answers that pass, a denser signal for multi-part problems at
the cost of rewarding a policy that gets one of three parts right. Pick one per experiment
and record it — the two are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .engine import SOLVER_TIMEOUT, extract_code
from .grading import TOLERANCE, UnitRule, verify
from .sandbox import run_solver_sandboxed

DEFAULT_UNIT_RULE: UnitRule = "strict"
Mode = Literal["strict", "partial"]

#: Why a rollout scored what it scored. `wrong_answer` is the only outcome in which the
#: policy produced runnable, well-formed code — the rest are failures of form, and a
#: training run whose `exec_error` share is not falling is not learning the contract.
#: Note `no_code` fires only on an empty completion: `extract_code` treats unfenced text
#: as raw code, so prose reaches the interpreter and surfaces as `exec_error`. That is
#: upstream behaviour and is kept — a model that emits bare Python is graded, not discarded.
OUTCOMES = ("passed", "wrong_answer", "no_code", "exec_error", "empty_result")


@dataclass(frozen=True)
class RewardResult:
    """One graded rollout."""

    reward: float
    passed: bool
    outcome: str
    n_subs: int = 0
    n_passed: int = 0
    error: str = ""
    details: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "passed": self.passed,
            "outcome": self.outcome,
            "n_subs": self.n_subs,
            "n_passed": self.n_passed,
            "error": self.error,
        }

    def trainer_metrics(self) -> dict[str, int | float]:
        """What the verl entry points log besides the score: sub-answer counts and a one-hot
        outcome, so verl's batch mean of `outcome/<name>` is that outcome's share."""
        metrics: dict[str, int | float] = {"n_subs": self.n_subs, "n_passed": self.n_passed}
        metrics.update({f"outcome/{name}": float(self.outcome == name) for name in OUTCOMES})
        return metrics


def _score(details: list[dict[str, Any]], mode: Mode) -> tuple[float, int, int]:
    n_subs = len(details)
    n_passed = sum(1 for detail in details if detail.get("passed"))
    if mode == "strict":
        return (1.0 if n_subs and n_passed == n_subs else 0.0), n_subs, n_passed
    if mode == "partial":
        return (n_passed / n_subs if n_subs else 0.0), n_subs, n_passed
    raise ValueError(f"unknown mode {mode!r}; expected 'strict' or 'partial'")


def grade(
    completion: str,
    sub_answers: Sequence[dict[str, Any]],
    *,
    tolerance: float = TOLERANCE,
    mode: Mode = "strict",
    units: UnitRule = DEFAULT_UNIT_RULE,
    timeout: float = SOLVER_TIMEOUT,
) -> RewardResult:
    """Grade one completion against one problem's stored answers.

    Never raises and never blocks indefinitely: extraction failures, syntax errors,
    runaway loops and malformed returns all come back as a zero-reward result carrying
    the reason, because an exception here would kill a training step.
    """
    code = extract_code(completion or "")
    if not code.strip():
        return RewardResult(0.0, False, "no_code", error="no code block in completion")

    try:
        ok, results, info = run_solver_sandboxed(code, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — e.g. OSError spawning the child; must not escape
        return RewardResult(0.0, False, "exec_error", error=f"{type(exc).__name__}: {exc}")
    if not ok:
        return RewardResult(0.0, False, "exec_error", error=info)
    if not results:
        return RewardResult(0.0, False, "empty_result", error="solve() returned an empty dict")

    passed, details = verify(results, list(sub_answers), tolerance, units)
    reward, n_subs, n_passed = _score(details, mode)
    return RewardResult(
        reward=reward,
        passed=passed,
        outcome="passed" if passed else "wrong_answer",
        n_subs=n_subs,
        n_passed=n_passed,
        details=details,
    )


def grade_problem(
    completion: str,
    problem: dict[str, Any],
    **kwargs: Any,
) -> RewardResult:
    """`grade` addressed by problem record rather than by answer list."""
    return grade(completion, problem["sub_answers"], **kwargs)


def reward_fn(
    completions: Sequence[str],
    problems: Sequence[dict[str, Any]],
    *,
    tolerance: float = TOLERANCE,
    mode: Mode = "strict",
    units: UnitRule = DEFAULT_UNIT_RULE,
    timeout: float = SOLVER_TIMEOUT,
) -> list[float]:
    """Batch reward, the shape verl / TRL / OpenRLHF expect.

    `completions[i]` is scored against `problems[i]`; the two must align. Each rollout
    costs one subprocess, so a trainer wanting throughput should call this from a thread
    pool — `run_solver` releases the GIL while the child runs.
    """
    if len(completions) != len(problems):
        raise ValueError(
            f"got {len(completions)} completions for {len(problems)} problems — these must align"
        )
    return [
        grade_problem(completion, problem, tolerance=tolerance, mode=mode,
                      units=units, timeout=timeout).reward
        for completion, problem in zip(completions, problems)
    ]
