"""One problem, one model: ask for a `solve()`, execute it, grade it, repair if it will not run.

The model sees the code-mode prompt (`prompts.py`, the same one training uses) and must answer
with a `solve()` function. The function is executed (`engine.run_solver`) and its return value
graded against the known answer with the benchmark's unit rule (`grading.verify`).

- **Self-repair.** If the response is *ungradable* — no code, or code that does not run — the
  failed code and its error are handed back in a fresh prompt (`REPAIR_PROMPT`), up to
  `max_attempts` content tries in total. Still ungradable after that is a **fail**: the model
  did not deliver. A gradable but wrong answer is final at once; it is never retried.
- **API errors are not the model's.** A failed call is retried blindly (`API_RETRIES`, on top
  of the client's own backoff) without counting as a try; one that never clears makes the
  record an excluded `error`.
- **`attempts` is a per-call log, not a chat.** Each call is stateless — no history is sent;
  a repair prompt restates the problem. Each entry is what one call sent and got back.

Execution is `engine.run_solver`, not the RL sandbox: a frozen model under evaluation has no
incentive to attack the grader (`sandbox.py` explains why training needs more).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..engine import extract_code, run_solver
from ..grading import verify
from ..prompts import SYSTEM, user_prompt
from .client import USAGE_KEYS, ChatModel, Completion, ModelError, PromptTooLongError

MAX_ATTEMPTS = 5    # content tries: the first answer plus up to four repairs
API_RETRIES = 2     # extra blind retries per call for an API error that outlived the client's backoff

REPAIR_PROMPT = """Your solve() function failed to run. Fix the bug and return the corrected function.

## Problem
{problem}

## Your code
{code}

## Error when it was executed
{error}

Return the COMPLETE corrected solve(): a dict keyed "1".."N" in the order asked, each
{{"value": <number>, "unit": "<unit>"}}, standard library only. Put all arithmetic in the function; give the corrected solve() in a single ```python code block."""


@dataclass
class Attempt:
    """One response turned into a gradable result, or the reason it could not be."""
    results: dict[str, Any] = field(default_factory=dict)   # sub -> {"value", "unit"}
    code: str = ""
    error: str | None = None                                  # set iff ungradable


def _attempt(text: str) -> Attempt:
    code = extract_code(text)
    if not code:
        return Attempt(error="no code in response")
    ran, results, error = run_solver(code)
    return Attempt(results=results, code=code) if ran else Attempt(code=code, error=error[:200])


def _generate(model: ChatModel, prompt: str) -> Completion:
    for retry in range(API_RETRIES + 1):
        try:
            return model.generate(prompt, SYSTEM)
        except PromptTooLongError:
            raise
        except ModelError:
            if retry == API_RETRIES:
                raise
    raise AssertionError("unreachable")


def _log(prompt: str, completion: Completion, outcome: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"prompt": prompt}
    if completion.reasoning:
        entry["reasoning"] = completion.reasoning
    entry.update(response=completion.text, outcome=outcome, usage=completion.usage)
    return entry


def _total_usage(calls: list[dict[str, Any]]) -> dict[str, int]:
    return {key: sum(call["usage"][key] for call in calls) for key in USAGE_KEYS}


def evaluate(problem: dict[str, Any], model: ChatModel, tolerance: float,
             max_attempts: int = MAX_ATTEMPTS) -> dict[str, Any]:
    """The stored record for one problem: `passed`, `details` (per-sub expected vs actual, with
    units), `num_attempts`, `attempts` and `usage` (provider-reported, summed over calls) — or
    `error` if the API never answered."""
    prompt = user_prompt(problem["problem"])
    calls: list[dict[str, Any]] = []
    outcome = Attempt(error="no response")
    try:
        while len(calls) < max_attempts:
            completion = _generate(model, prompt)
            outcome = _attempt(completion.text)
            calls.append(_log(prompt, completion, "graded" if not outcome.error
                              else f"ungradable: {outcome.error}"))
            if not outcome.error:
                break
            prompt = REPAIR_PROMPT.format(problem=problem["problem"],
                                          code=outcome.code or "(no code was produced)",
                                          error=outcome.error)
    except PromptTooLongError as error:
        failure = f"prompt too long: {error}"
    except ModelError as error:
        failure = str(error)
    else:
        record: dict[str, Any] = {"id": problem["id"]}
        if outcome.error:
            record.update(passed=False, num_attempts=len(calls), failed_reason=outcome.error)
        else:
            passed, details = verify(outcome.results, problem["sub_answers"], tolerance, units="benchmark")
            record.update(passed=passed, details=details, num_attempts=len(calls))
        record.update(usage=_total_usage(calls), attempts=calls)
        return record
    return {"id": problem["id"], "passed": False, "num_attempts": 0, "error": failure[:200],
            "usage": dict.fromkeys(USAGE_KEYS, 0), "attempts": []}
