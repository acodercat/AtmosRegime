"""The code-mode prompt: the one system string and user template every model sees.

Training rollouts (`export_verl`, `export_sft`, `sample_raft`) and evaluation
(`evaluation.episode`) both format problems here, so a trained model is always measured on
the prompt it was trained with. The text is AtmosCoder-Bench's code-mode prompt, verbatim,
so this project's numbers stay comparable to that benchmark's stored runs; changing a
character here breaks that comparison.
"""

from __future__ import annotations

SYSTEM = "You are an expert in atmospheric science."

_USER_TEMPLATE = """You are given an atmospheric science problem. Work out the solution and express it as a Python `solve()` function that computes the numerical answer(s).

## Rules
1. Put every given value in as a function parameter with a default.
2. Return a dict with one entry per quantity asked, keyed "1", "2", ..., "N" in the order asked, each mapping to {{"value": <number>, "unit": "<unit>"}} — exactly that many entries, no intermediate or unit-converted extras. Each "value" must be a single number, never a list; if one part asks for several values, give each its own entry.
3. Use only the standard library (math, etc.); do unit conversions explicitly. The function must COMPUTE each answer from its parameters — do not hard-code a precomputed number.

## Problem
{problem}

The graded answer is whatever solve() returns, so let the function do the arithmetic. Give your solve() in a single ```python code block."""


def user_prompt(problem_text: str) -> str:
    """The user turn for one problem, formatted the way the harness formats it."""
    return _USER_TEMPLATE.format(problem=problem_text)


def chat(problem_text: str) -> list[dict[str, str]]:
    """The two-message chat verl expects in a dataset's `prompt` column."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(problem_text)}]
