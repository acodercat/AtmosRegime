"""Shortcut capture: did a wrong answer on a trap reproduce the parent method's answer?

Each trap stores the full output vector of the shortcut (`shortcut_values`, keyed like the
answer's subs). A record *reproduces the shortcut* when every sub it answered that the
shortcut defines matches it within `SHORTCUT_TOL`, relatively — the scale is taken from the
shortcut value, never an absolute threshold (a correct answer of 7e-15 C is not zero) — or
matches after converting a commensurate declared unit under the **strict** unit rule. Always
test against `shortcut_values`, never the convenience scalar `shortcut_value`.

Why strict: the benchmark rule canonicalises °C to kelvin before a *relative* comparison, and
at 2 % of ~285 K any two Celsius values within ~6 °C "match" (it also reads a bare `C`, the
coulomb, as Celsius). Under that rule 70 of 6516 stored trap records, all correct answers,
were counted as reproducing a shortcut a few degrees away. The strict rule compares on the
expected unit's own scale and still credits a real conversion (283.25 K against 10.1 °C).

This is the one definition: `analysis traps`, the explore reports and anything new use it.
Unified 2026-09-26. Before that, `analysis` did not convert units and the explore reports
converted under the benchmark rule, so the stage-1 capture rates (explore/REPORT.md) were
computed with the Celsius inflation and are re-derived once those runs are local.
"""

from __future__ import annotations

from typing import Any

from .grading import reconcile_units
from .stats import majority

SHORTCUT_TOL = 0.02


def _close(actual: float, expected: float) -> bool:
    if expected == 0.0:
        return actual == 0.0
    return abs(actual - expected) <= SHORTCUT_TOL * abs(expected)


def _matches(detail: dict[str, Any], value: Any, unit: str) -> bool:
    try:
        actual, expected = float(detail.get("actual")), float(value)
    except (TypeError, ValueError):
        return False
    if _close(actual, expected):
        return True
    return bool(detail.get("actual_unit")) and reconcile_units(
        str(value), unit, actual, detail["actual_unit"], SHORTCUT_TOL, units="strict")


def shortcut_match(record: dict[str, Any], trap: dict[str, Any]) -> bool:
    """One record reproduces the trap's shortcut vector (see the module docstring)."""
    shortcut = trap.get("shortcut_values") or {}
    units = {str(sub["sub"]): sub.get("unit", "") for sub in trap["sub_answers"]}
    answered = {str(d["sub"]): d for d in record.get("details") or [] if "sub" in d}
    keys = [key for key in shortcut if key in answered]
    return bool(keys) and all(_matches(answered[key], shortcut[key], units.get(key, "")) for key in keys)


def shortcut_majority(records: list[dict[str, Any]], trap: dict[str, Any]) -> bool:
    """Majority over repetitions (ties count), the reduction every verdict here uses."""
    return bool(records) and majority(shortcut_match(record, trap) for record in records)
