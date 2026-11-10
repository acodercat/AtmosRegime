"""Grading rules: the benchmark's own, and a hardened one for reward.

`engine.py` is vendored bit-for-bit and must stay that way — it defines what the
benchmark paper's accuracy *means*. This module sits beside it and adds one correction
that matters only when the grader is used as a **reward**.

## The offset-scale hole

`engine._units_reconcile` documents itself as firing "only when both units parse to the
SAME dimensional signature, so it never loosens a same-unit compare". That invariant
holds for every multiplicative unit: scaling both sides by a common factor leaves the
relative error unchanged. It does **not** hold for the absolute temperature scales, which
canonicalise with an *offset* (°C → K adds 273.15). Adding a constant to both sides of a
relative comparison shrinks the error:

    expected 40.6 °C, model says 25.6 °C   — wrong by 15 °C
    canonicalised: 313.75 K vs 298.75 K    — 4.8% apart, so it PASSES at 5%

For typical atmospheric temperatures this inflates the effective tolerance on a
Celsius-declared answer by roughly 15×.

Measured on the benchmark's stored runs this is a rounding error: 18 of 436 core problems
have an offset-scale answer, and re-grading every model's stored completion on them moves
9 of 324 results (~0.11 pp of overall accuracy, concentrated in the weakest models). The
published leaderboard is not meaningfully affected.

As a **reward** it is a different object. The policy chooses the unit it declares, so
"answer temperatures in °C" is a free ~15× tolerance sitting in the gradient — precisely
the kind of shortcut RL discovers over thousands of steps, and precisely the kind of
process defect this project exists to study. Training against it would manufacture the
artefact we mean to measure.

## The rule

`units="strict"` (the default) converts the model's answer **into the unit the expected
answer declares**, then compares relatively against the expected value as stated. For
multiplicative units this is algebraically identical to the vendored rule; for offset
scales it removes the inflation while still crediting a correct answer given on the other
scale (313.6 K against an expected 40.6 °C still passes).

`units="benchmark"` calls the vendored rule unchanged, for reproducing published numbers.
Any accuracy this project reports must name which rule produced it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal, Sequence

from . import engine

UnitRule = Literal["strict", "benchmark"]

TOLERANCE = 0.05   # relative; the benchmark's and this project's grading tolerance


def _affine(unit: str) -> tuple[float, float]:
    """(scale, offset) of an absolute-temperature unit, read back from `engine._canonical`
    so this module never re-implements the engine's unit normalisation."""
    offset = engine._canonical(0.0, unit)[1]
    scale = engine._canonical(1.0, unit)[1] - offset
    return scale, offset


def _reconcile_strict(expected: str, expected_unit: str, actual: float,
                      actual_unit: str, tolerance: float) -> bool:
    """`engine._units_reconcile`, with offset scales compared on the expected's own scale."""
    try:
        expected_value = float(str(expected).strip())
    except ValueError:
        return False

    canon_expected = engine._canonical(expected_value, expected_unit)
    canon_actual = engine._canonical(actual, actual_unit)
    both_absolute_temperature = (
        canon_expected is not None
        and canon_actual is not None
        and canon_expected[0] == canon_actual[0] == "TEMPABS"
    )
    if both_absolute_temperature:
        scale, offset = _affine(expected_unit)
        actual_on_expected_scale = (canon_actual[1] - offset) / scale
        if expected_value == 0:
            return abs(actual_on_expected_scale) < tolerance
        return abs(actual_on_expected_scale - expected_value) / abs(expected_value) <= tolerance

    return engine._units_reconcile(expected, expected_unit, actual, actual_unit, tolerance)


def reconcile_units(expected: str, expected_unit: str, actual: float, actual_unit: str,
                    tolerance: float, *, units: UnitRule) -> bool:
    """True if `actual` in `actual_unit` equals `expected` in `expected_unit` within the
    relative tolerance, after converting between commensurate units under the given rule.
    False when the units do not parse to the same dimension."""
    reconcile = _reconcile_strict if units == "strict" else engine._units_reconcile
    return reconcile(expected, expected_unit, actual, actual_unit, tolerance)


def _matches(expected: str, expected_unit: str, actual: float, actual_unit: str,
             tolerance: float, units: UnitRule) -> bool:
    if engine.compare_values(expected, actual, tolerance):
        return True
    return reconcile_units(expected, expected_unit, actual, actual_unit, tolerance, units=units)


def verify(solver_results: dict[str, Any], expected_answers: Sequence[dict[str, Any]],
           tolerance: float = TOLERANCE, units: UnitRule = "strict") -> tuple[bool, list[dict]]:
    """Grade a solver's results: the one grading entry point (`check_reward` alone calls
    `engine.verify_solver` directly, to prove `units="benchmark"` reproduces it exactly).

    `engine.verify_solver` under the chosen unit rule.

    Mirrors the vendored function exactly — per-sub matching with positional fallback,
    a sub passing if any of its accepted values match — differing only in which
    reconciliation rule decides a single comparison.
    """
    if units not in ("strict", "benchmark"):
        raise ValueError(f"unknown unit rule {units!r}; expected 'strict' or 'benchmark'")
    if units == "benchmark":
        return engine.verify_solver(solver_results, list(expected_answers), tolerance)
    if not solver_results:
        return False, [{"error": "empty results"}]

    expected_by_sub: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for expected in expected_answers:
        expected_by_sub[str(expected["sub"])].append((expected["value"], expected.get("unit", "")))

    solver_keys = list(solver_results.keys())
    subs = list(expected_by_sub.keys())
    details: list[dict[str, Any]] = []
    all_passed = True

    for position, sub in enumerate(subs):
        accepted = expected_by_sub[sub]
        entry = solver_results.get(sub)
        if entry is None and position < len(solver_keys):  # positional fallback
            entry = solver_results[solver_keys[position]]

        detail: dict[str, Any] = {
            "sub": sub,
            "expected": [value for value, _ in accepted],
            "expected_units": [unit for _, unit in accepted],
        }
        if entry is None:
            detail |= {"actual": None, "actual_unit": "", "passed": False}
            details.append(detail)
            all_passed = False
            continue

        actual_value = entry.get("value") if isinstance(entry, dict) else entry
        actual_unit = entry.get("unit", "") if isinstance(entry, dict) else ""
        try:
            actual_float = float(actual_value)
        except (ValueError, TypeError):
            detail |= {"actual": str(actual_value), "actual_unit": actual_unit, "passed": False}
            details.append(detail)
            all_passed = False
            continue

        passed = any(_matches(value, unit, actual_float, actual_unit, tolerance, units)
                     for value, unit in accepted)
        detail |= {"actual": actual_float, "actual_unit": actual_unit, "passed": passed}
        details.append(detail)
        if not passed:
            all_passed = False

    return all_passed, details


#: Below this SI magnitude `engine._units_reconcile` compares against an absolute 5e-17 instead of 5 %
#: (it divides by max(|expected|, 1e-15)), so almost any small answer passes (docs/MEASUREMENT_NOTES.md §11).
SI_FLOOR = 1e-14


def si_magnitude(value: float, unit: str) -> float | None:
    """|value| in SI base units as the engine's unit reconciliation sees it, or None if the unit does not parse."""
    for canonical in (engine._canonical, engine._canonical_compound):
        try:
            c = canonical(float(value), unit)
        except (TypeError, ValueError):
            c = None
        if c is not None:
            return abs(c[1])
    return None

