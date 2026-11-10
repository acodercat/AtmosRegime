"""What a regime template is, and how one becomes matched problems.

A template is one physical relation with a validity condition. Its `trigger` parameter is
sampled once inside the condition and once outside it, on top of the same `base` parameters,
so the two problems differ in one given only:

    inside  -- the relation holds; the parent's method is right       (the control)
    outside -- the relation fails; the text does not say so           (the regime trap)
    given   -- the same numbers as `outside`, with the replacement stated in the text
               (the internal control: physics fixed, recognition step removed)

`correct` is the physics that holds on both sides; `shortcut` is the parent's method. A pair is
kept only if the shortcut agrees with `correct` inside (the parent is a fair control) and misses
it outside by a margin well beyond the 5 % grading tolerance (the trap is a trap). Fairness is
argued in each template's docstring from the given data, not from any model's solve.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

Params = dict[str, float]
VARIANTS = ("inside", "outside", "given")
INSIDE_MAX_GAP = 0.02      # shortcut within 2 % of correct: the parent's method is valid there
OUTSIDE_MIN_GAP = 0.15     # 3x the 5 % grading tolerance: a shortcut answer cannot pass by luck


@dataclass(frozen=True)
class Template:
    name: str
    category: str
    mechanism: str
    unit: str
    base: Callable[[random.Random], Params]
    trigger: Callable[[random.Random, str, Params], Params]  # (rng, "inside"|"outside", base)
    text: Callable[[Params, str], str]                  # variant: one of VARIANTS
    correct: Callable[[Params], float]
    shortcut: Callable[[Params], float]


def gap(correct: float, shortcut: float) -> float:
    return abs(shortcut - correct) / abs(correct)


def pair(t: Template, rng: random.Random, tries: int = 500) -> tuple[Params, Params]:
    """One (inside, outside) parameter pair sharing every base value, or raise."""
    for _ in range(tries):
        b = t.base(rng)
        tin, tout = t.trigger(rng, "inside", b), t.trigger(rng, "outside", b)
        # The pair must differ in exactly one given; enforced here, not left to each template.
        if len(tin) != 1 or tin.keys() != tout.keys() or tin.keys() & b.keys():
            raise ValueError(f"{t.name}: trigger must return one key, the same on both sides, not in base")
        pin, pout = {**b, **tin}, {**b, **tout}
        if gap(t.correct(pin), t.shortcut(pin)) <= INSIDE_MAX_GAP and \
           gap(t.correct(pout), t.shortcut(pout)) >= OUTSIDE_MIN_GAP:
            return pin, pout
    raise RuntimeError(f"{t.name}: no admissible pair in {tries} draws — the ranges are wrong")


def num(x: float) -> str:
    """A given as it appears in a problem: 3 significant figures, ×10^n outside [1e-3, 1e5)."""
    if x != 0 and not 1e-3 <= abs(x) < 1e5:
        e = math.floor(math.log10(abs(x)))
        return f"{x / 10**e:.3g} × 10^{e}"
    s = f"{x:.3g}"                          # 1150 -> "1.15e+03": write it out as 1150
    return f"{float(s):f}".rstrip("0").rstrip(".") if "e" in s else s


def rnd(x: float, sig: int = 2) -> float:
    """Round a sampled value to `sig` significant figures, so the text and the solver agree."""
    return float(f"{x:.{sig}g}")
