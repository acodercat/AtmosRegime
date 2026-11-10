"""Infinite-solenoid field -> field at the centre of a short solenoid (X5, electromagnetism).

Parent's method: B = mu_0 N I / l inside a solenoid (the infinite-solenoid result).
Where it fails: it needs l >> radius. At the centre of a finite single-layer solenoid of radius a,
B = mu_0 N I / sqrt(l^2 + 4 a^2) (integrating the on-axis loop field over the winding). With a / l = 0.7-1.0 the
infinite formula is 72-124 % high.
Fair from the text: every arm states the number of turns, length, radius (the trigger, ordinary geometry) and current.
Nothing is recalled.
Inside: a / l = 0.02-0.04 (<= 0.32 %). Outside: a / l = 0.7-1.0.
Conventions: none; the winding is a thin single layer as stated. Plausibility: 5-15 turns/cm and 0.3-2 A keep the wire's
current density at or below ~6 A/mm^2. Grading hazard (accepted): the grader does not convert mT <-> T; it affects all arms equally.
Asked quantity: magnetic field at the centre, mT.
check(): numerical sum of N single-loop on-axis fields mu_0 I a^2 / (2 (a^2 + z^2)^1.5), independent of the closed form.
Reference: Griffiths, Introduction to Electrodynamics, Problem 5.11; Jackson, Classical Electrodynamics, sec. 5.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

MU0 = 4e-7 * math.pi


def base(rng):
    l = rnd(rng.uniform(10, 30))
    # 5-15 turns per cm (wire >= ~0.67 mm) and 0.3-2 A keep the current density at or below ~6 A/mm^2 (review 2026-10-12)
    return {"turns": rnd(l * rng.uniform(5, 15)), "l": l, "i": rnd(rng.uniform(0.3, 2.0))}


def trigger(rng, regime, b):
    f = rng.uniform(0.02, 0.04) if regime == "inside" else rng.uniform(0.7, 1.0)
    return {"a": rnd(b["l"] * f)}                                       # cm


def correct(p):
    return MU0 * p["turns"] * p["i"] / math.sqrt((p["l"] / 100) ** 2 + 4 * (p["a"] / 100) ** 2) * 1000


def shortcut(p):
    return MU0 * p["turns"] * p["i"] / (p["l"] / 100) * 1000


def text(p, variant):
    given = ("The coil is not long compared with its radius: at its centre B = mu_0 N I / sqrt(l^2 + 4 a^2). "
             if variant == "given" else "")
    return (f"A single-layer coil of {num(p['turns'])} closely spaced turns is wound uniformly on a cylinder; the winding "
            f"is {num(p['l'])} cm long and has a radius of {num(p['a'])} cm. It carries a steady current of "
            f"{num(p['i'])} A in air (take mu_0 = 4 pi × 10^-7 T m A^-1). {given}What is the magnetic field at the centre "
            f"of the coil? Express your answer in mT.")


TEMPLATE = Template("solenoid_finite", "electromagnetism",
                    "infinite-solenoid field -> finite-solenoid centre field when the radius is comparable with the length",
                    "mT", base, trigger, text, correct, shortcut)


def _loops(p):
    n, l, a = int(p["turns"]), p["l"] / 100, p["a"] / 100
    s = 0.0
    for j in range(n):
        z = -l / 2 + (j + 0.5) * l / n
        s += MU0 * p["i"] * a * a / (2 * (a * a + z * z) ** 1.5)
    return s * 1000


def check() -> str:
    import random
    rng = random.Random("x5-check:solenoid_finite")
    worst = 0.0; gi, go = [], []
    for _ in range(100):
        pin, pout = pair(TEMPLATE, rng)
        worst = max(worst, abs(_loops(pout) / correct(pout) - 1))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert worst < 1e-4 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP
    return f"loop sum vs closed form {worst:.1e}, inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}"
