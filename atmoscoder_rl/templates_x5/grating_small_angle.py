"""Small-angle grating formula -> exact diffraction angle on a flat screen (X5, optics).

Parent's method: first-order maximum at y = L lambda / d (sin theta ~ tan theta ~ theta), the textbook screen formula.
Where it fails: for a fine grating lambda / d is not small; the order leaves at theta = arcsin(lambda / d) and lands at
y = L tan(theta). At lambda / d = 0.55-0.75 the small-angle answer is 16-34 % short.
Fair from the text: every arm states the wavelength, the line density (the trigger, ordinary data) and the screen
distance; normal incidence, flat screen perpendicular to the undiffracted beam. Nothing is recalled.
Inside: lambda / d = 0.02-0.05 (<= 0.13 %). Outside: lambda / d = 0.55-0.75.
Conventions: the grating equation is exact; "distance on the screen" is unambiguous for a flat screen. Using y = L theta
(arc length) is 12-25 % short (a second wrong method, check()).
Asked quantity: distance of the first-order maximum from the central maximum, cm.
check(): solve d sin(theta) = lambda by Newton on theta, then y = L sin/cos, independent of arcsin.
Reference: Hecht, Optics, sec. 10.2.8.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd


def base(rng):
    return {"lam": float(rng.choice(range(420, 661, 10))), "l": rnd(rng.uniform(0.5, 2.0))}


def trigger(rng, regime, b):
    x = rng.uniform(0.02, 0.05) if regime == "inside" else rng.uniform(0.55, 0.75)
    return {"n": rnd(x / (b["lam"] * 1e-6), 3)}                         # lines per mm


def _x(p):
    return p["lam"] * 1e-6 * p["n"]


def correct(p):
    return p["l"] * math.tan(math.asin(_x(p))) * 100


def shortcut(p):
    return p["l"] * _x(p) * 100


def text(p, variant):
    given = ("The diffraction angle is not small here: the first order leaves at sin(theta) = lambda / d and lands at "
             "y = L tan(theta). " if variant == "given" else "")
    return (f"Light of wavelength {num(p['lam'])} nm falls at normal incidence on a transmission grating with "
            f"{num(p['n'])} lines per mm. The pattern is observed on a flat screen {num(p['l'])} m from the grating, "
            f"perpendicular to the undiffracted beam. {given}How far from the central maximum is the first-order "
            f"maximum on the screen? Express your answer in cm.")


TEMPLATE = Template("grating_small_angle", "optics",
                    "small-angle grating screen formula -> exact arcsin / tan geometry at large diffraction angle",
                    "cm", base, trigger, text, correct, shortcut)


def check() -> str:
    import random
    rng = random.Random("x5-check:grating_small_angle")
    worst = 0.0; arc = []; gi, go = [], []
    for _ in range(200):
        pin, pout = pair(TEMPLATE, rng)
        x = _x(pout); th = x
        for _ in range(50):
            th -= (math.sin(th) - x) / math.cos(th)
        worst = max(worst, abs(pout["l"] * math.sin(th) / math.cos(th) * 100 / correct(pout) - 1))
        arc.append(gap(correct(pout), pout["l"] * th * 100))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert worst < 1e-12 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP and min(arc) > 0.05
    return (f"Newton vs arcsin {worst:.1e}, arc-length method miss {min(arc):.2f}-{max(arc):.2f}, "
            f"inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}")
