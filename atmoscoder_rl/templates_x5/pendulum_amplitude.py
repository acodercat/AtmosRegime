"""Small-angle pendulum period -> exact period at large amplitude (X5, mechanics).

Parent's method: T = 2 pi sqrt(L / g), the simple-pendulum period.
Where it fails: it linearises sin(theta) ~ theta. Released from rest at theta_0 the exact period is
T = 4 sqrt(L / g) K(sin(theta_0 / 2)), K the complete elliptic integral of the first kind; at theta_0 = 125-150 deg
it is 40-76 % longer.
Fair from the text: every arm states the rod length, g and the release angle (the trigger, ordinary data); a light
rigid rod lets the release exceed 90 deg. Nothing is recalled.
Inside: theta_0 = 3-8 deg (<= 0.13 %). Outside: theta_0 = 125-150 deg (120 deg excluded: sin 120 = sin 60, so a solver
who drops the half angle and writes K(sin theta_0) would land exactly on the key; review 2026-10-12).
Conventions: the series 1 + theta^2/16 + 11 theta^4/3072 is 2-9 % short (within 5 % up to ~135 deg) (a truncated method, not a
convention; check() reports it). The exact form is unique.
Asked quantity: period, s.
check(): direct integration of the equation of motion (RK4, time to the first return of theta' = 0, doubled),
independent of the elliptic integral (evaluated by the arithmetic-geometric mean).
Reference: Goldstein, Classical Mechanics; Marion & Thornton, Classical Dynamics, sec. 4.4.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

G = 9.81


def base(rng):
    return {"l": rnd(rng.uniform(0.4, 2.5))}


def trigger(rng, regime, b):
    return {"th": float(rng.choice(range(3, 9))) if regime == "inside" else float(rng.choice(range(125, 151, 5)))}


def _k(m):
    """Complete elliptic integral K(k) by the arithmetic-geometric mean."""
    a, b = 1.0, math.sqrt(1 - m * m)
    for _ in range(40):
        a, b = (a + b) / 2, math.sqrt(a * b)
    return math.pi / (2 * a)


def correct(p):
    return 4 * math.sqrt(p["l"] / G) * _k(math.sin(math.radians(p["th"]) / 2))


def shortcut(p):
    return 2 * math.pi * math.sqrt(p["l"] / G)


def text(p, variant):
    given = ("The amplitude is not small, so the small-angle approximation does not hold: T = 4 sqrt(L/g) K(k) with "
             "k = sin(theta_0/2) and K the complete elliptic integral of the first kind. " if variant == "given" else "")
    return (f"A small heavy bob is fixed to the end of a light rigid rod of length {num(p['l'])} m that pivots freely "
            f"about its other end. The bob is released from rest with the rod at {num(p['th'])} degrees from the "
            f"downward vertical. Neglect friction and air resistance and take g = 9.81 m s^-2. {given}What is the period "
            f"of the resulting oscillation? Express your answer in s.")


TEMPLATE = Template("pendulum_amplitude", "mechanics",
                    "small-angle pendulum period -> elliptic-integral period at large amplitude",
                    "s", base, trigger, text, correct, shortcut)


def _rk4(p, dt=2e-5):
    th, w, t = math.radians(p["th"]), 0.0, 0.0
    f = lambda th: -G / p["l"] * math.sin(th)
    while True:
        k1t, k1w = w, f(th)
        k2t, k2w = w + dt / 2 * k1w, f(th + dt / 2 * k1t)
        k3t, k3w = w + dt / 2 * k2w, f(th + dt / 2 * k2t)
        k4t, k4w = w + dt * k3w, f(th + dt * k3t)
        nth, nw = th + dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t), w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        if t > 0 and w < 0 <= nw:                                       # turning point at -theta_0: half period
            return 2 * (t + dt * (-w) / (nw - w))
        th, w, t = nth, nw, t + dt


def check() -> str:
    import random
    rng = random.Random("x5-check:pendulum_amplitude")
    worst = 0.0; series = []; gi, go = [], []
    for _ in range(8):
        pin, pout = pair(TEMPLATE, rng)
        worst = max(worst, abs(_rk4(pout) / correct(pout) - 1))
        th = math.radians(pout["th"])
        series.append(gap(correct(pout), shortcut(pout) * (1 + th ** 2 / 16 + 11 * th ** 4 / 3072)))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert abs(_k(math.sin(math.radians(45))) - 1.854075) < 1e-5     # K(sin 45 deg), tables
    assert worst < 1e-4 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP
    return (f"RK4 vs elliptic {worst:.1e}, two-term series miss {min(series):.2f}-{max(series):.2f}, "
            f"inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}")
