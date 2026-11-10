"""Uniform gravity -> inverse-square gravity for a vertical launch to great height (X5, mechanics).

Parent's method: launch speed to reach height h, v = sqrt(2 g h) (energy with constant g).
Where it fails: g falls off as (R/(R+h))^2; with g at the surface and the planet's radius R the exact speed (no drag,
non-rotating) is v = sqrt(2 g R h / (R + h)). At h = 3000-5000 km the constant-g answer is 21-34 % high.
Fair from the text: every arm states g at the surface and the Earth's radius; the trigger is the height reached, ordinary
data. Nothing is recalled.
Inside: h = 5-30 km (<= 0.24 %). Outside: h = 3000-5000 km.
Conventions: G M / R^2 = g is the stated datum; no competing convention.
Asked quantity: launch speed, km/s.
check(): numerical integration of the vertical motion in inverse-square gravity (RK4 from the computed speed, the
apex must equal h), independent of the energy formula.
Reference: any mechanics text, e.g. Kleppner & Kolenkow, An Introduction to Mechanics, ch. 4 and 9.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

G0, RE = 9.81, 6370.0                                                  # m s^-2, km


def base(rng):
    return {"g": G0}


def trigger(rng, regime, b):
    return {"h": rnd(rng.uniform(5, 30), 3) if regime == "inside" else rnd(rng.uniform(3000, 5000), 3)}


def correct(p):
    h, r = p["h"] * 1e3, RE * 1e3
    return math.sqrt(2 * p["g"] * r * h / (r + h)) / 1e3


def shortcut(p):
    return math.sqrt(2 * p["g"] * p["h"] * 1e3) / 1e3


def text(p, variant):
    given = ("Over this height gravity is not uniform; it falls off as the inverse square of the distance from the Earth's "
             "centre, so v = sqrt(2 g R h / (R + h)). " if variant == "given" else "")
    return (f"A probe is launched straight up from the Earth's surface and coasts, with its engine off, to a maximum "
            f"height of {num(p['h'])} km above the surface. Neglect air resistance and the Earth's rotation. The "
            f"gravitational acceleration at the surface is {num(p['g'])} m s^-2 and the Earth's radius is 6370 km. "
            f"{given}With what speed did the probe leave the surface? Express your answer in km/s.")


TEMPLATE = Template("gravity_altitude", "mechanics",
                    "uniform-g energy balance -> inverse-square gravity for heights comparable with the planet radius",
                    "km/s", base, trigger, text, correct, shortcut)


def _apex(v0, g, dt=0.5):
    r, v, mu = RE * 1e3, v0 * 1e3, g * (RE * 1e3) ** 2
    acc = lambda r: -mu / r ** 2
    while v > 0:
        k1r, k1v = v, acc(r)
        k2r, k2v = v + dt / 2 * k1v, acc(r + dt / 2 * k1r)
        k3r, k3v = v + dt / 2 * k2v, acc(r + dt / 2 * k2r)
        k4r, k4v = v + dt * k3v, acc(r + dt * k3r)
        r, v = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r), v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return (r - RE * 1e3) / 1e3


def check() -> str:
    import random
    rng = random.Random("x5-check:gravity_altitude")
    worst = 0.0; gi, go = [], []
    for _ in range(10):
        pin, pout = pair(TEMPLATE, rng)
        worst = max(worst, abs(_apex(correct(pout), pout["g"]) / pout["h"] - 1))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert worst < 2e-3 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP
    return f"RK4 apex vs h {worst:.1e}, inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}"
