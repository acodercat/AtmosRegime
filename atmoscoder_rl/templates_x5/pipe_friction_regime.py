"""Hagen-Poiseuille pressure drop -> turbulent Darcy friction in a smooth pipe (X5, fluid mechanics).

Parent's method: fully developed viscous pipe flow, dp = 32 mu L V / D^2 (f = 64 / Re), the closed form for
"viscous flow through a tube".
Where it fails: it holds only for laminar flow (Re < ~2000). At Re = 2-5 x 10^4 the flow is turbulent and
dp = f (L/D) rho V^2 / 2 with f from Colebrook (smooth: 1/sqrt f = -2 log10(2.51 / (Re sqrt f))); the laminar
formula is 8-17x too small.
Fair from the text: every arm states D, L, the mean velocity, density and kinematic viscosity, and that the pipe
is drawn copper (roughness ~1.5 um: f rises at most 1.6 % above the smooth-pipe key on the grid, so the smooth key
stands; review 2026-10-12 replaced 'hydraulically smooth', a phrase that only has meaning for turbulent flow). The trigger is the mean velocity, ordinary data.
Inside: Re = 400-1500 (laminar; the key is the Poiseuille value itself). Outside: Re = 2-5 x 10^4.
Conventions: Blasius (0.316 Re^-1/4), Petukhov ((0.790 ln Re - 1.64)^-2), Haaland (smooth) and Colebrook agree
within 3 % on the outside grid (check()).
Asked quantity: pressure drop over the length, kPa.
check(): Petukhov, Blasius and Haaland against the Colebrook key; Moody anchor f(Re = 10^5, smooth) = 0.0180.
Reference: White, Fluid Mechanics, sec. 6.4-6.7.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd


def base(rng):
    return {"d": rnd(rng.uniform(15, 40)), "l": float(rng.choice(range(10, 51, 5))),
            "rho": 998.0, "nu": rnd(rng.uniform(0.9, 1.1) * 1e-6)}


def trigger(rng, regime, b):
    re = rng.uniform(400, 1500) if regime == "inside" else rng.uniform(2e4, 5e4)
    return {"v": rnd(re * b["nu"] / (b["d"] * 1e-3))}


def _re(p):
    return p["v"] * p["d"] * 1e-3 / p["nu"]


def _colebrook(re):
    f = 0.02
    for _ in range(100):
        f = (-2 * math.log10(2.51 / (re * math.sqrt(f)))) ** -2
    return f


def _dp(p, f):
    return f * p["l"] / (p["d"] * 1e-3) * p["rho"] * p["v"] ** 2 / 2 / 1000


def correct(p):
    re = _re(p)
    return _dp(p, 64 / re if re < 2000 else _colebrook(re))


def shortcut(p):
    return _dp(p, 64 / _re(p))


def text(p, variant):
    given = ("At this Reynolds number the flow is turbulent, not laminar: use the Darcy-Weisbach equation with the "
             "smooth-pipe Colebrook friction factor, 1/sqrt(f) = -2 log10(2.51 / (Re sqrt(f))). " if variant == "given" else "")
    return (f"Water (density {num(p['rho'])} kg m^-3, kinematic viscosity {num(p['nu'])} m^2 s^-1) flows with a mean "
            f"velocity of {num(p['v'])} m s^-1 through a straight, horizontal drawn-copper tube of internal "
            f"diameter {num(p['d'])} mm and length {num(p['l'])} m. The flow is fully developed. {given}What is the "
            f"pressure drop along the pipe? Express your answer in kPa.")


TEMPLATE = Template("pipe_friction_regime", "fluid_mechanics",
                    "laminar Hagen-Poiseuille drop -> turbulent Darcy friction at Re of 10^4",
                    "kPa", base, trigger, text, correct, shortcut)


def check() -> str:
    import random
    assert abs(_colebrook(1e5) - 0.0180) < 3e-4                         # Moody chart, smooth, Re = 1e5
    rng = random.Random("x5-check:pipe_friction_regime")
    spread = 0.0; gi, go = [], []
    for _ in range(200):
        pin, pout = pair(TEMPLATE, rng)
        re, fc = _re(pout), _colebrook(_re(pout))
        for f in (0.316 * re ** -0.25, (0.790 * math.log(re) - 1.64) ** -2, (-1.8 * math.log10(6.9 / re)) ** -2):
            spread = max(spread, abs(f / fc - 1))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert spread < 0.05 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP
    return f"convention spread {spread:.3f}, inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}"
