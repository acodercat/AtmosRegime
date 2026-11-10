"""Dilute Fick diffusion -> Stefan diffusion through a stagnant gas at high vapour fraction (X5, chemical engineering).

Parent's method: evaporation flux up a tube N = c D (y_1 - y_2) / L (Fick's law, dilute vapour).
Where it fails: the stagnant gas cannot cross the liquid surface, so the vapour diffuses with a bulk (Stefan) flow:
N = (c D / L) ln((1 - y_2) / (1 - y_1)) (Bird, Stewart & Lightfoot, sec. 18.2). At y_1 = 0.40-0.53 the dilute answer is
22-30 % low.
Fair from the text: every arm states the total pressure, temperature, diffusivity, tube length and the liquid's vapour
pressure (the trigger, ordinary data); the vapour fraction at the open end is zero. Nothing is recalled.
Inside: y_1 = 0.010-0.025 (<= 1.3 %). Outside: y_1 = 0.40-0.53 (the arithmetic-mean y_B approximation, a published
form of the same physics, stays within ~5 % of the key there; review 2026-10-12).
Conventions: ideal gas, c = P / (R T); the log-mean form N = c D (y_1 - y_2) / (L y_B,lm) is the same expression.
Review (2026-10-12): the clause "the air does not dissolve in the liquid" was removed (the textbook signal phrase for
the non-diffusing-B case; steady state already forces N_air = 0). This setup is the canonical Stefan-tube example in
chemical-engineering texts, so the trap may be weak.
Asked quantity: molar evaporation flux, mmol m^-2 s^-1.
check(): shoot the steady profile dy/dz = -N (1 - y) / (c D) for N with y(0) = y_1, y(L) = 0 (RK4 + bisection).
Reference: Bird, Stewart & Lightfoot, Transport Phenomena, sec. 18.2; Welty et al., Fundamentals of Momentum, Heat and
Mass Transfer, sec. 26.1.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

RG = 8.314


def base(rng):
    return {"p": 101.0, "t": float(rng.choice(range(300, 351, 5))), "dab": rnd(rng.uniform(0.08, 0.20)),
            "l": rnd(rng.uniform(5, 20))}                               # kPa, K, cm^2/s, cm


def trigger(rng, regime, b):
    y = rng.uniform(0.010, 0.025) if regime == "inside" else rng.uniform(0.40, 0.53)
    return {"pv": rnd(b["p"] * y)}                                      # kPa


def _cdl(p):
    c = p["p"] * 1e3 / (RG * p["t"])
    return c * p["dab"] * 1e-4 / (p["l"] / 100)


def correct(p):
    return _cdl(p) * math.log(1 / (1 - p["pv"] / p["p"])) * 1000


def shortcut(p):
    return _cdl(p) * p["pv"] / p["p"] * 1000


def text(p, variant):
    given = ("The vapour is not dilute, and the air in the tube is stagnant, so the vapour diffuses with a bulk flow: "
             "N = (c D / L) ln((1 - y_2) / (1 - y_1)). " if variant == "given" else "")
    return (f"A volatile liquid fills the bottom of a vertical tube; its surface is {num(p['l'])} cm below the open top, "
            f"across which air flows so that the vapour mole fraction there is zero. At the liquid temperature the "
            f"liquid's vapour pressure is {num(p['pv'])} kPa. The gas in the tube is at {num(p['p'])} kPa and "
            f"{num(p['t'])} K, the vapour-air diffusivity is {num(p['dab'])} cm^2 s^-1. Treat the gas as ideal (R = 8.314 J mol^-1 K^-1) and the process as steady. {given}What is the "
            f"molar evaporation flux of the liquid? Express your answer in mmol m^-2 s^-1.")


TEMPLATE = Template("stefan_diffusion", "chemical_engineering",
                    "dilute Fick flux -> Stefan flow through stagnant gas at high vapour mole fraction",
                    "mmol m^-2 s^-1", base, trigger, text, correct, shortcut)


def _shoot(p):
    y1, L = p["pv"] / p["p"], p["l"] / 100
    cd = p["p"] * 1e3 / (RG * p["t"]) * p["dab"] * 1e-4
    def end(n, steps=2000):
        y, h = y1, L / steps
        f = lambda y: -n * (1 - y) / cd
        for _ in range(steps):
            k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
            y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y
    lo, hi = 0.0, 10 * correct(p) / 1000
    for _ in range(80):
        mid = (lo + hi) / 2
        if end(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * 1000


def check() -> str:
    import random
    rng = random.Random("x5-check:stefan_diffusion")
    worst = 0.0; gi, go = [], []
    for _ in range(20):
        pin, pout = pair(TEMPLATE, rng)
        worst = max(worst, abs(_shoot(pout) / correct(pout) - 1))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert worst < 1e-6 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP
    return f"shooting vs log formula {worst:.1e}, inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}"
