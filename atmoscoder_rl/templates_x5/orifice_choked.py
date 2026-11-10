"""Incompressible orifice equation -> choked (sonic) flow of a gas (X5, fluid mechanics).

Parent's method: discharge through a small ideal opening, m = A sqrt(2 rho_0 (p_0 - p_b)) (Bernoulli for a
liquid, or for a gas at small pressure drop).
Where it fails: a gas is compressible; once p_b / p_0 falls below (2/(gamma+1))^(gamma/(gamma-1)) = 0.528 (air)
the throat is sonic and m = A p_0 sqrt(gamma / (R T_0)) (2/(gamma+1))^((gamma+1)/(2(gamma-1))), independent of p_b.
The incompressible equation overestimates the flow by 73-96 % there.
Fair from the text: every arm states p_0, T_0, p_b, the opening area, gamma and R, and that the opening is
smoothly rounded, ideal and loss-free (discharge coefficient 1; review 2026-10-12 replaced 'convergent nozzle',
which frames the problem as compressible flow) -- stated in every arm. The trigger is the back pressure.
Inside: p_b / p_0 = 0.990-0.995 (compressibility correction <= 0.4 %, check()). Outside: p_b / p_0 = 0.10-0.30.
Conventions: none for the ideal nozzle; a discharge coefficient is excluded by the text.
Second wrong method (rule 11): the subsonic isentropic formula applied below the critical ratio is 12-48 % low (check()).
Asked quantity: mass flow rate, g/s.
check(): isentropic flow evaluated by marching the Mach number to the throat (area-Mach relation), independent of the
closed form; the critical ratio 0.5283 for gamma = 1.4.
Reference: Anderson, Modern Compressible Flow, ch. 5; White, Fluid Mechanics, sec. 9.4.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

GAMMA, R_AIR = 1.4, 287.0


def base(rng):
    return {"p0": float(rng.choice(range(300, 801, 25))), "t0": float(rng.choice(range(280, 341, 5))),
            "a": rnd(rng.uniform(5, 30))}                                # kPa, K, mm^2


def trigger(rng, regime, b):
    ratio = rng.uniform(0.990, 0.995) if regime == "inside" else rng.uniform(0.10, 0.30)
    return {"pb": rnd(b["p0"] * ratio, 3)}


def _rho0(p):
    return p["p0"] * 1e3 / (R_AIR * p["t0"])


def _isentropic(p, pr):
    g = GAMMA
    return p["a"] * 1e-6 * p["p0"] * 1e3 * math.sqrt(2 * g / ((g - 1) * R_AIR * p["t0"]) * (pr ** (2 / g) - pr ** ((g + 1) / g)))


def correct(p):
    crit = (2 / (GAMMA + 1)) ** (GAMMA / (GAMMA - 1))
    return 1000 * _isentropic(p, max(p["pb"] / p["p0"], crit))


def shortcut(p):
    return 1000 * p["a"] * 1e-6 * math.sqrt(2 * _rho0(p) * (p["p0"] - p["pb"]) * 1e3)


def text(p, variant):
    given = ("Because the pressure ratio is below the critical value, the flow at the nozzle exit is sonic (choked): "
             "m = A p_0 sqrt(gamma / (R T_0)) (2/(gamma+1))^((gamma+1)/(2(gamma-1))). " if variant == "given" else "")
    return (f"Air (gamma = 1.4, R = 287 J kg^-1 K^-1) flows from a large tank, where it is at rest at {num(p['p0'])} kPa "
            f"and {num(p['t0'])} K, through a small, smoothly rounded opening of area {num(p['a'])} mm^2 into surroundings at "
            f"{num(p['pb'])} kPa. Treat the opening as ideal and loss-free (discharge coefficient 1). {given}What is the "
            f"mass flow rate of air? Express your answer in g/s.")


TEMPLATE = Template("orifice_choked", "fluid_mechanics",
                    "incompressible orifice equation -> choked compressible nozzle flow below the critical pressure ratio",
                    "g/s", base, trigger, text, correct, shortcut)


def _by_mach(p):
    """Exit Mach from the pressure ratio (capped at 1), then m = rho V A from static state."""
    g = GAMMA
    pr = p["pb"] / p["p0"]
    m = 1.0 if pr <= (2 / (g + 1)) ** (g / (g - 1)) else math.sqrt(2 / (g - 1) * (pr ** (-(g - 1) / g) - 1))
    t = p["t0"] / (1 + (g - 1) / 2 * m * m)
    pe = p["p0"] * 1e3 * (t / p["t0"]) ** (g / (g - 1))
    return 1000 * pe / (R_AIR * t) * m * math.sqrt(g * R_AIR * t) * p["a"] * 1e-6


def check() -> str:
    import random
    crit = (2 / 2.4) ** 3.5
    assert abs(crit - 0.5283) < 1e-4
    rng = random.Random("x5-check:orifice_choked")
    worst = 0.0; wrong = []; gi, go = [], []
    for _ in range(200):
        pin, pout = pair(TEMPLATE, rng)
        for q in (pin, pout):
            worst = max(worst, abs(_by_mach(q) / correct(q) - 1))
        wrong.append(gap(correct(pout), 1000 * _isentropic(pout, pout["pb"] / pout["p0"])))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert worst < 1e-9 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP and min(wrong) > 0.05
    return (f"Mach route vs closed form {worst:.1e}, unchoked-formula miss {min(wrong):.2f}-{max(wrong):.2f}, "
            f"inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}")
