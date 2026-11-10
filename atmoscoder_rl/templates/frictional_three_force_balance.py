"""Geostrophic wind -> steady three-force (Guldberg-Mohn) balance wind (extended set, 2026-10-01).

Parent's method: with the pressure-gradient force balanced against Coriolis alone,
V_g = a/f, where a = (1/rho) dp/dn is the pressure-gradient force per unit mass.
Where it fails: the text also states a linear ("Rayleigh") frictional deceleration -kV acting on
the wind, steady in time. The force balance a - kV_x + fV_y = 0, -kV_y - fV_x = 0 (the same
du/dt = a - kU + fV, dv/dt = -kV - fU system as `damped_inertial_oscillation`, here at its t ->
infinity steady state) gives speed V = a/sqrt(f^2+k^2) = V_g * f/sqrt(f^2+k^2): friction turns the
wind sub-geostrophic and backs it toward low pressure (the classical Guldberg-Mohn triangle).
Nothing about the stated k looks extreme -- it is an ordinary near-surface linear-drag value, the
same order of magnitude as f itself -- but once k is comparable to f, V_g alone misses the balance
speed by well over the grading tolerance.
Fair from the text: every arm gives the latitude, density, the pressure gradient and the friction
coefficient k; nothing flags whether k matters relative to f, so the only way to know is to compare
them.
Inside: k is drawn as 0.1-1 % of f (k/f ~ 1e-3-1e-2), the balance speed is within 0.01 % of V_g --
far inside the 2 % bound, so the margin is not doing the work; see `check()`. Outside: k is drawn
as 0.65-1.6 times f (f is 7.3e-5-1.26e-4 s^-1 over 30-60 N, so k lands in 4.7e-5-2.0e-4 s^-1, an
ordinary near-surface drag coefficient, not an outlying number), and V_g misses the balance speed
by at least 19 %.
Conventions: none beyond Omega (7.27e-5 vs 7.292e-5 s^-1 moves the key by well under 1 %, checked
on the outside grid). Steady, linear (Rayleigh) friction is the textbook simplification (Guldberg &
Mohn, 1876; the same linearisation used for the Ekman-layer-averaged boundary wind); a more complete
treatment (quadratic drag, or an Ekman spiral integrated through depth) is a different physical
model, not an alternative numeric convention for this one, so it is not folded into the 5 % spread
check -- the docstring states this rather than hiding it.
Reference: Holton & Hakim (5th ed.), ch. 5 (planetary boundary layer, the linear-drag balance); Stull, An Introduction to Boundary Layer Meteorology
(1988), sec. 12.2 (the frictional triangle of forces). Related: `damped_inertial_oscillation` (the
time-dependent edge of the same force balance, trigger = k t rather than k alone) and
`isallobaric_wind` (another V_g-based shortcut that an unsteady or frictional complication breaks).
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

OMEGA = 7.292e-5


def f(p):
    return 2 * OMEGA * math.sin(math.radians(p["lat"]))


def base(rng):
    return {"lat": float(rng.randint(30, 60)), "rho": float(rng.choice(range(110, 126))) / 100,
            "dp": float(rng.choice([1.0, 1.5, 2.0, 2.5]))}


def trigger(rng, regime, base):
    ratio = rng.uniform(0.001, 0.01) if regime == "inside" else rng.uniform(0.65, 1.6)
    return {"k": rnd(ratio * f(base))}


def accel(p):
    return p["dp"] * 1e-3 / p["rho"]                                   # hPa/100 km -> Pa/m -> m/s^2


def shortcut(p):
    return accel(p) / f(p)


def correct(p):
    return accel(p) / math.hypot(f(p), p["k"])


def text(p, variant):
    given = ("Because k is not small compared with f, friction is not negligible: the steady force "
              "balance a - kV_x + fV_y = 0, -kV_y - fV_x = 0 gives a sub-geostrophic speed "
              "V = a/sqrt(f² + k²), backed from the geostrophic direction toward low pressure. "
              if variant == "given" else "")
    return (f"At latitude {num(p['lat'])}°N, air of density {num(p['rho'])} kg m^-3 is in steady horizontal "
            f"motion under a pressure gradient of {num(p['dp'])} hPa per 100 km and a linear frictional "
            f"deceleration -kV, with k = {num(p['k'])} s^-1. Take Ω = 7.292 × 10^-5 s^-1. {given}Find the steady wind speed. "
            f"Express your answer in m s^-1.")


def check() -> str:
    max_in = min_out = None
    spread = max_key = 0.
    min_key = 1e9
    for lat in range(30, 61):
        for rho in [1.10, 1.15, 1.20, 1.25]:
            for dp in [1.0, 1.5, 2.0, 2.5]:
                b = {"lat": lat, "rho": rho, "dp": dp}
                ff = f(b)
                for ratio in [0.001, 0.005, 0.01]:
                    p = {**b, "k": rnd(ratio * ff)}
                    key = correct(p)
                    # Independent: solve the 2x2 linear system [[−k, f], [−f, −k]] V = [−a, 0]
                    # by Cramer's rule, instead of the closed-form a/hypot(f, k).
                    det = (-p["k"]) * (-p["k"]) - f(p) * (-f(p))
                    vx = ((-accel(p)) * (-p["k"]) - f(p) * 0) / det
                    vy = ((-p["k"]) * 0 - (-f(p)) * (-accel(p))) / det
                    assert math.isclose(key, math.hypot(vx, vy), rel_tol=1e-9), p
                    g = gap(key, shortcut(p))
                    max_in = g if max_in is None else max(max_in, g)
                    min_key = min(min_key, key)
                for ratio in [0.65, 1.0, 1.6]:
                    p = {**b, "k": rnd(ratio * ff)}
                    key = correct(p)
                    det = (-p["k"]) * (-p["k"]) - f(p) * (-f(p))
                    vx = ((-accel(p)) * (-p["k"]) - f(p) * 0) / det
                    vy = ((-p["k"]) * 0 - (-f(p)) * (-accel(p))) / det
                    assert math.isclose(key, math.hypot(vx, vy), rel_tol=1e-9), p
                    g = gap(key, shortcut(p))
                    min_out = g if min_out is None else min(min_out, g)
                    max_key = max(max_key, key)
                    min_key = min(min_key, key)
                    ffalt = 2 * 7.27e-5 * math.sin(math.radians(lat))
                    alt = accel(p) / math.hypot(ffalt, p["k"])
                    spread = max(spread, abs(alt / key - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and spread < .05
    assert min_key > 1.0 and max_key < 45
    # Anchor: at k = f the balance speed is V_g/sqrt(2) (the classical 45-degree backing case).
    p = {"lat": 45, "rho": 1.2, "dp": 1.5}
    p["k"] = f(p)
    assert math.isclose(correct(p), shortcut(p) / math.sqrt(2), rel_tol=1e-12)
    return (f"Cramer's-rule solve agrees; max inside gap {max_in:.6f}, min outside gap {min_out:.3f}, "
            f"Omega-convention spread {spread:.4f}, key range {min_key:.1f}-{max_key:.1f} m/s; "
            f"k=f anchor gives V_g/sqrt(2)")


TEMPLATE = Template("frictional_three_force_balance", "boundary_layer",
                    "geostrophic wind -> steady Guldberg-Mohn balance wind when k is not small compared with f",
                    "m/s", base, trigger, text, correct, shortcut)
