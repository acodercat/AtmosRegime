"""Steady state with chemical loss only -> chemical loss plus ventilation in a one-box model
(extended, air quality, 2026-10-05).

Parent's method: in steady state a pollutant produced at rate P and destroyed with lifetime tau reaches
C = P tau. It is the first box-model result every air-chemistry course teaches.
Where it fails: a box with a wind blowing through it also loses the pollutant by export. For a
well-mixed box of length L along a wind u, with clean inflow, the steady state is
C = P / (1/tau + u/L); the parent's C = P tau holds only when tau << L/u. For a pollutant whose chemical
lifetime is comparable to the ventilation time (a day or so over a 200-350 km region in light wind), export
removes a large share and P tau overestimates C by 25-260 %.
Fair from the text: every arm states P, tau, the box length along the wind, the wind speed, that the box
is well mixed and that the air entering at its upwind edge is clean (the boundary condition any box model
needs; without it the answer is undefined). Nothing names ventilation, export or a residence time.
Inside: tau = 0.20-0.50 h against a ventilation time L/u = 28-97 h: tau u/L <= 1.8 %, so P tau is the key
within 1.8 %. Outside: tau = 24-72 h: tau u/L = 0.25-2.6, the key is 0.28-0.80 of P tau. The shortcut is
P tau = 48-360 ug m^-3, high but within the range of polluted-region secondary aerosol or ozone episodes;
the key is 26-207 ug m^-3.
Ranges: L 200-350 km, u 1.0-2.0 m/s, P 2.0-5.0 ug m^-3 h^-1. The smallest inside key (2 x 0.20 = 0.4 ug m^-3)
survives round(x, 2) within 1.3 % (rule 11).
Conventions: none that move the key. The ventilation rate of a well-mixed box is u/L (Jacob 1999, ch. 3);
a plug-flow (Lagrangian column) model would give a different, position-dependent answer, which is why "well
mixed" is stated in every arm (it moves the answer by more than 5 %, so it stays under rule 4).
Second wrong method (rule 11): using a residence time of L/(2u) (mean transit of an air parcel produced
uniformly in the box) doubles the export term and lands >= 16 % below the key on the outside grid
(`check()`) -- does not pass.
Reference: Jacob, *Introduction to Atmospheric Chemistry* (1999), ch. 3 (one-box model: sources, sinks,
flow in and out, residence time tau = 1/(k + u/L); chapter recalled, section not re-verified). Related:
`nonsteady_burden` (same shortcut C = P tau, but there the missing ingredient is *time* -- the approach to
steady state; here it is an extra *sink*, at steady state) and `cstr_vs_pfr` (well-mixed vs plug-flow
removal in a flushed chamber, a different relation and trigger).
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True


def _vent(p):
    """Ventilation rate u/L, h^-1."""
    return p["u"] * 3600 / (p["l"] * 1e3)


def shortcut(p):
    return p["p"] * p["tau"]


def correct(p):
    return p["p"] / (1 / p["tau"] + _vent(p))


def base(rng):
    return {"l": rnd(rng.uniform(200.0, 350.0)), "u": rnd(rng.uniform(1.0, 2.0)),
            "p": rnd(rng.uniform(2.0, 5.0))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"tau": rnd(rng.uniform(0.20, 0.50))}
    return {"tau": float(rng.randint(24, 72))}


def text(p, variant):
    given = ("The wind also carries the pollutant out of the box: in steady state production balances chemical "
             "loss C/τ plus export at the rate C u/L. " if variant == "given" else "")
    return (f"A polluted region {num(p['l'])} km long in the direction of the wind is treated as a single "
            f"well-mixed box. A secondary pollutant is produced uniformly in the box at {num(p['p'])} ug m^-3 h^-1 "
            f"and is removed chemically with a lifetime of {num(p['tau'])} h. A steady wind of {num(p['u'])} m/s "
            f"blows through the box, and the air entering at its upwind edge contains none of the pollutant. "
            f"{given}What is the steady-state concentration of the pollutant in the box? Express your answer in "
            f"ug m^-3.")


def _integrate(p):
    """Independent: integrate dC/dt = P - C/tau - C u/L from C = 0 with RK4 until it stops changing,
    instead of solving the steady-state algebra."""
    k = 1 / p["tau"] + _vent(p)
    dt = min(0.05 / k, 0.5)
    c = 0.0
    f = lambda x: p["p"] - k * x
    for _ in range(int(40 / (k * dt)) + 1):
        k1 = f(c)
        k2 = f(c + dt * k1 / 2)
        k3 = f(c + dt * k2 / 2)
        k4 = f(c + dt * k3)
        c += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return c


def check() -> str:
    max_in, min_out, max_err, min_wrong = 0.0, math.inf, 0.0, math.inf
    min_key, max_key, min_short, max_short, min_in_key = math.inf, 0.0, math.inf, 0.0, math.inf
    for l in (200.0, 275.0, 350.0):
        for u in (1.0, 1.5, 2.0):
            for pr in (2.0, 5.0):
                b = {"l": l, "u": u, "p": pr}
                for tau in (0.20, 0.50):
                    p = {**b, "tau": tau}
                    max_in = max(max_in, gap(correct(p), shortcut(p)))
                    max_err = max(max_err, gap(correct(p), _integrate(p)))
                    min_in_key = min(min_in_key, correct(p))
                for tau in (24.0, 48.0, 72.0):
                    p = {**b, "tau": tau}
                    c, s = correct(p), shortcut(p)
                    min_out = min(min_out, gap(c, s))
                    max_err = max(max_err, gap(c, _integrate(p)))
                    wrong = pr / (1 / tau + 2 * _vent(p))
                    min_wrong = min(min_wrong, gap(c, wrong))
                    min_key, max_key = min(min_key, c), max(max_key, c)
                    min_short, max_short = min(min_short, s), max(max_short, s)
    assert max_err < 1e-6 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    assert min_wrong > 0.05, min_wrong
    assert min_in_key >= 0.39 and abs(round(min_in_key, 2) / min_in_key - 1) < 0.02, min_in_key
    assert min_short > 0 and max_short <= 400, (min_short, max_short)      # shortcut plausible, not absurd
    # Anchor (Jacob 1999 ch. 3 style): L = 300 km, u = 2 m/s gives a ventilation time of 41.7 h.
    assert abs(1 / _vent({"l": 300.0, "u": 2.0}) - 41.67) < 0.01
    return (f"RK4 integration check {max_err:.1e}; inside max gap {max_in:.4f}; outside min gap {min_out:.3f}; "
            f"L/(2u) route min miss {min_wrong:.3f}; key {min_key:.0f}-{max_key:.0f}, shortcut "
            f"{min_short:.0f}-{max_short:.0f} ug/m^3; smallest inside key {min_in_key:.2f}")


TEMPLATE = Template("box_ventilation_loss", "air_quality",
                    "steady state with chemical loss only (C = P tau) -> chemical loss plus ventilation u/L",
                    "ug/m^3", base, trigger, text, correct, shortcut)
