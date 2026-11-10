"""Steady-state buildup S*tau -> exact non-steady approach to steady state (2026-10-01 extension).

Parent's method: a species with a constant source and first-order loss reaches a steady-state
concentration C_ss = S tau, with S the (constant) production rate and tau the first-order loss
lifetime; after "a while" a solver reports C_ss as the concentration at the stated time.
Where it fails: the system only approaches C_ss exponentially. Starting from C(0) = 0 (the
parcel's prior burden of this species has just been removed, e.g. by a rain-out or a clean-air
advection event), dC/dt = S - C/tau gives C(t) = S tau (1 - e^{-t/tau}); at t comparable with tau
the concentration is well below C_ss, and the shortcut overestimates it. Nothing about the
elapsed time looks extreme in either arm -- "after 2 days" against a 3-day lifetime is an entirely
ordinary way to pose an atmospheric box-model problem, and a solver that only checks whether S or
tau looks unusual will not notice that t itself is the quantity that matters.
Fair from the text: every variant states S, tau and t (t is the trigger and the only given that
differs between inside and outside). No variant mentions "steady state" or the exponential form;
"given" states the closed-form solution.
Inside: t/tau >= 4.5, so the shortcut's relative error is e^{-t/tau}/(1 - e^{-t/tau}) <= 1.2 %
(worst case t/tau = 4.5: e^-4.5 = 0.0111, gap = 0.0111/0.9889 = 1.12 %).
Outside: t/tau in [0.3, 2.0], so the shortcut's relative error ranges from about 286 % (t/tau=0.3)
down to 15.6 % (t/tau=2.0, the binding corner): check() confirms the floor stays above 15 %.
Conventions: none -- this is the exact solution of a linear first-order ODE with constant
coefficients, not a choice among competing closures; check() verifies it two ways (RK4 numerical
integration and the Laplace/undetermined-coefficients closed form agree to machine precision).
Anchor: the "number of lifetimes to approach steady state" rule of thumb in atmospheric box
models -- after 3 lifetimes a species is within 5 % of steady state, after 5 lifetimes within 1 %
(Jacob, *Introduction to Atmospheric Chemistry*, ch. 3, on the e-folding response of a reservoir to
a change in its source); check() reproduces both figures from the closed form.
Related: `henry_depletion` and `pseudo_first_order` are also exact-vs-approximate-kinetics traps,
but both turn on reservoir size/stoichiometry, not on elapsed time against a lifetime.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True


def base(rng):
    return {"s": rnd(rng.uniform(5, 30)), "tau": rnd(rng.uniform(1, 5))}


def trigger(rng, regime, base):
    ratio = rng.uniform(4.5, 8.0) if regime == "inside" else rng.uniform(0.3, 2.0)
    return {"t": rnd(base["tau"] * ratio)}


def shortcut(p):
    """Steady-state concentration S*tau, ug/m^3."""
    return p["s"] * p["tau"]


def correct(p):
    """Exact solution of dC/dt = S - C/tau, C(0) = 0, ug/m^3."""
    return p["s"] * p["tau"] * (1 - math.exp(-p["t"] / p["tau"]))


def text(p, variant):
    given = ("Because the elapsed time is not much longer than the lifetime, the concentration has not yet "
             "reached its steady-state value: it follows C(t) = S*tau*(1 - e^(-t/tau)), with t = 0 the moment the "
             "parcel's prior burden of the pollutant was removed. " if variant == "given" else "")
    return (f"A rain event scavenges a secondary pollutant from an air parcel to zero concentration. Right after "
            f"the rain stops, the pollutant begins forming again in the parcel at a constant rate of "
            f"{num(p['s'])} ug m^-3 day^-1, and is removed by deposition and chemical loss with an effective "
            f"first-order lifetime of {num(p['tau'])} days. {given}What is the pollutant's concentration "
            f"{num(p['t'])} days after the rain stops? Express your answer in ug m^-3.")


def _rk4(s, tau, t, steps=4000):
    """Independent numerical integration of dC/dt = S - C/tau, C(0) = 0."""
    dt = t / steps
    f = lambda c: s - c / tau
    c = 0.0
    for _ in range(steps):
        k1 = f(c); k2 = f(c + dt * k1 / 2); k3 = f(c + dt * k2 / 2); k4 = f(c + dt * k3)
        c += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return c


def check() -> str:
    max_in, min_out, max_err = 0.0, math.inf, 0.0
    for s in (5, 30):
        for tau in (1, 5):
            for ratio, inside in ((4.5, 1), (8.0, 1), (0.3, 0), (2.0, 0)):
                p = {"s": s, "tau": tau, "t": tau * ratio}
                max_err = max(max_err, gap(_rk4(s, tau, p["t"]), correct(p)))
                g = gap(correct(p), shortcut(p))
                if inside:
                    max_in = max(max_in, g)
                else:
                    min_out = min(min_out, g)
    assert max_err < 1e-6 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    # the rule-of-thumb anchor: after 3 lifetimes within 5 %, after 5 within 1 %
    g3 = gap(1.0, 1 - math.exp(-3))
    g5 = gap(1.0, 1 - math.exp(-5))
    assert g3 < 0.05 and g5 < 0.01, (g3, g5)
    smallest = correct({"s": 5, "tau": 5, "t": 5 * 0.3})
    assert smallest > 0.1 and abs(round(smallest, 2) / smallest - 1) < 0.05, smallest
    return (f"RK4 vs closed form {max_err:.1e}; inside max gap {max_in:.4f}; outside min gap {min_out:.3f}; "
            f"3-lifetime gap {g3:.3f} (anchor <5%), 5-lifetime gap {g5:.4f} (anchor <1%); "
            f"smallest key {smallest:.3f} ug/m^3")


TEMPLATE = Template("nonsteady_burden", "atmospheric_chemistry",
                    "steady-state buildup S*tau -> exact non-steady approach to steady state after a rain event",
                    "ug/m^3", base, trigger, text, correct, shortcut)
