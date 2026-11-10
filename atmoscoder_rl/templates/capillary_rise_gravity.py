"""Lucas-Washburn capillary rise h^2 = r sigma t/(2 mu) -> rise slowed by the weight of the water column
(extended, soil physics, 2026-10-05).

Parent's method: the rate of capillary uptake by a porous medium from the Lucas-Washburn law,
h = sqrt(r sigma t/(2 mu)) for a wetting liquid (contact angle 0) in pores of radius r -- the square-root
-of-time law of capillary absorption and sorptivity, quoted in every treatment of wicking.
Where it fails: the law balances the capillary pressure 2 sigma/r against viscous (Poiseuille) drag
only. For upward rise the hydrostatic weight of the risen column rho g h also opposes the suction:
8 mu h dh/dt = r^2 (2 sigma/r - rho g h). The rise slows and tends to the Jurin height
h_eq = 2 sigma/(rho g r); integrating, t = (8 mu h_eq/(r^2 rho g)) [-x - ln(1 - x)], x = h/h_eq
(Washburn 1921, his general equation; Lu & Likos 2004). Lucas-Washburn is the early-time limit
(x << 1) and overshoots by 20-33 % once the front is half-way to the Jurin height.
Fair from the text: every arm states that the soil's pores are treated as vertical cylindrical capillaries
of a stated radius, that water wets them completely, the surface tension and viscosity of water, and
the elapsed time since the column's base touched the water -- everything Lucas-Washburn needs. The
density of water and g are handbook constants used only by the gravity term, so they are stated only in
the given sentence (rule 3 as reworded 2026-09-27); 998-1000 kg/m^3 and g = 9.80-9.81 m/s^2 move the key by
< 0.1 % (checked). Nothing in the inside or outside text mentions gravity, the column's weight or the
equilibrium height; "rises" is the ordinary description of capillary rise and is in all arms.
Inside: elapsed time such that x = h/h_eq = 0.020-0.045 (seconds to about a minute); Lucas-Washburn is
within 1.5 % of the key.
Outside: x = 0.45-0.60 -- an elapsed time of minutes to several hours for pore radii 5-15 um (silt to very
fine sand; Jurin heights 1.0-3.0 m); the shortcut height is 20-33 % above the key and still below the
Jurin height (<= 0.81 h_eq), so nothing in its own number looks absurd.
Conventions: the capillary-tube model with Poiseuille flow is the textbook model and is stated; inertia
is negligible (rho r^2/mu ~ 1e-7 s). The density/g spread is the only recalled-constant spread (< 0.1 %).
Second wrong method (rule 11): answering the equilibrium Jurin height is 67-122 % above the key (`check()`).
check(): integrates the Poiseuille-capillary-gravity equation dh/dt numerically (RK4, started from the
Lucas-Washburn solution at a tiny time) -- independent of the implicit closed form, which correct() inverts
by bisection.
Reference: Washburn (1921), Phys. Rev. 17, 273 (dynamics of capillary flow, with the gravity term);
Lu & Likos, *Unsaturated Soil Mechanics* (2004), ch. 2 (capillarity; rate of capillary rise in a tube and in
soil); Hillel, *Environmental Soil Physics* (1998), ch. 3-4 (capillary-tube model of soil pores, Jurin
height). Chapter numbers not verified.

Cluster: new -- differs from size_vs_length_scale (kelvin_curvature also uses the capillary pressure
2 sigma/r, but as a static vapour-pressure shift with no flow) and from box_model (a relaxation toward a
steady value reported too early) because the parent is a transient early-time law (viscous drag vs
capillary suction) and the missing ingredient is a third force, the weight of the risen water, that
makes the front decelerate toward an equilibrium height the shortcut never has.
Independent review (2026-10-05): accepted (inside-arm difficulty for an 8B model is borderline -- it needs
the Lucas-Washburn law recalled; screening decides). Cluster ruling: new.
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

SIGMA, MU = 0.0728, 1.00e-3          # N/m, Pa s (stated in all arms)
RHO, G = 998.0, 9.81                  # kg/m^3, m/s^2 (stated only in the given sentence)


def _scales(r_um, rho=RHO, g=G):
    r = r_um * 1e-6
    h_eq = 2 * SIGMA / (rho * g * r)                       # m
    tau = 8 * MU * h_eq / (r * r * rho * g)               # s
    return h_eq, tau


def shortcut(p):
    """Lucas-Washburn, cm."""
    return math.sqrt(p["r"] * 1e-6 * SIGMA * p["t"] / (2 * MU)) * 100


def _height(p, rho=RHO, g=G):
    h_eq, tau = _scales(p["r"], rho, g)
    target = p["t"] / tau
    lo, hi = 0.0, 1.0
    for _ in range(100):                                  # -x - ln(1 - x) rises monotonically on (0, 1)
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if -mid - math.log(1 - mid) < target else (lo, mid)
    return (lo + hi) / 2 * h_eq * 100


def correct(p):
    """Capillary rise against gravity (Washburn's full equation), cm."""
    return _height(p)


def base(rng):
    return {"r": rnd(rng.uniform(5.0, 15.0))}             # um


def trigger(rng, regime, base):
    x = rng.uniform(0.020, 0.045) if regime == "inside" else rng.uniform(0.45, 0.60)
    _, tau = _scales(base["r"])
    return {"t": rnd(tau * (-x - math.log(1 - x)))}       # s


def text(p, variant):
    given = ("The weight of the risen water opposes the capillary suction: Poiseuille flow in the pore gives "
             "8μh dh/dt = r²(2σ/r − ρgh), with ρ = 998 kg m^-3 and g = 9.81 m s^-2, so the rise slows toward "
             "the height 2σ/(ρgr). " if variant == "given" else "")
    return (f"The base of a column of dry, uniform silty soil is placed in contact with a free water surface, and "
            f"water rises into the column by capillarity. Treat the soil's pores as vertical cylindrical capillaries "
            f"of radius {num(p['r'])} μm that water wets completely (contact angle 0); the water has a surface "
            f"tension of 0.0728 N m^-1 and a viscosity of 1.00 × 10^-3 Pa s. {given}How high above the free water "
            f"surface has the wetting front risen {num(p['t'])} s after contact? Express your answer in cm.")


TEMPLATE = Template("capillary_rise_gravity", "soil",
                    "Lucas-Washburn capillary rise -> rise against gravity toward the Jurin height",
                    "cm", base, trigger, text, correct, shortcut)


def _rk4(p, n=20000):
    """Independent: integrate dh/dt = r^2 (2 sigma/r - rho g h)/(8 mu h) in log time from a tiny t0."""
    r = p["r"] * 1e-6
    f = lambda h: r * r * (2 * SIGMA / r - RHO * G * h) / (8 * MU * h)
    t0 = p["t"] * 1e-9
    h = math.sqrt(r * SIGMA * t0 / (2 * MU))              # Lucas-Washburn start (gravity negligible)
    s0, s1 = math.log(t0), math.log(p["t"])
    ds = (s1 - s0) / n
    g = lambda s, h: math.exp(s) * f(h)                   # dh/ds with s = ln t
    s = s0
    for _ in range(n):
        k1 = g(s, h)
        k2 = g(s + ds / 2, h + ds * k1 / 2)
        k3 = g(s + ds / 2, h + ds * k2 / 2)
        k4 = g(s + ds, h + ds * k3)
        h += ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        s += ds
    return h * 100


def check() -> str:
    max_in, min_out, max_out, max_err, spread = 0.0, math.inf, 0.0, 0.0, 0.0
    min_wrong, max_frac, min_key = math.inf, 0.0, math.inf
    t_out = [math.inf, 0.0]
    for r in (5.0, 10.0, 15.0):
        h_eq, tau = _scales(r)
        for x, inside in ((0.020, 1), (0.045, 1), (0.45, 0), (0.52, 0), (0.60, 0)):
            p = {"r": r, "t": tau * (-x - math.log(1 - x))}
            c, s = correct(p), shortcut(p)
            max_err = max(max_err, gap(c, _rk4(p)))
            min_key = min(min_key, c)
            g = gap(c, s)
            if inside:
                max_in = max(max_in, g)
                continue
            min_out, max_out = min(min_out, g), max(max_out, g)
            max_frac = max(max_frac, s / (h_eq * 100))
            t_out = [min(t_out[0], p["t"]), max(t_out[1], p["t"])]
            min_wrong = min(min_wrong, gap(c, h_eq * 100))                     # Jurin height
            for rho, gg in ((1000.0, 9.81), (998.0, 9.80), (1000.0, 9.80665)):
                spread = max(spread, gap(c, _height(p, rho, gg)))
    rng = random.Random("capillary-rise-check")
    for _ in range(3000):
        b = base(rng)
        pin, pout = {**b, **trigger(rng, "inside", b)}, {**b, **trigger(rng, "outside", b)}
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        min_out = min(min_out, gap(correct(pout), shortcut(pout)))
        min_key = min(min_key, correct(pin))
        max_frac = max(max_frac, shortcut(pout) / (_scales(pout["r"])[0] * 100))
    assert max_err < 1e-5, max_err
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert spread < 0.05 and min_wrong > 0.10, (spread, min_wrong)
    assert max_frac < 0.85 and min_key > 1.0, (max_frac, min_key)      # below Jurin; cm clear of round(x, 2)
    # Anchor: Jurin height of water in a 10 um tube, 2 sigma/(rho g r) = 1.49 m (textbook ~1.5 m).
    h10 = _scales(10.0)[0]
    assert abs(h10 - 1.487) < 0.005, h10
    return (f"RK4 Washburn-with-gravity vs implicit solution {max_err:.1e}; inside max gap {max_in:.4f}; outside gap "
            f"{min_out:.3f}-{max_out:.3f}; outside t {t_out[0]:.0f}-{t_out[1]:.0f} s; rho/g spread {spread:.4f}; "
            f"Jurin-height min miss {min_wrong:.2f}; shortcut <= {max_frac:.2f} h_eq; min key {min_key:.1f} cm; "
            f"anchor h_eq(10 um) = {h10:.2f} m")
