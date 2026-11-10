"""Linear local-slope approximation -> logarithmic CO2 radiative forcing (2026-10-01 expansion).

Parent's method: scale the top-of-atmosphere forcing change linearly, using the per-ppm rate
measured at the reference concentration: dF = S (C - C0).
Where it fails: CO2 forcing is logarithmic in the concentration ratio, dF = K ln(C/C0) with
K = S C0 (the two expressions share the same tangent at C0 by construction); away from C0 the
log and its tangent line diverge.
Fair from the text: every arm states the reference level C0 = 280 ppm and the per-ppm forcing
rate S = 0.0191 W m^-2 measured at that level -- exactly what both the linear and the logarithmic
expression need (K = S C0 is derived from the stated S and C0, not from any separately-asserted
coefficient). Neither arm says the rate is a tangent, an approximation, or that it changes away
from C0 -- it is phrased only as "at 280 ppm, +1 ppm gives +0.0191 W m^-2", a plain data point. A
solver must recognise the logarithmic dependence unprompted, from general knowledge of CO2
forcing, not from a cue in the text.
Inside: C = 287-291 ppm (0.025-0.038 x C0 above the reference), where the log and its tangent
line agree within 1.9 %.
Outside: C = 420-700 ppm, ordinary present-day-to-end-of-century concentrations; the linear
extrapolation misses by 23-64 %.
Conventions: none beyond S and C0, both stated; S C0 = 5.348 W m^-2 is the tangent-matched
coefficient used throughout, not a separately asserted "standard" value.
Anchor: the standard simplified expression dF = 5.35 ln(C/C0) (Myhre et al. 1998) gives S C0 =
5.348 here; a doubling from 280 ppm gives 3.71 W m^-2, matching the commonly quoted ~3.7 W m^-2
forcing for a CO2 doubling.
Related mechanism: none in this set; the closest analogue is planck_linearization (also a
Taylor-expansion-vs-exact trap, different physics).

Design note (2026-10-01, in response to a review that rejected sibling templates for this):
inside/outside never say the stated rate is local, approximate, or a tangent, and never use the
word "linear" -- the given sentence is the only place the logarithmic form and its formula appear.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

C0 = 280.0
S = 0.0191           # W m^-2 ppm^-1, the stated per-ppm rate at C0
K = S * C0            # W m^-2, the tangent-matched logarithmic coefficient (~5.35, Myhre et al. 1998)


def base(rng):
    return {}


def trigger(rng, regime, base):
    # The trigger is the increment above C0, not the absolute concentration: C is a 3-digit
    # number, so rounding it to 3 sig figs over the narrow inside band leaves only ~5 distinct
    # values (too few for 100 built rows); the increment has smaller magnitude and so keeps
    # 3-sig-fig precision without widening the gap-safe range.
    x = rng.uniform(0.025, 0.038) if regime == "inside" else rng.uniform(0.5, 1.5)
    return {"dc": rnd(C0 * x, 3)}


def shortcut(p):
    return S * p["dc"]


def correct(p):
    return K * math.log(1 + p["dc"] / C0)


def text(p, variant):
    given = ("Because this forcing increases with the logarithm of the concentration ratio rather than "
             "linearly, scale it as ΔF = 5.35 ln(C/C0) W m^-2 (the coefficient matched so the two "
             "expressions agree at C0). " if variant == "given" else "")
    return (f"At a CO2 concentration of 280 ppm, increasing the concentration by 1 ppm raises the "
            f"top-of-atmosphere radiative forcing by about 0.0191 W m^-2. {given}If the CO2 concentration "
            f"rises by {num(p['dc'])} ppm above this 280 ppm reference, what is the increase in radiative "
            f"forcing? Express your answer in W m^-2.")


TEMPLATE = Template("co2_forcing_log", "climate_dynamics",
                    "linear local-slope CO2 forcing -> logarithmic forcing dF = K ln(C/C0)",
                    "W m^-2", base, trigger, text, correct, shortcut)


def check() -> str:
    max_in, min_out = 0.0, 1.0
    xs_in = [0.025, 0.030, 0.035, 0.038]
    xs_out = [0.5, 0.75, 1.0, 1.25, 1.5]
    for x in xs_in + xs_out:
        c = C0 * (1 + x)
        p = {"dc": c - C0}
        # Independent: Simpson's-rule integration of dF/dC = K/C from C0 to c.
        n = 2000
        h = (c - C0) / n
        s = 1 / C0 + 1 / c
        for i in range(1, n):
            xi = C0 + i * h
            s += (4 if i % 2 else 2) / xi
        integral = K * h / 3 * s
        assert math.isclose(integral, correct(p), rel_tol=1e-6)
        g_ = gap(correct(p), shortcut(p))
        if x in xs_in:
            max_in = max(max_in, g_)
        else:
            min_out = min(min_out, g_)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP
    assert abs(K - 5.35) < 0.01
    assert abs(K * math.log(2) - 3.7) < 0.02
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, "
            f"K={K:.3f} W/m^2 (doubling forcing {K*math.log(2):.3f})")
