"""Optically thin emission T_b = tau T -> isothermal-layer emission T_b = T (1 - e^-tau)
(spec: MECHANISMS.md).

Parent's method: a thin isothermal layer seen against a cold background has brightness
temperature tau T in the Rayleigh-Jeans limit (the linear, optically thin form used for thin cloud
and trace-gas lines in microwave remote sensing).
Where it fails: once the optical depth is not small, the layer absorbs part of its own emission and
T_b = T (1 - e^-tau); at tau = 0.4-1.5 the thin form overestimates by 21-93 %.
Fair from the text: both variants give the layer temperature, its zenith optical depth, a cold
background and the Rayleigh-Jeans definition of brightness temperature, which fix the answer.
Inside: tau = 0.005-0.035; tau / (1 - e^-tau) - 1 < 1.8 %.
Outside: tau = 0.4-1.5. The upper bound keeps an 'optically thick, T_b = T' answer at least 29 %
off, so the outside variant has one right answer, not two.

Conventions: the text fixes the Rayleigh-Jeans brightness temperature, the zenith path and a
background that contributes nothing, so no alternative definition applies. check() recomputes
T_b by integrating the radiative-transfer equation dI/dtau' = B - I through the layer numerically.
The answer is a brightness temperature in kelvin: the outside gap of >= 21 % is far beyond the 5 %
tolerance, so the grader's kelvin canonicalisation cannot let the shortcut pass.
Reference: Petty, A First Course in Atmospheric Radiation (2006), sec. 8.2 (isothermal layer,
T_b = T(1 - t) with transmittance t = e^-tau).
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, num, rnd


def base(rng):
    return {"t": float(rng.randint(250, 290))}


def trigger(rng, regime, base):
    return {"tau": rnd(rng.uniform(0.005, 0.035)) if regime == "inside" else rnd(rng.uniform(0.4, 1.5))}


def shortcut(p):
    return p["tau"] * p["t"]


def correct(p):
    return p["t"] * (1 - math.exp(-p["tau"]))


def text(p, variant):
    given = ("Because the layer is not optically thin, it absorbs part of its own emission: use "
             "T_b = T(1 − e^(−τ)) for an isothermal layer of temperature T and optical depth τ. "
             if variant == "given" else "")
    return (f"A ground-based microwave radiometer points at the zenith through an isothermal cloud layer at a "
            f"temperature of {num(p['t'])} K. At the observing frequency the layer has a zenith optical depth "
            f"of {num(p['tau'])}, and nothing else along the line of sight, including the sky beyond the layer, "
            f"contributes radiation. Use the Rayleigh–Jeans definition of brightness temperature. {given}What "
            f"brightness temperature does the radiometer measure? Express your answer in K.")


def _integrated(t, tau, n=2000):
    """RK4 integration of dI/dtau' = B - I from the far edge (I = 0) to the radiometer, B = T."""
    i, h = 0.0, tau / n
    for _ in range(n):
        k1 = t - i
        k2 = t - (i + h * k1 / 2)
        k3 = t - (i + h * k2 / 2)
        k4 = t - (i + h * k3)
        i += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return i


def check() -> str:
    max_in = max_indep = 0.
    min_out = min_thick = math.inf
    for t in range(250, 291):
        for tau in [i / 1000 for i in range(5, 36)] + [i / 100 for i in range(40, 151)]:
            p = {"t": t, "tau": tau}
            c = correct(p)
            max_indep = max(max_indep, abs(_integrated(t, tau) / c - 1))
            g = abs(shortcut(p) - c) / c
            if tau < .1:
                max_in = max(max_in, g)
            else:
                min_out = min(min_out, g)
                min_thick = min(min_thick, abs(t - c) / c)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and min_thick >= .15 and max_indep < 1e-9
    assert abs(correct({"t": 280, "tau": 1}) / 280 - (1 - 1 / math.e)) < 1e-12
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, optically-thick answer >= "
            f"{min_thick:.3f} off; RK4 transfer integral {max_indep:.1e}")


TEMPLATE = Template("optically_thin_emission", "atmospheric_radiation",
                    "optically thin emission T_b = tau T -> self-absorbing isothermal layer T(1 - e^-tau)",
                    "K", base, trigger, text, correct, shortcut)
