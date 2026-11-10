"""Henry's law with an unchanged gas reservoir -> closed-parcel partitioning (2026-09-26 expansion).

Parent's method: the dissolved concentration is H p, with p the partial pressure the gas had before
the cloud formed.
Where it fails: for a very soluble gas in a closed parcel, dissolution removes a large share of
the gas, and the partial pressure falls. Conserving the total amount per volume of air,
p/(RT) + H p L = p_0/(RT), so [X(aq)] = H p_0 / (1 + H R T L), with L the liquid-water volume per
volume of air (1 g m^-3 -> 10^-6).
Fair from the text: both variants state the mixing ratio *before* the cloud forms, the pressure,
the temperature, the liquid water content, R and that no air or gas enters or leaves the parcel.
The closed-parcel clause is kept in all arms because open vs closed changes the answer by far
more than 5 %. The gas is left unnamed so that no known solubility is recalled.
Inside: H = 1-100 M atm^-1, so H R T L <= 2.4e-3 and the shortcut is within 0.25 %.
Outside: H = 5e4-1e5 M atm^-1 (H2O2-like); with L = 0.2-1.0 g m^-3, H R T L = 0.23-2.4.
Conventions: none that move the answer; the volume the liquid takes from the air (10^-6) is
negligible. check() solves the SI mass balance by bisection, independently of the closed form.
Anchor: the fraction-dissolved relation f = H R T w_L / (1 + H R T w_L) (Seinfeld & Pandis 2016,
ch. 7): for w_L = 1e-6 and T = 298 K, f = 0.5 at H = 4.1e4 M atm^-1, the order of the solubility
above which textbooks describe a species as residing mainly in cloud water (not verified against
the exact S&P wording).

Review (2026-09-26, independent reviewer): the inside/outside texts no longer state
R = 0.08206 L atm mol^-1 K^-1. The shortcut H p never uses R, only the depletion term H R T L
does, so an otherwise unused given pointed at the missing ingredient. R is a universal constant
(as c is in line_broadening); the given arm still states it. Fairness: R is standard knowledge.

Review (2026-09-27, domain + measurement-design review): keyed in nmol/L (was umol/L) and the
inside Henry's constant starts at 2 (was 1) M atm^-1. In umol/L the smallest inside key was
~4e-4, which round(x, 2) turns into 0 (the calibration model rounds in 17 % of its solvers). Now
the smallest inside key is ~0.8 nmol/L. The outside H tops out at 9.4e4 (was 1e5, which rendered
as "1 × 10^5" against plain inside numbers). Depletion inside stays negligible (H R T L < 3e-3).
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

R_LATM = 0.08206                    # L atm mol^-1 K^-1
ATM = 1013.25                       # hPa


def base(rng):
    return {"xi": rnd(rng.uniform(0.5, 5)), "p": float(rng.choice(range(800, 1050, 50))),
            "t": float(rng.randint(275, 290)), "lwc": rnd(rng.uniform(0.2, 1.0))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"h": rnd(10 ** rng.uniform(math.log10(2), 2))}
    return {"h": rnd(rng.uniform(5e4, 9.4e4))}     # below 1e5, which would render as "1 × 10^5"


def _p0(p):
    return p["xi"] * 1e-9 * p["p"] / ATM         # atm


def shortcut(p):
    return p["h"] * _p0(p) * 1e9                 # nmol/L


def correct(p):
    return p["h"] * _p0(p) / (1 + p["h"] * R_LATM * p["t"] * p["lwc"] * 1e-6) * 1e9


def text(p, variant):
    given = ("Because the gas is very soluble, dissolving removes a large fraction of it from the air: conserve "
             "the total amount, so the equilibrium partial pressure is p_0/(1 + H R T L), where p_0 is the partial "
             "pressure before the cloud forms, R = 0.08206 L atm mol^-1 K^-1, and L is the volume of liquid water "
             "per volume of air (1 g m^-3 corresponds to L = 10^-6). " if variant == "given" else "")
    return (f"Before a cloud forms, an air parcel at {num(p['p'])} hPa and {num(p['t'])} K contains a trace gas at a "
            f"mixing ratio of {num(p['xi'])} ppb. A cloud with a liquid water content of {num(p['lwc'])} g m^-3 then "
            f"forms in the parcel at the same pressure and temperature; no air or gas enters or leaves the parcel. "
            f"The gas dissolves in the cloud water without reacting, with a Henry's law constant of "
            f"{num(p['h'])} M atm^-1. Take 1 atm = 1013.25 hPa. {given}What is "
            f"the equilibrium concentration of the dissolved gas in the cloud water? Express your answer in "
            f"nmol L^-1.")


def _bisect(p):
    """Mass balance per m^3 of air in SI: gas moles + dissolved moles = initial gas moles."""
    rsi = R_LATM * 101325 * 1e-3                  # J mol^-1 K^-1
    n0 = _p0(p) * 101325 / (rsi * p["t"])         # mol m^-3
    water_l = p["lwc"] * 1e-3                     # L of water per m^3 of air
    lo, hi = 0.0, _p0(p)
    for _ in range(200):
        mid = (lo + hi) / 2
        total = mid * 101325 / (rsi * p["t"]) + p["h"] * mid * water_l
        lo, hi = (mid, hi) if total < n0 else (lo, mid)
    return p["h"] * (lo + hi) / 2 * 1e9


def check() -> str:
    max_in, min_out, max_err = 0.0, math.inf, 0.0
    for pr in (800, 1000):
        for t in (275, 290):
            for lwc in (0.2, 1.0):
                for h, inside in ((2.0, 1), (100.0, 1), (5e4, 0), (9.4e4, 0)):
                    p = {"xi": 1.0, "p": pr, "t": t, "lwc": lwc, "h": h}
                    max_err = max(max_err, gap(_bisect(p), correct(p)))
                    g = gap(correct(p), shortcut(p))
                    if inside:
                        max_in = max(max_in, g)
                    else:
                        min_out = min(min_out, g)
    assert max_err < 1e-9 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    # Rounding: the smallest inside key (xi 0.5 ppb, 800 hPa, H = 2) survives round(x, 2) within 1 %.
    smallest = correct({"xi": 0.5, "p": 800, "t": 290, "lwc": 1.0, "h": 2.0})
    assert smallest > 0.5 and abs(round(smallest, 2) / smallest - 1) < 0.01, smallest
    h_half = 1 / (R_LATM * 298 * 1e-6)
    assert 3.5e4 < h_half < 4.5e4, h_half
    return (f"bisection vs closed form {max_err:.1e}; inside max gap {max_in:.4f}; outside min gap {min_out:.3f}; "
            f"50 % dissolved at H = {h_half:.3g} M/atm (S&P ~4e4); smallest inside key {smallest:.2f} nmol/L")


TEMPLATE = Template("henry_depletion", "atmospheric_chemistry",
                    "Henry's law with a fixed gas reservoir -> closed-parcel partitioning for a very soluble gas",
                    "nmol/L", base, trigger, text, correct, shortcut)
