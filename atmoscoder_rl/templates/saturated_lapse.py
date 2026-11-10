"""Dry adiabatic lapse rate -> saturated adiabatic lapse rate (source trap: holton_50_gen).

Parent's method: a rising parcel cools at g / c_p.
Where it fails: once the parcel is saturated, condensation releases latent heat and it cools at
Gamma_s = g (1 + L_v r_s / (R_d T)) / (c_p + L_v^2 r_s / (R_v T^2)).
Fair from the text: both variants state the relative humidity and give every constant and the
saturation-vapour-pressure formula, so r_s is computable; outside, the parcel is at 100 %.
The answer is a lapse rate, not a temperature, on purpose: the grader converts degC and K to
kelvin before comparing, so a final temperature would be graded to within 5 % of ~290 K and the
dry-adiabat answer would pass. The textbook dilute-vapor approximation is used; the fuller moist-mixture
expression differs by at most 2.02 % across this grid.
Inside: relative humidity 30-70 %, so the parcel starts unsaturated and cools at g / c_p exactly.

Review (2026-09-24):
Revised checks 4, 5, 10 and 11: disambiguate Celsius T_C in the vapor-pressure fit
from kelvin T_K in thermodynamics, and define r_s and K/m in the given formula.
Specify initially zero condensate, equilibrium liquid-water phase changes, no
entrainment and the dilute-vapor model in ALL arms. No parameter range or solver
changed. Initially zero condensate makes reversible and pseudoadiabatic ascent
have the same initial derivative; a finite pre-existing liquid loading would
not. The answer is the positive initial cooling gradient, not a finite-ascent
temperature or an environmental lapse rate. The dilute first law
c_p*dT + g*dz = -L_v*dr_s with Clausius-Clapeyron gives the implemented relation.
At 26 C and 1013.25 hPa it gives 3.75638 K/km, consistent with Stull
Practical Meteorology, eq. (4.37), worked value 3.73 K/km:
https://www.eoas.ubc.ca/books/Practical_Meteorology/mse3/Ch04-Moist.pdf
Across the grid a full moist-mixture formula (c_pv=1850 J/kg/K) differs by
0.54-2.02%; using specific humidity instead of mixing ratio in the dilute formula
also stays within tolerance. A supplied constant L_v must not be replaced by a
temperature-dependent value. All constants remain identical across the arms.

Trim (2026-09-24): qualifiers the review added in both arms were removed where they do not
move the answer outside the 5 % tolerance (the review's own alternative-convention numbers).
In a trap arm, every extra qualifier is a hint about the physics the solver should notice.
Removed: 'relative to liquid water', no condensate, no entrainment, hydrostatic
environment, equilibrium phase change, dilute-vapour thermodynamics (review: all conventions
within 2.0 %). The given arm still defines T_K and r_s.
"""
from __future__ import annotations

import math
from itertools import product

from .base import OUTSIDE_MIN_GAP, Template, gap, num

G, CP, RD, RV, LV = 9.81, 1005.0, 287.0, 461.5, 2.5e6


def base(rng):
    return {"p": float(rng.choice(range(850, 1010, 10))), "t": float(rng.randint(10, 30))}


def trigger(rng, regime, base):
    return {"rh": float(rng.choice(range(30, 75, 5)) if regime == "inside" else 100)}


def shortcut(p):
    return G / CP * 1000                                      # K per km


def correct(p):
    if p["rh"] < 100:
        return shortcut(p)
    es = 6.112 * math.exp(17.67 * p["t"] / (p["t"] + 243.5))
    rs = RD / RV * es / (p["p"] - es)
    tk = p["t"] + 273.15
    return G * (1 + LV * rs / (RD * tk)) / (CP + LV ** 2 * rs / (RV * tk ** 2)) * 1000


def text(p, variant):
    given = ("Because the parcel is saturated, condensation releases latent heat as it rises: use the saturated "
             "adiabatic lapse rate Γ_s = g(1 + L_v r_s/(R_d T_K))/(c_p + L_v^2 r_s/(R_v T_K^2)), "
             "where T_K is the temperature in kelvin, r_s = (R_d/R_v)e_s/(p - e_s) in kg/kg of dry air, with p and e_s in the same "
             "pressure units; this expression gives Γ_s in K/m. " if variant == "given" else "")
    return (f"An air parcel at a pressure of {num(p['p'])} hPa has a temperature of {num(p['t'])} °C and a relative "
            f"humidity of {num(p['rh'])}%. It begins to rise adiabatically. Take g = 9.81 m s^-2, "
            f"c_p = 1005 J kg^-1 K^-1, R_d = 287 J kg^-1 K^-1, R_v = 461.5 J kg^-1 K^-1, L_v = 2.5 × 10^6 J kg^-1, "
            f"and the saturation vapour pressure over liquid water e_s = 6.112 exp[17.67 T/(T + 243.5)] hPa with "
            f"T in °C. {given}At what rate does the parcel's temperature decrease with height at the start of "
            f"its ascent? Express your answer in K km^-1.")


TEMPLATE = Template("saturated_lapse", "atmospheric_thermodynamics",
                    "dry adiabatic lapse rate -> saturated adiabatic lapse rate for a saturated parcel",
                    "K/km", base, trigger, text, correct, shortcut)


def check() -> str:
    """Full moist-mixture first law, specific-humidity form, the literal fit derivative, and
    the moist dry adiabat inside; Stull's worked value."""
    max_full = max_q = max_fit = max_dry = 0.0
    eps = RD / RV
    for temp, press in product(range(10, 31), range(850, 1010, 10)):
        tk = temp + 273.15
        es = 6.112 * math.exp(17.67 * temp / (temp + 243.5))
        r = eps * es / (press - es)
        key = correct({"t": temp, "p": press, "rh": 100})
        full = G * (1 + r) * (1 + LV * r / (RD * tk)) / (
            CP + 1850 * r + LV ** 2 * r * (1 + r / eps) / (RV * tk ** 2)) * 1000
        max_full = max(max_full, abs(full / key - 1))
        q = r / (1 + r)
        by_q = G * (1 + LV * q / (RD * tk)) / (CP + LV ** 2 * q / (RV * tk ** 2)) * 1000
        max_q = max(max_q, abs(by_q / key - 1))
        dr_dt = r * (1 + r / eps) * 17.67 * 243.5 / (temp + 243.5) ** 2
        fit = G * (1 + r) * (1 + LV * r / (RD * tk)) / (CP + 1850 * r + LV * dr_dt) * 1000
        max_fit = max(max_fit, abs(fit / key - 1))
        for rh in range(30, 75, 5):
            ru = eps * es * rh / 100 / (press - es * rh / 100)
            max_dry = max(max_dry, abs(G * (1 + ru) / (CP + 1850 * ru) * 1000 / shortcut({}) - 1))
        assert gap(key, shortcut({})) >= OUTSIDE_MIN_GAP
    assert max(max_full, max_q, max_fit) < .05 and max_dry < .02
    assert math.isclose(correct({"t": 26, "p": 1013.25, "rh": 100}), 3.73, rel_tol=.01)
    return (f"deviations: full mixture {max_full:.4f}, specific humidity {max_q:.4f}, "
            f"literal fit {max_fit:.4f}, moist dry adiabat {max_dry:.4f}")
