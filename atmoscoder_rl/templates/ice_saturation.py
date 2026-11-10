"""Saturation vapour pressure over liquid water -> over ice (spec: MECHANISMS.md, added 2026-09-27).

Parent's method: the air in equilibrium with a surface holds the saturation vapour pressure the
problem states, e_s = 6.112 exp[17.67 T/(T + 243.5)] hPa (Bolton 1980), which is the value over a
plane liquid-water surface.
Where it fails: below 0 degC the equilibrium vapour pressure over ice is lower than over
(supercooled) water, because the latent heat of sublimation exceeds that of vaporisation; at
-20 to -30 degC it is 18-25 % lower (the difference that drives the Wegener-Bergeron-Findeisen
process).
Fair from the text: every arm names the surface — ice crystals — and labels the stated formula
"over a plane liquid-water surface". The water formula is stated in all arms because the inside
arm needs some e_s and the parent problem states it; stating it *unlabelled* would be an unfair
bait (a solver could fairly read it as the intended formula for the crystal), so the label is
kept in every arm. The ice formula is used only by the replacement and is a handbook relation, so
by the reworded rule 3 it is stated only in the given sentence; the standard ice formulas a
solver may recall agree within 1.4 % of the key over the outside range (checked below).
Inside: T = -0.2 to -1.5 degC, where the shortcut is within 1.5 % of the key — crystals just above
the melting layer; the water formula is then a valid control.
Outside: T = -20.0 to -30.0 degC; e_i is 18-25 % below e_w, so the shortcut overshoots the
key by 22-34 % (gap relative to the key). Not colder: a blind solver (2026-09-27) derived e_i from
the stated e_w with a constant latent heat of fusion, e_w exp[-(L_f/R_v)(1/T - 1/T_0)], which falls
3.2 % below the key at -30 degC but 4.6 % at -35 degC — too close to the 5 % tolerance.

Conventions (all within 5 % of the key, Magnus-over-ice with Alduchov & Eskridge's coefficients as
used in WMO-style tables, 6.112 exp[22.46 T/(272.62 + T)]): Buck (1981) ice, Buck (1996) ice,
Alduchov & Eskridge (1996) AERKi, Tetens/Murray (1967) ice, Goff-Gratch ice (List 1951), Murphy &
Koop (2005) ice, and Clausius-Clapeyron integrated from the triple point with constant
L_s = 2.834e6 J/kg, and the stated e_w times the constant-L_f ice/water ratio. Ratio forms (e_i = e_w x something) reduce to the same physics.
Anchor: Smithsonian Meteorological Tables (List 1951): e_i(-20 degC) = 1.0326 hPa,
e_w(-20 degC) = 1.2540 hPa; e_i(-30 degC) = 0.3801 hPa.
Rule 11: the smallest outside key is 0.380 hPa (-30 degC); round(x, 2) moves it by < 1.4 %.
Rule 10: the key is the vapour pressure itself, not a ratio 1 - small.
Temperatures carry one decimal in both arms (integers outside were a format cue). The cloud's
pressure level (600-850 hPa) is a second, non-trigger given: it does not enter the key (the
enhancement factor of moist air is 1.004 at these pressures), and without it the text depends on T
alone, so 20 pairs repeated items (review 2026-09-27). The frame was also reworded away from
kelvin_curvature's ("what vapour pressure must the surrounding air have for the droplet to be in
equilibrium"), which it shares a stated flat-water e_s with.

Cluster: with kelvin_curvature (and so in "size_vs_length_scale"): both ask for the equilibrium
vapour pressure of a surface that is not the flat water the stated e_s describes. Added 2026-09-27, after the exploratory A/B
registration; by the A/B definition it is kind A (a new ingredient — the ice phase and its
latent heat of sublimation — must be recalled), but it is not part of the registered A/B lists.
Expected lure (prior, not a measurement): weak. The word "ice" and the "liquid-water" label
sit in the same problem, so a solver that reads carefully sees the mismatch; the lure rests on
the stated formula being the only one on the page.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, num


def ew(t):
    return 6.112 * math.exp(17.67 * t / (t + 243.5))                     # hPa, t in degC


def ei(t):
    return 6.112 * math.exp(22.46 * t / (272.62 + t))                    # hPa, t in degC


def base(rng):
    return {"p": float(rng.choice(range(600, 860, 10)))}          # the cloud's level, hPa: not in the key


def trigger(rng, regime, base):
    # One decimal in both arms, so the number format does not tell the arms apart.
    return {"t": -rng.randint(2, 15) / 10 if regime == "inside" else -rng.randint(200, 300) / 10}


def shortcut(p):
    return ew(p["t"])


def correct(p):
    return ei(p["t"])


def text(p, variant):
    given = ("Over ice the equilibrium vapour pressure is lower than over liquid water: use "
             "e_i = 6.112 exp[22.46 T/(T + 272.62)] hPa with T in °C. " if variant == "given" else "")
    return (f"Ice crystals are falling through a layer of cloud at the {num(p['p'])} hPa level, where the "
            f"temperature is {num(p['t'])} °C. The saturation vapour pressure over a plane liquid-water "
            f"surface is e_s = 6.112 exp[17.67 T/(T + 243.5)] hPa with T in °C. {given}At what "
            f"water-vapour pressure would the crystals neither gain mass by deposition nor lose it by "
            f"sublimation? Express your answer in hPa.")


def _conventions(t):
    tk = t + 273.15
    gg = 10 ** (-9.09718 * (273.16 / tk - 1) - 3.56654 * math.log10(273.16 / tk)
                + 0.876793 * (1 - tk / 273.16) + math.log10(6.1071))       # Goff-Gratch ice (List 1951)
    mk = math.exp(9.550426 - 5723.265 / tk + 3.53068 * math.log(tk) - 0.00728332 * tk) / 100
    return {"Buck 1981": 6.1115 * math.exp(22.452 * t / (272.55 + t)),
            "Buck 1996": 6.1115 * math.exp((23.036 - t / 333.7) * (t / (279.82 + t))),
            "AERKi": 6.1128 * math.exp(22.571 * t / (273.71 + t)),
            "Tetens/Murray": 6.1078 * math.exp(21.875 * t / (t + 265.5)),
            "Goff-Gratch": gg, "Murphy-Koop": mk}


def check() -> str:
    max_in, min_out, max_out, spread, max_cc = 0., math.inf, 0., 0., 0.
    for t in [-i / 10 for i in range(2, 16)] + [-i / 10 for i in range(200, 301)]:
        p = {"t": t}
        key = correct(p)
        g = abs(shortcut(p) - key) / key
        # Independent: Clausius-Clapeyron from the triple point, constant L_s.
        cc = 6.112 * math.exp(2.834e6 / 461.5 * (1 / 273.16 - 1 / (t + 273.15)))
        # The route a blind solver took: the stated e_w times the constant-L_f ice/water ratio.
        ratio = ew(t) * math.exp(-3.34e5 / 461.5 * (1 / (t + 273.15) - 1 / 273.15))
        max_cc = max(max_cc, abs(cc / key - 1), abs(ratio / key - 1))
        if t > -2:
            max_in = max(max_in, g)
            continue
        min_out, max_out = min(min_out, g), max(max_out, g)
        spread = max(spread, *(abs(v / key - 1) for v in _conventions(t).values()))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP
    assert spread < .05 and max_cc < .05
    assert abs(ei(-20) / 1.0326 - 1) < .01 and abs(ew(-20) / 1.2540 - 1) < .01    # List (1951)
    assert abs(ei(-30) / 0.3801 - 1) < .01
    return (f"max inside gap {max_in:.4f}, outside gap {min_out:.3f}-{max_out:.3f}; ice-formula spread "
            f"{spread:.4f}; Clausius-Clapeyron routes {max_cc:.4f}; e_i(-20) {ei(-20):.4f} vs 1.0326, "
            f"e_w(-20) {ew(-20):.4f} vs 1.2540, min key {ei(-30):.3f} hPa")


TEMPLATE = Template("ice_saturation", "cloud_physics",
                    "saturation vapour pressure over liquid water -> over ice below 0 degC",
                    "hPa", base, trigger, text, correct, shortcut)
