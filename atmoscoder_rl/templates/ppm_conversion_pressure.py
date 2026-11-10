"""ppb -> ug m^-3 with the standard molar volume -> ideal gas at the actual pressure.

Parent's method: the air-quality conversion C = (mixing ratio) x M / 24.45 L mol^-1, which holds
at 25 degC and 1 atm.
Where it fails: at a high-elevation site the molar concentration of air is p/(RT), so the mass
concentration scales with p; at 770-850 hPa the standard factor overstates it by 19-32 %.
Fair from the text: every variant states the pressure, the temperature (25 degC in every arm, so
the lure is exactly the 1-atm factor) and the molar mass of ozone; R is universal and stated only
in the given sentence (2026-09-27 note).
Inside: p = 1000-1030 hPa, within 1.6 % of 1 atm.
Conventions: 24.45 vs 24.47 L/mol, R = 8.314 vs 8.3145, M = 48.00 vs 47.998 all change the key by
< 0.1 %.
Anchor: 1 ppb O3 = 1.96 ug m^-3 at 25 degC and 1 atm (the tabulated US EPA / WHO factor).
Reference: Seinfeld & Pandis, ch. 1 (units of concentration).
Related mechanism: none in this set.

Review (2026-09-26, independent reviewer): at a "monitoring station" ug m^-3 is conventionally
reported at reference conditions (EU 293 K / 101.3 kPa, US 25 degC / 1 atm), which made the
shortcut a legitimate convention choice in the trap arm. All arms now say "measurement site" and ask
for the mass of ozone per cubic metre of the ambient air at the site. Outside pressure moved from
500-800 to 650-850 hPa: 25 degC air does not occur at 500-600 hPa (4.5-5.5 km); the gap stays
>= 19 %. check() now compares with the tabulated 1.96 ug m^-3 per ppb (25 degC, 1 atm) scaled by
p/1013.25, instead of the same ideal-gas algebra written with k_B N_A.

Review (2026-09-27, domain + measurement-design review): (1) R is removed from the inside/outside
texts and stated only in the given sentence: the shortcut uses 24.45 L/mol and never R, so stating
R announced the replacement (the new rule for constants used only by the replacement). Recalled
values 8.314-8.3145 (and 0.08206 L atm/mol/K) move the key by < 0.01 % (checked below).
(2) Outside pressure is now 760-850 hPa: 25 degC ambient air essentially does not occur at 650-750
hPa (3-3.6 km); ~780 hPa is Mexico City, ~840 hPa Denver. The gap stays >= 19 %.
Blind re-check (2026-09-27): 760 hPa reads as '760 mmHg = 1 atm' to a solver; the outside range
now starts at 770 hPa.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num

R, T, M = 8.314, 298.15, 48.0


def base(rng):
    return {"ppb": float(rng.randint(30, 120))}


def trigger(rng, regime, base):
    return {"p": float(rng.choice([1000, 1010, 1020, 1030]) if regime == "inside" else rng.choice(range(770, 860, 10)))}


def shortcut(p):
    return p["ppb"] * M / 24.45


def correct(p):
    return p["ppb"] * 1e-9 * p["p"] * 100 / (R * T) * M * 1e6


def text(p, variant):
    given = ("Because the pressure differs from 1 atm, the standard molar volume of 24.45 L mol^-1 does not apply: "
             "compute the molar concentration of air from the ideal gas law, n/V = p/(RT), with "
             "R = 8.314 J mol^-1 K^-1. "
             if variant == "given" else "")
    return (f"At a measurement site the surface pressure is {num(p['p'])} hPa and the air temperature is 25 °C. "
            f"The ozone mixing ratio there is {num(p['ppb'])} ppb by volume. Take the molar mass of ozone to be "
            f"48.0 g mol^-1. {given}Find the mass of ozone per cubic metre of "
            f"the ambient air at the site. Express your answer in μg m^-3.")


def check() -> str:
    max_in, min_out, max_recall = 0.0, math.inf, 0.0
    for ppb in range(30, 125, 5):
        for pr in [1000, 1010, 1020, 1030] + list(range(770, 860, 10)):
            p = {"ppb": float(ppb), "p": float(pr)}
            key = correct(p)
            # Independent: the tabulated factor 1.96 ug m^-3 per ppb (25 degC, 1 atm), scaled by pressure.
            alt = ppb * 1.96 * pr / 1013.25
            assert math.isclose(alt, key, rel_tol=5e-3), p
            g = gap(key, shortcut(p))
            if pr >= 1000:
                max_in = max(max_in, g)
            else:
                min_out = min(min_out, g)
                for recalled in (8.3145, 8.31446, 0.08206 * 101.325):    # R no longer stated
                    max_recall = max(max_recall, gap(key * R / recalled, key))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_recall < .05, max_recall
    anchor = correct({"ppb": 1.0, "p": 1013.25})
    assert math.isclose(anchor, 1.96, rel_tol=0.005), anchor
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; recalled R {max_recall:.1e}; tabulated 1.96 x p/1013.25 agrees within 0.5 %; "
            f"1 ppb O3 at 1 atm = {anchor:.3f} ug/m^3 (EPA/WHO 1.96)")


TEMPLATE = Template("ppm_conversion_pressure", "air_quality",
                    "standard 24.45 L/mol conversion -> ideal gas at the actual (high-elevation) pressure",
                    "ug/m^3", base, trigger, text, correct, shortcut)
