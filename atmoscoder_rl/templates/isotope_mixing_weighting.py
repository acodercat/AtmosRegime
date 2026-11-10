"""Linear two-end-member mixing of delta values -> mixing weighted by the solute concentration (extended,
environmental chemistry / isotope hydrology, 2026-10-05).

Parent's method: two-component mixing (hydrograph separation) with an isotope tracer,
f = (delta_M - delta_B)/(delta_A - delta_B), the fraction of the mixed discharge supplied by source A. It
is the formula of every hydrograph-separation exercise, and for delta18O or delta2H of the water itself
it is exact: every litre carries the same amount of oxygen and hydrogen.
Where it fails: the isotope ratio of a *dissolved* species (here delta13C of dissolved inorganic carbon,
DIC) mixes in proportion to the amount of that species each water carries. Conserving total carbon and
13C, delta_M = [f C_A delta_A + (1 - f) C_B delta_B]/[f C_A + (1 - f) C_B], so
f/(1 - f) = C_B (delta_B - delta_M)/[C_A (delta_M - delta_A)]. The linear form is right only when the two
waters carry the same DIC concentration; when the tributary is dilute it gives the fraction of the
*carbon* from the tributary, not of the water, and underestimates the water fraction badly.
Fair from the text: every arm states both inflows' DIC concentrations and delta13C, the delta13C of the
fully mixed water below the confluence, and that DIC and its isotopes are conservative (no degassing,
exchange or reaction) -- everything either answer needs; nothing is recalled. The concentrations are
ordinary water-chemistry data; the inside and outside texts never say how the isotope ratios combine.
Inside: the tributary's DIC is within 1.3 % of the river's; the linear fraction (0.17-0.48) is within
1.2 % of the key.
Outside: the tributary is dilute, DIC 0.07-0.15 times the river's (e.g. 0.3 vs 3.5 mmol/L -- an ordinary
contrast between a stream on silicate rock and a river fed by carbonate groundwater); the key is 0.57-0.93
and the linear fraction 0.17-0.48, 44-77 % low and still an ordinary-looking fraction.
Conventions: mixing the exact 13C atom fractions (R = R_VPDB(1 + delta/1000), F = R/(1 + R), weighted by
total C) instead of delta values moves the key by < 0.1 % (`check()`); molar vs mass concentrations of
carbon give the same ratio. No other convention exists for conservative two-end-member mixing.
Second wrong method (rule 11): weighting by the inverse concentrations gives f = 0.01-0.12, far from the
key (`check()`). The carbon fraction is the shortcut itself.
check(): solves the 12C and 13C mass balances for f by bisection on the exact atom fractions, an
independent route to the closed form.
Reference: Clark & Fritz, *Environmental Isotopes in Hydrogeology* (1997), ch. 1 and 5 (mixing and
hydrograph separation; carbon-13 of DIC); Kendall & McDonnell (eds.), *Isotope Tracers in Catchment
Hydrology* (1998), ch. 2 and 16 (two-component separation; concentration-weighted mixing of solute
isotopes); Faure & Mensing, *Isotopes: Principles and Applications* (3rd ed., 2005), mixing of two
components with different concentrations. Chapter numbers not verified.

Cluster: new -- differs from the nearest, lognormal_mass (a moment of a size distribution), and from
aqueous_equilibria (speciation by mass action) because the parent is a linear two-end-member mass balance
and the missing step is weighting an intensive ratio by the amount of the carrier species; no existing
template has a mixing or tracer mass balance.
Independent review (2026-10-05): with outside keys of 0.57-0.93 a "tributary" supplied most of the flow
below the confluence, which reads oddly. The text now says "Two streams join: stream 1 ... stream 2 ..." and
asks for the fraction of the combined discharge from stream 1; numbers and keys unchanged. Cluster ruling: new.
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

R_VPDB = 0.0111802


def shortcut(p):
    """Linear delta mixing, fraction of discharge from the tributary A."""
    return (p["dm"] - p["db"]) / (p["da"] - p["db"])


def correct(p):
    """Concentration-weighted mixing."""
    odds = p["cb"] * (p["db"] - p["dm"]) / (p["ca"] * (p["dm"] - p["da"]))
    return odds / (1 + odds)


def base(rng):
    da = rnd(rng.uniform(-24.0, -19.0), 3)
    db = rnd(rng.uniform(-12.0, -9.0), 3)
    s = rng.uniform(0.17, 0.48)
    return {"da": da, "db": db, "dm": rnd(db + s * (da - db), 3),
            "cb": rnd(rng.uniform(2.0, 5.0))}                 # mmol/L


def trigger(rng, regime, base):
    if regime == "inside":
        return {"ca": rnd(base["cb"] * rng.uniform(0.987, 1.013), 3)}
    return {"ca": rnd(base["cb"] * rng.uniform(0.07, 0.15))}


def text(p, variant):
    given = ("The carbon isotope ratios mix in proportion to the amount of DIC each water carries: "
             "δ_mix = [f C_1 δ_1 + (1 − f) C_2 δ_2]/[f C_1 + (1 − f) C_2], with f stream 1's share of the "
             "combined discharge, C the DIC concentrations and subscripts 1 and 2 for the streams. " if variant == "given" else "")
    return (f"Two streams join. Stream 1 has a dissolved inorganic carbon (DIC) concentration of "
            f"{num(p['ca'])} mmol/L and a δ13C of DIC of {num(p['da'])} ‰; stream 2 has a DIC "
            f"concentration of {num(p['cb'])} mmol/L and a δ13C of DIC of {num(p['db'])} ‰. Below the confluence, "
            f"where the two waters are completely mixed, the δ13C of DIC is {num(p['dm'])} ‰. Treat DIC and its "
            f"carbon isotopes as conservative (no degassing, exchange or reaction) and both flows as steady. "
            f"{given}What fraction of the combined discharge below the confluence comes from stream 1? Express your "
            f"answer as a fraction.")


TEMPLATE = Template("isotope_mixing_weighting", "environmental_chemistry",
                    "linear delta mixing -> concentration-weighted two-end-member mixing of a solute isotope",
                    "", base, trigger, text, correct, shortcut)


def _atom_fraction(delta):
    r = R_VPDB * (1 + delta / 1000)
    return r / (1 + r)


def _exact(p):
    """Independent: bisection on f with total-C and 13C mass balances in exact atom fractions."""
    fa, fb, fm = _atom_fraction(p["da"]), _atom_fraction(p["db"]), _atom_fraction(p["dm"])

    def mixed(f):
        c13 = f * p["ca"] * fa + (1 - f) * p["cb"] * fb
        return c13 / (f * p["ca"] + (1 - f) * p["cb"])

    lo, hi = 0.0, 1.0                     # 13C fraction falls monotonically as f rises (da < db)
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mixed(mid) > fm else (lo, mid)
    return (lo + hi) / 2


def check() -> str:
    max_in, min_out, max_out, max_err = 0.0, math.inf, 0.0, 0.0
    min_wrong, min_key, max_key = math.inf, math.inf, 0.0
    for da in (-24.0, -19.0):
        for db in (-12.0, -9.0):
            for s in (0.17, 0.3, 0.48):
                for cb in (2.0, 5.0):
                    dm = db + s * (da - db)
                    for ratio, inside in ((0.987, 1), (1.013, 1), (0.07, 0), (0.11, 0), (0.15, 0)):
                        p = {"da": da, "db": db, "dm": dm, "cb": cb, "ca": cb * ratio}
                        c, sh = correct(p), shortcut(p)
                        max_err = max(max_err, gap(c, _exact(p)))
                        min_key, max_key = min(min_key, c), max(max_key, c)
                        g = gap(c, sh)
                        if inside:
                            max_in = max(max_in, g)
                            continue
                        min_out, max_out = min(min_out, g), max(max_out, g)
                        odds_inv = (db - dm) / (dm - da) * ratio       # inverse-concentration weighting
                        min_wrong = min(min_wrong, gap(c, odds_inv / (1 + odds_inv)))
    rng = random.Random("isotope-mixing-check")
    for _ in range(3000):
        b = base(rng)
        pin, pout = {**b, **trigger(rng, "inside", b)}, {**b, **trigger(rng, "outside", b)}
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        min_out = min(min_out, gap(correct(pout), shortcut(pout)))
        min_key = min(min_key, correct(pin), correct(pout))
        max_err = max(max_err, gap(correct(pout), _exact(pout)))
    assert max_err < 1e-3, max_err                          # delta vs exact atom-fraction mixing
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert min_wrong > 0.10, min_wrong
    assert min_key > 0.15 and max_key < 0.97, (min_key, max_key)   # clear of round(x, 2); fraction < 1
    # Anchor (hand-worked): C_t = 0.5, C_r = 4 mmol/L, delta -22 / -10, mixed -13 permil:
    # f/(1-f) = 4*3/(0.5*9) = 2.667 -> f = 0.727 (linear: 3/12 = 0.25).
    anchor = correct({"da": -22.0, "db": -10.0, "dm": -13.0, "cb": 4.0, "ca": 0.5})
    assert abs(anchor - 8 / 11) < 1e-12, anchor
    return (f"exact 13C atom-fraction mixing vs delta form {max_err:.1e}; inside max gap {max_in:.4f}; outside gap "
            f"{min_out:.3f}-{max_out:.3f}; inverse weighting min miss {min_wrong:.2f}; key {min_key:.2f}-"
            f"{max_key:.2f}; anchor f = {anchor:.3f}")
