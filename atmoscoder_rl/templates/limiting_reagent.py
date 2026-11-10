"""Oxidant assumed in excess -> stoichiometric limiting-reagent sulfate production (2026-10-01
extension).

Parent's method: in cloud water, dissolved SO2 (as S(IV)) is oxidized by dissolved H2O2,
S(IV) + H2O2 -> S(VI) + H2O, irreversibly and essentially to completion. Textbook treatments of
this reaction routinely assume H2O2 is in large excess over S(IV) (the usual case in clean to
moderately polluted clouds), so every mole of S(IV) is converted and the sulfate produced is set
by the S(IV) amount alone.
Where it fails: during high-SO2 episodes the dissolved H2O2 can itself be the smaller amount, in
which case it -- not S(IV) -- caps how much sulfate can form (the well-documented "oxidant-limited"
or H2O2-limited regime of in-cloud sulfate production). The correct amount of S(VI) formed is the
stoichiometric minimum of the two reactant amounts, not the S(IV) amount alone. Nothing about the
H2O2 concentration given in the outside arm is extreme -- it is simply somewhat less than the
S(IV) concentration, which is an entirely ordinary way for cloud water composition to come out
during a pollution episode.
Fair from the text: every variant states the S(IV) concentration, the H2O2 concentration (the
trigger), the liquid water content and the molar mass used to convert to a mass concentration;
no variant says which reactant is more abundant. "given" states the limiting-reagent rule.
Inside: dissolved H2O2 exceeds S(IV) by a factor of 2-5. Whenever H2O2 is in excess at all, the
stoichiometric minimum equals the S(IV) amount exactly (gap = 0), so the 2x floor is a comfortable
margin on top of an exact match, not a tuned-small gap.
Outside: dissolved H2O2 is 30-80 % of the S(IV) amount (an unremarkable composition during a
high-SO2 episode): the shortcut then overstates the sulfate formed by (1 - r)/r, 25-233 % for
r in [0.3, 0.8]; check() confirms the floor stays above 15 %.
Conventions: none -- "the smaller of two stoichiometric amounts limits the product" has no
competing convention; check() recomputes it via min(a,b) = (a+b-|a-b|)/2, independent of Python's
min().
Anchor: the sulfate ion molar mass, 32.06 (S) + 4*16.00 (O) = 96.06 g/mol. The regime itself is
textbook material on in-cloud SO2 oxidation (e.g. Seinfeld & Pandis, *Atmospheric Chemistry and
Physics*, ch. 7): H2O2 is normally the more abundant oxidant, but its supply is finite and can be
exhausted before all the dissolved SO2 is oxidized when SO2 loading is high.
Related: `henry_depletion` is also a closed-reservoir trap in cloud water, but it turns on a
single species' own solubility depleting the gas phase, not on a second reactant's stoichiometry.
"""
from __future__ import annotations

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

M_SO4 = 96.06   # g/mol = ug/umol


def base(rng):
    return {"cs": rnd(rng.uniform(25, 80)), "lwc": rnd(rng.uniform(0.4, 1.2))}


def trigger(rng, regime, base):
    ratio = rng.uniform(2.0, 5.0) if regime == "inside" else rng.uniform(0.3, 0.8)
    return {"ch": rnd(base["cs"] * ratio)}


def _n(conc_um, lwc):
    """umol/L (aqueous) * L-water/m3-air -> umol/m3-air. 1 g/m^3 water = 1e-3 L/m^3 air."""
    return conc_um * lwc * 1e-3


def shortcut(p):
    """Sulfate formed assuming H2O2 is in excess: all S(IV) converts. ug/m^3."""
    return _n(p["cs"], p["lwc"]) * M_SO4


def correct(p):
    """Sulfate formed is capped by whichever reactant amount is smaller. ug/m^3."""
    return min(_n(p["cs"], p["lwc"]), _n(p["ch"], p["lwc"])) * M_SO4


def text(p, variant):
    given = ("Because the dissolved H2O2 amount is less than the dissolved S(IV) amount, H2O2 limits how much "
             "sulfate can form: the amount of sulfate produced equals the smaller of the two reactant amounts "
             "(1:1 stoichiometry), not the S(IV) amount alone. " if variant == "given" else "")
    return (f"A cloud parcel has a liquid water content of {num(p['lwc'])} g m^-3. Its cloud water contains "
            f"dissolved S(IV) (from absorbed SO2) at a concentration of {num(p['cs'])} umol L^-1 and dissolved "
            f"H2O2 at a concentration of {num(p['ch'])} umol L^-1. The reaction S(IV) + H2O2 -> S(VI) + H2O goes "
            f"to completion (irreversible, 1:1 stoichiometry), and no more SO2 or H2O2 enters the parcel. Take "
            f"the molar mass of the sulfate ion (SO4^2-) formed as 96.06 g mol^-1. {given}What mass concentration "
            f"of sulfate, per cubic meter of air, is formed? Express your answer in ug m^-3.")


def _min_alt(a, b):
    """min(a, b) via (a + b - |a - b|) / 2, independent of Python's min()."""
    return (a + b - abs(a - b)) / 2


def check() -> str:
    max_in, min_out, max_err = 0.0, float("inf"), 0.0
    for cs in (25, 80):
        for lwc in (0.4, 1.2):
            for ratio, inside in ((2.0, 1), (5.0, 1), (0.3, 0), (0.8, 0)):
                p = {"cs": cs, "lwc": lwc, "ch": cs * ratio}
                alt = _min_alt(_n(p["cs"], p["lwc"]), _n(p["ch"], p["lwc"])) * M_SO4
                max_err = max(max_err, gap(alt, correct(p)) if correct(p) else abs(alt - correct(p)))
                g = gap(correct(p), shortcut(p))
                if inside:
                    max_in = max(max_in, g)
                else:
                    min_out = min(min_out, g)
    assert max_err < 1e-9 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    assert max_in == 0.0, max_in   # H2O2 in excess at all makes the shortcut exact, not just close
    smallest = correct({"cs": 25, "lwc": 0.4, "ch": 25 * 0.3})
    assert smallest > 0.1 and abs(round(smallest, 2) / smallest - 1) < 0.05, smallest
    return (f"min() vs (a+b-|a-b|)/2 {max_err:.1e}; inside max gap {max_in:.4f} (exact when H2O2 in excess); "
            f"outside min gap {min_out:.3f}; smallest key {smallest:.3f} ug/m^3")


TEMPLATE = Template("limiting_reagent", "atmospheric_chemistry",
                    "oxidant assumed in excess -> stoichiometric limiting-reagent sulfate production",
                    "ug/m^3", base, trigger, text, correct, shortcut)
