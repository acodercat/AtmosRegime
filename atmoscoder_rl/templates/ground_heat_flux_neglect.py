"""Surface energy balance with the ground heat flux neglected -> the full balance.

Added 2026-10-01 (extended set, docs/PROPOSAL.md E0d / E4'), replacing the planned
hot_object_self_emission: that mechanism's trigger (a hot surface, ~800-1200 K) makes the missing
physical process (thermal self-emission) obvious from the scenario itself ("lava"/"fire") even
though the number is plausible, which is a contextual cue this batch is meant to avoid (rule 4's
spirit: no qualifier should move the answer by more than 5 % tolerance without being needed, and a
setting that announces "this object radiates" is the same kind of giveaway). This mechanism's
trigger is a flux given directly in ordinary W/m^2 units with no such contextual tell.

Parent's method: a common simplification of the surface energy balance keeps only the two largest
terms over vegetated or well-watered surfaces, Rn = H + LE (net radiation partitions into sensible
and latent heat alone), solving for the latent heat flux as LE = Rn - H.
Where it fails: the full balance is Rn = H + LE + G, where G is the ground (soil) heat flux --
energy conducted into or out of the substrate (Oke, Boundary Layer Climates, 2nd ed., 1987, ch. 1;
Arya, Introduction to Micrometeorology, ch. 4; Campbell & Norman, An Introduction to Environmental
Biophysics, ch. 15). Over bare or sparsely vegetated, dry soil at midday G is routinely tens of
W/m^2 -- an ordinary flux, not an extreme one -- and LE = Rn - H - G. A solver who recalls "G is
often neglected" as a blanket rule, rather than checking the stated G against the other two terms,
drops a term that the text hands them as a plain number.
Fair from the text: every arm states Rn, H and G (with the sign convention spelled out: G positive
means a flux into the ground), the data needed for LE = Rn - H - G in every case; the inside arm
carries the same G given even though it barely matters there (rule 3). Neither arm says that G can
be neglected or writes the three-term balance -- only the given sentence does, and the problem
itself never hints at which term, if any, is meant to be dropped.
Inside: G = 1-3 W/m^2 (a near-zero conductive flux, e.g. a moist, densely vegetated surface near
energy-balance closure, or a transition period around sunrise/sunset when G crosses zero); it is
below 1.6 % of LE anywhere on the grid, so Rn = H + LE is a fair control.
Outside: G = 58-90 W/m^2, an ordinary flux for dry or sparsely vegetated soil at midday (11.6-
22.5 % of the sampled Rn, squarely inside the ~10-50 % G/Rn range commonly cited for bare-to-
partly-vegetated soils, Oke 1987 table 1.1). Rn (400-500 W/m^2) and H (120-200 W/m^2) are ordinary
midday clear-sky values; the key is at least 18 % below the shortcut everywhere on the grid.
Conventions: the balance Rn = H + LE + G (all of H, LE, G taken positive away from the surface) is
essentially universal across the cited references for this scenario -- there is no second textbook
formula to disagree with the way there is for, e.g., a vapour-pressure fit. check()'s "convention"
entry instead verifies (a) a different grouping of the same arithmetic (available energy Rn - G
computed first, then minus H) and (b) the balance holds after converting every flux to MJ m^-2
day^-1 (1 W/m^2 = 0.0864 MJ m^-2 day^-1, a common agronomy/evapotranspiration unit, e.g. FAO-56).
Rule 6: the key (LE) is an energy flux, never an absolute temperature.
Reference: Oke (1987), Boundary Layer Climates, 2nd ed., ch. 1; Arya, Introduction to
Micrometeorology, ch. 4; Campbell & Norman (1998), An Introduction to Environmental Biophysics,
ch. 15. Related mechanism: none yet in this set addresses a missing surface-balance term; the
family is otherwise phase-change energetics (freezing_latent_heat, evaporative_cooling_limit) and
loading (condensate_loading).

Judgement call (flagged): because the physics here is a linear balance with one essentially
universal textbook form, check()'s "independent recomputation" is a different-unit-system cross
check rather than a different formula or different numerics on a transcendental function (unlike,
say, the Newton/bisection pairs used elsewhere in this module); this is noted rather than
overstated as equivalent rigor. I also flag that G's sign convention has to be stated explicitly
in the text for the problem to be unambiguous, which I have done without hinting at whether G
should or should not be used.
"""
from __future__ import annotations

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True   # extended set (templates/__init__.py); not one of the original 30

W_TO_MJ_DAY = 0.0864   # 1 W/m^2 = 0.0864 MJ/m^2/day


def base(rng):
    return {"rn": rnd(rng.uniform(400.0, 500.0), 3), "h": rnd(rng.uniform(120.0, 200.0), 3)}


def trigger(rng, regime, base):
    return {"g": rnd(rng.uniform(1.0, 3.0) if regime == "inside" else rng.uniform(58.0, 90.0), 2)}


def shortcut(p):
    return p["rn"] - p["h"]                                          # W/m^2, G neglected


def correct(p):
    return p["rn"] - p["h"] - p["g"]                                  # W/m^2


def text(p, variant):
    given = ("Because part of the net radiation also goes into heating the ground, the balance is "
             "Rn = H + LE + G, so LE = Rn - H - G rather than Rn - H. " if variant == "given" else "")
    return (f"At a field site the net radiation is Rn = {num(p['rn'])} W m^-2 and the sensible heat "
            f"flux is H = {num(p['h'])} W m^-2. A soil heat-flux plate measures a ground heat flux "
            f"of G = {num(p['g'])} W m^-2 (positive G denotes a flux into the soil). Using the "
            f"surface energy balance, estimate the latent heat flux LE -- the energy going into "
            f"evaporating water at the site. {given}Express your answer in W m^-2.")


TEMPLATE = Template("ground_heat_flux_neglect", "boundary_layer",
                    "surface energy balance with the ground heat flux neglected -> the full "
                    "Rn = H + LE + G balance",
                    "W/m^2", base, trigger, text, correct, shortcut)


def _other_units(p):
    """Independent: the same balance carried out in MJ/m^2/day instead of W/m^2."""
    rn, h, g = (p["rn"] * W_TO_MJ_DAY, p["h"] * W_TO_MJ_DAY, p["g"] * W_TO_MJ_DAY)
    return (rn - h - g) / W_TO_MJ_DAY


def check() -> str:
    max_in, min_out, max_units, max_group = 0.0, float("inf"), 0.0, 0.0
    keys, ratios = [], []
    for rn in (400.0, 500.0):
        for h in (120.0, 200.0):
            p = {"rn": rn, "h": h}
            for g_ in (1.0, 3.0):
                pin = {**p, "g": g_}
                max_in = max(max_in, gap(correct(pin), shortcut(pin)))
                max_units = max(max_units, abs(_other_units(pin) - correct(pin)) / abs(correct(pin)))
            for g_ in (58.0, 90.0):
                pout = {**p, "g": g_}
                gap_ = gap(correct(pout), shortcut(pout))
                min_out = min(min_out, gap_)
                keys.append(correct(pout))
                ratios.append(g_ / rn)
                max_units = max(max_units, abs(_other_units(pout) - correct(pout)) / abs(correct(pout)))
                available = pout["rn"] - pout["g"]                     # group G with Rn first
                max_group = max(max_group, abs((available - pout["h"]) - correct(pout)))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_units < 1e-9 and max_group < 1e-9
    assert 0.10 < min(ratios) and max(ratios) < 0.30, ratios   # ordinary bare/partly-vegetated G/Rn
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; MJ/m^2/day cross-check "
            f"{max_units:.1e}; regrouped arithmetic {max_group:.1e}; outside G/Rn "
            f"{min(ratios):.3f}-{max(ratios):.3f}; outside keys {min(keys):.1f}-{max(keys):.1f} W/m^2")
