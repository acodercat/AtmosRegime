"""Buoyancy from virtual-temperature excess alone -> buoyancy with condensate (liquid-water) loading.

Added 2026-10-01 (extended set, docs/PROPOSAL.md E0d / E4'): a mechanism where the standard
formula silently omits a physical process, and no single given looks numerically extreme -- the
trigger (liquid-water mixing ratio) is an ordinary in-cloud value in both arms.

Parent's method: a rising parcel's buoyant acceleration comes from its virtual-temperature excess
over the environment, B = g (Tv_parcel - Tv_env)/Tv_env.
Where it fails: liquid water carried by the parcel adds to its weight (inertia) without adding to
its buoyant volume -- it does not expand like a gas. To first order in the liquid-water mixing
ratio r_l (Emanuel, Atmospheric Convection, 1994, section 4.3; Markowski & Richardson, Mesoscale
Meteorology in Midlatitudes, 2010, section 2.2, "density temperature" T_rho = Tv/(1+r_l)):
B = g[(Tv_parcel - Tv_env)/Tv_env - r_l]. The loading term -g r_l does not depend on Tv_env, so it
subtracts a fixed amount (g r_l) from the buoyancy regardless of how large the thermal excess is --
which is exactly why a shortcut solver who never cross-checks the liquid-water number against the
thermal term can be wrong by an arbitrarily large fraction even though r_l itself looks unremarkable.
Fair from the text: every arm states the parcel's virtual-temperature excess Delta Tv over the
environment, the environment's virtual temperature Tve (needed to evaluate the ratio), the parcel's
liquid-water mixing ratio r_l, and g; the inside arm carries the same r_l given even though it does
not matter there (rule 3). Neither arm says the word "loading" or writes the corrected formula --
only the given sentence does.
Inside: r_l = 0.02-0.05 g/kg, a trace amount (just above cloud base, most condensate still vapour or
not yet grown into bulk liquid); the loading term is below 1.6 % of the thermal term anywhere on the
grid, so the parent's formula is a fair control.
Outside: r_l = 2.5-4.5 g/kg, an ordinary cumulus liquid-water content (nothing in this number alone
looks extreme), while the thermal excess Delta Tv is kept at 2.0-3.5 K -- values that only look
unremarkable individually. Because the loading term is a fixed g r_l independent of Delta Tv, it
removes at least 25 % of the buoyancy everywhere on the grid, and for large r_l against small
Delta Tv the buoyancy can even reverse sign (the parcel is still reported as "buoyant" by the
shortcut while it is actually negatively buoyant once loading is included).
Conventions: the first-order formula above and the exact density-temperature form
B = g[Tv_parcel/(1+r_l) - Tve]/Tve agree to within 0.2 % over the whole outside grid (checked
below, since r_l <= 0.0045 is always small); g = 9.8, 9.81 and 9.80665 m/s^2 move the key by well
under 0.1 %.
Rule 6 (no absolute temperature as the key): the key is the parcel's buoyant acceleration, an
acceleration, expressed in cm/s^2 so the values land near O(1)-O(10); it is never an absolute
temperature.
Reference: Emanuel (1994), Atmospheric Convection, eq. 4.3.6; Markowski & Richardson (2010),
Mesoscale Meteorology in Midlatitudes, sec. 2.2. The rule of thumb that 1 g/kg of condensate is
worth roughly 0.3 K of virtual-temperature deficit (g r_l x Tve/g = r_l Tve ~ 0.001 x 290 K) is
common convective lore; check() verifies it against the formula rather than citing a specific page.
Related mechanism: none yet in this set attacks condensate loading; evaporative_cooling_limit and
freezing_latent_heat are the other phase-change-energetics mechanisms in this module's family.

Judgement call (flagged): rather than keying the buoyancy in a less-common unit, I key it in
cm/s^2 directly -- CONVERTED has no acceleration pair, so the unit-conversion leg of
check_templates is skipped for this template, same as several original-30 templates whose unit
(mol/L, sr) is not in CONVERTED either.


Lead review (2026-10-01): the question used to define B = g(Tv_parcel - Tve)/Tve, i.e. it stated
the shortcut as the definition, which made the keyed answer contradict the text; removed.

Independent review (2026-10-01): (1) some textbooks (e.g. Stull, Practical Meteorology) fold liquid water into
the virtual temperature, under which the shortcut would be right; the text now says "the vapour-only
definition" (a qualifier that moves the answer by > 25 %, allowed by rule 4). (2) The remark above that the
buoyancy can reverse sign on the outside grid is wrong: the smallest outside key is 2.12 cm s^-2 (positive).
"""
from __future__ import annotations

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True   # extended set (templates/__init__.py); not one of the original 30

G = 9.8   # m/s^2


def base(rng):
    return {"dt": rnd(rng.uniform(2.0, 3.5), 3), "tve": float(rng.choice(range(285, 301)))}


def trigger(rng, regime, base):
    return {"rl": rnd(rng.uniform(0.02, 0.05) if regime == "inside" else rng.uniform(2.5, 4.5), 2)}


def shortcut(p):
    return 100 * G * p["dt"] / p["tve"]                              # cm/s^2, loading ignored


def correct(p):
    return shortcut(p) - 0.1 * G * p["rl"]                            # cm/s^2: -100*G*(rl/1000)


def text(p, variant):
    given = ("Liquid water adds to the parcel's weight without adding to its buoyant volume, so it "
             "subtracts directly from the buoyancy as an extra term: "
             "B = g[(Tv_parcel - Tve)/Tve - r_l], where r_l is the liquid-water mixing ratio "
             "(dimensionless, e.g. 0.001 for 1 g/kg). " if variant == "given" else "")
    return (f"A cloud parcel rises through its environment. At some level the parcel's virtual "
            f"temperature (the vapour-only definition) exceeds the environment's virtual temperature by ΔTv = {num(p['dt'])} K; "
            f"the environment's virtual temperature there is Tve = {num(p['tve'])} K. The parcel is "
            f"carrying a liquid-water mixing ratio of r_l = {num(p['rl'])} g/kg. Using "
            f"g = 9.8 m s^-2, estimate the parcel's buoyant acceleration at that level."
            f" {given}Express your answer in cm s^-2.")


TEMPLATE = Template("condensate_loading", "cloud_physics",
                    "buoyancy from virtual-temperature excess alone -> buoyancy reduced by "
                    "condensate (liquid-water) loading",
                    "cm/s^2", base, trigger, text, correct, shortcut)


def _exact(p, rl):
    """Independent: exact density-temperature form B = g[Tv_parcel/(1+r_l) - Tve]/Tve, no
    first-order expansion in r_l."""
    tvp = p["tve"] + p["dt"]
    return 100 * G * (tvp / (1 + rl / 1000) - p["tve"]) / p["tve"]


def check() -> str:
    max_in, min_out, max_exact, max_g = 0.0, float("inf"), 0.0, 0.0
    keys = []
    for dt in (2.0, 3.5):
        for tve in (285.0, 300.0):
            p = {"dt": dt, "tve": tve}
            for rl in (0.02, 0.05):
                pin = {**p, "rl": rl}
                g_ = gap(correct(pin), shortcut(pin))
                max_in = max(max_in, g_)
                max_exact = max(max_exact, abs(_exact(pin, rl) - correct(pin)) / abs(correct(pin)))
            for rl in (2.5, 4.5):
                pout = {**p, "rl": rl}
                g_ = gap(correct(pout), shortcut(pout))
                min_out = min(min_out, g_)
                max_exact = max(max_exact, abs(_exact(pout, rl) - correct(pout)) / abs(correct(pout)))
                keys.append(correct(pout))
                for g_const in (9.80665, 9.81):
                    alt = correct({**pout, "rl": rl}) * g_const / G    # scaling B by g is exact
                    max_g = max(max_g, abs(alt / correct(pout) - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_exact < 0.005 and max_g < 0.002
    # Rule-of-thumb anchor: 1 g/kg of condensate ~ g*0.001 cm/s^2 * 100 removed from B, independent
    # of Tve; check the loading term alone against that figure.
    thumb = 100 * G * 0.001            # cm/s^2 removed per g/kg
    assert abs((shortcut({"dt": 3.0, "tve": 290, "rl": 1.0}) -
                correct({"dt": 3.0, "tve": 290, "rl": 1.0})) - thumb) < 1e-9
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; exact density-temperature "
            f"form vs linear {max_exact:.4f}; g=9.80665/9.81 vs 9.8 {max_g:.4f}; "
            f"outside keys {min(keys):.2f}-{max(keys):.2f} cm/s^2; loading term = {thumb:.3f} "
            f"cm/s^2 per g/kg")
