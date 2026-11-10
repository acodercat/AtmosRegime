"""Sea-level equivalent of the whole ice volume -> only the ice above flotation (extended, climate,
2026-10-05).

Parent's method: the sea-level rise if a body of grounded ice melts is its mass spread as water over
the ocean, Delta h = rho_i A H / (rho_w A_ocean), with H the ice thickness (surface minus bed elevation)
and A its area -- the "sea-level equivalent" taught with ice-sheet volumes.
Where it fails: ice grounded on a bed below sea level already displaces ocean water down to the bed.
When it melts, the meltwater adds rho_i H A / rho_w of water, but the basin it vacates below sea level
(volume d A, d = depth of the bed below sea level) fills with ocean water. The net rise is
(rho_i H / rho_w - d) A / A_ocean, i.e. only the ice thickness above flotation, H - (rho_w/rho_i) d,
counts. For a bed above sea level (d = 0) the two methods agree exactly.
Fair from the text: every arm states the area, the mean ice-surface elevation, the mean bed elevation
relative to sea level (the ice thickness is their difference, so the shortcut uses the bed too), that
the ice is grounded, the densities of ice and water, the ocean area, and that isostatic rebound and
changes of ocean area are neglected. Nothing mentions flotation, displacement or a marine basin.
Inside: bed elevation +50 to +300 m (above sea level): the shortcut is exact (gap 0).
Outside: surface 500-1000 m above sea level over a bed 1.05-1.35 times as deep below it (about -530
to -1350 m) -- a marine sector near the coast of the West Antarctic Ice Sheet (Thwaites/Pine Island
beds lie ~1000-1200 m below sea level). The whole-volume answer is 2.3-2.7 x the key; it is still an
ordinary sea-level figure of tens to a few hundred mm, not absurd.
Second wrong method (rule 11): counting only the ice above sea level, rho_i s A/(rho_w A_ocean), is off
by 0.083 d / (0.917 s - 0.083 d); it is > 10 % off only when the bed is deeper below sea level than the
surface is above it (d > ~1.0 s), hence the d/s range (asserted > 10 % in check(), grid and rendered
draws).
Conventions (rule 5): the text gives one water density, 1000 kg m^-3. A solver who uses sea water
(1028 kg m^-3) for the flotation/cavity term but 1000 for the meltwater lands <= 4.9 % from the key
(checked over the outside grid; this caps d/s at 1.35). Using 1028 everywhere (Goelzer et al. 2020's
"above flotation" volume converted with ocean density, without their density correction) exceeds 5 %
on this grid; it is not checked, because it contradicts the stated water density of 1000 kg m^-3.
Goelzer et al.'s full formula (above-flotation volume + potential-ocean-volume + density terms) with
rho_ocean = rho_fresh reduces to the key.
check(): an independent volume budget in SI (meltwater volume minus the cavity volume refilled by the
ocean, divided by ocean area), and the Archimedes anchor: ice exactly at flotation thickness adds
nothing.
Reference: Cuffey & Paterson, *The Physics of Glaciers* (4th ed.), sec. 1.3 and ch. 11 (ice above
flotation, marine ice sheets); Goelzer et al. (2020), The Cryosphere 14, 833 ("Brief communication: On
calculating the sea-level contribution in marine ice-sheet models"); IPCC AR6 WG1 ch. 9.4.
Related: none in the set. New cluster "buoyancy / displacement already counted" (Archimedes).

Independent review (2026-10-05): with bed -400 to -800 m under a 1200-2000 m surface, counting only the
ice above sea level was within 5 % of the key on 93 % of outside draws. The reviewer proposed
d = (0.65-1.0) s, but that route is then only 6-10 % off; the outside draw is now d = (1.05-1.35) s with
s = 500-1000 m, where it is > 10 % off and the sea-water-cavity convention stays < 5 %. The
"1028 everywhere" convention check was dropped (reason above).
"""
from __future__ import annotations

import math

import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

EXTENDED = True

RHO_I, RHO_W = 917.0, 1000.0
A_OCEAN = 3.61e8            # km^2


def _thickness(p):
    return p["s"] - p["b"]     # m


def shortcut(p):
    """Whole ice volume as water spread over the ocean, mm."""
    return p["area"] * _thickness(p) * RHO_I / RHO_W / A_OCEAN * 1e3


def correct(p):
    """Only the ice above flotation, mm."""
    d = max(0.0, -p["b"])
    return p["area"] * (_thickness(p) - RHO_W / RHO_I * d) * RHO_I / RHO_W / A_OCEAN * 1e3


def base(rng):
    return {"area": rnd(rng.uniform(2.0e4, 9.0e4)), "s": rnd(rng.uniform(500.0, 1000.0))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"b": rnd(rng.uniform(50.0, 300.0))}
    return {"b": -rnd(rng.uniform(1.05, 1.35) * base["s"])}


def text(p, variant):
    given = ("Ice resting on a bed below sea level already displaces ocean water, and when it melts the basin it "
             "leaves is refilled by the ocean, so only the ice thickness above flotation, "
             "H − (ρ_water/ρ_ice) × (depth of the bed below sea level), contributes to the rise. "
             if variant == "given" else "")
    return (f"A sector of an ice sheet covers {num(p['area'])} km^2. Averaged over the sector, the ice surface is "
            f"{num(p['s'])} m above sea level and the bed beneath the ice is at an elevation of {num(p['b'])} m "
            f"relative to sea level; the ice is grounded on its bed everywhere. Take the density of ice as "
            f"917 kg m^-3, the density of water as 1000 kg m^-3 and the area of the global ocean as "
            f"3.61 × 10^8 km^2, and neglect isostatic rebound and any change in ocean area. {given}If all of the "
            f"ice in this sector melted, by how much would global mean sea level rise? Express your answer in mm.")


TEMPLATE = Template("ice_above_flotation", "climate_dynamics",
                    "sea-level equivalent of the whole ice volume -> ice above flotation only",
                    "mm", base, trigger, text, correct, shortcut)


def _budget(p, rho_cavity=RHO_W, rho_melt=RHO_W, rho_convert=None):
    """Independent SI volume budget: meltwater volume minus the below-sea-level cavity refilled from the
    ocean. `rho_cavity` is the water density used for the cavity (flotation), `rho_melt` for meltwater."""
    area = p["area"] * 1e6                              # m^2
    ice_mass = RHO_I * area * _thickness(p)             # kg
    melt_volume = ice_mass / rho_melt                   # m^3
    cavity = area * max(0.0, -p["b"])                   # m^3 of ocean water drawn in
    if rho_convert is not None:                         # Goelzer-style: above-flotation mass / rho_convert
        af_mass = ice_mass - rho_cavity * cavity
        return af_mass / rho_convert / (A_OCEAN * 1e6) * 1e3
    return (melt_volume - cavity * rho_cavity / RHO_W) / (A_OCEAN * 1e6) * 1e3


def _above_sea_level(p):
    """Second wrong method: count only the ice above sea level, mm."""
    return p["area"] * p["s"] * RHO_I / RHO_W / A_OCEAN * 1e3


def check() -> str:
    max_in, min_out, max_out = 0.0, math.inf, 0.0
    max_err = spread_cav = 0.0
    above_sl = math.inf
    keys = [math.inf, 0.0]
    for area in (2.0e4, 5.5e4, 9.0e4):
        for s in (500.0, 750.0, 1000.0):
            for b, inside in ((50.0, 1), (300.0, 1), (-1.05 * s, 0), (-1.2 * s, 0), (-1.35 * s, 0)):
                p = {"area": area, "s": s, "b": b}
                c, sc = correct(p), shortcut(p)
                max_err = max(max_err, gap(c, _budget(p)))
                keys = [min(keys[0], c), max(keys[1], c)]
                g = gap(c, sc)
                if inside:
                    max_in = max(max_in, g)
                    continue
                min_out, max_out = min(min_out, g), max(max_out, g)
                assert 0 < sc < 2000                    # an ordinary sea-level figure, not absurd
                # rule 5: sea water (1028) for the cavity, fresh for meltwater
                spread_cav = max(spread_cav, gap(c, _budget(p, rho_cavity=1028.0)))
                above_sl = min(above_sl, gap(c, _above_sea_level(p)))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_err < 1e-12, max_err
    rng = random.Random("ice_above_flotation:rounded")
    for _ in range(2000):
        _, pout = pair(TEMPLATE, rng)
        c = correct(pout)
        above_sl = min(above_sl, gap(c, _above_sea_level(pout)))
        spread_cav = max(spread_cav, gap(c, _budget(pout, rho_cavity=1028.0)))
    assert spread_cav < 0.05, spread_cav
    assert above_sl > 0.10, above_sl                     # ice-above-sea-level route must fail
    assert keys[0] > 5, keys                             # mm: round(x, 2) moves it < 0.1 %
    # Anchor (Archimedes): ice exactly at flotation thickness (rho_i H = rho_w d) adds nothing to sea level.
    d = 600.0
    floating = {"area": 5e4, "s": RHO_W / RHO_I * d - d, "b": -d}
    assert abs(correct(floating)) < 1e-9 and abs(_budget(floating)) < 1e-9
    return (f"SI volume budget vs closed form {max_err:.1e}; inside max gap {max_in:.4f}; outside gap "
            f"{min_out:.3f}-{max_out:.3f}; ice-above-sea-level route >= {above_sl:.3f} off; spread sea-water "
            f"cavity {spread_cav:.3f}; key {keys[0]:.0f}-{keys[1]:.0f} mm; at flotation thickness -> 0")
