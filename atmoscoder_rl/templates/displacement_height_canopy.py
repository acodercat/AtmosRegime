"""Neutral log wind profile referenced to the ground -> zero-plane-displaced log profile over a
tall canopy (extended, 2026-10-01).

Parent's method: the neutral logarithmic wind profile referenced to the ground surface,
u(z) = (u*/kappa) ln(z/z0).
Where it fails: once the canopy is tall enough that the measurement height is only a few canopy
heights above the canopy top, the flow no longer sees the ground as its effective origin -- the
profile is displaced upward by the zero-plane displacement height d, the level at which the
canopy's drag acts as if concentrated: u(z) = (u*/kappa) ln((z-d)/z0), with d given in the
"given" text only as the standard rule of thumb, two-thirds of the canopy height. Nothing about
the stated canopy height looks extreme -- 15-18 m is an ordinary stand of trees, shorter than a
mature forest -- but at a fixed 28 m mast the parent's formula misses by 18-26 %.
Fair from the text: both inside and outside state the friction velocity u*, the von Karman
constant, and the canopy height h; both methods also need the roughness length z0, which is
stated identically in every arm as a *formula in words* -- "one-tenth of the canopy height" --
not as a separately rendered number. This is deliberate (see the Design note below): z0 is
needed by the shortcut too, so stating it is neutral, not a hint, and phrasing it as a ratio
keeps the text's only numeric difference between inside and outside to the canopy height itself
(`check_templates.check_texts` requires exactly one differing number). The displacement height d
is never stated in inside or outside -- a solver must recall it unprompted.
Inside: h = 0.05-0.3 m (short grass or a mown field), so d = 0.03-0.2 m is negligible next to the
28 m mast; the stability-free log law is correct to within 0.11 %.
Outside: h = 15-18 m, so d = 10-12 m is a third to nearly half of the mast height; the
ground-referenced log law misses the true wind speed by 18-26 %.
Conventions: d/h is published in the range 0.6-0.75 (rule 5 spread, checked below); across the
outside grid the resulting key varies by at most 4.5 % relative to the 2/3h value used here,
comfortably under the 5 % bound. z0 = 0.1h is likewise a standard rule of thumb (not swept here,
since it enters both the shortcut and the correct formula identically and so cannot bias the gap).
Reference: Stull, R.B. (1988), An Introduction to Boundary Layer Meteorology, sec. 9.7
(roughness-length and displacement-height rules of thumb for vegetated surfaces); Thom, A.S.
(1971), Q. J. R. Meteorol. Soc. 97, 414-428 (d/h = 0.63 for an oat crop, the original empirical
basis for the 2/3h rule, inside the 0.6-0.75 spread checked here); Garratt, J.R. (1992), The
Atmospheric Boundary Layer, table 4.1 (z0/h and d/h by canopy type).
Related mechanism: log_wind_stability (same neutral log law, a different missing ingredient).

Design note (2026-10-01): the mast height is a fixed 28 m rather than a sampled base value. A
sampled mast height would have to satisfy the rule-5 convention-spread bound (d/h = 0.6-0.75
within 5 %) across its own range as well as the trigger's, and the two constraints trade off
directly against each other (a taller mast widens the safe canopy-height range but shrinks the
shortcut's miss); 28 m keeps both comfortably satisfied without adding a parameter that does not
change the physics being tested.

Independent review (2026-10-01): published forest d/h reach 0.8 (Jarvis et al. 1976: 0.61-0.92; dense canopies
~0.75-0.8); with d = 0.8h the key error exceeded 5 % above h ~ 15.3 m, so the outside canopy height is now 15.0-15.3 m (d = 0.8h stays within 5 % only up to ~15.3 m)
and check() includes d/h = 0.8. The given sentence no longer says the mast is "a few canopy heights" above the
canopy (it is 10-13 m above a 15-16 m canopy). Note: z/h ~ 1.75-1.87 lies in the roughness sublayer (~2-3 h), where
the log law is a textbook idealisation rather than the real profile; textbook problems routinely make it.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

KAPPA = 0.4
Z = 28.0          # m, fixed mast height (see Design note)
D_FRAC = 2 / 3    # zero-plane displacement, d = D_FRAC * h (the ingredient to recall)


def base(rng):
    return {"ustar": rnd(rng.uniform(0.3, 0.6), 2)}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"h": rnd(rng.uniform(0.05, 0.3), 3)}
    return {"h": rnd(rng.uniform(15.0, 15.3), 3)}


def _z0(h):
    return 0.1 * h


def shortcut(p):
    return p["ustar"] / KAPPA * math.log(Z / _z0(p["h"]))


def correct(p, dfrac=D_FRAC):
    d = dfrac * p["h"]
    return p["ustar"] / KAPPA * math.log((Z - d) / _z0(p["h"]))


def text(p, variant):
    given = ("Because the mast is less than two canopy heights above the ground, the profile is "
             "referenced not to the ground but to an effective origin raised into the canopy: use "
             "u(z) = (u*/κ) ln((z − d)/z0), where the zero-plane displacement height d is two-thirds of "
             "the canopy height. " if variant == "given" else "")
    return (f"Wind blows over a uniformly vegetated surface with a canopy height of {num(p['h'])} m. A sonic "
            f"anemometer measures a friction velocity of {num(p['ustar'])} m s^-1, and the surface's "
            f"aerodynamic roughness length is one-tenth of the canopy height. Take the von Kármán constant to "
            f"be 0.4. {given}Find the mean wind speed at a height of 28 m above the ground. Express your "
            f"answer in m s^-1.")


TEMPLATE = Template("displacement_height_canopy", "boundary_layer",
                    "ground-referenced log wind profile -> zero-plane-displaced profile over a tall canopy",
                    "m/s", base, trigger, text, correct, shortcut)


def _simpson_integral(p, dfrac=D_FRAC, n=100000):
    """Independent: integrate du/dz = u*/(kappa (z-d)) from z = d+z0 to the mast height Z
    (Simpson's rule over z, not the closed-form antiderivative)."""
    d = dfrac * p["h"]
    z0 = _z0(p["h"])
    lo, hi = d + z0, Z

    def dudz(z):
        return p["ustar"] / (KAPPA * (z - d))

    h_step = (hi - lo) / n
    s = dudz(lo) + dudz(hi)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * dudz(lo + i * h_step)
    return s * h_step / 3


def check() -> str:
    max_in = 0.0
    for h in [0.05, 0.1, 0.2, 0.3]:
        for ustar in [0.3, 0.45, 0.6]:
            p = {"h": h, "ustar": ustar}
            max_in = max(max_in, gap(correct(p), shortcut(p)))

    min_out, max_out, max_spread = 1.0, 0.0, 0.0
    for h10 in range(1500, 1531):
        h = h10 / 100
        p = {"h": h, "ustar": 0.45}
        key = correct(p)
        g = gap(key, shortcut(p))
        min_out, max_out = min(min_out, g), max(max_out, g)
        spread = max(abs(correct(p, df) - key) / key for df in (0.6, 0.75, 0.8))
        max_spread = max(max_spread, spread)

    max_indep = 0.0
    for h in [0.05, 0.3, 15.0, 15.3]:
        for ustar in [0.3, 0.6]:
            p = {"h": h, "ustar": ustar}
            c = correct(p)
            s = _simpson_integral(p)
            max_indep = max(max_indep, abs(c - s) / c)

    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_spread < 0.05 and max_indep < 1e-6, (max_spread, max_indep)
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f} (max {max_out:.3f}), "
            f"d/h=0.6-0.8 spread {max_spread:.4f}, du/dz integral check {max_indep:.1e}")
