"""Tropospheric barometric formula -> isothermal stratosphere above the ICAO tropopause
(extended set, 2026-10-01).

Parent's method: in the troposphere, a constant lapse rate Gamma gives the barometric formula
p(z) = p0 (1 - Gamma z / T0)^(g / (R Gamma)), derived by combining the hydrostatic equation
dp/p = -g dz / (R T) with T(z) = T0 - Gamma z.
Where it fails: the standard atmosphere's lapse rate holds only up to the tropopause at
z1 = 11 km; above it the temperature is constant at T1 = 216.65 K, so the hydrostatic equation
instead integrates to the isothermal (exponential) form p(z) = p1 exp(-g (z - z1) / (R T1)), with
p1 = p(z1) given by the same formula evaluated at z1 (the two pieces agree exactly at 11 km, so
the profile is continuous; only the functional form above it changes). Applying the tropospheric
power law at z > 11 km anyway -- a constant-lapse-rate extrapolation into a regime where the lapse
rate is actually zero -- is the shortcut this template traps.
Fair from the text: every arm states the sea-level pressure and temperature, the tropospheric
lapse rate, the tropopause height and its temperature, and g and R (both formulas need them: the
troposphere exponent is g/(R Gamma), the stratosphere decay scale is g/(R T1)). Nothing in the
text says which formula to use at the asked height; the solver must compare it with 11 km itself.
Inside: z = 2-10 km, strictly below the tropopause, so the two formulas are identical by
construction (gap = 0 to floating-point precision).
Outside: z = 18.8-20 km. The naive extrapolation first clears the 15 % floor at z ~= 18.6 km
(found by bisection in check()), so the suggested 14-20 km band is narrowed upward to start at
18.8 km; the upper end stays at 20 km rather than moving higher, because above 20 km the real
(US Standard Atmosphere 1976) profile stops being isothermal and starts warming again at
+1.0 K/km, so this template's two-layer model would itself become the thing a well-informed solver
could legitimately disagree with (MECHANISMS.md rule 5).
Text diversity: with z confined to a 1.2 km band and rounded to the required 2-3 significant
figures (tenths, at this magnitude), only ~13 distinct outside heights exist -- far short of the
100 distinct problems check_templates builds per template. Rather than widen z past its physical
ceiling, the sea-level pressure p0 is made a second, genuinely varying given: p0 = 970-1060 hPa (an
ordinary day-to-day range; 1013.25 hPa is the ICAO value but real station pressure routinely spans
this band under passing weather systems). p0 is a pure multiplicative scale on both the shortcut and
the correct formula (T0, Gamma, z1, T1 -- everything that sets the *shape* of the profile and hence
the gap -- stays fixed at the standard values), so it changes the key but leaves every gap bound in
this docstring exactly as computed; check() confirms the gap is independent of p0 on the sampled
grid. The p0 x z_km product gives >400 representable (p0, z) combinations, comfortably above 100.
Conventions: R = 287.05 J/(kg K) and g = 9.80665 m/s^2 are the ICAO/US Standard Atmosphere values;
check() shows that using R = 287.0 or g = 9.81 instead moves the key by well under 1 %.
Reference: ICAO Doc 7488 / US Standard Atmosphere 1976 (tropopause at 11 km, 216.65 K, isothermal
to 20 km; p(11 km)/p0 = 0.2234 and p(20 km)/p0 = 0.05403 are the published table ratios at the
standard p0 = 1013.25 hPa, matched here to 4 significant figures -- see check()). Related mechanism:
none in this set.

Independent review (2026-10-01): the text said the temperature "remains constant above" the tropopause, which
states the replacement, and called a sea level of 970-1060 hPa "the standard atmosphere" (whose p0 is 1013.25);
now the day's temperature profile follows the ICAO standard atmosphere (whose isothermal layer above 11 km is
standard knowledge) with that day's sea-level pressure. Geometric vs geopotential height moves the key ~1 %.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

G, R, T0, GAMMA, Z1, T1 = 9.80665, 287.05, 288.15, 6.5e-3, 11000.0, 216.65
EXP_TROP = G / (R * GAMMA)
TROP_RATIO = (1 - GAMMA * Z1 / T0) ** EXP_TROP                 # p(11 km) / p0, independent of p0


def base(rng):
    return {"p0": rnd(rng.uniform(970.0, 1060.0), 3)}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"z_km": rnd(rng.uniform(2.0, 10.0), 3)}
    return {"z_km": rnd(rng.uniform(18.8, 20.0), 3)}


def shortcut(p):
    """The tropospheric power law, applied at any height (the parent's method)."""
    z = p["z_km"] * 1000.0
    return p["p0"] * (1 - GAMMA * z / T0) ** EXP_TROP


def correct(p):
    z = p["z_km"] * 1000.0
    if z <= Z1:
        return p["p0"] * (1 - GAMMA * z / T0) ** EXP_TROP
    return p["p0"] * TROP_RATIO * math.exp(-G * (z - Z1) / (R * T1))


def text(p, variant):
    given = ("Because this height is above the tropopause, the pressure there follows the isothermal "
             "form instead: p(z) = p(11 km) exp[-g(z - 11 km)/(R T1)], where p(11 km) is given by the "
             "tropospheric formula evaluated at 11 km. " if variant == "given" else "")
    return (f"On a certain day the temperature profile follows the ICAO standard atmosphere: 288.15 K at sea "
            f"level, decreasing at 6.5 K/km up to the tropopause at 11 km, where the temperature is 216.65 K. "
            f"The sea-level pressure that day is {num(p['p0'])} hPa. Take g = 9.80665 m/s^2 and the specific gas "
            f"constant for dry air R = 287.05 J/(kg K). {given}What is the pressure at a height of "
            f"{num(p['z_km'])} km? Express your answer in hPa.")


def _hydrostatic_rk4(p0, z_target, steps=4000):
    """Independent: integrate dp/dz = -g p / (R T(z)) from the surface, T(z) piecewise."""
    def temp(z):
        return T0 - GAMMA * z if z <= Z1 else T1

    def dpdz(z, p_):
        return -G * p_ / (R * temp(z))

    dz = z_target / steps
    p_ = p0
    z = 0.0
    for _ in range(steps):
        k1 = dpdz(z, p_)
        k2 = dpdz(z + dz / 2, p_ + dz / 2 * k1)
        k3 = dpdz(z + dz / 2, p_ + dz / 2 * k2)
        k4 = dpdz(z + dz, p_ + dz * k3)
        p_ += dz / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        z += dz
    return p_


def check() -> str:
    max_in = 0.0
    for p0 in [970.0, 1013.25, 1060.0]:
        for z_km in [2.0, 4.0, 6.0, 8.0, 10.0]:
            p = {"p0": p0, "z_km": z_km}
            max_in = max(max_in, gap(correct(p), shortcut(p)))
    min_out, spread, min_key, max_key, gap_p0_spread = 1.0, 0.0, 1e18, 0.0, 0.0
    for z_km in [18.8, 19.0, 19.5, 20.0]:
        gaps_here = []
        for p0 in [970.0, 1013.25, 1060.0]:
            p = {"p0": p0, "z_km": z_km}
            c = correct(p)
            g_ = gap(c, shortcut(p))
            gaps_here.append(g_)
            min_out = min(min_out, g_)
            min_key, max_key = min(min_key, c), max(max_key, c)
            rk4 = _hydrostatic_rk4(p0, z_km * 1000.0)
            assert math.isclose(rk4, c, rel_tol=1e-6), (p0, z_km, rk4, c)
            for r_alt, g_alt in [(287.0, G), (R, 9.81)]:
                z = z_km * 1000.0
                ratio_alt = (1 - GAMMA * Z1 / T0) ** (g_alt / (r_alt * GAMMA))
                alt = p0 * ratio_alt * math.exp(-g_alt * (z - Z1) / (r_alt * T1))
                spread = max(spread, gap(c, alt))
        gap_p0_spread = max(gap_p0_spread, max(gaps_here) - min(gaps_here))
    # inside: the two formulas must also match at 11 km itself (continuity) and well below it
    assert math.isclose(correct({"p0": 1013.25, "z_km": 11.0}), shortcut({"p0": 1013.25, "z_km": 11.0}),
                         rel_tol=1e-12)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and spread < 0.05, (max_in, min_out, spread)
    assert gap_p0_spread < 1e-9, gap_p0_spread          # gap is exactly independent of p0
    # Anchors: published ICAO/US Standard Atmosphere 1976 table ratios at the standard p0
    assert abs(TROP_RATIO - 0.2234) / 0.2234 < 1e-3
    assert abs(correct({"p0": 1013.25, "z_km": 20.0}) / 1013.25 - 0.05403) / 0.05403 < 1e-3
    return (f"RK4 hydrostatic integration agrees; max inside gap {max_in:.5f}, min outside gap "
            f"{min_out:.3f}, g/R-convention spread {spread:.4f}, gap is p0-independent (spread "
            f"{gap_p0_spread:.2e}); outside key range {min_key:.1f}-{max_key:.1f} hPa; "
            f"p(11 km)/p0 {TROP_RATIO:.4f} (table 0.2234)")


TEMPLATE = Template("isa_above_tropopause", "atmospheric_thermodynamics",
                    "tropospheric barometric formula -> isothermal stratosphere above the ICAO tropopause",
                    "hPa", base, trigger, text, correct, shortcut)
