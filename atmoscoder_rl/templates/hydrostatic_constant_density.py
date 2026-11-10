"""Hydrostatic pressure drop with constant density -> isothermal exponential profile (2026-09-26).

Parent's method: over a shallow layer the density is taken as its surface value, so the pressure
falls linearly with height: dp = rho_0 g z with rho_0 = p_0 / (R_d T).
Where it fails: over a depth comparable to the scale height H = R_d T / g (about 8 km) the density
itself falls with the pressure; the hydrostatic equation with the ideal-gas law gives, for an
isothermal layer, p(z) = p_0 exp(-z/H), so the drop is p_0 (1 - exp(-z/H)).
Fair from the text: all arms state the surface pressure, that the temperature is the same at every
level of the layer, dry air, g and R_d, so the exponential profile follows from the givens.
Outside z = 3-6 km, z/H = 0.34-0.75 and the linear drop overshoots by 18-43 %.
Inside: z = 30-150 m, z/H < 0.019, so the linear drop is within 1 % of the exponential one.
Conventions: g is stated, so the decrease of g with height (0.2 % at 6 km) is the only alternative
and it is far inside the tolerance; virtual-temperature effects are excluded by "dry air".
Anchor: scale height of an isothermal 273.15 K dry atmosphere, R_d T / g = 7.99 km (Wallace &
Hobbs 2006, eq. 3.8 and the ~8 km value quoted there).
Related mechanism: anelastic_continuity (also neglects the decrease of density with height).

Review (2026-09-27, domain + measurement-design review): "the air temperature is X °C at every level
... up to a height of z m" served only the isothermal model and read unnaturally. The text now gives
"the mean temperature of the air between the surface and a height of z m", the textbook hypsometric
input. The key is unchanged: with a height-mean temperature T and any lapse rate from isothermal to
dry adiabatic, the exact pressure drop lies within 0.3 % of p_0 (1 - exp(-g z/(R_d T))) over the whole
outside range (measured in check()). The shortcut reads the mean temperature as the surface value.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

G, RD = 9.81, 287.0


def base(rng):
    return {"p0": float(rng.choice(range(980, 1040, 10))), "t": float(rng.randint(0, 25))}


def trigger(rng, regime, base):
    return {"z": rnd(rng.uniform(30, 150)) if regime == "inside" else rnd(rng.uniform(3000, 6000))}


def scale_height(p):
    return RD * (p["t"] + 273.15) / G


def shortcut(p):
    rho0 = p["p0"] * 100 / (RD * (p["t"] + 273.15))
    return rho0 * G * p["z"] / 100                                # hPa


def correct(p):
    return p["p0"] * (1 - math.exp(-p["z"] / scale_height(p)))


def text(p, variant):
    given = ("Because the layer is deep, the density cannot be treated as constant: combining the hydrostatic "
             "equation with the ideal-gas law for air at a single temperature T (in kelvin) gives "
             "p(z) = p_0 exp(-g z/(R_d T)). " if variant == "given" else "")
    return (f"At a station the surface pressure is {num(p['p0'])} hPa, and the mean temperature of the air "
            f"between the surface and a height of {num(p['z'])} m is {num(p['t'])} °C. Treat the air "
            f"as dry, with R_d = 287 J kg^-1 K^-1, and take g = 9.81 m s^-2. {given}By how much does the "
            f"pressure decrease between the surface and that height? Express your answer in hPa.")


TEMPLATE = Template("hydrostatic_constant_density", "atmospheric_thermodynamics",
                    "constant-density hydrostatic drop -> isothermal exponential pressure profile",
                    "hPa", base, trigger, text, correct, shortcut)


def check() -> str:
    # Independent: RK4 integration of dp/dz = -p g / (R_d T), not the closed form.
    def integrate(p, steps=2000):
        pr, dz, tk = p["p0"], p["z"] / steps, p["t"] + 273.15
        f = lambda x: -x * G / (RD * tk)
        for _ in range(steps):
            k1 = f(pr); k2 = f(pr + dz * k1 / 2); k3 = f(pr + dz * k2 / 2); k4 = f(pr + dz * k3)
            pr += dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return p["p0"] - pr
    max_in, min_out, max_g, max_lapse = 0.0, 1.0, 0.0, 0.0
    for p0 in range(980, 1040, 10):
        for t in range(0, 26):
            for z in [30, 60, 100, 150, 3000, 4500, 6000]:
                p = {"p0": p0, "t": t, "z": z}
                c = correct(p)
                assert math.isclose(c, integrate(p), rel_tol=1e-9)
                g_ = gap(c, shortcut(p))
                if z <= 150:
                    max_in = max(max_in, g_)
                else:
                    min_out = min(min_out, g_)
                    # Alternative: g falling as (a/(a+z))^2, i.e. geopotential height; tiny.
                    zg = 6.371e6 * z / (6.371e6 + z)
                    max_g = max(max_g, gap(c, p0 * (1 - math.exp(-zg / scale_height(p)))))
                    # The text gives the layer's height-mean temperature: with a real lapse rate the
                    # exact drop must stay at the key (isothermal to dry adiabatic, mean = t).
                    for lapse in (0.004, 0.0065, 0.0098):
                        t0 = t + 273.15 + lapse * z / 2
                        exact = p0 - p0 * ((t0 - lapse * z) / t0) ** (G / (RD * lapse))
                        max_lapse = max(max_lapse, abs(exact / c - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and max_g < .05 and max_lapse < .01
    assert math.isclose(scale_height({"t": 0.0}), 7991.2, rel_tol=1e-4)
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, variable-g spread {max_g:.4f}, "
            f"lapse-rate spread at the stated mean temperature {max_lapse:.4f}")
