"""Geostrophic wind -> gradient wind around a tightly curved low (added 2026-09-26).

Parent's method: flow parallel to the isobars balances the pressure-gradient and Coriolis forces,
V_g = (1/(rho f)) dp/dn.
Where it fails: when the isobars are strongly curved, the centripetal acceleration V^2/R is not
small compared with f V (Rossby number V_g/(f R) not small). For cyclonic flow around a low the
balance V^2/R + f V = f V_g gives the gradient wind V = -fR/2 + sqrt(f^2 R^2/4 + f R V_g) < V_g.
Fair from the text: every arm gives the latitude, Omega, the air density, the pressure gradient and
the radius of curvature of the isobars, and says the flow is steady, frictionless and parallel to
the isobars around a low, so the three-force balance is fully specified.
Inside: R = 6000-9000 km with gradients of 0.6-1.0 hPa per 100 km, Rossby number <= 0.0173, the
gradient wind is within 1.7 % of V_g. Outside: R = 60-150 km (a weak mesoscale low, winds 3-7 m/s), Rossby number
>= 0.22, V_g overestimates the gradient wind by >= 19 %.
Conventions: Omega = 7.292e-5 vs 7.27e-5 s^-1 changes the key by < 0.3 %. The first-order
curvature correction V ~ V_g(1 - Ro) is an approximation, not a convention, and is not credited.
Reference: Holton & Hakim, An Introduction to Dynamic Meteorology (5th ed.), sec. 3.2.5,
eq. (3.15) and the normal-low root. Related mechanism: `coriolis_neglect` (horizontal balance).

Wording (2026-09-26, lead review): the text no longer says the air circles 'a centre of low
pressure' -- with inside radii of 6000-9000 km that described a physically odd low. It now says the
isobars curve cyclonically with lower pressure on the inside; the physics and ranges are unchanged.

Review (2026-09-26, independent reviewer): the outside flow was described as "a small intense low";
with 0.6-1.0 hPa per 100 km and R = 60-150 km the keyed winds are only 2.9-6.6 m/s, a weak mesoscale
low. Docstring only; no code change.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

OMEGA = 7.292e-5


def f(p):
    return 2 * OMEGA * math.sin(math.radians(p["lat"]))


def base(rng):
    return {"lat": float(rng.randint(40, 55)), "rho": float(rng.choice(range(110, 126))) / 100,
            "dp": float(rng.choice(range(6, 11))) / 10}                     # hPa per 100 km


def trigger(rng, regime, base):
    return {"r": rnd(rng.uniform(6000, 9000)) if regime == "inside" else rnd(rng.uniform(60, 150))}


def shortcut(p):
    return p["dp"] * 1e-3 / (p["rho"] * f(p))                               # hPa/100 km -> Pa/m


def correct(p):
    fr = f(p) * p["r"] * 1e3
    return -fr / 2 + math.sqrt(fr ** 2 / 4 + fr * shortcut(p))


def text(p, variant):
    given = ("Because the isobars are strongly curved, the centripetal acceleration cannot be neglected: "
             "use the gradient-wind balance V^2/R + fV = (1/ρ)(∂p/∂n), where R is the radius of curvature, "
             "f = 2Ω sin φ, and ∂p/∂n is the pressure gradient toward lower pressure; its positive root is "
             "V = -fR/2 + sqrt(f^2 R^2/4 + fR V_g) with V_g = (1/(ρf))(∂p/∂n). " if variant == "given" else "")
    return (f"At latitude {num(p['lat'])}°N, air flows steadily parallel to isobars that curve "
            f"cyclonically, with lower pressure on the inside of the curve; their radius of curvature is "
            f"{num(p['r'])} km. The pressure decreases toward the inside of the curve at {num(p['dp'])} hPa per 100 km, and the air density is {num(p['rho'])} kg m^-3. Neglect friction "
            f"and take the Earth's angular velocity to be 7.292 × 10^-5 s^-1. {given}Find the wind speed. "
            f"Express your answer in m s^-1.")


def check() -> str:
    max_in = min_out = None
    spread = 0.
    for lat in range(40, 56):
        for rho in [x / 100 for x in range(110, 126)]:
            for dp in [x / 10 for x in range(6, 11)]:
                for r in [6000, 9000, 60, 150]:
                    p = {"lat": lat, "rho": rho, "dp": dp, "r": r}
                    key, vg = correct(p), shortcut(p)
                    # Independent: bisection on the force balance V^2/R + fV - f V_g = 0 on (0, V_g].
                    lo, hi = 0., vg
                    for _ in range(200):
                        mid = (lo + hi) / 2
                        if mid ** 2 / (r * 1e3) + f(p) * mid - f(p) * vg > 0:
                            hi = mid
                        else:
                            lo = mid
                    assert math.isclose(key, lo, rel_tol=1e-9), (p, key, lo)
                    g = gap(key, vg)
                    if r >= 6000:
                        max_in = g if max_in is None else max(max_in, g)
                    else:
                        min_out = g if min_out is None else min(min_out, g)
                        fr = 2 * 7.27e-5 * math.sin(math.radians(lat)) * r * 1e3
                        alt = -fr / 2 + math.sqrt(fr ** 2 / 4 + fr * dp * 1e-3 / (rho * fr / (r * 1e3)))
                        spread = max(spread, abs(alt / key - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and spread < .05
    # Limits: R -> infinity recovers V_g; R -> 0 recovers cyclostrophic sqrt(R dp/dn / rho).
    p = {"lat": 45, "rho": 1.2, "dp": 1.0, "r": 1e9}
    assert math.isclose(correct(p), shortcut(p), rel_tol=1e-5)
    p = {"lat": 45, "rho": 1.2, "dp": 1.0, "r": 1e-6}                      # R = 1 mm
    assert math.isclose(correct(p), math.sqrt(1e-3 * 1.0e-3 / 1.2), rel_tol=1e-3)
    return (f"bisection agrees; max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, "
            f"Omega-convention spread {spread:.4f}; geostrophic and cyclostrophic limits recovered")


TEMPLATE = Template("gradient_wind_curvature", "atmospheric_dynamics",
                    "geostrophic wind -> gradient wind when the isobars are strongly curved",
                    "m/s", base, trigger, text, correct, shortcut)
