"""Neutral logarithmic wind profile -> stable Monin-Obukhov profile.

Parent's method: the neutral surface-layer log law, u(z) = (u*/kappa) ln(z/z0).
Where it fails: when the Obukhov length L is not large compared with z, the stable stratification
steepens the profile; Monin-Obukhov similarity with the Businger-Dyer form phi_m = 1 + 5 z/L gives
u(z) = (u*/kappa) [ln(z/z0) + 5 (z - z0)/L].
Fair from the text: both variants state u*, z0, kappa and L, so z/L is computable from the givens;
outside, z/L = 0.25-0.33 at z = 10 m, well inside the range (z/L < 1) where the linear form is
the textbook stable law.
Inside: L = 2000-10000 m, z/L <= 0.005; the stability term changes u by < 0.8 %.
Conventions: Businger (1971, 4.7 z/L), Dyer (1974, 5 z/L), Hogstrom (1988, 6 z/L with kappa = 0.4)
and Beljaars & Holtslag (1991) all land within 5 % of the key on the outside grid; `check` reports
the spread. That is why z/L is capped at 1/3 and z0 kept at 0.01-0.03 m: at z/L = 0.5 or
z0 = 0.1 m the Hogstrom coefficient moves the answer by more than 5 %, and below z0 = 0.01 m the
log term is so large that L would have to fall under 37 m to keep a 15 % gap.
References: Dyer (1974) BLM 7, 363; Beljaars & Holtslag (1991) JAM 30, 327 (psi_m, a=1, b=2/3,
c=5, d=0.35); Stull (1988), An Introduction to Boundary Layer Meteorology, sect. 9.7.
Related mechanism: none in this set.

Review (2026-09-26, independent reviewer): u* was 0.15-0.50 m/s independently of L; with L = 30-40 m
that implies a surface heat flux of about -200 to -360 W m^-2 (w'theta' = -u*^3 theta/(kappa g L)),
unphysical for a stable surface layer, and 10-m winds of 8-11 m/s under strong stability. u* is now
0.15-0.30 m/s (about -10 to -80 W m^-2). The gap does not depend on u*, so the other ranges are
unchanged (they are not widened: the Hogstrom 6 z/L spread is already 4.45 %).
"""
from __future__ import annotations

import math

from .base import OUTSIDE_MIN_GAP, Template, gap, num, rnd

KAPPA, Z = 0.4, 10.0


def base(rng):
    return {"ustar": rnd(rng.choice([0.15, 0.2, 0.25, 0.3])),
            "z0": rng.choice([0.01, 0.015, 0.02, 0.03])}


def trigger(rng, regime, base):
    return {"L": rnd(rng.uniform(2000, 10000)) if regime == "inside" else rnd(rng.uniform(30, 40))}


def shortcut(p):
    return p["ustar"] / KAPPA * math.log(Z / p["z0"])


def correct(p):
    return p["ustar"] / KAPPA * (math.log(Z / p["z0"]) + 5 * (Z - p["z0"]) / p["L"])


def text(p, variant):
    given = ("Because the surface layer is stably stratified and z/L is not small, include the Monin-Obukhov "
             "stability correction: u(z) = (u*/κ)[ln(z/z0) + 5z/L], where L is the Obukhov length. "
             if variant == "given" else "")
    return (f"Over flat, uniform terrain with an aerodynamic roughness length of {num(p['z0'])} m, the friction "
            f"velocity is {num(p['ustar'])} m s^-1 and the Obukhov length is {num(p['L'])} m. Take the von Kármán "
            f"constant to be 0.4. {given}Find the mean wind speed at a height of 10 m. "
            f"Express your answer in m s^-1.")


def _psi_bh(zeta):
    a, b, c, d = 1.0, 2 / 3, 5.0, 0.35
    return -(a * zeta + b * (zeta - c / d) * math.exp(-d * zeta) + b * c / d)


def _profile(p, psi):
    return p["ustar"] / KAPPA * (math.log(Z / p["z0"]) - psi(Z / p["L"]) + psi(p["z0"] / p["L"]))


def check() -> str:
    # Independent: integrate du/dz = (u*/kappa z) phi_m(z/L) from z0 to z (Simpson, log-spaced).
    def integral(p, beta):
        n, lo, hi = 2000, math.log(p["z0"]), math.log(Z)
        h = (hi - lo) / n
        f = [1 + beta * math.exp(lo + i * h) / p["L"] for i in range(n + 1)]   # phi_m * dz/z in ln z
        s = f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2])
        return p["ustar"] / KAPPA * s * h / 3
    max_in, min_out, spread = 0.0, 1.0, {"4.7": 0.0, "6": 0.0, "BH": 0.0}
    for us in [0.15, 0.2, 0.25, 0.3]:
        for z0 in [0.01, 0.015, 0.02, 0.03]:
            for L in [2000, 5000, 10000] + [30, 35, 40]:
                p = {"ustar": us, "z0": z0, "L": L}
                key = correct(p)
                assert math.isclose(integral(p, 5), key, rel_tol=1e-9), p
                g = gap(key, shortcut(p))
                if L >= 2000:
                    max_in = max(max_in, g)
                    continue
                min_out = min(min_out, g)
                spread["4.7"] = max(spread["4.7"], gap(key, integral(p, 4.7)))
                spread["6"] = max(spread["6"], gap(key, integral(p, 6)))
                spread["BH"] = max(spread["BH"], gap(key, _profile(p, _psi_bh)))
    assert max_in <= 0.008 and min_out >= OUTSIDE_MIN_GAP and max(spread.values()) < 0.05, (max_in, min_out, spread)
    # Anchor: Beljaars-Holtslag chose their constants so that phi_m -> 1 + 5 zeta as zeta -> 0 (Dyer).
    eps = 1e-6
    assert math.isclose(-(_psi_bh(eps) - _psi_bh(0)) / eps, 5.0, rel_tol=1e-4)
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, outside spread vs key "
            + ", ".join(f"{k} {v:.4f}" for k, v in spread.items()) + "; BH slope at 0 = 5 (Dyer)")


TEMPLATE = Template("log_wind_stability", "boundary_layer",
                    "neutral log wind profile -> stable Monin-Obukhov profile when z/L is not small",
                    "m/s", base, trigger, text, correct, shortcut)
