"""Kinematic vertical velocity with constant density -> with density decreasing with height
(added 2026-09-26).

Parent's method: with uniform horizontal convergence C (divergence -C) and w = 0 at the ground, the
incompressible continuity equation dw/dz = C gives w = C z.
Where it fails: mass continuity is d(rho w)/dz = rho C. When the layer is not thin compared with
the density scale height H = R_d T/g, rho falls with height and w grows faster than linearly:
in an isothermal atmosphere rho w = C rho0 H (1 - e^{-z/H}), so w = C H (e^{z/H} - 1).
Fair from the text: every arm says the atmosphere is isothermal and gives its temperature, and says
the convergence is the same throughout the layer; R_d and g are universal and stated only in the
given sentence (2026-09-27 note). The target height is the only difference.
Inside: z = 50-250 m, z/H <= 0.034, the linear law is within 1.8 %. Outside: z = 3-6 km,
z/H >= 0.35, C z underestimates w by >= 16 %.
Conventions: the pressure-coordinate kinematic method (omega = -C (p0 - p), w = -omega/(rho g))
gives the identical answer (checked); R_d = 287.05 vs 287 and g = 9.8 vs 9.81 move the key < 0.2 %.
Reference: Holton & Hakim (5th ed.), sec. 3.5 (kinematic method, continuity in isobaric
coordinates); Wallace & Hobbs (2nd ed.), sec. 3.2 (isothermal scale height). Related:
`hydrostatic_constant_density` (the same neglect of the density decrease with height).

Review (2026-09-26, independent reviewer): "the divergence is -C at every height" read literally
means the whole column converges, which with w = 0 at the ground and steady density would make
surface pressure rise at p0 C (36-108 hPa/h). All arms now say the convergence holds throughout the
layer between the ground and the level considered (the Holton kinematic-method premise), without
printing the height twice. Keys unchanged.

Review (2026-09-27, domain + measurement-design review): (1) outside heights are capped at 6 km:
uniform low-level convergence to 8 km is not a realistic column (deep convergence to ~6 km is
defensible for tropical disturbances); the gap stays >= 16 %. (2) R_d and g are removed from the
inside/outside texts and stated only in the given sentence (the new rule for constants used only
by the replacement -- the linear law needs neither). Recalled values R_d = 287-287.05 and
g = 9.80-9.81 move the outside key by < 0.2 % (checked below).
Rule 11 (2026-09-27): answer in mm/s, not cm/s — the smallest inside key was 0.05 cm/s, which a
solver's round(x, 2) fails; in mm/s every key survives it.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

RD, G = 287.0, 9.81


def scale_height(p):
    return RD * p["t"] / G


def base(rng):
    return {"t": float(rng.choice(range(250, 295, 5))), "c": float(f"{rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])}e-5")}


def trigger(rng, regime, base):
    return {"z": rnd(rng.uniform(50, 250)) if regime == "inside" else rnd(rng.uniform(3000, 6000))}


def shortcut(p):
    return p["c"] * p["z"] * 1000                                            # mm/s


def correct(p):
    h = scale_height(p)
    return p["c"] * h * math.expm1(p["z"] / h) * 1000


def text(p, variant):
    given = ("Because the layer is not thin compared with the density scale height H = R_d T/g "
             "(R_d = 287 J kg^-1 K^-1, g = 9.81 m s^-2), use mass continuity, d(ρw)/dz = -ρ × (horizontal divergence), with the density decreasing as "
             "ρ = ρ_0 exp(-z/H); this gives w = C H (exp(z/H) - 1), where C is the convergence. "
             if variant == "given" else "")
    return (f"Over a region, the horizontal wind converges uniformly: the horizontal velocity divergence is "
            f"−{num(p['c'])} s^-1 throughout the layer between the ground and the level considered. The atmosphere is isothermal at {num(p['t'])} K. "
            f"The vertical velocity is zero at the ground. "
            f"{given}Find the vertical velocity at a height of {num(p['z'])} m above the ground. Express your "
            f"answer in mm s^-1.")


def check() -> str:
    max_in = min_out = None
    spread = 0.
    for t in range(250, 295, 5):
        for c in (1e-5, 3e-5):
            for z in (50, 250, 3000, 6000):
                p = {"t": t, "c": c, "z": z}
                key = correct(p)
                # Independent 1: trapezoid integration of rho w = C * int_0^z rho dz, rho from p = rho R T.
                n, h = 20000, scale_height(p)
                rho = [math.exp(-i * z / n / h) for i in range(n + 1)]
                mass = sum((rho[i] + rho[i + 1]) / 2 for i in range(n)) * z / n
                assert math.isclose(key, c * mass / rho[-1] * 1000, rel_tol=1e-6), p
                # Independent 2: isobaric kinematic method with p0 = 1000 hPa.
                p0 = 1e5
                pz = p0 * math.exp(-z / h)
                omega = -c * (p0 - pz)                                       # Pa/s, upward negative
                w = -omega / (pz / (RD * t) * G) * 1000
                assert math.isclose(key, w, rel_tol=1e-9), p
                g = gap(key, shortcut(p))
                if z <= 250:
                    max_in = g if max_in is None else max(max_in, g)
                else:
                    min_out = g if min_out is None else min(min_out, g)
                    for rd, g0 in ((287.05, 9.8), (287.0, 9.80665), (287.04, 9.81), (287.0, 9.8)):
                        ha = rd * t / g0                                     # R_d, g no longer stated
                        spread = max(spread, abs(c * ha * math.expm1(z / ha) * 1000 / key - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and spread < .05
    return (f"trapezoid and isobaric-kinematic checks agree; max inside gap {max_in:.4f}, "
            f"min outside gap {min_out:.3f}, R_d/g spread {spread:.4f}")


TEMPLATE = Template("anelastic_continuity", "atmospheric_dynamics",
                    "incompressible kinematic w -> mass continuity with density decreasing over a deep layer",
                    "mm/s", base, trigger, text, correct, shortcut)
