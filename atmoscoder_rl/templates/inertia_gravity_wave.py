"""Non-rotating gravity-wave period -> inertia-gravity wave period for long waves (added 2026-09-26).

Parent's method: a hydrostatic internal gravity wave in the mid-frequency range has
omega = N k/m, so its period is 2 pi lambda_x/(N lambda_z).
Where it fails: when omega is not large compared with f (horizontal wavelengths of order 1000 km),
rotation enters: omega^2 = (N^2 k^2 + f^2 m^2)/(k^2 + m^2) ~ f^2 + N^2 k^2/m^2 (hydrostatic), and the
period cannot exceed the inertial period 2 pi/f. The non-rotating period is then far too long.
Fair from the text: every arm gives the latitude, so f is computable with the universal Omega, plus N
and both wavelengths; the horizontal wavelength is the only difference.
Inside: lambda_x = 8-12 lambda_z (lambda_z = 2-4 km, so 16-48 km) with N = 0.015-0.020 s^-1 at
30-50 N. The ratio is kept >= 8 so the non-hydrostatic factor (1 + k^2/m^2)^(-1/2) and the rotation
term f^2 m^2/(N^2 k^2) together move the period by < 0.8 %; a wider inside range is not possible,
because a longer wave feels rotation and a shorter one is non-hydrostatic. Outside:
lambda_x = 1000-3000 km, N k/m <= 1.1 f, the non-rotating period is >= 35 % too long.
Conventions: the key is the full Boussinesq relation. The hydrostatic form f^2 + N^2 k^2/m^2
differs by < 1e-5 outside; the compressible 1/(4 H^2) addition to m^2 (H = 7 km) by < 0.3 %.
Reference: Fritts & Alexander (2003), Rev. Geophys. 41, 1003, eq. (26); Holton & Hakim (5th ed.),
sec. 7.4 (inertia-gravity waves). Related: `coriolis_neglect` (the same ingredient, rotation f).

Review (2026-09-27, domain + measurement-design review): Omega is used only by the replacement (the
non-rotating period never needs it), so it now appears only in the given sentence; the other texts
give the latitude alone. check() measures the Omega-convention spread (7.27e-5 vs 7.292e-5 s^-1).
The pilot `nonhydrostatic_wave` this docstring used to cite was removed from the set the same day.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

OMEGA = 7.292e-5


def f(p):
    return 2 * OMEGA * math.sin(math.radians(p["lat"]))


def base(rng):
    return {"lat": float(rng.randint(30, 50)), "n": float(rng.choice(range(15, 21))) / 1000,
            "lz": float(rng.choice([2.0, 2.5, 3.0, 3.5, 4.0]))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"lx": max(rnd(base["lz"] * rng.uniform(8, 12)), 8 * base["lz"])}
    return {"lx": rnd(rng.uniform(1000, 3000))}


def shortcut(p):
    return 2 * math.pi * p["lx"] / (p["n"] * p["lz"]) / 3600                 # hours


def correct(p):
    k, m = 2 * math.pi / p["lx"], 2 * math.pi / p["lz"]                     # per km; only ratios matter
    omega = math.sqrt((p["n"] ** 2 * k ** 2 + f(p) ** 2 * m ** 2) / (k ** 2 + m ** 2))
    return 2 * math.pi / omega / 3600


def text(p, variant):
    given = ("Because the wave frequency is not large compared with the Coriolis parameter f = 2Ω sin φ "
             "(Ω = 7.292 × 10^-5 s^-1), "
             "rotation cannot be neglected: use the inertia–gravity-wave dispersion relation "
             "ω^2 = f^2 + N^2 k^2/m^2, where k and m are the horizontal and vertical wavenumbers. "
             if variant == "given" else "")
    return (f"In the lower stratosphere at latitude {num(p['lat'])}°N, the buoyancy (Brunt–Väisälä) frequency "
            f"is {num(p['n'])} s^-1. An internal gravity wave there has a horizontal wavelength of "
            f"{num(p['lx'])} km and a vertical wavelength of {num(p['lz'])} km. {given}What is the intrinsic period of the wave? Express "
            f"your answer in hours.")


def check() -> str:
    max_in = min_out = None
    spread = 0.
    for lat in range(30, 51):
        for n in [x / 1000 for x in range(15, 21)]:
            for lz in (2.0, 2.5, 3.0, 3.5, 4.0):
                for lx in (8 * lz, 12 * lz, 1000, 3000):
                    p = {"lat": lat, "n": n, "lz": lz, "lx": lx}
                    key = correct(p)
                    # Independent: angular form omega^2 = N^2 cos^2(a) + f^2 sin^2(a), a = angle of the
                    # wavevector from the horizontal (Holton & Hakim eq. 7.64 with rotation).
                    a = math.atan2(1 / lz, 1 / lx)
                    om = math.sqrt(n ** 2 * math.cos(a) ** 2 + f(p) ** 2 * math.sin(a) ** 2)
                    assert math.isclose(key, 2 * math.pi / om / 3600, rel_tol=1e-9), p
                    g = gap(key, shortcut(p))
                    if lx < 100:
                        max_in = g if max_in is None else max(max_in, g)
                    else:
                        min_out = g if min_out is None else min(min_out, g)
                        k, m = 2 * math.pi / lx, 2 * math.pi / lz
                        hydro = 2 * math.pi / math.sqrt(f(p) ** 2 + n ** 2 * k ** 2 / m ** 2) / 3600
                        m2 = m ** 2 + 1 / (4 * 7.0 ** 2)
                        comp = 2 * math.pi / math.sqrt((n ** 2 * k ** 2 + f(p) ** 2 * m2) / (k ** 2 + m2)) / 3600
                        f_alt = f(p) * 7.27 / 7.292                   # the other common Omega
                        omega_alt = 2 * math.pi / math.sqrt((n ** 2 * k ** 2 + f_alt ** 2 * m ** 2)
                                                            / (k ** 2 + m ** 2)) / 3600
                        spread = max(spread, abs(hydro / key - 1), abs(comp / key - 1),
                                     abs(omega_alt / key - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and spread < .05
    # Anchor: the long-wave limit is the inertial period 2 pi/f = 12 h/sin(phi) (24 h at 30 N).
    p = {"lat": 30, "n": .02, "lz": 3.0, "lx": 1e9}
    assert math.isclose(correct(p), 2 * math.pi / OMEGA / 3600, rel_tol=1e-6)      # 23.9 h
    return (f"angular-form check agrees; max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, "
            f"hydrostatic/compressible/Omega spread {spread:.2e}; long-wave limit = inertial period")


TEMPLATE = Template("inertia_gravity_wave", "atmospheric_dynamics",
                    "non-rotating gravity-wave period -> inertia-gravity-wave period when omega ~ f",
                    "h", base, trigger, text, correct, shortcut)
