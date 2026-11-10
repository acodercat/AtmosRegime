"""Non-divergent Rossby wave -> free-surface (divergent) Rossby wave (added 2026-09-26).

Parent's method: a barotropic Rossby wave with no mean flow and no north-south variation moves
westward at c = beta/k^2 (non-divergent vorticity equation), beta = 2 Omega cos(phi0)/a.
Where it fails: in a shallow layer with a free surface, stretching by the surface displacement adds
a term to the potential vorticity; the phase speed becomes c = beta/(k^2 + 1/L_d^2) with the
Rossby radius of deformation L_d = sqrt(gH)/f0. When the wavelength is not short compared with
L_d (here 2200-3300 km), beta/k^2 overestimates the speed.
Fair from the text: every arm says the fluid is a shallow layer with a free surface, gives its mean
depth, Omega, Earth's radius and the latitude of the beta-plane; with the universal g (given only in
the given sentence, see the 2026-09-27 note) L_d is computable.
Inside: zonal wavelength 1000-1800 km, k^2 L_d^2 >= 60, the correction is < 1.7 %. Outside:
10 000-15 000 km, the non-divergent speed is >= 22 % too fast. 15 000 km fits inside the latitude
circle at 60 N (20 015 km).
Conventions: the beta-plane is stated in every arm, so the spherical Rossby-Haurwitz formula is not
an alternative; g = 9.8 vs 9.81 and Omega = 7.27e-5 vs 7.292e-5 move the key by < 0.5 %. Meridional
wavenumber is zero by the text ("no north-south variation").
Reference: Vallis, Atmospheric and Oceanic Fluid Dynamics (2nd ed.), sec. 6.4.1, eq. (6.60):
omega = -beta k/(k^2 + l^2 + k_d^2); Holton & Hakim (5th ed.), sec. 5.7.

Review (2026-09-26, independent reviewer): a competent solver may use the full linear shallow-water
beta-plane dispersion relation, omega^3 - omega (gH k^2 + f0^2) - beta gH k = 0 (slowest root),
instead of the QG form. It lay up to 4.63 % from the key at 40N, H = 10 km, 18 000 km -- at the
tolerance once combined with g = 9.8 or Omega = 7.27e-5. The outside wavelength is now capped at
15 000 km and check() measures this root's spread (asserted < 5 %).

Review (2026-09-27, domain + measurement-design review): g is used only by the replacement (through
L_d; the non-divergent beta/k^2 never needs it), so stating it in every arm hinted at the free-surface
term. It now appears only in the given sentence. Omega and a stay in every arm: the shortcut needs them
for beta. check() measures the g-convention spread (9.78-9.83 m s^-2) on every outside key.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

OMEGA, A, G = 7.292e-5, 6.371e6, 9.81


def f0(p):
    return 2 * OMEGA * math.sin(math.radians(p["lat"]))


def beta(p):
    return 2 * OMEGA * math.cos(math.radians(p["lat"])) / A


def k(p):
    return 2 * math.pi / (p["lx"] * 1e3)


def base(rng):
    return {"lat": float(rng.randint(40, 60)), "h": float(rng.choice([8, 9, 10]))}


def trigger(rng, regime, base):
    return {"lx": rnd(rng.uniform(1000, 1800)) if regime == "inside" else rnd(rng.uniform(10000, 15000))}


def shortcut(p):
    return beta(p) / k(p) ** 2


def correct(p):
    ld2 = G * p["h"] * 1e3 / f0(p) ** 2
    return beta(p) / (k(p) ** 2 + 1 / ld2)


def text(p, variant):
    given = ("Because the wavelength is not short compared with the Rossby radius of deformation, the "
             "free-surface displacement cannot be neglected: use c = β/(k^2 + 1/L_d^2), where "
             "L_d = sqrt(gH)/f_0 with g = 9.81 m s^-2, f_0 = 2Ω sin φ_0, β = 2Ω cos φ_0/a and k is the "
             "zonal wavenumber. "
             if variant == "given" else "")
    return (f"A shallow layer of fluid with a free surface has a mean depth of {num(p['h'])} km and no mean "
            f"flow. On a β-plane centred at {num(p['lat'])}°N, a Rossby wave in the layer has a zonal "
            f"wavelength of {num(p['lx'])} km and no north–south variation. Take the "
            f"Earth's angular velocity Ω = 7.292 × 10^-5 s^-1 and the Earth's radius a = 6371 km. {given}"
            f"At what speed do the wave crests move westward? Express your answer in m s^-1.")


def fd_speed(p):
    """Independent: phase speed from the linear QG PV equation d/dt(psi_xx - psi/L_d^2) + beta psi_x = 0,
    evaluated by centred finite differences on psi = cos(kx - omega t) at an arbitrary point."""
    ld2 = G * p["h"] * 1e3 / f0(p) ** 2
    kk, dx, x0 = k(p), p["lx"] * 1e3 / 2000, 0.37 * p["lx"] * 1e3
    def psi(x):
        return math.cos(kk * x)
    psi_x = (psi(x0 + dx) - psi(x0 - dx)) / (2 * dx)
    def q(x):                                       # PV at t = 0
        return (psi(x + dx) - 2 * psi(x) + psi(x - dx)) / dx ** 2 - psi(x) / ld2
    # For psi = cos(kx - w t): dq/dt = -c dq/dx with c = w/k; solve -c q_x + beta psi_x = 0 for c.
    q_x = (q(x0 + dx) - q(x0 - dx)) / (2 * dx)
    return -(beta(p) * psi_x / q_x)                 # westward positive


def sw_speed(p):
    """Slowest (Rossby) root of the linear shallow-water beta-plane relation with l = 0,
    omega^3 - omega (gH k^2 + f0^2) - beta gH k = 0, by Newton from the QG estimate; westward positive."""
    gh, kk, ff, bb = G * p["h"] * 1e3, k(p), f0(p), beta(p)
    w = -bb * gh * kk / (gh * kk ** 2 + ff ** 2)
    for _ in range(50):
        w -= (w ** 3 - w * (gh * kk ** 2 + ff ** 2) - bb * gh * kk) / (3 * w ** 2 - (gh * kk ** 2 + ff ** 2))
    return -w / kk


def check() -> str:
    max_in = min_out = None
    spread = sw_spread = 0.
    for lat in range(40, 61):
        for h in (8, 9, 10):
            for lx in (1000, 1800, 10000, 15000):
                p = {"lat": lat, "h": h, "lx": lx}
                key = correct(p)
                assert math.isclose(key, fd_speed(p), rel_tol=1e-4), (p, key, fd_speed(p))
                g = gap(key, shortcut(p))
                if lx <= 1800:
                    max_in = g if max_in is None else max(max_in, g)
                else:
                    min_out = g if min_out is None else min(min_out, g)
                    om = 7.27e-5
                    fa, ba = 2 * om * math.sin(math.radians(lat)), 2 * om * math.cos(math.radians(lat)) / A
                    alt = ba / (k(p) ** 2 + fa ** 2 / (9.8 * h * 1e3))
                    spread = max(spread, abs(alt / key - 1))
                    for g_alt in (9.78, 9.83):                    # g, now stated only in the given arm
                        alt_g = beta(p) / (k(p) ** 2 + f0(p) ** 2 / (g_alt * h * 1e3))
                        spread = max(spread, abs(alt_g / key - 1))
                    sw_spread = max(sw_spread, abs(sw_speed(p) / key - 1))
                    assert lx < 2 * math.pi * A / 1e3 * math.cos(math.radians(lat))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and spread < .05 and sw_spread < .05, sw_spread
    # Anchors: beta at 45 N is 1.62e-11 m^-1 s^-1 (Holton & Hakim); L_d for H = 10 km at 45 N is
    # about 3000 km, the external (barotropic) deformation radius quoted in Vallis sec. 3.8.
    p = {"lat": 45, "h": 10}
    assert abs(beta(p) - 1.62e-11) < .005e-11
    ld = math.sqrt(G * 1e4) / f0(p) / 1e3
    assert 2900 < ld < 3200
    return (f"finite-difference PV check agrees; max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, "
            f"g/Omega spread {spread:.4f}, full shallow-water root spread {sw_spread:.4f}; beta(45N) = {beta(p):.3g}, L_d(10 km, 45N) = {ld:.0f} km")


TEMPLATE = Template("rossby_deformation", "atmospheric_dynamics",
                    "non-divergent Rossby wave speed -> free-surface Rossby wave when the wave is not short "
                    "compared with the deformation radius",
                    "m/s", base, trigger, text, correct, shortcut)
