"""Acceleration from rest without rotation -> inertial oscillation about geostrophy (added 2026-09-26).

Parent's method: air at rest under a uniform pressure-gradient force per unit mass
a = (1/rho) dp/dx accelerates as v = a t.
Where it fails: once f t is of order 1, the Coriolis force turns the flow. On an f-plane,
du/dt = a + f v, dv/dt = -f u from rest give u = (a/f) sin ft, v = -(a/f)(1 - cos ft), so the speed
is |V| = (2a/f)|sin(ft/2)|: it oscillates between 0 and twice the geostrophic speed a/f.
Fair from the text: every arm gives the latitude, the density and the pressure gradient,
says the air starts from rest and the gradient then stays constant, and says to neglect friction;
the elapsed time is the only difference.
Inside: t = 0.2-0.5 h, f t <= 0.23, the speed is within 0.3 % of a t. Outside: f t = 2.2-4.2 (the
time is drawn as a target f t divided by f, rounded to 2 significant figures), a t exceeds the
speed by >= 20 %; f t stays below 2 pi so the key is never near zero.
Conventions: none beyond Omega (7.292e-5 vs 7.27e-5 s^-1 changes the key by < 1 %). The f-plane
is the textbook model and the text puts the problem at one latitude. It is an idealisation: over
the outside times a parcel drifts up to about 1100 km along the isobars and up to about 300 km
across them (worst case 40N, 2 hPa per 100 km, ft = 4.4), and the text gives no orientation, so if
the drift were meridional f would change by up to ~20 %. The problem does not ask for that
refinement and no text answer depends on it.
Reference: Holton & Hakim (5th ed.), sec. 3.2.4 (inertial oscillations); Vallis, Atmospheric and
Oceanic Fluid Dynamics (2nd ed.), sec. 3.8.3 (geostrophic adjustment from rest: the flow oscillates
about the geostrophic value, peaking at 2 a/f when ft = pi). Related: `gradient_wind_curvature`.

Review (2026-09-26, independent reviewer): at 30N, 3 hPa per 100 km and ft = 3.2 the key was 75 m/s
(shortcut up to ~150 m/s) and the parcel moved ~1000 km across the isobars, contrary to the old
docstring's "< 1000 km, f changes < 10 %". Latitude is now 40-60N and the gradient 1.0-2.0 hPa per
100 km, so keys stay below 40 m/s and the cross-isobar drift below ~300 km; the docstring states the
actual worst-case displacement. "Treat f as constant" is deliberately not added to the text (a cue).

Review (2026-09-27, domain + measurement-design review): Omega is a universal constant used only by
the replacement (the shortcut a t never needs it), so stating it in every arm hinted that rotation
matters. It now appears only in the given sentence; the inside and outside texts give the latitude
alone. check() measures the Omega-convention spread (7.27e-5 vs 7.292e-5 s^-1) on every outside key.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

OMEGA = 7.292e-5


def f(p):
    return 2 * OMEGA * math.sin(math.radians(p["lat"]))


def base(rng):
    return {"lat": float(rng.randint(40, 60)), "rho": float(rng.choice(range(110, 126))) / 100,
            "dp": float(rng.choice([1.0, 1.5, 2.0]))}            # hPa per 100 km


def trigger(rng, regime, base):
    if regime == "inside":
        return {"t": float(rng.choice([0.2, 0.25, 0.3, 0.4, 0.5]))}
    return {"t": rnd(rng.uniform(2.2, 4.2) / f(base) / 3600)}             # hours


def accel(p):
    return p["dp"] * 1e-3 / p["rho"]                                        # hPa/100 km -> Pa/m


def shortcut(p):
    return accel(p) * p["t"] * 3600


def correct(p):
    return 2 * accel(p) / f(p) * abs(math.sin(f(p) * p["t"] * 3600 / 2))


def text(p, variant):
    given = ("Because the time is not short compared with 1/f, the Coriolis force cannot be neglected: the "
             "air performs an inertial oscillation about the geostrophic wind, and its speed is "
             "|V| = (2a/f)|sin(ft/2)|, where a is the pressure-gradient force per unit mass and "
             "f = 2Ω sin φ with Ω = 7.292 × 10^-5 s^-1. " if variant == "given" else "")
    return (f"At latitude {num(p['lat'])}°N, air of density {num(p['rho'])} kg m^-3 is initially at rest. A "
            f"uniform horizontal pressure gradient of {num(p['dp'])} hPa per 100 km is suddenly established "
            f"and then stays constant. Neglect friction. {given}What is the wind speed {num(p['t'])} h after the pressure gradient "
            f"is established? Express your answer in m s^-1.")


def rk4_speed(p, steps=4000):
    """Independent: integrate du/dt = a + f v, dv/dt = -f u from rest."""
    a, ff, dt = accel(p), f(p), p["t"] * 3600 / steps
    u = v = 0.
    def rhs(u, v):
        return a + ff * v, -ff * u
    for _ in range(steps):
        k1 = rhs(u, v)
        k2 = rhs(u + dt / 2 * k1[0], v + dt / 2 * k1[1])
        k3 = rhs(u + dt / 2 * k2[0], v + dt / 2 * k2[1])
        k4 = rhs(u + dt * k3[0], v + dt * k3[1])
        u += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return math.hypot(u, v)


def check() -> str:
    max_in = min_out = None
    spread = max_key = 0.
    for lat in range(40, 61):
        for dp in [1.0, 1.5, 2.0]:
            b = {"lat": lat, "rho": 1.1, "dp": dp}
            ts = [0.2, 0.5] + [rnd(x / f(b) / 3600) for x in (2.2, 3.2, 4.2)]
            for t in ts:
                p = {**b, "t": t}
                key = correct(p)
                assert math.isclose(key, rk4_speed(p), rel_tol=1e-7), p
                g = gap(key, shortcut(p))
                if t <= .5:
                    max_in = g if max_in is None else max(max_in, g)
                else:
                    min_out = g if min_out is None else min(min_out, g)
                    max_key = max(max_key, key)
                    assert f(p) * t * 3600 < 2 * math.pi - 1.5            # key well away from zero
                    alt = 2 * accel(p) / (f(p) * 7.27 / 7.292) * abs(math.sin(f(p) * 7.27 / 7.292 * t * 1800))
                    spread = max(spread, abs(alt / key - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and spread < .05 and max_key < 40
    # Anchor: at ft = pi the speed peaks at twice the geostrophic speed a/f.
    p = {"lat": 45, "rho": 1.2, "dp": 2.0}
    p["t"] = math.pi / f(p) / 3600
    assert math.isclose(correct(p), 2 * accel(p) / f(p), rel_tol=1e-12)
    return (f"RK4 agrees; max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, "
            f"Omega-convention spread {spread:.4f}, max outside key {max_key:.1f} m/s; peak 2a/f at ft = pi")


TEMPLATE = Template("coriolis_neglect", "atmospheric_dynamics",
                    "acceleration from rest without rotation -> inertial oscillation when ft is of order 1",
                    "m/s", base, trigger, text, correct, shortcut)
