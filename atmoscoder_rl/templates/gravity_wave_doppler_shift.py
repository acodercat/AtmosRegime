"""Intrinsic gravity-wave period -> ground-observed (Doppler-shifted) period (extended, dynamics, 2026-10-05).

Parent's method: the period of an internal gravity wave follows from its dispersion relation. For a
hydrostatic wave in a layer of buoyancy frequency N with horizontal and vertical wavenumbers k and m,
omega = N k/m, so the period is 2 pi m/(N k) = (2 pi/N)(lambda_x/lambda_z). This is the textbook
one-line answer and the number a solver writes for "the period of the wave".
Where it fails: the dispersion relation gives the *intrinsic* frequency, seen by an observer drifting
with the air. An instrument fixed to the ground sees the wave advected by the mean wind,
omega_obs = omega_int + k . U = omega_int + k U cos(theta), with theta the angle between the wind
and the wave's horizontal direction of travel. When the wind has a sizeable component along the
propagation the observed period is several times shorter.
Fair from the text: every arm states N, both wavelengths, the wind speed, the angle between the wind
and the wave's direction of travel (relative to the air) and that the period is recorded by an
instrument fixed to the ground. Nothing names the Doppler shift; the wind and its direction are
ordinary sounding data. The only trigger is the angle.
Inside: theta = 90 deg (wind perpendicular to the propagation), so k.U = 0 and the intrinsic period
is the observed one exactly. A near-perpendicular range would not do: with U/c_int up to 2.3 the
Doppler term reaches 2 % already 0.5 deg off perpendicular.
Outside: theta = 5-25 deg (wind blowing nearly along the propagation), U = 8-15 m/s against an
intrinsic horizontal phase speed c_int = N/m = 6.4-13.4 m/s, so the observed period is 1.5-3.3x
shorter than the intrinsic one; the shortcut (52-126 min) is an ordinary gravity-wave period.
Opposing winds are excluded (they approach a critical level, omega_obs -> 0).
Ranges: N 0.010-0.014 s^-1, lambda_z 4.0-6.0 km, lambda_x 7-12 lambda_z (28-72 km): k/m <= 0.143,
so the non-hydrostatic form N k/sqrt(k^2+m^2) moves the key by <= 1.0 %, and the intrinsic frequency
(>= 8.3e-4 s^-1) is far above f, so adding rotation (omega^2 = f^2 + N^2 k^2/m^2, f at 45 deg) moves
it by <= 0.8 %. "Hydrostatic" and "non-rotating" are therefore not stated (rule 4: they would only be
extra words); `check()` measures both spreads on the outside grid.
Conventions: the angle is between the wind vector (the direction the wind blows *toward*) and the
direction in which the crests move relative to the air; the text says "blows at an angle of theta to
that direction", which a solver reads as the angle between the two vectors. A meteorological
"from" reading would put the wind against the wave (theta + 180), giving omega_obs < omega_int;
the text's "blows ... to that direction" is the vector reading, recorded here as a residual risk.
Reference: Holton & Hakim, *An Introduction to Dynamic Meteorology* (5th ed., 2013), sec. 7.4
(internal gravity waves, intrinsic vs observed frequency, omega = U k +- N k/m); Nappo,
*An Introduction to Atmospheric Gravity Waves* (2nd ed., 2012), ch. 2 (Doppler shifting by the mean
wind). Related: `inertia_gravity_wave` (adds rotation to the same dispersion relation; here rotation
is negligible and the missing ingredient is advection, not a term of the dispersion relation).
Independent review (2026-10-05): "its crests move horizontally" was wrong for an internal wave (phase
lines are tilted) -- now "the wave propagates horizontally in a fixed direction"; the wind is now said to
blow "toward a direction theta from that direction", removing the from/toward ambiguity. `check()` asserts
the "wind from" reading gives omega_obs <= 0 or a period > 2x the shortcut on the outside grid.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True


def _k_m(p):
    return 2 * math.pi / (p["lx"] * 1e3), 2 * math.pi / (p["lz"] * 1e3)


def shortcut(p):
    """Intrinsic hydrostatic period, min."""
    k, m = _k_m(p)
    return 2 * math.pi / (p["n"] * k / m) / 60


def correct(p):
    """Ground-observed period, min: omega_obs = N k/m + k U cos(theta)."""
    k, m = _k_m(p)
    w = p["n"] * k / m + k * p["u"] * math.cos(math.radians(p["theta"]))
    return 2 * math.pi / w / 60


def base(rng):
    lz = rnd(rng.uniform(4.0, 6.0))
    return {"n": rnd(rng.uniform(0.010, 0.014)), "lz": lz, "lx": rnd(lz * rng.uniform(7.0, 12.0)),
            "u": rnd(rng.uniform(8.0, 15.0))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"theta": 90.0}
    return {"theta": float(rng.randint(5, 25))}


def text(p, variant):
    given = ("An instrument fixed to the ground sees the wave Doppler-shifted by the wind: the observed "
             "frequency is the intrinsic frequency N k/m plus k U cos θ, where k and m are the horizontal and "
             "vertical wavenumbers, U the wind speed and θ the angle between the wind and the propagation direction. " if variant == "given" else "")
    return (f"An internal gravity wave travels through a stably stratified layer with a buoyancy (Brunt–Väisälä) "
            f"frequency of {num(p['n'])} s^-1. Its horizontal wavelength is {num(p['lx'])} km and its vertical "
            f"wavelength is {num(p['lz'])} km. Relative to the air, the wave propagates horizontally in a fixed "
            f"direction, and a uniform wind of {num(p['u'])} m/s blows toward a direction {num(p['theta'])}° from "
            f"that direction. {given}What wave period does a microbarograph fixed to the ground record? Express your "
            f"answer in min.")


def _vector_period(p):
    """Independent: sample the wave at a fixed ground point and time its zero crossings. In the air's
    frame the field is cos(k.x' - omega_int t); a ground point sits at x' = -U t (2-D vectors, Galilean
    transform), so the recorded signal is cos(-k.U t - omega_int t). The period is read off the spacing
    of the first and last of many up-crossings, located by bisection -- numerics, not the k.U formula."""
    lx, lz = p["lx"] * 1e3, p["lz"] * 1e3
    kvec = (2 * math.pi / lx, 0.0)
    w_int = p["n"] * lz / lx                                       # N k/m written via the wavelengths
    th = math.radians(p["theta"])
    uvec = (p["u"] * math.cos(th), p["u"] * math.sin(th))

    def signal(t):
        xp = (-uvec[0] * t, -uvec[1] * t)
        return math.cos(kvec[0] * xp[0] + kvec[1] * xp[1] - w_int * t)

    step, t, prev, ups = 5.0, 0.0, signal(0.0), []
    while len(ups) < 6 and t < 2e5:
        t += step
        cur = signal(t)
        if prev < 0 <= cur:                                          # up-crossing in (t - step, t]
            lo, hi = t - step, t
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if signal(mid) >= 0 else (mid, hi)
            ups.append(hi)
        prev = cur
    return (ups[-1] - ups[0]) / (len(ups) - 1) / 60


def _nonhydrostatic(p):
    k, m = _k_m(p)
    w = p["n"] * k / math.hypot(k, m) + k * p["u"] * math.cos(math.radians(p["theta"]))
    return 2 * math.pi / w / 60


def _rotating(p, lat=45.0):
    k, m = _k_m(p)
    f = 2 * 7.292e-5 * math.sin(math.radians(lat))
    w = math.sqrt(f ** 2 + (p["n"] * k / m) ** 2) + k * p["u"] * math.cos(math.radians(p["theta"]))
    return 2 * math.pi / w / 60


def check() -> str:
    max_in, min_out, max_err, spread = 0.0, math.inf, 0.0, 0.0
    min_key, max_key, min_short, max_short = math.inf, 0.0, math.inf, 0.0
    for n in (0.010, 0.012, 0.014):
        for lz in (4.0, 5.0, 6.0):
            for ratio in (7.0, 9.5, 12.0):
                for u in (8.0, 11.5, 15.0):
                    b = {"n": n, "lz": lz, "lx": lz * ratio, "u": u}
                    p = {**b, "theta": 90.0}
                    max_in = max(max_in, gap(correct(p), shortcut(p)))
                    max_err = max(max_err, gap(correct(p), _vector_period(p)))
                    for th in (5.0, 15.0, 25.0):
                        p = {**b, "theta": th}
                        c, s = correct(p), shortcut(p)
                        min_out = min(min_out, gap(c, s))
                        max_err = max(max_err, gap(c, _vector_period(p)))
                        spread = max(spread, gap(c, _nonhydrostatic(p)), gap(c, _rotating(p)))
                        min_key, max_key = min(min_key, c), max(max_key, c)
                        min_short, max_short = min(min_short, s), max(max_short, s)
    assert max_err < 1e-6 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    assert spread < 0.05, spread                                    # rule 5: hydrostatic / rotation conventions
    # Misreading the wind as blowing *from* theta (vector reversed): omega_obs = omega_int - k U cos(theta)
    # is <= 0 or gives a period at least 2x the shortcut on the whole outside grid -- far off, not a rival.
    for n in (0.010, 0.014):
        for lz in (4.0, 6.0):
            for ratio in (7.0, 12.0):
                for u in (8.0, 15.0):
                    for th in (5.0, 25.0):
                        p = {"n": n, "lz": lz, "lx": lz * ratio, "u": u, "theta": th + 180.0}
                        k, m = _k_m(p)
                        w = p["n"] * k / m + k * p["u"] * math.cos(math.radians(p["theta"]))
                        assert w <= 0 or 2 * math.pi / w / 60 > 2 * shortcut(p), (p, w)
    assert min_key > 10 and max_key < 120, (min_key, max_key)       # rounding-safe, ordinary periods
    assert 40 < min_short and max_short < 150, (min_short, max_short)   # shortcut is a plausible period
    # Anchor: N = 0.01 s^-1, lambda_x = 10 lambda_z gives an intrinsic period 2 pi/(0.001 s^-1) = 104.7 min.
    anchor = shortcut({"n": 0.01, "lz": 5.0, "lx": 50.0, "u": 10.0, "theta": 90.0})
    assert abs(anchor - 104.72) < 0.01, anchor
    return (f"zero-crossing check {max_err:.1e}; inside max gap {max_in:.4f}; outside min gap {min_out:.3f}; "
            f"non-hydrostatic / rotating spread {spread:.4f}; key {min_key:.1f}-{max_key:.1f} min, shortcut "
            f"{min_short:.0f}-{max_short:.0f} min; anchor intrinsic period {anchor:.1f} min")


TEMPLATE = Template("gravity_wave_doppler_shift", "atmospheric_dynamics",
                    "intrinsic gravity-wave period -> ground-observed period Doppler-shifted by the wind",
                    "min", base, trigger, text, correct, shortcut)
