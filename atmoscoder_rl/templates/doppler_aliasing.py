"""Doppler radial velocity reported as the true velocity -> folded (aliased) velocity (2026-09-26).

Parent's method: a pulsed Doppler radar reports the target's radial velocity.
Where it fails: the radar infers velocity from the phase change between successive pulses, which is
known only modulo 2 pi, so it can report only velocities within +/- v_N = lambda PRF / 4 (the Nyquist
velocity); a true velocity outside that interval is folded back by a multiple of 2 v_N.
Fair from the text: all arms give the wavelength, the pulse repetition frequency, the true radial
velocity and the sign convention, so v_N and the folding are fully specified.
Outside the true speed is 1.2-1.8 v_N away from the radar, so the reported value is negative,
between -0.8 v_N and -0.2 v_N: never near zero, never within the tolerance of the true speed.
Inside the true speed is 0.2-0.9 v_N and is reported unchanged (gap exactly zero).
Conventions: whether the interval is closed at +v_N or -v_N does not matter away from its edges.
Anchor: lambda = 10 cm, PRF = 1000 Hz gives v_N = 25 m/s (Doviak & Zrnic 1993, sec. 3.5,
v_a = lambda / (4 T_s)).

Review (2026-09-26, independent reviewer): operational radars routinely dealias, so "what does the
radar report" admitted the true velocity as a processed-product answer. All arms now say the radar
uses a single pulse repetition frequency and reports its estimate without further processing, and
the motion is "along the beam" (no elevation or fall-speed reading). The base is restricted to
lambda*PRF/4 <= 25.7 m/s so the outside true speed (<= 1.75 v_N) stays at or below 45 m/s, a
realistic precipitation speed; the old corner (11 cm, 1200 Hz) gave 58 m/s.

Review (2026-09-27, domain + measurement-design review): the wavelength was drawn continuously from
5.0-11.0 cm, so draws like 7.4 cm named no real radar and announced a synthetic problem. It is now drawn
from operational weather-radar bands: C band 5.3 and 5.5 cm, S band 10.0 and 10.7 cm. The v_N cap, the
outside-speed cap (<= 45 m/s) and |reported| >= 0.19 v_N are re-verified in check().
"""
from __future__ import annotations

import math

from .base import OUTSIDE_MIN_GAP, Template, gap, num, rnd


def nyquist(p):
    return p["lam"] / 100 * p["prf"] / 4


VN_MAX = 45 / 1.75                     # outside true speed <= 1.75 v_N <= 45 m/s
WAVELENGTHS = (5.3, 5.5, 10.0, 10.7)   # cm: operational C- and S-band weather radars


def base(rng):
    while True:
        p = {"lam": rng.choice(WAVELENGTHS), "prf": float(rng.choice(range(600, 1250, 50)))}
        if nyquist(p) <= VN_MAX:
            return p


def trigger(rng, regime, base):
    ratio = rng.uniform(0.2, 0.9) if regime == "inside" else rng.uniform(1.25, 1.75)
    return {"v": rnd(ratio * nyquist(base))}


def shortcut(p):
    return p["v"]


def correct(p):
    vn = nyquist(p)
    return p["v"] - 2 * vn * math.floor((p["v"] + vn) / (2 * vn))


def text(p, variant):
    given = ("Because the radar infers velocity from the phase change between successive pulses, it can only "
             "report velocities between -v_N and +v_N, where v_N = λ·PRF/4 is the Nyquist velocity; a true "
             "velocity outside this interval is folded back into it by adding or subtracting 2v_N. "
             if variant == "given" else "")
    return (f"A pulsed Doppler weather radar operates at a wavelength of {num(p['lam'])} cm with a single pulse "
            f"repetition frequency of {num(p['prf'])} Hz, and it reports its velocity estimate without further "
            f"processing. A region of precipitation is moving directly away from the radar along the beam "
            f"at {num(p['v'])} m s^-1. Radial velocities away from the radar are counted as positive. {given}"
            f"What radial velocity does the radar report for this region? Express your answer in m s^-1.")


TEMPLATE = Template("doppler_aliasing", "observation_and_modeling",
                    "reported radial velocity = true velocity -> Nyquist-folded (aliased) velocity",
                    "m/s", base, trigger, text, correct, shortcut)


def check() -> str:
    max_in, min_out, min_abs, max_speed = 0.0, math.inf, math.inf, 0.0
    for lam in WAVELENGTHS:
        for prf in range(600, 1250, 50):
            p = {"lam": lam, "prf": prf}
            vn = nyquist(p)
            if vn > VN_MAX:
                continue
            for ratio in [0.2, 0.5, 0.9, 1.25, 1.5, 1.75]:
                p["v"] = rnd(ratio * vn)
                # Independent: wrap the pulse-pair phase 4 pi v Ts / lambda into (-pi, pi].
                lam, ts = p["lam"] / 100, 1 / prf
                phase = 4 * math.pi * p["v"] * ts / lam
                wrapped = math.atan2(math.sin(phase), math.cos(phase))
                assert math.isclose(correct(p), wrapped * lam / (4 * math.pi * ts), rel_tol=1e-9, abs_tol=1e-9)
                g_ = gap(correct(p), shortcut(p))
                if ratio < 1:
                    max_in = max(max_in, g_)
                else:
                    min_out = min(min_out, g_)
                    min_abs = min(min_abs, abs(correct(p)) / vn)
                    max_speed = max(max_speed, p["v"])
    assert max_in == 0 and min_out >= OUTSIDE_MIN_GAP and min_abs >= .19
    assert nyquist({"lam": 10, "prf": 1000}) == 25
    assert max_speed <= 45, max_speed
    return f"max inside gap {max_in}, min outside gap {min_out:.2f}, min |reported|/v_N outside {min_abs:.2f}, max outside true speed {max_speed:.1f} m/s"
