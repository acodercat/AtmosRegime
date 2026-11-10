"""Uniformly filling bay (tidal prism) -> co-oscillating standing wave (extended, oceanography,
2026-10-05).

Parent's method: a bay whose surface rises and falls in step with the tide at its mouth (amplitude
A, frequency omega) stores and releases the tidal prism through the mouth, so continuity gives the
depth-averaged current amplitude at the mouth U = omega L A / H for a bay of length L and uniform
depth H (the width cancels for a bay of uniform cross-section) -- the standard "pumping mode" /
tidal-prism estimate of inlet currents.
Where it fails: the bay surface moves in step only if the bay is short compared with the tidal
wavelength. A frictionless, linear shallow-water tide in a closed, uniform channel is a standing wave
with an antinode at the head: eta(x) = eta_head cos(kx), x from the head, k = omega / sqrt(g H).
Matching eta = A at the mouth and integrating continuity gives U = A sqrt(g/H) tan(kL) = (omega L A/H)
tan(kL)/(kL). The amplitude grows toward the head, so the bay stores more water than the uniform rise
assumes and the mouth current is larger.
Fair from the text: every arm states the bay's length, its uniform depth (and width), that it is
closed at the head, the tidal amplitude at the mouth, the tidal period (12.42 h) and that friction is
neglected -- everything the tidal-prism estimate needs (it uses L, H, A and the period). g is used
only by the replacement; it is a universal constant and is stated only in the given sentence (rule 3
as reworded 2026-09-27; 9.80 vs 9.81 moves the key by < 0.1 %). Neither arm mentions waves,
resonance or wavelength.
Inside: kL = 0.10-0.20 (bay 9-28 km long); tan(kL)/(kL) - 1 <= 1.4 %.
Outside: kL = 0.85-1.05 (bay 73-150 km long, 15-40 m deep -- the scale of real tidal embayments such
as the Bristol Channel or the Gulf of Maine-Bay of Fundy system, though here idealised); the
tidal-prism current is 25-40 % below the key, and both are ordinary tidal currents (tens of cm/s to
~1.7 m/s), so nothing in the shortcut's own number looks absurd. kL stays well below pi/2, where the
frictionless response diverges.
Conventions: the M2 period is stated (12.42 h), so 12 h vs 12.42 h does not arise. The amplitude at
the head A/cos(kL) is a by-product, not the key. Friction is excluded by the text in all arms
(frictional damping changes the answer by more than 5 % in real bays). No published alternative for
the frictionless linear response of a uniform closed channel exists besides this standing wave.
check(): independent numerical route -- integrate eta'' = -k^2 eta from the head by RK4, scale to A
at the mouth, and integrate the rate of change of the stored volume along the bay (Simpson) to get the
mouth flux; plus the literature quarter-wave resonance length.
Reference: Pugh & Woodworth, *Sea-Level Science* (2014), sec. 6.4 (co-oscillating tides in a
uniform channel, A_head = A_mouth / cos(kL)); Knauss, *Introduction to Physical Oceanography*,
ch. 11; Garrett (1972), Nature 238, 441, for the Bay of Fundy quarter-wave resonance.
Related: shallow_water_wave_speed uses the same sqrt(gH) but there the omitted ingredient is finite
depth relative to wavelength; here sqrt(gH) is exact and the omitted ingredient is the basin length
relative to the tidal wavelength (the reflection at the closed head). New cluster "basin length vs
wavelength (co-oscillation)".

Second wrong method (rule 11): the progressive-wave relation U = A sqrt(g/H) (an incoming wave with no
reflection) equals the key where tan(kL) = 1. With the outside range 0.85-1.05 it is >= 11.5 % off
(asserted > 10 % in check(), on the grid and on 2000 rendered draws).
Independent review (2026-10-05): with outside kL = 0.75-1.05 the progressive-wave relation landed within
5 % of the key on 15.5 % of outside draws; the range now starts at kL = 0.85 and check() asserts the
progressive route stays > 10 % off. The bay length is now rendered to 3 significant figures (at 2 the
rounding moved kL by up to 5 % and let rendered draws fall back toward tan kL = 1).
"""
from __future__ import annotations

import math

import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

EXTENDED = True

G = 9.81
PERIOD_H = 12.42
OMEGA = 2 * math.pi / (PERIOD_H * 3600)


def _k(h):
    return OMEGA / math.sqrt(G * h)


def shortcut(p):
    """Tidal-prism (uniform rise) current amplitude at the mouth, cm/s."""
    return OMEGA * p["L"] * 1e3 * p["a"] / p["h"] * 100


def correct(p):
    """Standing-wave current amplitude at the mouth, cm/s."""
    return p["a"] * math.sqrt(G / p["h"]) * math.tan(_k(p["h"]) * p["L"] * 1e3) * 100


def base(rng):
    return {"a": rnd(rng.uniform(0.5, 1.2)), "h": rnd(rng.uniform(15.0, 40.0))}


def trigger(rng, regime, base):
    kl = rng.uniform(0.10, 0.20) if regime == "inside" else rng.uniform(0.85, 1.05)
    return {"L": rnd(kl / _k(base["h"]) / 1e3, 3)}       # km, 3 s.f.: 2 s.f. moved kL by up to 5 %


def text(p, variant):
    given = ("The bay is long enough that its surface does not rise uniformly: the tide forms a standing wave "
             "with elevation amplitude proportional to cos(kx), x measured from the closed head, "
             "k = ω/√(gH) and g = 9.81 m s^-2; the flow through the mouth must supply the changing volume of "
             "the whole bay. " if variant == "given" else "")
    return (f"A narrow bay of uniform width and uniform depth {num(p['h'])} m extends {num(p['L'])} km from its "
            f"mouth to its closed head. At the mouth the sea level rises and falls with the semidiurnal tide "
            f"(period 12.42 h) with an amplitude of {num(p['a'])} m. Neglect friction. {given}What is the "
            f"amplitude of the depth-averaged tidal current through the mouth of the bay? Express your answer "
            f"in cm/s.")


TEMPLATE = Template("bay_tidal_resonance", "oceanography",
                    "uniformly filling bay (tidal prism) -> co-oscillating standing wave in a closed channel",
                    "cm/s", base, trigger, text, correct, shortcut)


def _numerical(p, n=2000):
    """Independent: RK4 for eta'' = -k^2 eta from the head (eta = 1, eta' = 0) to the mouth; scale to A
    at the mouth; mouth flux per unit width = omega * integral of eta over the bay (Simpson)."""
    k = _k(p["h"])
    length = p["L"] * 1e3
    dx = length / n
    eta, deta = 1.0, 0.0
    profile = [eta]
    for _ in range(n):
        def f(y, dy):
            return dy, -k * k * y
        k1 = f(eta, deta)
        k2 = f(eta + dx / 2 * k1[0], deta + dx / 2 * k1[1])
        k3 = f(eta + dx / 2 * k2[0], deta + dx / 2 * k2[1])
        k4 = f(eta + dx * k3[0], deta + dx * k3[1])
        eta += dx / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        deta += dx / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        profile.append(eta)
    scale = p["a"] / profile[-1]
    simpson = profile[0] + profile[-1] + sum((4 if i % 2 else 2) * v for i, v in enumerate(profile[1:-1], 1))
    volume_amp = simpson * dx / 3 * scale              # m^2 per unit width
    return OMEGA * volume_amp / p["h"] * 100


def _progressive(p):
    """Second wrong method: a progressive wave, U = A sqrt(g/H), cm/s."""
    return p["a"] * math.sqrt(G / p["h"]) * 100


def check() -> str:
    max_in, min_out, max_out = 0.0, math.inf, 0.0
    max_err = 0.0
    prog = math.inf
    l_in, l_out, u_rng = [math.inf, 0.0], [math.inf, 0.0], [math.inf, 0.0]
    for a in (0.5, 0.85, 1.2):
        for h in (15.0, 25.0, 40.0):
            for kl, inside in ((0.10, 1), (0.15, 1), (0.20, 1), (0.85, 0), (0.95, 0), (1.05, 0)):
                p = {"a": a, "h": h, "L": kl / _k(h) / 1e3}
                c, s = correct(p), shortcut(p)
                max_err = max(max_err, gap(c, _numerical(p)))
                g = gap(c, s)
                rng_ = l_in if inside else l_out
                rng_[0], rng_[1] = min(rng_[0], p["L"]), max(rng_[1], p["L"])
                u_rng[0], u_rng[1] = min(u_rng[0], c), max(u_rng[1], c)
                if inside:
                    max_in = max(max_in, g)
                else:
                    min_out, max_out = min(min_out, g), max(max_out, g)
                    assert 0 < s < c                  # shortcut positive, an ordinary current
                    prog = min(prog, gap(c, _progressive(p)))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_err < 1e-6, max_err
    rng = random.Random("bay_tidal_resonance:rounded")
    for _ in range(2000):
        _, pout = pair(TEMPLATE, rng)
        prog = min(prog, gap(correct(pout), _progressive(pout)))
    assert prog > 0.10, prog                              # progressive-wave route must fail
    assert u_rng[0] > 2.0 and u_rng[1] < 200.0, u_rng     # cm/s: clear of round(x, 2), not absurd
    # Anchor: quarter-wave resonance length sqrt(gH) T / 4 for H = 70 m, T = 12.42 h is ~291 km -- the
    # Bay of Fundy-Gulf of Maine system's ~300 km scale near M2 resonance (Garrett 1972).
    quarter = math.sqrt(G * 70) * PERIOD_H * 3600 / 4 / 1e3
    assert 280 < quarter < 300, quarter
    return (f"RK4 + Simpson volume flux vs closed form {max_err:.1e}; inside max gap {max_in:.4f}; outside gap "
            f"{min_out:.3f}-{max_out:.3f}; progressive wave >= {prog:.3f} off; L inside {l_in[0]:.0f}-{l_in[1]:.0f} km, outside {l_out[0]:.0f}-"
            f"{l_out[1]:.0f} km; key {u_rng[0]:.1f}-{u_rng[1]:.0f} cm/s; quarter-wave length (H=70 m) "
            f"{quarter:.0f} km")
