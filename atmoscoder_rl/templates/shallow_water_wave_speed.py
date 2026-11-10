"""Shallow-water wave speed sqrt(g H) -> full linear dispersion relation (extended, oceanography,
2026-10-05).

Parent's method: a surface gravity wave over water of depth H travels at c = sqrt(g H), so a wave of
wavelength lambda has period T = lambda / sqrt(g H) -- the long-wave (shallow-water) result taught
for tides, tsunamis and waves approaching a beach.
Where it fails: sqrt(g H) is the kH -> 0 limit of the linear (Airy) dispersion relation
omega^2 = g k tanh(kH), k = 2 pi / lambda, i.e. c = sqrt((g/k) tanh(kH)). It needs the wavelength to be
long compared with the depth (lambda >~ 20 H). When the depth is a sizeable fraction of the wavelength
the orbital motion no longer reaches the bottom, the wave is slower than sqrt(g H), and the period
is longer.
Fair from the text: every arm states the wavelength, the uniform depth, g and that the waves are of
small amplitude. The period asked for needs the wavelength under either method (lambda / c), so no
given is there only for the replacement. Neither arm says "shallow" or "deep".
Inside: kH = 0.12-0.30 (depth about 1/50-1/20 of the wavelength); sqrt(g H) is within 1.5 % of the
dispersion-relation speed, so the long-wave period is within 1.5 % of the key.
Outside: kH = 1.2-1.35 (wavelength 50-150 m over 10-32 m of water -- ordinary swell over a shelf); the
long-wave period is 16-20 % short of the key. It is still an ordinary wave period (a few seconds to
~10 s), so nothing in the shortcut's own number looks absurd.
Conventions: g = 9.81 m s^-2 is stated. Surface tension changes c by < 0.01 % at lambda >= 50 m;
second-order (Stokes) amplitude dispersion does not enter at small amplitude. The only published
alternative for the same physics is the dispersion relation itself in its other forms (Fenton & McKee's
explicit approximation, within 1.7 %; checked).
Second wrong method (rule 11 hazard): the deep-water form c = sqrt(g/k) is off by 1 - sqrt(tanh kH),
which falls below 5 % for kH >~ 1.5. The outside range stops at kH = 1.35 so that the deep-water answer
also fails (>= 6.4 % off on the grid, asserted in check()); a first draft with kH up to 2.6 let it pass on most
of the grid.
Independent review (2026-10-05): the deep-water margin was checked only on the unrounded kH grid;
check() now also draws 2000 rendered (rounded) outside items and asserts the deep-water form stays
> 5 % off there too; at kH up to 1.4 rounded draws came within 5.4 %, so the outside range
now stops at 1.35.
check(): independent round trip -- from the keyed period, solve the dispersion relation for the
wavelength by Newton iteration in L (L = g T^2/(2 pi) tanh(2 pi H/L)) and recover the stated wavelength.
Reference: Knauss, *Introduction to Physical Oceanography* (2nd ed.), ch. 10 (the dispersion relation,
deep- and shallow-water limits, validity at H/lambda < 1/20 and > 1/2); Dean & Dalrymple, *Water Wave
Mechanics for Engineers and Scientists*, ch. 3; Fenton & McKee (1990), Coastal Eng. 14, 499.
Related: none in the set -- rossby_deformation and inertia_gravity_wave are rotating/internal waves
whose omitted term is rotation or a deformation radius; here the omitted ingredient is the finite
depth-to-wavelength ratio in a non-rotating surface wave. New cluster "finite water depth vs wavelength".
"""
from __future__ import annotations

import math

import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

EXTENDED = True

G = 9.81


def _k(p):
    return 2 * math.pi / p["lam"]


def shortcut(p):
    """Long-wave period lambda / sqrt(g H), s."""
    return p["lam"] / math.sqrt(G * p["h"])


def correct(p):
    """Linear dispersion relation, s."""
    k = _k(p)
    return p["lam"] / math.sqrt(G / k * math.tanh(k * p["h"]))


def base(rng):
    return {"lam": rnd(rng.uniform(50.0, 150.0))}


def trigger(rng, regime, base):
    kh = rng.uniform(0.12, 0.30) if regime == "inside" else rng.uniform(1.2, 1.35)
    return {"h": rnd(kh * base["lam"] / (2 * math.pi))}


def text(p, variant):
    given = ("The depth is not small compared with the wavelength, so use the full linear dispersion relation "
             "ω² = gk tanh(kH), with k = 2π/λ, rather than the shallow-water speed √(gH). "
             if variant == "given" else "")
    return (f"A train of small-amplitude surface gravity waves with a crest-to-crest wavelength of {num(p['lam'])} m "
            f"travels over a flat sea floor where the water is {num(p['h'])} m deep. Take g = 9.81 m s^-2. {given}"
            f"What is the period of the waves? Express your answer in s.")


TEMPLATE = Template("shallow_water_wave_speed", "oceanography",
                    "shallow-water speed sqrt(gH) -> full linear dispersion omega^2 = g k tanh(kH)",
                    "s", base, trigger, text, correct, shortcut)


def _wavelength_from_period(period, h):
    """Independent: Newton iteration on f(L) = L - g T^2/(2 pi) tanh(2 pi h / L), started at deep water."""
    l0 = G * period ** 2 / (2 * math.pi)
    lw = l0
    for _ in range(100):
        th = math.tanh(2 * math.pi * h / lw)
        f = lw - l0 * th
        df = 1 + l0 * (1 - th ** 2) * 2 * math.pi * h / lw ** 2
        lw -= f / df
    return lw


def _fenton_mckee(period, h):
    l0 = G * period ** 2 / (2 * math.pi)
    return l0 * math.tanh((2 * math.pi / period * math.sqrt(h / G)) ** 1.5) ** (2 / 3)


def check() -> str:
    max_in, min_out, max_out = 0.0, math.inf, 0.0
    max_err = fm = 0.0
    deep_min, deep_max = math.inf, 0.0
    min_key = math.inf
    for lam in (50.0, 75.0, 100.0, 125.0, 150.0):
        for kh, inside in ((0.12, 1), (0.21, 1), (0.30, 1), (1.2, 0), (1.275, 0), (1.35, 0)):
            p = {"lam": lam, "h": kh * lam / (2 * math.pi)}
            c, s = correct(p), shortcut(p)
            max_err = max(max_err, gap(lam, _wavelength_from_period(c, p["h"])))
            fm = max(fm, gap(lam, _fenton_mckee(c, p["h"])))
            min_key = min(min_key, c)
            g = gap(c, s)
            if inside:
                max_in = max(max_in, g)
            else:
                min_out, max_out = min(min_out, g), max(max_out, g)
                assert 1.5 < s < 20                  # an ordinary wave period, not absurd
                d = gap(c, lam / math.sqrt(G / _k(p)))
                deep_min, deep_max = min(deep_min, d), max(deep_max, d)
    assert deep_min > 0.06, deep_min                # the deep-water form must not pass either
    # The same on draws as rendered (lambda and H rounded to 2 significant figures shift kH).
    rng = random.Random("shallow_water_wave_speed:rounded")
    rounded_deep, rounded_gap = math.inf, math.inf
    for _ in range(2000):
        _, pout = pair(TEMPLATE, rng)
        c = correct(pout)
        rounded_deep = min(rounded_deep, gap(c, pout["lam"] / math.sqrt(G / _k(pout))))
        rounded_gap = min(rounded_gap, gap(c, shortcut(pout)))
    assert rounded_deep > 0.05 and rounded_gap >= OUTSIDE_MIN_GAP, (rounded_deep, rounded_gap)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_err < 1e-10 and fm < 0.02, (max_err, fm)
    assert min_key > 2
    # Anchor: deep-water wavelength of a 10 s wave, L0 = g T^2 / (2 pi) = 156.1 m (the "1.56 T^2" rule).
    l10 = _wavelength_from_period(10.0, 5000.0)
    assert abs(l10 - 156.1) < 0.1, l10
    return (f"Newton round trip {max_err:.1e}; Fenton-McKee {fm:.3f}; inside max gap {max_in:.4f}; outside gap "
            f"{min_out:.3f}-{max_out:.3f}; deep-water form off by {deep_min:.3f}-{deep_max:.3f} "
            f"(rounded draws: >= {rounded_deep:.3f}, shortcut gap >= {rounded_gap:.3f}); "
            f"min key {min_key:.1f} s; L(10 s, deep) = {l10:.1f} m")
