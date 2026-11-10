"""Flat-Earth radar beam height -> 4/3-effective-Earth-radius beam height (2026-09-26).

Parent's method: at short range the beam centre rises as h = r sin(theta) above the radar.
Where it fails: at long range the Earth curves away under the beam; with standard refraction
represented by an effective radius kR (k = 4/3) the height is
h = sqrt(r^2 + (kR)^2 + 2 r kR sin(theta)) - kR (Doviak & Zrnic 1993, eq. 2.28b).
Fair from the text: all arms state the effective-radius model with k = 4/3 and R = 6371 km, the
elevation angle and the slant range, so the geometry is fully specified.
Outside r = 150-250 km at 1.5-2.5 degrees, the flat-Earth height is 17-45 % low.
Inside r = 3-8 km, the curvature term r^2/(2kR) is under 2 % of the height.
Conventions: the answer is defined by the stated 4/3 model; using the true radius (no refraction)
would be a different model, which the text excludes. The small-range expansion
r sin(theta) + r^2/(2kR) is within 5 % of the key outside (`check` prints the spread).
Anchor: the formula reduces exactly to r sin(theta) for R -> infinity and to the law-of-cosines
geometry of a straight ray over a sphere of radius kR (the independent recomputation below).
Related mechanism: spherical_slant_path (neglected curvature of the Earth along a long path).

Review (2026-09-26, independent reviewer): the Cartesian check below is the same law-of-cosines
algebra as the key, not an independent check. check() now also traces the ray numerically over the
TRUE-radius Earth (R = 6371 km) with a constant ray curvature of 1/(4R) toward the ground -- the
refraction that the 4/3 model represents (1/(kR) = 1/R - 1/(4R)) -- stepping the path in 1 km (or
finer) increments, and compares the height at the stated range with the key.
Rule 11 (2026-09-27): answer in m, not km — an inside height of 0.084 km rounded to 0.08 km fails
the 5 % tolerance; the physics and check() stay in km.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

KR = 4 / 3 * 6371.0


def base(rng):
    return {"el": rng.randint(15, 25) / 10}


def trigger(rng, regime, base):
    return {"r": rnd(rng.uniform(3, 8)) if regime == "inside" else rnd(rng.uniform(150, 250))}


def shortcut(p):
    return p["r"] * math.sin(math.radians(p["el"]))


def correct(p):
    r, s = p["r"], math.sin(math.radians(p["el"]))
    return math.sqrt(r ** 2 + KR ** 2 + 2 * r * KR * s) - KR


def text(p, variant):
    given = ("Because the range is long, the curvature of the Earth under the beam matters: the height is "
             "h = sqrt(r^2 + (kR)^2 + 2 r kR sin θ) - kR, where r is the slant range, θ the elevation angle and "
             "kR the effective Earth radius. " if variant == "given" else "")
    return (f"A weather radar at ground level scans at an elevation angle of {num(p['el'])}°. Assume standard "
            f"atmospheric refraction, represented by an effective Earth radius equal to 4/3 of the true radius, "
            f"with the true radius of the Earth 6371 km. {given}How high above the ground is the centre of the "
            f"beam at a slant range of {num(p['r'])} km? Express your answer in m.")


def correct_m(p):
    """The key in metres (rule 11: the inside height is ~0.1 km, which round(x, 2) in km fails)."""
    return 1000 * correct(p)


def shortcut_m(p):
    return 1000 * shortcut(p)


TEMPLATE = Template("radar_beam_height", "observation_and_modeling",
                    "flat-Earth beam height r sin(theta) -> 4/3-effective-radius beam height at long range",
                    "m", base, trigger, text, correct_m, shortcut_m)


def _traced(el, r, steps=2000):
    """Height of a ray with curvature 1/(4R) toward the ground over a true-radius Earth, after path r."""
    big_r = 6371.0
    kappa, ds = 1 / (4 * big_r), r / steps
    x, y, ang = 0.0, big_r, math.radians(el)             # Earth's centre at the origin; ang from local horizontal at x = 0
    for _ in range(steps):
        # direction in fixed axes: rotate by -kappa*ds each step (the ray bends toward the ground)
        x += math.cos(ang) * ds
        y += math.sin(ang) * ds
        ang -= kappa * ds
    return math.hypot(x, y) - big_r


def check() -> str:
    max_in, min_out, max_series, max_trace = 0.0, 1.0, 0.0, 0.0
    for el10 in range(15, 26):
        th = math.radians(el10 / 10)
        for r in [3, 5, 8, 150, 200, 250]:
            p = {"el": el10 / 10, "r": r}
            # Independent: ray point in Cartesian coordinates about the Earth's centre.
            x, y = r * math.cos(th), KR + r * math.sin(th)
            assert math.isclose(correct(p), math.hypot(x, y) - KR, rel_tol=1e-9)
            max_trace = max(max_trace, gap(correct(p), _traced(el10 / 10, r)))
            g_ = gap(correct(p), shortcut(p))
            if r <= 8:
                max_in = max(max_in, g_)
            else:
                min_out = min(min_out, g_)
                max_series = max(max_series, gap(correct(p), shortcut(p) + r ** 2 / (2 * KR)))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and max_series < .05 and max_trace < .01, max_trace
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, r^2/(2kR) expansion spread "
            f"{max_series:.4f}, ray traced over the true-radius Earth {max_trace:.4f}")
