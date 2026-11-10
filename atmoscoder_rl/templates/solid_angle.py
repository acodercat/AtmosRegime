"""Small-angle solid angle pi R^2 / d^2 -> exact cone solid angle of a nearby sphere
(spec: MECHANISMS.md).

Parent's method: a distant sphere of radius R at centre distance d subtends Omega = pi R^2 / d^2
(its cross-section area over distance squared), as used for the Sun seen from Earth or the Earth
seen from geostationary orbit.
Where it fails: close to the sphere the small-angle form breaks down; the exact solid angle of
the visible cap is Omega = 2 pi (1 - cos alpha), sin alpha = R/d. From low Earth orbit the Earth
fills about a third of the sky and the small-angle value is 19-35 % too low.
Fair from the text: both variants give the altitude and the Earth's radius, which fix the
geometry completely; only the altitude changes.
Inside: altitude 30 000-400 000 km (from near geostationary height to lunar distance), where
R/d <= 0.175 and the two forms agree to better than 0.8 %.
Outside: altitude 300-1800 km (low Earth orbit).

Conventions: none -- the cone solid angle is exact geometry. check() recomputes it by direct
numerical integration of sin(theta) dtheta dphi over the cap. The small-angle form treats the Earth
as a flat disc of area pi R^2 at distance d; there is no competing correct form.
Anchor: from geostationary orbit (altitude 35 786 km) the Earth's angular diameter is 17.4 degrees
(e.g. NOAA GOES documentation), and the solid angle 0.0722 sr.

Review (2026-09-27, domain + measurement-design review): keyed in square degrees (deg^2), not sr,
and the inside altitude is 20,000-90,000 km (GEO at 35,786 km is inside the range). In sr the
inside key went down to ~8e-4, which round(x, 2) turns into 0 or a wrong value (the calibration
model rounds in 17 % of its solvers), and the grader cannot parse "sr" at all; deg^2 it converts.
Inside altitudes above 1e5 km also rendered as "2.6 × 10^5 km" against plain outside numbers.
The small-angle gap at 20,000 km is under 1.5 %.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, num, rnd

R = 6371.0
DEG2_PER_SR = (180 / math.pi) ** 2


def base(rng):
    return {}


def trigger(rng, regime, base):
    h = rnd(rng.uniform(2e4, 9e4), 3) if regime == "inside" else rnd(rng.uniform(300, 1800), 3)
    return {"h": h}


def shortcut(p):
    """Small-angle solid angle, deg^2."""
    return math.pi * R ** 2 / (R + p["h"]) ** 2 * DEG2_PER_SR


def correct(p):
    """Exact cone solid angle, deg^2."""
    s = R / (R + p["h"])
    return 2 * math.pi * (1 - math.sqrt(1 - s * s)) * DEG2_PER_SR


def text(p, variant):
    given = ("Because the spacecraft is close to the Earth compared with the Earth's radius, the small-angle "
             "approximation fails: use the exact solid angle of a cone, Ω = 2π(1 − cos α), where "
             "sin α = R/(R + h) is the angular radius of the Earth. " if variant == "given" else "")
    return (f"A spacecraft is at an altitude of {num(p['h'])} km above the Earth's surface. Take the Earth to "
            f"be a sphere of radius 6371 km. {given}What solid angle does the Earth subtend as seen from the "
            f"spacecraft? Express your answer in square degrees (deg^2).")


def _integrated(h, n=2000):
    """Solid angle of the cap by Simpson integration of 2 pi sin(theta) over [0, alpha]."""
    alpha = math.asin(R / (R + h))
    step = alpha / n
    s = sum((1 if i in (0, n) else 4 if i % 2 else 2) * math.sin(i * step) for i in range(n + 1))
    return 2 * math.pi * s * step / 3


def check() -> str:
    max_in = max_indep = 0.
    min_out = math.inf
    grid = [float(f"{x:.2g}") for x in [2e4 * 1.01 ** i for i in range(160)] if x <= 9e4] + \
           [float(f"{x:.2g}") for x in range(300, 1801, 10)]
    keys = []
    for h in grid:
        p = {"h": h}
        c = correct(p)
        keys.append(c)
        max_indep = max(max_indep, abs(_integrated(h) * DEG2_PER_SR / c - 1))
        g = abs(shortcut(p) - c) / c
        if h >= 2e4:
            max_in = max(max_in, g)
        else:
            min_out = min(min_out, g)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and max_indep < 1e-9
    # Rounding and rendering: every key survives round(x, 2) to within 1 %; no altitude uses x 10^n.
    assert min(keys) > 10 and all(abs(round(k, 2) / k - 1) < 0.01 for k in keys)
    assert all("×" not in num(h) for h in grid)
    geo = correct({"h": 35786}) / DEG2_PER_SR
    assert abs(2 * math.degrees(math.asin(R / (R + 35786))) - 17.4) < .05 and abs(geo - .0723) < 3e-4
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; numerical cap integral "
            f"{max_indep:.1e}; GEO anchor 17.4 deg, {geo:.4f} sr = {geo * DEG2_PER_SR:.0f} deg^2")


TEMPLATE = Template("solid_angle", "observation_and_modeling",
                    "small-angle solid angle pi R^2/d^2 -> exact cone solid angle close to the sphere",
                    "deg^2", base, trigger, text, correct, shortcut)
