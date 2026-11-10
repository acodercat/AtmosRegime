"""Plane-parallel slant path H sec(theta) -> spherical-shell path near the horizon
(source traps: p_55_regime, 7.6_regime).

Parent's method: a layer of depth H is traversed along H / cos(theta).
Where it fails: as theta approaches 90 degrees, sec(theta) diverges, while the real path through a
spherical shell stays finite: L = sqrt((R + H)^2 - R^2 sin^2 theta) - R cos theta.
Fair from the text: both variants give the Earth's radius and say to neglect refraction, so the
spherical geometry is fully specified; outside, theta = 87.5-89.5 degrees.
Inside: theta = 10-60 degrees; the two paths agree to better than 0.5 %.

Review (2026-09-24):
Revised checks 4 and 11: make the spherical Earth explicit in ALL arms, and add
the actual one-sided shell-path formula to given; previously given only named
the geometry and still required deriving the replacement relation. No sampler
or numerical formula changed. A ray from the surface obeys
R^2 + L^2 + 2*R*L*cos(theta)=(R+H)^2, selecting the positive root.
This is one surface-to-top segment, not the doubled limb chord. At H=10 km,
theta=89 degrees, L=262.82015 km versus H*sec(theta)=572.98688 km.
At theta=90 degrees, H=1 km, the exact 112.88490 km agrees with NASA
horizon geometry (112.88 km in the thin-shell limit):
https://cdaweb.gsfc.nasa.gov/pub/documents/archived_websites/pwg.gsfc.nasa.gov/stargaze/Shorizon.htm
The geometry stays spherical in both arms; only the angle crosses the
plane-parallel approximation range, so this is not a geometry-definition switch.

Trim (2026-09-24): qualifiers the review added in both arms were removed where they do not
move the answer outside the 5 % tolerance (the review's own alternative-convention numbers).
In a trap arm, every extra qualifier is a hint about the physics the solver should notice.
Removed: 'spherical Earth' -- the stated Earth radius already fixes the geometry,
and naming sphericity in the trap arm points at the replacement.
"""
from __future__ import annotations

import math
from itertools import product

from .base import OUTSIDE_MIN_GAP, Template, gap, num

R = 6371.0


def base(rng):
    return {"h": float(rng.randint(5, 15))}


def trigger(rng, regime, base):
    t = rng.randint(10, 60) if regime == "inside" else round(rng.uniform(87.5, 89.5), 1)
    return {"theta": float(t)}


def correct(p):
    th = math.radians(p["theta"])
    return math.sqrt((R + p["h"]) ** 2 - (R * math.sin(th)) ** 2) - R * math.cos(th)


def shortcut(p):
    return p["h"] / math.cos(math.radians(p["theta"]))


def text(p, variant):
    given = ("Because the Sun is close to the horizon, the curvature of the layer cannot be neglected: treat "
             "the layer as a spherical shell between radii R and R + H and use "
             "L = sqrt((R + H)^2 - R^2 sin^2(theta)) - R cos(theta), where R is Earth's radius, "
             "H is the layer height, and theta is the surface zenith angle. " if variant == "given" else "")
    return (f"A layer of absorbing gas of uniform density extends from the Earth's surface up to a height of "
            f"{num(p['h'])} km. At a point on the surface, the Sun is at a zenith angle of {num(p['theta'])}°. "
            f"Take the Earth's radius to be 6371 km and neglect atmospheric refraction. {given}Find the length "
            f"of the path that sunlight reaching this point travels through the layer. Express your answer in km.")


TEMPLATE = Template("spherical_slant_path", "atmospheric_radiation",
                    "plane-parallel sec(theta) path -> spherical-shell path near the horizon",
                    "km", base, trigger, text, correct, shortcut)


def check() -> str:
    """Independent ray-sphere intersection by bisection; NASA's thin-shell horizon value."""
    max_in, min_out = 0.0, math.inf
    for h, theta in product(range(5, 16), list(range(10, 61)) + [87.5 + i / 10 for i in range(21)]):
        lo, hi = 0.0, 2000.0
        for _ in range(60):
            mid = (lo + hi) / 2
            radius = math.hypot(R + mid * math.cos(math.radians(theta)), mid * math.sin(math.radians(theta)))
            lo, hi = (mid, hi) if radius < R + h else (lo, mid)
        p = {"h": h, "theta": theta}
        assert math.isclose(correct(p), (lo + hi) / 2, rel_tol=1e-8)
        g = gap(correct(p), shortcut(p))
        if theta <= 60:
            max_in = max(max_in, g)
        else:
            min_out = min(min_out, g)
    assert max_in < .005 and min_out >= OUTSIDE_MIN_GAP
    assert math.isclose(correct({"h": 1, "theta": 90}), 112.88, rel_tol=5e-5)
    return f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; 1 km horizon 112.88 km"
