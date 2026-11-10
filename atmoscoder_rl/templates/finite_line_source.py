"""Infinite line source -> finite line source of length comparable to the crosswind spread.

Parent's method: a ground-level line source perpendicular to the wind, receptor at ground level,
C = sqrt(2/pi) q/(u sigma_z) (Turner's infinite line source).
Where it fails: when the segment is not long compared with sigma_y, the ends contribute nothing
at the receptor; integrating the Gaussian point-source contributions over the segment multiplies
the infinite-line value by erf(L/(2 sqrt(2) sigma_y)) for a receptor opposite the midpoint.
Fair from the text: both variants state the segment length, q, u, and sigma_y, sigma_z at the
receptor, so L/sigma_y is computable.
Inside: L = 2-10 km against sigma_y <= 30 m, erf = 1 to machine precision.
Outside: L = 1-2.5 sigma_y (a 12-75 m segment), erf = 0.38-0.79, so the parent's value is
at least 24 % too high (grid bound, allowing for rounding of L).
Conventions: along-wind diffusion neglected (standard slender-plume approximation, the same in
both formulas); sigma values are given, so no stability-class choice enters.
Reference: Turner (1970), Workbook of Atmospheric Dispersion Estimates, ch. 7 (line sources);
Seinfeld & Pandis, ch. 18.
Related mechanism: plume_mixing_lid (same Gaussian-plume family, different failure).

Wording (2026-09-26, lead review): 'road segment' became 'a straight section of road ... no other
sources nearby', so a 12-75 m outside section reads as a stretch of road rather than an odd road;
the physics and ranges are unchanged.

Review (2026-09-26, independent reviewer): the receptor distance x (50-200 m) was stated but unused,
and sigma_y, sigma_z were sampled independently of it, giving combinations such as x = 50 m with
sigma_y = 30 m, sigma_z = 21 m that no stability class produces at that distance. The distance is
removed from the text and the parameters: the receptor is "downwind of the midpoint", and the
dispersion parameters there are the givens. Keys and gaps are unchanged.

Review (2026-09-27, domain + measurement-design review): an isolated 12-75 m "section of road" with
no other sources is not something a practitioner meets. The source is now a queue of idling vehicles
along a straight stretch of road (a short queue at a toll booth or drive-through outside; a long
standing queue of traffic inside); the physics, the givens and the keys are unchanged.
An expert who notices the short source may instead treat it as a ground-level point source of strength
qL, C = qL/(pi u sigma_y sigma_z). Across the outside range that reading lies 4-27 % above the key
(1.04 at L = sigma_y, 1.27 at L = 2.5 sigma_y; measured in check()), so it fails the tolerance for L above
about 1.1 sigma_y. In screening such answers are tagged "recognised finite source, point-source
approximation": reported, never counted as captures of the infinite-line shortcut.
"""
from __future__ import annotations

import math

from .base import OUTSIDE_MIN_GAP, Template, gap, num, rnd


def base(rng):
    sy = rnd(rng.uniform(12, 30))
    return {"q": rnd(rng.uniform(0.002, 0.02)), "u": rng.choice([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]),
            "sy": sy, "sz": rnd(sy * rng.uniform(0.4, 0.7))}


def trigger(rng, regime, base):
    return {"L": rnd(rng.uniform(2000, 10000)) if regime == "inside" else rnd(rng.uniform(1.0, 2.5) * base["sy"])}


def shortcut(p):
    return math.sqrt(2 / math.pi) * p["q"] * 1e6 / (p["u"] * p["sz"])


def correct(p):
    return shortcut(p) * math.erf(p["L"] / (2 * math.sqrt(2) * p["sy"]))


def text(p, variant):
    given = ("Because the queue is not long compared with the crosswind spread σ_y, the infinite-line-source "
             "formula does not apply: integrating the Gaussian point-source contributions along the queue "
             "multiplies the infinite-line result √(2/π) q/(u σ_z) by erf(L/(2√2 σ_y)), where L is the length "
             "of the queue. " if variant == "given" else "")
    return (f"Vehicles stand in a queue with their engines idling along a straight {num(p['L'])} m stretch of "
            f"road, and there are no other sources nearby. The queue lies perpendicular to a steady wind of "
            f"{num(p['u'])} m s^-1. The idling engines emit carbon monoxide at {num(p['q'])} g m^-1 s^-1, "
            f"uniformly along the queue and at ground level. A receptor at ground level stands downwind of "
            f"the midpoint of the queue, where the dispersion parameters are σ_y = {num(p['sy'])} m and "
            f"σ_z = {num(p['sz'])} m. {given}Estimate the carbon monoxide concentration at the receptor. Express your answer in μg m^-3.")


def _numeric(p):
    """Independent: Simpson sum over the segment of ground-level point sources (with ground reflection)."""
    n, half = 4000, p["L"] / 2
    h = 2 * half / n
    point = lambda y: 2 * math.exp(-y ** 2 / (2 * p["sy"] ** 2)) / (2 * math.pi * p["u"] * p["sy"] * p["sz"])
    f = [point(-half + i * h) for i in range(n + 1)]
    return p["q"] * 1e6 * (f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2])) * h / 3


def point_source(p):
    """The alternative reading: the whole queue as one ground-level point source of strength qL."""
    return p["q"] * 1e6 * p["L"] / (math.pi * p["u"] * p["sy"] * p["sz"])


def check() -> str:
    max_in, min_out = 0.0, math.inf
    point_lo, point_hi = math.inf, 0.0
    for q in [0.002, 0.02]:
        for u in [1.5, 6.0]:
            for sy in [12.0, 20.0, 30.0]:
                for sz in [0.4 * sy, 0.7 * sy]:
                    for L in [2000.0, 10000.0] + [0.95 * sy, 1.7 * sy, 2.6 * sy]:
                        p = {"q": q, "u": u, "sy": sy, "sz": sz, "L": L}
                        key = correct(p)
                        if L < 1000:        # Simpson over a very long segment would need a finer grid
                            assert math.isclose(_numeric(p), key, rel_tol=1e-8), p
                        g = gap(key, shortcut(p))
                        if L >= 2000:
                            max_in = max(max_in, g)
                        else:
                            min_out = min(min_out, g)
                    for ratio in [1.0 + 0.1 * i for i in range(16)]:          # the outside range L/sigma_y
                        r = point_source({**p, "L": ratio * sy}) / correct({**p, "L": ratio * sy})
                        point_lo, point_hi = min(point_lo, r), max(point_hi, r)
    assert max_in <= 1e-12 and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert 1.03 < point_lo < 1.05 and 1.25 < point_hi < 1.28, (point_lo, point_hi)
    # Anchor: infinite limit of the segment integral is Turner's 2q/(sqrt(2 pi) sigma_z u).
    p = {"q": 0.01, "u": 3.0, "sy": 20.0, "sz": 10.0, "L": 400.0}
    assert math.isclose(_numeric(p), 2 * 0.01e6 / (math.sqrt(2 * math.pi) * 10 * 3), rel_tol=1e-6)
    return (f"max inside gap {max_in:.1e}, min outside gap {min_out:.3f}; Simpson segment integral = key; "
            f"Turner infinite-line anchor; point-source reading {point_lo:.3f}-{point_hi:.3f} x key (tagged)")


TEMPLATE = Template("finite_line_source", "air_quality",
                    "infinite line source -> finite segment when its length is comparable to sigma_y",
                    "ug/m^3", base, trigger, text, correct, shortcut)
