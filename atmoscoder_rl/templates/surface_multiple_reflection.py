"""Single-pass transmitted flux -> flux absorbed after multiple reflections between the surface and
an overlying scattering layer (spec: MECHANISMS.md; extended set, 2026-10-01).

Parent's method: of the flux S incident on top of a cloud or haze layer of transmittance T, a
fraction T S reaches the surface, and the surface (albedo alpha) absorbs T S (1 - alpha).
Where it fails: the fraction T S alpha reflected back up by the surface is not simply lost -- the
layer reflects a fraction R of it back down again, where the surface absorbs a further
T S alpha R (1 - alpha), and so on. Summing the resulting geometric series gives the correct
absorbed flux, T S (1 - alpha)/(1 - alpha R); the single-pass estimate omits every bounce after the
first.
Fair from the text: every arm states the incident flux S, the layer's transmittance T and
reflectance R, and the surface albedo alpha -- R is stated in every arm even though the single-pass
estimate does not use it (rule 3). Only the surface albedo changes between the arms.
Inside: alpha = 3-5.5 %, an ordinary dark-ocean value; with R = 26-32 %, alpha R <= 1.8 %, so the
single bounce the surface reflects back is negligible and the single-pass estimate is correct.
Outside: alpha = 62-85 %, ordinary snow or ice -- nothing about the stated albedo looks extreme by
itself. alpha R is then 16-27 %, and the single-pass estimate misses 16-27 % of the absorbed flux
(gap = alpha R exactly, since the shortcut-to-correct ratio is 1 - alpha R).
Conventions: the closed form is the sum of the full geometric series T S (1-alpha) sum_k (alpha R)^k;
truncating that series at 5 reflections instead of summing it exactly changes the key by under 0.2 %
on the outside grid (checked below) -- a solver who reasons about a handful of bounces rather than
the infinite series lands in the same place.
Reference: the geometric series for flux trapped between a reflective surface and an overlying
scattering layer is standard (e.g. Liou, An Introduction to Atmospheric Radiation, 2nd ed., sec. 6.5,
and Petty, A First Course in Atmospheric Radiation, 2nd ed., sec. 6.4, on multiple reflection between
a cloud layer and the surface); this module states it from the series itself rather than citing a
specific equation number.
Answer: absorbed flux in W m^-2, 20-700 across the grid.

Independent review (2026-10-01): T (0.55-0.75) and R (0.26-0.32) were drawn independently, so ~20 % of draws had
T + R > 1 (energy creation). R is now drawn first and T from 0.55 to 0.95 - R; check() asserts T + R <= 0.95 over
20,000 draws. The gap depends on alpha R only, so the gap bounds are unchanged.
"""
from __future__ import annotations

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True


def base(rng):
    # T + R <= 0.95 (the layer also absorbs): R first, then T up to 0.95 - R (review 2026-10-01).
    r = rnd(rng.uniform(0.26, 0.32), 3)
    return {"s": rnd(rng.uniform(200, 1000), 3), "t": rnd(rng.uniform(0.55, 0.95 - r), 3), "r": r}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"a": rnd(rng.uniform(0.03, 0.055), 3)}
    return {"a": rnd(rng.uniform(0.62, 0.85), 3)}


def shortcut(p):
    return p["t"] * p["s"] * (1 - p["a"])


def correct(p):
    return p["t"] * p["s"] * (1 - p["a"]) / (1 - p["a"] * p["r"])


def text(p, variant):
    given = ("Because the flux the surface reflects upward is itself partly reflected back down by "
             "the layer, and this repeats, the surface absorbs the full series of multiple "
             "reflections, not just the first pass: the absorbed flux is "
             "T S (1 - alpha) / (1 - alpha R). " if variant == "given" else "")
    return (f"A solar flux of {num(p['s'])} W m^-2 is incident on top of a cloud layer that "
            f"transmits a fraction T = {num(p['t'])} and reflects a fraction R = {num(p['r'])} of "
            f"the flux incident on it. The surface beneath the layer has an albedo of "
            f"alpha = {num(p['a'])}. {given}What is the flux absorbed by the surface? Express your "
            f"answer in W m^-2.")


TEMPLATE = Template("surface_multiple_reflection", "atmospheric_radiation",
                    "single-pass transmitted flux -> flux absorbed after multiple reflections "
                    "between the surface and the overlying layer",
                    "W/m^2", base, trigger, text, correct, shortcut)


def check() -> str:
    """Independent series-sum recomputation, gap bounds, and the finite-bounce convention."""
    import random as _random
    _rng = _random.Random(0)
    worst = max(b["t"] + b["r"] for b in (base(_rng) for _ in range(20000)))
    assert worst <= 0.95 + 1e-9, worst                     # energy: T + R <= 0.95 on every draw
    max_in, min_out, max_trunc = 0.0, float("inf"), 0.0
    for s in (200.0, 600.0, 1000.0):
        for t in (0.55, 0.6, 0.63):
            for r in (0.26, 0.29, 0.32):
                for a in (0.03, 0.04, 0.055):
                    p = {"s": s, "t": t, "r": r, "a": a}
                    max_in = max(max_in, gap(correct(p), shortcut(p)))
                for a in (0.62, 0.7, 0.85):
                    p = {"s": s, "t": t, "r": r, "a": a}
                    key = correct(p)
                    min_out = min(min_out, gap(key, shortcut(p)))
                    # Independent recomputation: sum the geometric series directly.
                    series = sum((a * r) ** k for k in range(200))
                    indep = t * s * (1 - a) * series
                    assert abs(indep / key - 1) < 1e-9
                    trunc = t * s * (1 - a) * sum((a * r) ** k for k in range(5))
                    max_trunc = max(max_trunc, abs(trunc / key - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP
    assert max_trunc < 0.05
    return f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; 5-term truncation {max_trunc:.4f}"
