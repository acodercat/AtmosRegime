"""Gaussian plume without the ground -> Gaussian plume with a reflecting ground (extended,
2026-10-01).

Parent's method: the Gaussian-plume concentration from an elevated point source as if the plume
spread freely in the vertical, with no ground beneath it,
C(x, z) = Q/(2 pi u sigma_y sigma_z) exp(-(z-H)^2/(2 sigma_z^2)).
Where it fails: the ground is not transparent to the plume -- it reflects material back upward,
equivalent (by the method of images) to adding a mirror-image source at -H:
C(x, z) = Q/(2 pi u sigma_y sigma_z) [exp(-(z-H)^2/(2 sigma_z^2)) + exp(-(z+H)^2/(2 sigma_z^2))].
A receptor at the plume's own centreline height (z = H, an elevated sampler -- see Fair from the
text) makes the direct term a constant exp(0) = 1 regardless of sigma_z, so the miss is driven
entirely by how large the image term exp(-2H^2/sigma_z^2) has grown, i.e. by how far sigma_z has
spread relative to H: the downwind distance. Nothing about that distance looks extreme -- a few
hundred metres to a few kilometres downwind of an ordinary stack -- but once sigma_z exceeds H the
image term is no longer negligible and the parent's formula misses by 20-40 %.
Fair from the text: every arm states Q, the effective stack height H, the wind speed u, and the
sigma_y/sigma_z power laws -- everything both formulas need. The receptor height is not a separate
given: it is stated as "equal to the stack's effective height", reusing H rather than introducing
a new number, so there is nothing for the trap arm to omit.
Inside: sigma_z/H = 0.40-0.65 (close enough to the stack that the image term is small); the
image-free formula misses the true concentration by at most 0.87 %.
Outside: sigma_z/H = 1.20-2.20 (far enough downwind that the ground reflection has become
significant); the parent's formula misses by 20-40 %.
Conventions: the reflecting-ground formula is the standard, single-image closed form for a
non-absorbing, non-depositing ground (no higher-order images are needed, since there is no second
reflecting boundary here); check() verifies it independently from the boundary condition it is
built to satisfy, not a competing formula, since there is no competing convention for this step.
Reference: Turner, D.B. (1970), Workbook of Atmospheric Dispersion Estimates, eqs. 5.1-5.2 (the
single ground-reflection term); Seinfeld & Pandis, Atmospheric Chemistry and Physics, ch. 18
(the exact factor of 2 at the ground, used as the check() anchor below).
Related mechanism: plume_mixing_lid (same image-source family, a second reflecting boundary).

Independent review (2026-10-01): the pollutant is now sulfur dioxide, not particulate matter, so a partial-reflection
(settling) argument is not a defensible alternative.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

A, P, C, Q_EXP = 0.16, 0.9, 0.14, 0.85      # sigma_y = A x^P, sigma_z = C x^Q_EXP (m)


def base(rng):
    return {"q": rnd(rng.uniform(15.0, 40.0), 2), "u": rnd(rng.uniform(4.0, 9.0), 2),
            "H": rnd(rng.uniform(50.0, 90.0), 3)}


def trigger(rng, regime, base):
    ratio = rng.uniform(0.4, 0.65) if regime == "inside" else rng.uniform(1.2, 2.2)
    return {"x": rnd((ratio * base["H"] / C) ** (1 / Q_EXP))}


def _sig(p):
    return A * p["x"] ** P, C * p["x"] ** Q_EXP


def shortcut(p):
    sy, sz = _sig(p)
    return p["q"] * 1e6 / (2 * math.pi * p["u"] * sy * sz)


def correct(p):
    sy, sz = _sig(p)
    image = math.exp(-2 * p["H"] ** 2 / sz ** 2)
    return p["q"] * 1e6 / (2 * math.pi * p["u"] * sy * sz) * (1 + image)


def text(p, variant):
    given = ("Because this height is comparable to the vertical spread of the plume, the ground acts as a "
             "reflecting boundary and an image source at -H must be added: "
             "C = Q/(2π u σy σz) [exp(-(z-H)²/2σz²) + exp(-(z+H)²/2σz²)], evaluated at the receptor height "
             "z = H. " if variant == "given" else "")
    return (f"A stack emits sulfur dioxide at {num(p['q'])} g s^-1 from an effective height of "
            f"{num(p['H'])} m into a mean wind of {num(p['u'])} m s^-1. For the prevailing stability the "
            f"dispersion parameters are σy = 0.16x^0.9 and σz = 0.14x^0.85, with σ and x in metres. An "
            f"instrument mounted on a tower records the plume concentration at a height equal to the stack's "
            f"effective height, {num(p['x'])} m downwind of the stack. {given}What concentration does the "
            f"instrument record? Express your answer in μg m^-3.")


TEMPLATE = Template("plume_ground_reflection", "air_quality",
                    "Gaussian plume with no ground -> ground-reflected plume once sigma_z is comparable to H",
                    "ug/m^3", base, trigger, text, correct, shortcut)


def _general_correct(p, z):
    sy, sz = _sig(p)
    direct = math.exp(-(z - p["H"]) ** 2 / (2 * sz ** 2))
    image = math.exp(-(z + p["H"]) ** 2 / (2 * sz ** 2))
    return p["q"] * 1e6 / (2 * math.pi * p["u"] * sy * sz) * (direct + image)


def _general_shortcut(p, z):
    sy, sz = _sig(p)
    direct = math.exp(-(z - p["H"]) ** 2 / (2 * sz ** 2))
    return p["q"] * 1e6 / (2 * math.pi * p["u"] * sy * sz) * direct


def check() -> str:
    max_in, min_out, max_out = 0.0, 1.0, 0.0
    for q in (15.0, 40.0):
        for u in (4.0, 9.0):
            for H in (50.0, 90.0):
                for ratio in (0.4, 0.5, 0.65):
                    x = (ratio * H / C) ** (1 / Q_EXP)
                    p = {"q": q, "u": u, "H": H, "x": x}
                    max_in = max(max_in, gap(correct(p), shortcut(p)))
                for ratio in (1.2, 1.7, 2.2):
                    x = (ratio * H / C) ** (1 / Q_EXP)
                    p = {"q": q, "u": u, "H": H, "x": x}
                    g = gap(correct(p), shortcut(p))
                    min_out, max_out = min(min_out, g), max(max_out, g)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)

    # Independent: the correct() formula must satisfy the zero-flux (reflecting) ground boundary
    # condition dC/dz = 0 at z = 0 -- a different check from recomputing the gap, since it tests
    # the physics the image term is required to satisfy rather than the formula's own algebra.
    p = {"q": 20.0, "u": 6.0, "H": 70.0, "x": 500.0}
    eps = 1e-3
    flux_correct = abs((_general_correct(p, eps) - _general_correct(p, -eps)) / (2 * eps))
    flux_shortcut = abs((_general_shortcut(p, eps) - _general_shortcut(p, -eps)) / (2 * eps))
    assert flux_correct < 1e-6, flux_correct
    assert flux_shortcut > 1.0, flux_shortcut   # the parent's formula violates the boundary condition

    # Anchor (Turner 1970 / Seinfeld & Pandis): at the ground itself (z=0) the reflecting-ground
    # concentration is EXACTLY twice the image-free value, independent of sigma_z, Q, u, H.
    anchors = []
    for H in (50.0, 90.0):
        for x in (200.0, 2000.0):
            p = {"q": 20.0, "u": 6.0, "H": H, "x": x}
            anchors.append(_general_correct(p, 0.0) / _general_shortcut(p, 0.0))
    assert all(math.isclose(a, 2.0, rel_tol=1e-9) for a in anchors), anchors

    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f} (max {max_out:.3f}), "
            f"ground flux: correct {flux_correct:.1e}, shortcut {flux_shortcut:.3f}; "
            f"ground-level factor-of-2 anchor holds for all {len(anchors)} cases")
