"""Normal-incidence reflectance -> angle-dependent Fresnel reflectance of a calm sea (extended, radiation,
2026-10-05).

Parent's method: the reflectance of a water surface from its refractive index, R0 = ((n - 1)/(n + 1))^2
(about 2 % for sea water), so the fraction of direct sunlight entering the sea is 1 - R0. It is the
one reflectance formula everybody remembers and the textbook value quoted for "water reflects ~2 %".
Where it fails: R0 is the normal-incidence limit. For unpolarised light the Fresnel reflectance
R = (1/2)[sin^2(i - t)/sin^2(i + t) + tan^2(i - t)/tan^2(i + t)], sin i = n sin t, stays near R0 out to
an incidence angle of ~40 deg and then rises steeply toward 1 at grazing incidence: with the sun 5-14 deg
above the horizon a calm sea reflects 24-59 % of the direct beam.
Fair from the text: every arm states the refractive index of sea water and the solar elevation, and says
the surface is calm and flat and the light is the direct beam. Nothing else is needed for either answer.
The text never mentions reflectance at normal incidence or angles of incidence; the solar elevation is
ordinary data.
Inside: solar elevation 60-88 deg (incidence 2-30 deg): 1 - R0 is within 0.11 % of the key.
Outside: elevation 5.0-14.0 deg (incidence 76-85 deg): the key is 0.41-0.76 and the shortcut 0.979,
28-136 % high -- a plausible-looking transmission (the shortcut never exceeds 1).
Conventions: n of sea water is quoted as 1.33-1.34 (fresh water 1.333 at 589 nm, sea water ~1.339);
the text states 1.34. Recomputing the key with n = 1.33 instead moves it by <= 0.7 % on the outside grid
(`check()`), so a solver who substitutes the familiar 1.33 still passes. Schlick's approximation
R0 + (1 - R0)(1 - cos i)^5 is not textbook physics in this field and lands up to 14 % off at grazing,
so it is not counted as a convention. The answer is a fraction (grader also accepts the percentage).
Second wrong method (rule 11): using one polarisation only gives T_s = 0.32-0.66 or T_p = 0.51-0.87 on
the outside grid, each >= 14 % from the unpolarised key, so neither passes (`check()`).
Reference: Petty, *A First Course in Atmospheric Radiation* (2nd ed., 2006), ch. 4 (reflection and
refraction at a plane boundary: Fresnel relations and the reflectivity of water versus incidence angle;
section and figure numbers not verified); Mobley, *Light and Water* (1994), ch. 4 (the air-water surface). Related: `surface_multiple_reflection` (a stated albedo, no angular dependence) and
`sun_photometer_airmass` (low sun lengthens the atmospheric path, a different relation).
"""
from __future__ import annotations

import cmath
import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

N_SEA = 1.34


def _fresnel(n, inc_deg):
    """Unpolarised Fresnel reflectance, Snell + the sin/tan forms."""
    i = math.radians(inc_deg)
    if i < 1e-9:
        return ((n - 1) / (n + 1)) ** 2
    t = math.asin(math.sin(i) / n)
    rs = math.sin(i - t) ** 2 / math.sin(i + t) ** 2
    rp = math.tan(i - t) ** 2 / math.tan(i + t) ** 2
    return (rs + rp) / 2


def shortcut(p):
    return 1 - ((N_SEA - 1) / (N_SEA + 1)) ** 2


def correct(p):
    return 1 - _fresnel(N_SEA, 90 - p["elev"])


def base(rng):
    return {}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"elev": rnd(rng.uniform(60.0, 88.0), 3)}
    return {"elev": rnd(rng.uniform(5.0, 14.0), 3)}


def text(p, variant):
    given = ("At low sun the reflectance is far above its value for vertical light: use the Fresnel equations "
             "for unpolarised light, R = ½[sin²(i − t)/sin²(i + t) + tan²(i − t)/tan²(i + t)], where i is the "
             "angle of incidence from the vertical and sin i = n sin t. " if variant == "given" else "")
    return (f"Direct sunlight falls on a calm, flat sea surface while the Sun is {num(p['elev'])}° above the "
            f"horizon. The refractive index of sea water is 1.34. {given}What fraction of the direct-beam solar "
            f"power reaching the sea surface passes into the water? Express your answer as a fraction.")


def _fresnel_amplitudes(n, inc_deg):
    """Independent: the amplitude coefficients in cosine form (r_s = (cos i - n cos t)/(cos i + n cos t),
    r_p = (n cos i - cos t)/(n cos i + cos t)) with cos t from complex sqrt, and T from the transmitted
    amplitudes with the n cos t / cos i flux factor -- energy conservation is checked, not assumed."""
    i = math.radians(inc_deg)
    ci = math.cos(i)
    ct = cmath.sqrt(1 - (math.sin(i) / n) ** 2).real
    rs = (ci - n * ct) / (ci + n * ct)
    rp = (n * ci - ct) / (n * ci + ct)
    ts = 2 * ci / (ci + n * ct)
    tp = 2 * ci / (n * ci + ct)
    flux = n * ct / ci
    t_s, t_p = flux * ts ** 2, flux * tp ** 2
    assert abs(rs ** 2 + t_s - 1) < 1e-12 and abs(rp ** 2 + t_p - 1) < 1e-12
    return (t_s + t_p) / 2, 1 - rs ** 2, 1 - rp ** 2


def check() -> str:
    max_in, min_out, max_err, spread, min_wrong = 0.0, math.inf, 0.0, 0.0, math.inf
    min_key, max_key = math.inf, 0.0
    for e10 in range(600, 881, 5):
        p = {"elev": e10 / 10}
        max_in = max(max_in, gap(correct(p), shortcut(p)))
        max_err = max(max_err, gap(correct(p), _fresnel_amplitudes(N_SEA, 90 - p["elev"])[0]))
    for e10 in range(50, 141, 1):
        p = {"elev": e10 / 10}
        c, s = correct(p), shortcut(p)
        min_out = min(min_out, gap(c, s))
        t_unpol, t_s, t_p = _fresnel_amplitudes(N_SEA, 90 - p["elev"])
        max_err = max(max_err, gap(c, t_unpol))
        spread = max(spread, gap(c, 1 - _fresnel(1.33, 90 - p["elev"])),
                     gap(c, 1 - _fresnel(1.333, 90 - p["elev"])), gap(c, 1 - _fresnel(1.339, 90 - p["elev"])))
        min_wrong = min(min_wrong, gap(c, t_s), gap(c, t_p))
        min_key, max_key = min(min_key, c), max(max_key, c)
    assert max_err < 1e-9 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    assert spread < 0.05, spread                         # rule 5: n = 1.33-1.34
    assert min_wrong > 0.10, min_wrong                   # single-polarisation answers do not pass
    assert 0.3 < min_key and max_key < 0.85 and shortcut({}) < 1, (min_key, max_key)
    # Anchors: R0 = 2.1 % for n = 1.34; Brewster angle atan(n) = 53.3 deg, where r_p = 0.
    assert abs((1 - shortcut({})) - 0.0211) < 2e-4
    assert _fresnel_amplitudes(N_SEA, math.degrees(math.atan(N_SEA)))[2] > 1 - 1e-12
    r80 = _fresnel(1.333, 80.0)
    assert 0.33 < r80 < 0.36, r80                         # tabulated Fresnel value for water: R(80 deg) ~ 0.35
    return (f"amplitude-form check {max_err:.1e}; inside max gap {max_in:.5f}; outside min gap {min_out:.3f}; "
            f"n 1.33-1.339 spread {spread:.4f}; single-polarisation min miss {min_wrong:.2f}; key "
            f"{min_key:.2f}-{max_key:.2f} (shortcut {shortcut({}):.3f}); R(80 deg, n=1.333) = {r80:.3f}")


TEMPLATE = Template("fresnel_reflectance_grazing", "atmospheric_radiation",
                    "normal-incidence reflectance of water -> unpolarised Fresnel reflectance at low sun",
                    "", base, trigger, text, correct, shortcut)
