"""Photolysis rate from a flat-sensor photon flux -> from the actinic flux (extended, atmospheric chemistry /
photochemistry, 2026-10-05).

Parent's method: the photolysis frequency is J = sigma phi F, cross-section x quantum yield x photon
flux, and the photon flux is what the radiometer reports; the photolytic lifetime is 1/J.
Where it fails: a molecule absorbs photons arriving from every direction alike, so J needs the actinic
flux (radiance integrated over solid angle with no cosine factor), while a flat sensor reports the
irradiance on its plane (radiance weighted by cos of the angle to its normal). For a direct beam at solar
zenith angle theta the upward-facing sensor reads F_beam cos(theta), so the actinic flux is the reading
divided by cos(theta) (Madronich 1987). With the Sun high the two agree; at theta = 65-76 deg the actinic
flux is 2.4-4.1 times the reading and the lifetime 2.4-4.1 times shorter than the shortcut.
Fair from the text: every arm states sigma, phi, the measured flat-sensor flux, that the sensor faces
straight up, the solar zenith angle, and that at this wavelength and altitude essentially all the light
arrives in the direct solar beam (light near 205 nm at 35 km, in the O2-Schumann-Runge / O3-Hartley window that
reaches the middle stratosphere: Rayleigh optical depth above ~0.01, and light scattered or reflected from
below is absorbed by O2 and O3 beneath) -- the qualifier fixes
the angular distribution, which the key needs, and is the same in all arms. The zenith angle is the only
difference between arms and is ordinary observing metadata. Nothing names actinic flux or the cosine.
Inside: theta = 2.0-9.9 deg, cos >= 0.985: shortcut within 1.5 %.
Outside: theta = 65.0-76.0 deg: key 0.24-0.42 x the shortcut lifetime (gap 137-313 % of the key); the shortcut is the
ordinary high-Sun lifetime, not absurd.
Ranges: sigma 1.0-4.0 x 10^-19 cm^2, phi 0.50-1.0, flux 1.0-3.0 x 10^14 photons cm^-2 s^-1 (J = 5e-6 to
1.2e-4 s^-1, typical of stratospheric photolysis in a narrow UV band); keys 0.56-56 h.
Conventions (rule 5): replacing 1/cos by an air-mass formula (Kasten & Young 1989) moves the key <= 1.6 %
at theta <= 76 deg; a finite solar disk (0.53 deg) instead of a point beam changes nothing measurable.
Second wrong methods (rule 11): the diffuse-light rule "actinic flux = 2 x irradiance" (exact for isotropic
radiance) is >= 15 % off because the outside range starts at 65 deg (1/cos 65 = 2.37) and excludes the
56-63 deg window where 1/cos is near 2; multiplying the reading by cos(theta) again (treating it as the
beam flux and projecting it) is far off. Both asserted > 10 % in check().
check(): the actinic flux and the sensor reading are recomputed by numerically integrating a uniform
solar-disk radiance over solid angle (with and without the cosine weighting) -- a different route from
the closed form; the anchor is Madronich's ratio actinic/irradiance = 2 for isotropic radiance over a
hemisphere, recovered by the same integrator.
Reference: Madronich (1987), J. Geophys. Res. 92, 9740 (actinic flux vs irradiance); Seinfeld & Pandis,
*Atmospheric Chemistry and Physics* (3rd ed., 2016), ch. 4 (actinic flux, photolysis rate coefficient);
Finlayson-Pitts & Pitts, *Chemistry of the Upper and Lower Atmosphere* (2000), ch. 3 (actinic flux and
its distinction from irradiance). Section numbers not verified.
Cluster: new -- differs from sun_photometer_airmass / spherical_slant_path (nearest: low Sun and
1/cos(theta)) because there sec(theta) is the parent's path length and the correction is Earth curvature
near the horizon, whereas here the parent has no angle at all and the missing ingredient is the
definition of the actinic flux (radiance over solid angle without the cosine a flat sensor applies); the
outside range stops at 76 deg, where the curvature habit learned there changes nothing.
Independent review (2026-10-05): the band was "near 250 nm" at 35 km -- Hartley-band ozone above 35 km leaves
essentially no direct beam there, contradicting the stated flux; moved to the 205 nm window, docstring
Rayleigh optical depth corrected (~0.01, not 0.005). No key or range changed. Cluster ruling: joins
diffuse_transmission (cos-theta weighting between radiance and flux), not a new cluster as argued above.
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

EXTENDED = True


def _j_reading(p):
    """sigma * phi * (flat-sensor reading), s^-1."""
    return p["sigma"] * 1e-19 * p["phi"] * p["e"] * 1e14


def shortcut(p):
    return 1 / _j_reading(p) / 3600                                  # h


def correct(p):
    return math.cos(math.radians(p["sza"])) / _j_reading(p) / 3600    # h


def base(rng):
    return {"sigma": rnd(rng.uniform(1.0, 4.0)), "phi": rnd(rng.uniform(0.5, 1.0)),
            "e": rnd(rng.uniform(1.0, 3.0))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"sza": rnd(rng.uniform(2.0, 9.9), 2)}
    return {"sza": rnd(rng.uniform(65.0, 76.0), 3)}


def text(p, variant):
    given = ("A molecule absorbs photons arriving from any direction equally, so photolysis depends on the actinic "
             "flux, which for a direct beam is the reading of the flat sensor divided by the cosine of the solar "
             "zenith angle. " if variant == "given" else "")
    return (f"At an altitude of 35 km, a trace gas absorbs sunlight in a narrow band near 205 nm, where its "
            f"absorption cross-section is {num(p['sigma'])} × 10^-19 cm^2 and its photodissociation quantum yield "
            f"is {num(p['phi'])}. A flat radiometer on a balloon gondola, with its sensor facing straight up, "
            f"measures a photon flux of {num(p['e'])} × 10^14 photons cm^-2 s^-1 in this band while the solar "
            f"zenith angle is {num(p['sza'])}°. At this wavelength and altitude essentially all of the light "
            f"arrives in the direct solar beam. {given}What is the e-folding lifetime of the gas against "
            f"photolysis? Express your answer in hours.")


TEMPLATE = Template("actinic_flux_horizontal", "atmospheric_chemistry",
                    "photolysis rate from a flat-sensor photon flux -> from the actinic flux (direct beam / cos SZA)",
                    "h", base, trigger, text, correct, shortcut)


# ---- check --------------------------------------------------------------------------------------------
def _integrate(sza_deg, radiance, n=400):
    """Integrate a radiance field L(theta, phi) over the upper hemisphere by the midpoint rule in
    (cos theta, phi): returns (actinic flux, upward-facing flat-sensor irradiance)."""
    act = irr = 0.0
    dmu, dphi = 1.0 / n, 2 * math.pi / n
    for i in range(n):
        mu = (i + 0.5) * dmu
        th = math.acos(mu)
        for j in range(n):
            ph = (j + 0.5) * dphi - math.pi
            ell = radiance(th, ph, sza_deg)
            if ell:
                act += ell * dmu * dphi
                irr += ell * mu * dmu * dphi
    return act, irr


def _disk_ratio(sza_deg):
    """Actinic / irradiance for a solar disk, by direct integration on a fine local grid around the Sun."""
    s, half, n = math.radians(sza_deg), math.radians(0.6), 200
    act = irr = 0.0
    for i in range(n):                       # polar coordinates (r, a) on the sky around the disk centre
        r = (i + 0.5) / n * half
        for j in range(n):
            a = (j + 0.5) / n * 2 * math.pi
            mu = math.cos(s) * math.cos(r) + math.sin(s) * math.sin(r) * math.cos(a)   # cos of zenith angle
            dw = math.sin(r) * (half / n) * (2 * math.pi / n)
            act += dw
            irr += mu * dw
    return act / irr


def _kasten_young(sza_deg):
    return 1 / (math.cos(math.radians(sza_deg)) + 0.50572 * (96.07995 - sza_deg) ** -1.6364)


def check() -> str:
    max_in, min_out, max_err, spread = 0.0, math.inf, 0.0, 0.0
    min_wrong = math.inf
    keys = [math.inf, 0.0]
    corners = [{"sigma": s, "phi": f, "e": e} for s in (1.0, 4.0) for f in (0.5, 1.0) for e in (1.0, 3.0)]
    for b in corners:
        for sza in (2.0, 6.0, 9.9):
            p = {**b, "sza": sza}
            max_in = max(max_in, gap(correct(p), shortcut(p)))
            keys = [min(keys[0], correct(p)), max(keys[1], correct(p))]
        for sza in (65.0, 70.0, 76.0):
            p = {**b, "sza": sza}
            c = correct(p)
            keys = [min(keys[0], c), max(keys[1], c)]
            min_out = min(min_out, gap(c, shortcut(p)))
    for sza in (5.0, 65.0, 70.0, 76.0):
        p = {"sigma": 2.0, "phi": 1.0, "e": 2.0, "sza": sza}
        independent = shortcut(p) / _disk_ratio(sza)       # lifetime from integrated actinic / reading
        max_err = max(max_err, gap(correct(p), independent))
        if sza >= 65:
            spread = max(spread, gap(correct(p), shortcut(p) / _kasten_young(sza)))
    rng = random.Random("actinic_flux_horizontal:rounded")
    for _ in range(2000):
        pin, pout = pair(TEMPLATE, rng)
        c = correct(pout)
        min_out = min(min_out, gap(c, shortcut(pout)))
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        spread = max(spread, gap(c, shortcut(pout) / _kasten_young(pout["sza"])))
        cos = math.cos(math.radians(pout["sza"]))
        min_wrong = min(min_wrong, gap(c, shortcut(pout) / 2),          # isotropic "2 x irradiance" rule
                        gap(c, shortcut(pout) / cos))                   # reading projected again
        keys = [min(keys[0], c), max(keys[1], c, correct(pin))]
    assert max_err < 2e-3, max_err
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert spread < 0.05, spread
    assert min_wrong > 0.10, min_wrong
    assert keys[0] > 0.5, keys                          # hours: round(x, 2) moves it < 1 %
    act, irr = _integrate(0.0, lambda th, ph, s: 1.0, n=200)   # isotropic hemisphere: 2 pi vs pi
    assert abs(act / irr - 2) < 1e-3, act / irr
    return (f"solar-disk integration vs 1/cos {max_err:.1e}; inside max gap {max_in:.4f}; outside min gap "
            f"{min_out:.3f}; Kasten-Young spread {spread:.4f}; '2x' or double-cosine route >= {min_wrong:.2f} off; "
            f"key {keys[0]:.2f}-{keys[1]:.1f} h; isotropic actinic/irradiance = {act / irr:.4f} (Madronich: 2)")
