"""Mass of median-diameter particles -> mass of a lognormal mode (Hatch-Choate third moment)
(spec: MECHANISMS.md; added 2026-09-27, after the A/B registration).

Parent's method: every particle is treated as having the count median diameter,
m = N rho_p (pi/6) D_g^3.
Where it fails: the mass is the third moment of the number distribution, and for a broad
lognormal mode it is dominated by the large-particle tail: the mean of D^3 is
D_g^3 exp(4.5 ln^2 sigma_g) (Hatch-Choate), a factor 2.1-6.4 at sigma_g = 1.50-1.90.
Fair from the text: every arm states N, D_g, sigma_g and rho_p and says the distribution is
lognormal; the mean volume follows from the lognormal alone, no constant is needed beyond pi. The
Hatch-Choate relation is the replacement and appears only in the given sentence. "Geometric
standard deviation" is unambiguous (a ratio, see Conventions); an earlier gloss "(ln sigma_g is the
standard deviation of ln D)" was dropped in review (2026-09-27) because it drew attention to the
width in every arm. Only sigma_g changes between the arms; nothing in the text says "laboratory" or
"ambient", which would be a cue.
Inside: sigma_g = 1.03-1.06 (a near-monodisperse, e.g. classified calibration aerosol): the
Hatch-Choate factor is at most exp(4.5 ln^2 1.06) = 1.0154, so the median-particle mass is within
1.6 % of the key.
Outside: sigma_g = 1.50-1.90 (ambient accumulation modes): factor 2.09-6.38, gap 52-84 %.
Keys: N = 1000-3000 cm^-3 (5000 at D_g = 0.25 um was severe-haze level), D_g = 0.10-0.25 um, rho_p = 1.2-2.0 g cm^-3 give 0.63-314 ug m^-3
(rule 11: the smallest key survives round(x, 2) within 1 %; all givens render plainly).

Conventions: the geometric standard deviation is a ratio, identical whether the width is
parametrised in ln D or log10 D (sigma_g = e^{SD(ln D)} = 10^{SD(log10 D)}), and identical for
diameter and radius; the key does not depend on either choice. The alternative readings that do
move the key are errors, not conventions, and are recorded in check(): taking the stated sigma_g as
ln sigma_g (factor exp(4.5 sigma_g^2), > 25x), cubing the arithmetic mean diameter
D_g exp(0.5 ln^2 sigma_g) (factor exp(1.5 ln^2 sigma_g), 22-64 % low outside), or using the mass
median diameter D_g exp(3 ln^2 sigma_g) as the representative size (exp(9 ln^2 sigma_g), far high).
Cubing the diameter of average volume D_g exp(1.5 ln^2 sigma_g) is the same key exactly.
Anchor: the Hatch-Choate conversion MMD = CMD exp(3 ln^2 sigma_g) (Hinds 1999, Aerosol Technology,
section 4.5; Seinfeld & Pandis 2016, section 8.1), checked below by taking the median of the
numerically integrated mass distribution.
Related mechanism: none in this set; new cluster "size-distribution moments".
"""
from __future__ import annotations

import math
from itertools import product

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num

INSIDE_SIGMA = (1.03, 1.04, 1.05, 1.06)
OUTSIDE_SIGMA = tuple(round(1.5 + 0.05 * i, 2) for i in range(9))    # 1.50-1.90


def base(rng):
    return {"n": float(rng.choice(range(1000, 3250, 250))),                     # cm^-3
            "d": float(rng.choice([0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25])),  # um
            "rho": float(rng.choice([1.2, 1.4, 1.5, 1.6, 1.7, 1.77, 1.8, 2.0]))}  # g cm^-3


def trigger(rng, regime, base):
    return {"sg": float(rng.choice(INSIDE_SIGMA if regime == "inside" else OUTSIDE_SIGMA))}


def _median_mass(p):
    """Mass concentration if every particle had the count median diameter, ug m^-3."""
    volume = math.pi / 6 * (p["d"] * 1e-6) ** 3                      # m^3 per particle
    return p["n"] * 1e6 * volume * p["rho"] * 1e3 * 1e9               # m^-3 * m^3 * kg m^-3 -> ug m^-3


def shortcut(p):
    return _median_mass(p)


def correct(p):
    return _median_mass(p) * math.exp(4.5 * math.log(p["sg"]) ** 2)


def text(p, variant):
    given = ("Because the distribution is broad, the mass is not that of N particles of the median "
             "diameter: by the Hatch-Choate relation the mean of D^3 over a lognormal distribution is "
             "D_g^3 exp(4.5 ln^2 σ_g). " if variant == "given" else "")
    return (f"An aerosol population of spherical particles follows a lognormal number size distribution "
            f"with a total number concentration of {num(p['n'])} cm^-3, a count median diameter "
            f"D_g = {num(p['d'])} μm and a geometric standard deviation σ_g = {num(p['sg'])}. The particle density is {num(p['rho'])} g cm^-3. {given}"
            f"Find the mass concentration of the aerosol. Express your answer in μg m^-3.")


TEMPLATE = Template("lognormal_mass", "atmospheric_aerosols",
                    "median-particle mass -> third moment of a broad lognormal mode (Hatch-Choate)",
                    "ug/m^3", base, trigger, text, correct, shortcut)


def _moments(p, steps=4000):
    """Numerically integrated mass (ug m^-3) and mass median diameter (um) of the lognormal mode."""
    s, mu = math.log(p["sg"]), math.log(p["d"])
    lo, hi = mu - 12 * s, mu + 12 * s
    h = (hi - lo) / steps
    cum, mass, prev = [], 0.0, None
    for i in range(steps + 1):
        x = lo + i * h
        dn = p["n"] / (math.sqrt(2 * math.pi) * s) * math.exp(-(x - mu) ** 2 / (2 * s * s))   # cm^-3 per ln D
        dm = dn * 1e6 * math.pi / 6 * (math.exp(x) * 1e-6) ** 3 * p["rho"] * 1e12             # ug m^-3 per ln D
        if prev is not None:
            mass += (prev + dm) / 2 * h
        cum.append((x, mass))
        prev = dm
    half = next(x for x, c in cum if c >= mass / 2)
    return mass, math.exp(half)


def check() -> str:
    """Numerical third moment, gap bounds, error readings, the MMD anchor and rounding safety."""
    worst_in, worst_out, max_dev = 0.0, 1.0, 0.0
    for n, d, rho in product((1000.0, 3000.0), (0.10, 0.25), (1.2, 2.0)):
        for sg in INSIDE_SIGMA + OUTSIDE_SIGMA:
            p = {"n": n, "d": d, "rho": rho, "sg": sg}
            mass, mmd = _moments(p)
            max_dev = max(max_dev, abs(mass / correct(p) - 1))
            assert math.isclose(mmd, d * math.exp(3 * math.log(sg) ** 2), rel_tol=0.01), (p, mmd)
            g = gap(correct(p), shortcut(p))
            if sg in INSIDE_SIGMA:
                worst_in = max(worst_in, g)
            else:
                worst_out = min(worst_out, g)
                ln2 = math.log(sg) ** 2
                assert gap(correct(p), _median_mass(p) * math.exp(1.5 * ln2)) >= OUTSIDE_MIN_GAP   # mean-diameter cube
                assert math.exp(4.5 * sg ** 2) > 25                                                 # sigma read as ln sigma
    assert max_dev < 1e-3 and worst_in <= INSIDE_MAX_GAP and worst_out >= OUTSIDE_MIN_GAP
    # Anchor: sigma_g = 2 -> MMD / CMD = exp(3 ln^2 2) = 4.23 (Hatch-Choate), from the integral.
    _, mmd = _moments({"n": 1000.0, "d": 1.0, "rho": 1.0, "sg": 2.0})
    assert abs(mmd - 4.23) < 0.02, mmd
    smallest = correct({"n": 1000.0, "d": 0.10, "rho": 1.2, "sg": 1.03})
    largest = correct({"n": 3000.0, "d": 0.25, "rho": 2.0, "sg": 1.9})
    assert 0.5 < smallest and abs(round(smallest, 2) / smallest - 1) < 0.01, smallest
    return (f"integral vs closed form {max_dev:.1e}; inside gap <= {worst_in:.4f}, outside >= {worst_out:.3f}; "
            f"MMD/CMD at sigma_g 2 = {mmd:.3f}; keys {smallest:.2f}-{largest:.0f} ug/m^3")
