"""Radar reflectivity of an exponential drop size distribution integrated to infinity -> integrated only up
to the largest drop present (extended, remote sensing / radar meteorology, 2026-10-05).

Parent's method: for N(D) = N0 exp(-Lambda D) the radar reflectivity factor is the sixth moment,
Z = integral N(D) D^6 dD = Gamma(7) N0 / Lambda^7 = 720 N0 / Lambda^7 -- the closed form behind every
Marshall-Palmer Z exercise.
Where it fails: the closed form integrates to D = infinity. The D^6 weight puts the integrand's peak at
D = 6/Lambda, so when the largest drops present are not several times larger than that, a large part of
the analytic moment comes from drops that do not exist. With a largest diameter D_max the moment is
N0 gamma(7, Lambda D_max)/Lambda^7 (lower incomplete gamma), i.e. the closed form times the Poisson
probability P(X >= 7; Lambda D_max). Ulbrich (1985) showed the truncation matters for the high moments
whenever Lambda D_max is below ~10-15.
Fair from the text: every arm states N0, Lambda (both used by the shortcut too) and the diameter of the
largest drops present, D_max, with the exponential fitting all smaller sizes. D_max is the trigger (the
only difference between arms) and ordinary disdrometer output. Nothing says "truncated" or warns about the
tail; the definition of Z as the sixth moment is stated in every arm so units and definition are fixed.
Inside: Lambda = 3.0-4.0 mm^-1 (light to moderate rain, Marshall-Palmer R ~ 1-5 mm/h) with D_max = 5.0-6.0
mm: Lambda D_max >= 15, the missing tail is <= 0.8 % of the moment.
Outside: the same Lambda with D_max = (6.5-9.5)/Lambda = 1.6-3.2 mm (largest drops of 2-3 mm are typical of
light stratiform rain): the closed form is 1.18-2.3 x the key (gap 18-126 %) -- 0.7-3.5 dB, an ordinary
reflectivity, not absurd. N0 = 5000-12000 m^-3 mm^-1 (Marshall-Palmer 8000). Keys ~100-4000 mm^6 m^-3.
Conventions (rule 5): none move the key -- Z's definition, units (N0 in m^-3 mm^-1, D in mm -> mm^6 m^-3)
and the Rayleigh regime are fixed by the text; a coarse midpoint sum in 0.1 mm bins agrees within 4 %.
Second wrong methods (rule 11): number concentration times the sixth power of the median volume diameter
(3.67/Lambda) is >= 3 x the key; number concentration times D_max^6 is far off. Asserted > 10 %.
check(): Simpson integration of N(D) D^6 to D_max against the incomplete-gamma series; Poisson anchor
P(X <= 6; 10) = 0.13014; Marshall-Palmer anchor Z(R = 1 mm/h) = 720 x 8000 / 4.1^7 = 296 mm^6 m^-3 (the
exponential-DSD relation Z = 296 R^1.47).
Grading unit: keyed "mm^6/m^3 (Z)", a label the grader's unit parser rejects on purpose. A parseable
"mm^6/m^3" is 1e-18 m^3 in SI, so a key of ~200 mm^6 m^-3 is 2e-16 -- below the 1e-15 floor in
engine._units_reconcile (abs diff / max(|expected|, 1e-15)). That floor acts as an absolute tolerance of
~50 mm^6 m^-3, so for keys below ~1000 mm^6 m^-3 it can pass a wrong answer, the shortcut included (found
2026-10-05 by check_templates). With the label unparseable only the plain
relative comparison decides; a correct number passes, the shortcut fails. No commensurate unit is converted.
Reference: Marshall & Palmer (1948), J. Meteor. 5, 165; Ulbrich (1985), J. Climate Appl. Meteor. 24, 580
("The effects of drop size distribution truncation on rainfall integral parameters and empirical
relations"); Rinehart, *Radar for Meteorologists* (5th ed.), ch. 5 (reflectivity factor from the DSD);
Rogers & Yau, *A Short Course in Cloud Physics* (3rd ed.), ch. 10. Section numbers not verified.
Cluster: new -- differs from lognormal_mass (nearest: a moment of a size distribution) because there the
parent replaces the distribution by its median size and the missing ingredient is the distribution's
width, whereas here the moment is integrated over the distribution but over sizes that do not exist; the
habit learned on lognormal_mass ("integrate the moment analytically over the whole distribution") produces
exactly this template's shortcut, so it cannot solve this one.
Independent review (2026-10-05): accepted; grader-hazard sentence reworded (absolute ~50 mm^6 m^-3 tolerance).
Cluster ruling: joins lognormal_mass (moment of a size distribution), not a new cluster as argued above.
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

EXTENDED = True


def _tail_fraction(x):
    """P(X <= 6) for Poisson(x) = 1 - gamma(7, x)/Gamma(7): the share of the moment beyond D_max."""
    return math.exp(-x) * sum(x ** k / math.factorial(k) for k in range(7))


def shortcut(p):
    return 720 * p["n0"] / p["lam"] ** 7


def correct(p):
    return shortcut(p) * (1 - _tail_fraction(p["lam"] * p["dmax"]))


def base(rng):
    return {"n0": rnd(rng.uniform(5000.0, 12000.0)), "lam": rnd(rng.uniform(3.0, 4.0))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"dmax": rnd(rng.uniform(5.0, 6.0))}
    return {"dmax": rnd(rng.uniform(6.5, 9.5) / base["lam"])}


def text(p, variant):
    given = ("No drops are larger than the largest ones observed, so the sixth-moment integral runs only from D = 0 "
             f"to {num(p['dmax'])} mm, not to infinity. " if variant == "given" else "")
    return (f"A disdrometer in rain finds that the drop size distribution is exponential, N(D) = N0 exp(−ΛD), with "
            f"N0 = {num(p['n0'])} m^-3 mm^-1 and Λ = {num(p['lam'])} mm^-1, over all sizes up to the largest drops "
            f"present, which are {num(p['dmax'])} mm in diameter. A weather radar observes this rain. {given}What "
            f"is the radar reflectivity factor Z, the sixth moment of the drop size distribution (Z = ∫ N(D) D^6 dD)? "
            f"Express your answer in mm^6 m^-3.")


TEMPLATE = Template("drop_size_truncation", "observation_and_modeling",
                    "sixth moment of an exponential DSD to infinity -> only up to the largest drop present",
                    "mm^6/m^3 (Z)", base, trigger, text, correct, shortcut)


# ---- check --------------------------------------------------------------------------------------------
def _simpson(p, h=0.01):
    n = max(2, int(round(p["dmax"] / h)))
    n += n % 2
    h = p["dmax"] / n
    f = [p["n0"] * math.exp(-p["lam"] * i * h) * (i * h) ** 6 for i in range(n + 1)]
    return h / 3 * (f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2]))


def _binned(p, w=0.1):
    """Disdrometer-style midpoint sum in w-mm bins."""
    n = int(round(p["dmax"] / w))
    return sum(p["n0"] * math.exp(-p["lam"] * (i + 0.5) * w) * ((i + 0.5) * w) ** 6 * w for i in range(n))


def _wrong(p):
    n_t = p["n0"] / p["lam"] * (1 - math.exp(-p["lam"] * p["dmax"]))          # number concentration, m^-3
    return n_t * (3.67 / p["lam"]) ** 6, n_t * p["dmax"] ** 6


def check() -> str:
    max_in, min_out, max_out, max_err, spread = 0.0, math.inf, 0.0, 0.0, 0.0
    min_wrong = math.inf
    keys = [math.inf, 0.0]
    for n0 in (5000.0, 12000.0):
        for lam in (3.0, 3.5, 4.0):
            for d in (5.0, 5.5, 6.0):
                p = {"n0": n0, "lam": lam, "dmax": d}
                max_in = max(max_in, gap(correct(p), shortcut(p)))
                max_err = max(max_err, gap(correct(p), _simpson(p)))
                keys = [min(keys[0], correct(p)), max(keys[1], correct(p))]
            for x in (6.3, 8.0, 9.7):                      # rounding of D_max moves x by <= 0.2
                p = {"n0": n0, "lam": lam, "dmax": x / lam}
                c = correct(p)
                max_err = max(max_err, gap(c, _simpson(p)))
                g = gap(c, shortcut(p))
                min_out, max_out = min(min_out, g), max(max_out, g)
                spread = max(spread, gap(c, _binned(p)))
                min_wrong = min(min_wrong, *(gap(c, v) for v in _wrong(p)))
                keys = [min(keys[0], c), max(keys[1], c)]
    rng = random.Random("drop_size_truncation:rounded")
    for _ in range(2000):
        pin, pout = pair(TEMPLATE, rng)
        c = correct(pout)
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        min_out = min(min_out, gap(c, shortcut(pout)))
        spread = max(spread, gap(c, _binned(pout)))
        min_wrong = min(min_wrong, *(gap(c, v) for v in _wrong(pout)))
        keys = [min(keys[0], c), max(keys[1], c, correct(pin))]
    assert max_err < 1e-6, max_err
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert spread < 0.05, spread                            # a coarse midpoint sum still passes
    assert min_wrong > 0.10, min_wrong
    assert keys[0] > 50, keys                               # mm^6 m^-3: round(x, 2) is irrelevant
    assert abs(_tail_fraction(10.0) - 0.130141) < 1e-6      # Poisson CDF anchor
    z_mp = shortcut({"n0": 8000.0, "lam": 4.1})
    assert 290 < z_mp < 300, z_mp                           # Marshall-Palmer: Z(1 mm/h) ~ 296 mm^6 m^-3
    return (f"Simpson vs incomplete gamma {max_err:.1e}; inside max gap {max_in:.4f}; outside gap {min_out:.3f}-"
            f"{max_out:.2f}; 0.1-mm bins {spread:.4f}; median-volume / D_max routes >= {min_wrong:.1f} off; key "
            f"{keys[0]:.0f}-{keys[1]:.0f} mm^6 m^-3; MP Z(1 mm/h) = {z_mp:.0f}")
