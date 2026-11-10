"""Stokes settling speed -> slip-corrected settling speed (source trap: air170_cunningham).

Parent's method: Stokes drag, v = 2 rho g r^2 / (9 mu).
Where it fails: when the particle is not large compared with the mean free path of air
(Knudsen number Kn = lambda / r not small), the continuum drag overestimates resistance; the
speed must be multiplied by the Cunningham factor C = 1 + Kn [1.257 + 0.4 exp(-1.1/Kn)].
Fair from the text: both variants state the mean free path (0.066 um), so the ratio lambda/r is
computable from the givens; at r = 0.05-0.2 um it is 0.3-1.3. Stating lambda in the inside
variant too keeps it from being a cue. Other published coefficient sets (e.g. 1.142/0.558/0.999)
change C by under 3 % in this range, inside the 5 % tolerance.
Inside: r = 7-9 um, Kn < 0.01, C - 1 < 1.2 %; Reynolds number stays below 0.030
(using sea-level air density 1.225 kg/m^3, conservatively).

Review (2026-09-24):
Revised checks 3, 5 and 8: reduce only the inside radius upper bound from 15 to 9 um
to keep the independent inertial-drag correction small (old maximum Re = 0.1383;
new maximum 0.02999). This is a creeping-flow validity repair, not difficulty tuning;
the outside distribution is unchanged. Specify a rigid sphere and mean settling
speed so interfacial circulation and Brownian instantaneous velocity are not
alternative interpretations. Both arms retain viscosity and mean free path.
Balance 4*pi*r^3*rho*g/3 against 6*pi*mu*r*v/C to recover the solver. At r=0.1 um,
rho=1000 kg/m^3, C=1.879483 and v=2.263687e-6 m/s. Across the outside range,
Allen/Raabe (1.142,0.558,0.999) and Kim (1.165,0.483,0.997) coefficient sets
differ from the key by at most 2.54% and 2.16%, respectively, with the stated lambda.
Reference: Kim et al. (2005), NIST JRES, doi:10.6028/jres.110.005.
Stokes settling remains the leading-order inside method; a solver can also use
the corrected law inside, and no behavioral lure is claimed without measurement.

Review (2026-09-27, domain + measurement-design review): keyed in um/s (was m/s). The outside key
was ~1e-6 m/s, which a solver's habitual round(x, 2)/round(x, 4) turns into 0 (the calibration
model rounds in 17 % of its solvers); in um/s outside keys are O(1) and inside O(10^4). Physics
and ranges unchanged; the force-balance anchor is checked in m/s.
"""
from __future__ import annotations

import math
from itertools import product

from .base import OUTSIDE_MIN_GAP, Template, gap, num, rnd

MU, LAM, G = 1.81e-5, 0.066, 9.81          # Pa s, um, m s^-2


def base(rng):
    return {"rho": float(rng.choice(range(1000, 2550, 50)))}


def trigger(rng, regime, base):
    return {"r": rnd(rng.uniform(7, 9)) if regime == "inside" else rnd(rng.uniform(0.05, 0.2))}


def _stokes(p):
    """Stokes settling speed, m/s."""
    return 2 * p["rho"] * G * (p["r"] * 1e-6) ** 2 / (9 * MU)


def _slip(p):
    """Slip-corrected settling speed, m/s."""
    kn = LAM / p["r"]
    return _stokes(p) * (1 + kn * (1.257 + 0.4 * math.exp(-1.1 / kn)))


def shortcut(p):
    return _stokes(p) * 1e6                                        # um/s


def correct(p):
    return _slip(p) * 1e6                                          # um/s


def text(p, variant):
    given = ("Because the particle is not large compared with the mean free path, correct the Stokes "
             "drag for slip: multiply the Stokes settling speed by the Cunningham factor "
             "C = 1 + Kn[1.257 + 0.4 exp(-1.1/Kn)], where Kn = λ/r is the mean free path divided by the "
             "particle radius. " if variant == "given" else "")
    return (f"A rigid spherical particle of radius {num(p['r'])} μm and density {num(p['rho'])} kg m^-3 settles "
            f"at its terminal velocity in still air at sea level. The dynamic viscosity of air is "
            f"1.81 × 10^-5 kg m^-1 s^-1 and the mean free path of air molecules is 0.066 μm. Neglect the "
            f"density of air compared with that of the particle and take g = 9.81 m s^-2. {given}"
            f"Find the mean terminal settling speed of the particle. Express your answer in μm s^-1.")


TEMPLATE = Template("cunningham_slip", "atmospheric_aerosols",
                    "continuum Stokes drag -> slip-corrected drag at Knudsen number of order 1",
                    "um/s", base, trigger, text, correct, shortcut)


def check() -> str:
    """Creeping flow inside, coefficient sets outside, and the force balance at one point."""
    max_re = max_oseen = 0.0
    for rho, radius in product(range(1000, 2550, 50), [7 + i / 10 for i in range(21)]):
        p = {"rho": rho, "r": radius}
        speed = _slip(p)
        max_re = max(max_re, 1.225 * speed * 2 * radius * 1e-6 / MU)   # sea-level air, conservative
        assert gap(speed, _stokes(p)) < .012
        a = 3 / 16 * 1.225 * 2 * radius * 1e-6 / MU                    # leading inertial drag, 1 + 3Re/16
        max_oseen = max(max_oseen, abs(2 * speed / (1 + math.sqrt(1 + 4 * a * speed)) / speed - 1))
    assert max_re < .03 and max_oseen < .006
    alt_max = [0.0, 0.0]                                                # Allen/Raabe and NIST Kim et al.
    for i in range(1501):
        kn = LAM / (.05 + .0001 * i)
        ref = 1 + kn * (1.257 + .4 * math.exp(-1.1 / kn))
        for j, (a, b, c) in enumerate(((1.142, .558, .999), (1.165, .483, .997))):
            alt_max[j] = max(alt_max[j], abs((1 + kn * (a + b * math.exp(-c / kn))) / ref - 1))
        assert 1 - 1 / ref >= OUTSIDE_MIN_GAP                          # every outside size, any density
    assert max(alt_max) < .05
    r = .1e-6
    balance = 4 * math.pi * r ** 3 * 1000 / 3 * G / (6 * math.pi * MU * r) * 1.8794831591491163
    assert math.isclose(_slip({"r": .1, "rho": 1000}), balance, rel_tol=1e-10)
    # Rounding: the smallest outside key (r 0.05 um, rho 1000) survives round(x, 2) within 1 %.
    smallest = correct({"r": .05, "rho": 1000})
    assert 0.5 < smallest and abs(round(smallest, 2) / smallest - 1) < 0.01, smallest
    return (f"max Re {max_re:.4f}, Oseen deviation {max_oseen:.4f}, "
            f"coefficient-set deviations {alt_max[0]:.4f} / {alt_max[1]:.4f}; smallest outside key {smallest:.2f} um/s")
