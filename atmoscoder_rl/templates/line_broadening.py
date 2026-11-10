"""Pressure-scaled Lorentz half-width -> Voigt (Doppler-dominated) half-width aloft
(spec: MECHANISMS.md).

Parent's method: the half-width of a pressure-broadened line scales in proportion to pressure,
alpha_L = alpha_0 p / p_0.
Where it fails: at low pressure the Lorentz width falls below the Doppler width
alpha_D = (nu_0/c) sqrt(2 ln2 R T / M), which does not depend on pressure; the line is then a
Voigt profile whose half-width is set by Doppler broadening. For the 667 cm^-1 CO2 line the two
widths cross near 8 hPa (about 33 km); at 0.1-0.4 hPa the pressure-scaled width is 19-85 times
too small.
Fair from the text: both variants give the line centre, the reference Lorentz width and its
pressure scaling, the temperature and the identity of the gas (CO2), which fix both widths. The Boltzmann
(gas) constant and the speed of light are not stated: they are universal constants, and stating
c in a pressure-broadening problem would point at the Doppler shift in every arm.
Inside: p = 400-1000 hPa, alpha_D / alpha_L <= 0.021; the Voigt half-width exceeds alpha_L by < 0.1 %.
Outside: p = 0.10-0.40 hPa, alpha_L / alpha_D <= 0.052.

Key: the Olivero & Longbothum (1977, JQSRT 17, 233) Voigt half-width
alpha_V = 0.5346 alpha_L + sqrt(0.2166 alpha_L^2 + alpha_D^2), accurate to 0.02 %.
Conventions: over the outside grid a Doppler-only answer is within 2.8 % of the key, and Whiting's
simpler alpha_L/2 + sqrt(alpha_L^2/4 + alpha_D^2) within 0.2 % -- all pass at 5 %. check() also
computes the half-width of the numerically convolved Voigt profile at sample points.
Anchor: the standard Doppler HWHM shortcut alpha_D = 3.58e-7 nu_0 sqrt(T/M[amu]) (e.g. Liou,
An Introduction to Atmospheric Radiation, 2002, sec. 1.3) agrees with the formula here to < 0.1 %;
Lorentz and Doppler widths of mid-infrared lines are comparable near 30-40 km (Liou; Goody & Yung).

Review (2026-09-26, independent reviewer): (1) The molar mass of CO2 is no longer stated in the
inside/outside texts: it served only the Doppler width, so it cued Doppler broadening in every
arm, inconsistently with leaving c and R unstated. M(CO2) = 44 g/mol is standard knowledge like c
and R; the given arm still states it. (2) Width convention: Liou, Petty and Goody & Yung
define the "Doppler width" as the 1/e half-width (nu_0/c) sqrt(2kT/m), which is sqrt(1/ln2) =
1.20 times the HWHM. A solver who recognises Doppler broadening but uses that width answers
17-19 % above the key (measured in check()) and fails. In screening, outside failures near
1.19 x key must be tagged "Doppler recognised, wrong width convention" and not counted as
regime misses; the given arm supplies the ln2 form, so part of any outside -> given improvement
can be this convention rather than recognition.

Review (2026-09-27, domain + measurement-design review): the answer and the reference Lorentz
half-width are now in MHz (1 cm^-1 = 29979.2458 MHz; the reference is 2100 MHz = 0.0700 cm^-1),
the line centre stays 667 cm^-1. In cm^-1 the outside key was ~5e-4, which a solver's habitual
round(x, 2) or round(x, 4) turns into a wrong answer (and the shortcut, ~2e-5, into 0, so a capture
could not be detected); the calibration model rounds in 17 % of its solvers. In MHz the keys are
O(10) outside and O(10^3) inside. The base temperature is now 245-270 K: 220 K at 1000 hPa was a
polar-winter-only atmosphere. The 1/e-width band and the Doppler-only tolerance are re-checked.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, num, rnd

MHZ_PER_CM = 29979.2458                                     # 1 cm^-1 in MHz
NU0, ALPHA0_MHZ, P0, MOLAR = 667.0, 2100.0, 1013.0, 0.044    # cm^-1, MHz, hPa, kg/mol
ALPHA0 = ALPHA0_MHZ / MHZ_PER_CM                             # the same reference width in cm^-1
R_GAS, C_LIGHT = 8.314, 2.998e8


def lorentz(p):
    """Lorentz HWHM in MHz."""
    return ALPHA0_MHZ * p["p"] / P0


def doppler(p):
    """Doppler HWHM in MHz."""
    return NU0 * MHZ_PER_CM / C_LIGHT * math.sqrt(2 * math.log(2) * R_GAS * p["t"] / MOLAR)


def base(rng):
    return {"t": float(rng.choice(range(245, 275, 5)))}


def trigger(rng, regime, base):
    return {"p": rnd(rng.uniform(400, 1000)) if regime == "inside" else rnd(rng.uniform(0.1, 0.4))}


def correct(p):
    al, ad = lorentz(p), doppler(p)
    return 0.5346 * al + math.sqrt(0.2166 * al * al + ad * ad)


def text(p, variant):
    given = ("Because the pressure is low, Doppler broadening cannot be neglected: combine the Lorentz half-width "
             "α_L with the Doppler half-width α_D = (ν_0/c)√(2 ln2 RT/M), where ν_0 is the line centre as a "
             "frequency (1 cm^-1 = 29979.2458 MHz) and M = 44.0 g mol^-1 is the molar mass of "
             "CO2, using the Voigt approximation "
             "α_V = 0.5346 α_L + √(0.2166 α_L² + α_D²). " if variant == "given" else "")
    return (f"A carbon dioxide absorption line is centred at 667 cm^-1. At a temperature of {num(p['t'])} K its "
            f"pressure-broadened (Lorentz) half-width at half maximum is 2100 MHz at a pressure of 1013 hPa, "
            f"and this width is proportional to pressure. {given}What is "
            f"the half-width at half maximum of the line at a level in the atmosphere where the pressure is "
            f"{num(p['p'])} hPa and the temperature is {num(p['t'])} K? Express your answer in MHz.")


def shortcut(p):
    return lorentz(p)


def _voigt(x, al, ad, n=20000):
    """Voigt profile at offset x by direct convolution, integrating over the narrower component."""
    sig = ad / math.sqrt(2 * math.log(2))

    def gauss(u):
        return math.exp(-u * u / (2 * sig * sig)) / (sig * math.sqrt(2 * math.pi))
    if al <= ad:                     # substitute u = al tan(phi): the Lorentzian becomes uniform in phi
        lo, hi = -math.pi / 2, math.pi / 2
        f = lambda phi: gauss(x - al * math.tan(phi)) / math.pi
    else:
        lo, hi = -12 * sig, 12 * sig
        f = lambda u: gauss(u) * al / math.pi / ((x - u) ** 2 + al * al)
    h = (hi - lo) / n
    ends = range(1, n) if al <= ad else range(n + 1)        # tan(phi) diverges at +-pi/2, where f = 0
    return h / 3 * sum((1 if i in (0, n) else 4 if i % 2 else 2) * f(lo + i * h) for i in ends)


def _numeric_hwhm(al, ad):
    peak = _voigt(0, al, ad)
    lo, hi = 0.0, 4 * max(al, ad)
    for _ in range(40):
        mid = (lo + hi) / 2
        if _voigt(mid, al, ad) > peak / 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def check() -> str:
    max_in = max_dopp = max_whit = 0.
    min_out, e_lo, e_hi = math.inf, math.inf, 0.
    for t in range(245, 275, 5):
        for pr in list(range(400, 1000, 10)) + [1000] + [i / 100 for i in range(10, 41)]:
            p = {"t": t, "p": pr}
            c, al, ad = correct(p), lorentz(p), doppler(p)
            g = abs(shortcut(p) - c) / c
            whit = al / 2 + math.sqrt(al * al / 4 + ad * ad)
            max_whit = max(max_whit, abs(whit / c - 1))
            if pr >= 400:
                max_in = max(max_in, g)
            else:
                min_out = min(min_out, g)
                max_dopp = max(max_dopp, abs(ad / c - 1))
                efold = ad / math.sqrt(math.log(2)) / c              # 1/e Doppler half-width as the answer
                e_lo, e_hi = min(e_lo, efold), max(e_hi, efold)
        # Anchor: alpha_D = 3.58e-7 nu0 sqrt(T/M[amu]) (in the units of nu0).
        assert abs(doppler({"t": t}) / (3.58e-7 * NU0 * MHZ_PER_CM * math.sqrt(t / 44.0)) - 1) < 1e-3
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and max(max_dopp, max_whit) < .05
    # Numerical Voigt half-width at the corners of both ranges.
    max_num = 0.
    for t, pr in ((245, 400), (270, 1000), (245, .1), (270, .4), (255, .25)):
        p = {"t": t, "p": pr}
        max_num = max(max_num, abs(_numeric_hwhm(lorentz(p), doppler(p)) / correct(p) - 1))
    assert max_num < 1e-3
    cross = P0 * doppler({"t": 255}) / ALPHA0_MHZ                          # hPa where alpha_L = alpha_D
    assert 4 < cross < 15
    # Rounding: the smallest key in either arm survives round(x, 2) to within 1 %.
    keys = [correct({"t": t, "p": pr}) for t in (245, 270) for pr in (0.1, 0.4, 400, 1000)]
    assert min(keys) > 10 and all(abs(round(k, 2) / k - 1) < 0.01 for k in keys)
    assert abs(ALPHA0 - 0.0700) < 5e-5                                     # 2100 MHz is the old 0.070 cm^-1
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.2f}; Doppler-only {max_dopp:.4f}, "
            f"Whiting {max_whit:.4f}, numerical Voigt {max_num:.1e}; widths cross at {cross:.1f} hPa; "
            f"1/e Doppler width answers {e_lo:.3f}-{e_hi:.3f} x key (fail: tag in screening)")


TEMPLATE = Template("line_broadening", "atmospheric_radiation",
                    "pressure-scaled Lorentz line width -> Doppler-dominated Voigt width at low pressure",
                    "MHz", base, trigger, text, correct, shortcut)
