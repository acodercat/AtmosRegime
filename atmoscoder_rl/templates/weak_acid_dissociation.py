"""Small-dissociation approximation [H+] = sqrt(K_a C) -> exact weak-acid equilibrium
(2026-09-26 expansion).

Parent's method: for a weak acid at formal concentration C, the dissociated amount x is small
compared with C, so K_a = x^2/(C - x) ~ x^2/C and [H+] = sqrt(K_a C).
Where it fails: when C is comparable with K_a (a dilute solution, e.g. cloud or rain water), most
of the acid dissociates and x must come from the quadratic x^2 + K_a x - K_a C = 0.
Fair from the text: both variants give K_a and C; outside, C = 1.5-4e-5 M against K_a = 1.75e-5 M.
Inside: C = 0.02-0.05 M; the relative error of sqrt(K_a C) is about (1/2) sqrt(K_a/C) <= 1.5 %.
Outside: the shortcut overestimates [H+] by 38-86 %. Water's own dissociation is negligible in
both arms ([OH-] < 2e-9 M), which check() confirms by solving the full charge balance
[H+] = K_a C/(K_a + [H+]) + K_w/[H+] by bisection.
Conventions: activity corrections are not expected at these concentrations (and are not in any
textbook treatment of this exercise); they would change the inside answer by a few per cent in
both key and shortcut alike.
Anchor: K_a(acetic acid, 25 degC) = 1.75e-5 (pK_a 4.756, CRC Handbook); a 0.1 M solution has
pH 2.88, the standard textbook value.

Review (2026-09-26, independent reviewer): two alternative readings were at the range edges.
(1) "Pure water" read as water in equilibrium with atmospheric CO2 (420 ppm) raises [H+] by 9 %
at C = 1e-5 M; the outside floor is now 1.5e-5 M, where it stays under 5 %. (2) K_a read as a
thermodynamic (activity) constant, with Debye-Hueckel limiting-law activity coefficients, moved
the inside answer by up to 5.1 % at C = 0.1 M; the inside cap is now 0.05 M. check() measures
both readings over the new ranges.

Review (2026-09-27, domain + measurement-design review): the concentration is now stated in
mmol/L (inside 20-50, outside 0.015-0.040) and [H+] is keyed in umol/L. In mol/L the outside
key was ~1e-5 and the outside concentration rendered as "3.8 × 10^-5" while the inside rendered
plain: a solver's round(x, 2)/round(x, 4) turned every outside key into 0 (the calibration model
rounds in 17 % of its solvers) and the notation switch marked the trap. Physics, ranges in mol/L
and the alternative-reading checks are unchanged; they run in mol/L and convert.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

KA, KW = 1.75e-5, 1.0e-14


def base(rng):
    return {}


def trigger(rng, regime, base):
    """The formal concentration in mmol/L."""
    return {"c": rnd(rng.uniform(20, 50) if regime == "inside" else rng.uniform(0.015, 0.040), 3)}


def _molar(p):
    return p["c"] * 1e-3


def shortcut(p):
    """[H+] in umol/L."""
    return math.sqrt(KA * _molar(p)) * 1e6


def _exact(c):
    """[H+] in mol/L for a formal concentration c in mol/L."""
    return (-KA + math.sqrt(KA * KA + 4 * KA * c)) / 2


def correct(p):
    """[H+] in umol/L."""
    return _exact(_molar(p)) * 1e6


def text(p, variant):
    given = ("Because the solution is dilute, a large fraction of the acid dissociates, so [H^+] is not small "
             "compared with the total concentration C: solve K_a = [H^+]^2/(C − [H^+]) for [H^+] without "
             "approximation. " if variant == "given" else "")
    return (f"Acetic acid, whose acid dissociation constant is K_a = 1.75 × 10^-5 at 25 °C, is dissolved in pure "
            f"water at 25 °C to a total concentration of {num(p['c'])} mmol L^-1. {given}What is the hydrogen-ion "
            f"concentration of the solution? Express your answer in μmol L^-1.")


def _charge_balance(c):
    lo, hi = 1e-12, 1.0
    for _ in range(300):
        h = math.sqrt(lo * hi)
        excess = h - KA * c / (KA + h) - KW / h
        lo, hi = (lo, h) if excess > 0 else (h, hi)
    return math.sqrt(lo * hi)


K1_CO2, KH_CO2, P_CO2 = 4.45e-7, 3.4e-2, 420e-6        # M, M atm^-1, atm (25 degC)


def _with_co2(c):
    """[H+] if the water were in equilibrium with 420 ppm CO2 (an alternative reading of 'pure water')."""
    lo, hi = 1e-12, 1.0
    for _ in range(300):
        h = math.sqrt(lo * hi)
        excess = h - KA * c / (KA + h) - K1_CO2 * KH_CO2 * P_CO2 / h - KW / h
        lo, hi = (lo, h) if excess > 0 else (h, hi)
    return math.sqrt(lo * hi)


def _activity(c, exact=True):
    """[H+] if K_a is read as an activity constant (Debye-Hueckel limiting law, I = [H+])."""
    h = _exact(c)
    for _ in range(100):
        gamma = 10 ** (-0.509 * math.sqrt(h))
        kc = KA / gamma ** 2
        h = (-kc + math.sqrt(kc * kc + 4 * kc * c)) / 2 if exact else math.sqrt(kc * c)
    return h


def check() -> str:
    max_in, min_out, max_err, max_co2, max_act = 0.0, math.inf, 0.0, 0.0, 0.0
    inside = [0.02 + 0.005 * i for i in range(7)]
    outside = [1.5e-5 + 2.5e-6 * i for i in range(11)]
    keys = []
    for c in inside + outside:                                   # mol/L here; the template takes mmol/L
        p = {"c": c * 1e3}
        exact = correct(p) * 1e-6
        keys.append(correct(p))
        max_err = max(max_err, gap(_charge_balance(c), exact))
        g = gap(correct(p), shortcut(p))
        if c >= 0.02:
            max_in = max(max_in, g)
            max_act = max(max_act, gap(exact, _activity(c)), gap(exact, _activity(c, exact=False)))
        else:
            min_out = min(min_out, g)
            max_co2 = max(max_co2, gap(exact, _with_co2(c)))
    assert max_err < 1e-3 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    assert max_co2 < 0.05 and max_act < 0.05, (max_co2, max_act)
    # Rounding: every key survives round(x, 2) to within 1 %; both arms' concentrations render plain.
    assert min(keys) > 5 and all(abs(round(k, 2) / k - 1) < 0.01 for k in keys)
    assert all("×" not in num(c * 1e3) for c in inside + outside)
    ph = -math.log10(_exact(0.1))
    assert abs(ph - 2.88) < 0.01, ph
    return (f"full charge balance (with K_w) vs quadratic {max_err:.1e}; inside max gap {max_in:.4f}; "
            f"outside min gap {min_out:.3f}; CO2-equilibrated reading {max_co2:.3f}; activity reading "
            f"{max_act:.3f}; pH of 0.1 M = {ph:.3f}")


TEMPLATE = Template("weak_acid_dissociation", "atmospheric_chemistry",
                    "small-dissociation approximation sqrt(K_a C) -> exact weak-acid equilibrium in dilute solution",
                    "umol/L", base, trigger, text, correct, shortcut)
