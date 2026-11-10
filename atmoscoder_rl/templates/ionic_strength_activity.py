"""Concentration-based dissociation constant -> activity-corrected equilibrium in a saline droplet
(2026-10-01 extension).

Parent's method: for a weak acid HA (HA <-> H+ + A-) at total concentration C with dissociation
constant K_a, solve the mass-action law treating concentrations as activities: K_a = x^2/(C - x),
giving x = [H+] = (-K_a + sqrt(K_a^2 + 4 K_a C)) / 2 (the exact quadratic solution; "exact" here
means only that the small-x approximation is not used -- see `weak_acid_dissociation` for that
separate trap -- not that activities are accounted for).
Where it fails: K_a is a thermodynamic (activity-based) constant, K_a = {H+}{A-}/{HA}. In a
droplet that also carries dissolved ions unrelated to this equilibrium (sea-salt or other aerosol
background), the ionic strength is high enough that activity coefficients depart from 1, and
treating concentrations as activities under- or over-states [H+]. The correct treatment solves
K_a = gamma_pm^2 x^2 / (C - x) self-consistently with the ionic strength I = I_bg + x (I_bg the
background, x the acid's own contribution; HA is neutral, so its activity coefficient is taken as
1, the standard approximation for a neutral solute at these ionic strengths). Nothing about the
outside background ionic strength looks extreme -- sea-spray and marine cloud/fog water routinely
carry ionic strengths from a few hundredths to several tenths of a molar, an entirely ordinary
range for atmospheric liquid water, not a contrived edge case.
Fair from the text: every variant states C, K_a and the background ionic strength (the trigger,
the only given that differs between inside and outside); the temperature (25 C, needed to fix the
activity-coefficient parameter a competent solver would use) is stated in all three. No variant
names an activity-coefficient model or gives its parameter; "given" states both.
Inside: I_bg in [1e-6, 1e-4] M. Even including the acid's own (sub-dominant) contribution to I,
check() finds the shortcut's gap stays <=1.3% over the sampled C range -- consistent with the
textbook rule that activity corrections are negligible below I ~ 1e-4 M.
Outside: I_bg in [0.08, 0.16] M (ordinary marine/cloud-water salinity): the shortcut is off by
16-21 %; check() confirms the floor stays above 15 %.
Conventions: the Davies equation (log10 gamma_pm = -A z+z- (sqrt(I)/(1+sqrt(I)) - 0.3 I), A = 0.51
at 25 C) is the standard extension of Debye-Hueckel used up to I ~ 0.5 M (Stumm & Morgan,
*Aquatic Chemistry*; Butler, *Ionic Equilibrium*). check() compares it against the plain
(Guentelberg) form without the -0.3 I term, log10 gamma_pm = -A z+z- sqrt(I)/(1+sqrt(I)); the two
agree within 5 % over the outside grid used here (both monovalent ions, z+z- = 1), which is why
the outside range stops at I_bg = 0.16 M rather than running to the literature's 0.5 M ceiling --
the two conventions diverge faster than the key does at higher I, and only Davies is checked as
the literature anchor.
Anchor: the Davies equation's own textbook check -- gamma_pm(NaCl, I = 0.1 M, 25 C) = 0.781 here,
against the experimental value 0.778 (Robinson & Stokes, *Electrolyte Solutions*), both monovalent.
Related: `weak_acid_dissociation` is the small-x-approximation trap for the same mass-action law;
this template holds the quadratic exact in both arms and turns only on activity vs. concentration.

Independent review (2026-10-01): cloud and fog water have ionic strengths of ~1e-4-1e-2 M, so 0.08-0.16 M was
implausible for a "cloud/fog droplet" (the claim above that cloud water routinely carries tenths of a molar is
wrong); the droplet is now an aqueous haze droplet formed on sea salt, where such values are ordinary. K_a is now
called thermodynamic (a stoichiometric constant for the medium would make the shortcut right). The outside range
is capped at 0.14 M for margin on the Davies/Guentelberg spread (4.8 % at 0.16 M). The Davies A (0.51) is not
stated: it is standard textbook knowledge, and stating it would point to the activity correction.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

KA = 2.0e-5   # M, a generic weak monoprotic organic acid (order of acetic/formic acid)
A_DAVIES = 0.51   # Davies parameter, 25 C


def base(rng):
    return {"c": rnd(rng.uniform(100, 300))}   # total acid concentration, uM


def trigger(rng, regime, base):
    if regime == "inside":
        return {"i": rnd(10 ** rng.uniform(-6, -4), 3)}
    return {"i": rnd(rng.uniform(0.08, 0.14))}


def _gamma_davies(ionic_strength):
    f = math.sqrt(ionic_strength) / (1 + math.sqrt(ionic_strength)) - 0.3 * ionic_strength
    return 10 ** (-A_DAVIES * f)   # z+ = z- = 1


def _exact_x(c_molar, ka):
    """K_a = x^2/(c - x), concentrations treated as activities."""
    return (-ka + math.sqrt(ka * ka + 4 * ka * c_molar)) / 2


def shortcut(p):
    """[H+] ignoring activity coefficients, uM."""
    return _exact_x(p["c"] * 1e-6, KA) * 1e6


def correct(p, gamma_fn=_gamma_davies, iters=150):
    """[H+] with Davies activity coefficients, self-consistent in I = I_bg + x. uM."""
    c_molar = p["c"] * 1e-6
    x = _exact_x(c_molar, KA)
    for _ in range(iters):
        ionic_strength = p["i"] + x
        gamma = gamma_fn(ionic_strength)
        ka_eff = KA / gamma ** 2
        x = _exact_x(c_molar, ka_eff)
    return x * 1e6


def text(p, variant):
    given = ("Because the solution is not dilute, ion activities differ from concentrations: using the Davies "
             "equation, log10(gamma_pm) = -0.51 (sqrt(I)/(1+sqrt(I)) - 0.3 I) for these singly-charged ions at "
             "25 C, with I the total ionic strength (the stated background plus the acid's own dissociated ions), "
             "solve K_a = gamma_pm^2 [H+][A-]/[HA] self-consistently for [H+]. " if variant == "given" else "")
    return (f"A weak monoprotic organic acid HA, with thermodynamic dissociation constant K_a = 2.0 × 10^-5 at 25 °C, is "
            f"dissolved in an aqueous haze droplet at a total concentration of {num(p['c'])} umol L^-1. The droplet "
            f"also carries other dissolved ions, unrelated to this equilibrium (it formed on a sea-salt particle), "
            f"giving it a background ionic strength of {num(p['i'])} mol L^-1. {given}What is the hydrogen-ion "
            f"concentration [H+] in the droplet? Express your answer in umol L^-1.")


def _bisect_x(c_molar, ka, ionic_bg, gamma_fn, iters=200):
    """Independent recomputation: bisect directly on x for F(x) = gamma(I_bg+x)^2 x^2/(c-x) - ka = 0."""
    lo, hi = 0.0, c_molar
    for _ in range(iters):
        mid = (lo + hi) / 2
        gamma = gamma_fn(ionic_bg + mid)
        f = gamma * gamma * mid * mid / (c_molar - mid) - ka
        lo, hi = (mid, hi) if f < 0 else (lo, mid)
    return (lo + hi) / 2


def _gamma_guntelberg(ionic_strength):
    return 10 ** (-A_DAVIES * math.sqrt(ionic_strength) / (1 + math.sqrt(ionic_strength)))


def check() -> str:
    max_in, min_out, max_err, max_spread = 0.0, math.inf, 0.0, 0.0
    for c in (100, 200, 300):
        for i_in in (1e-6, 1e-5, 1e-4):
            p = {"c": c, "i": i_in}
            cx = correct(p)
            max_err = max(max_err, gap(cx, _bisect_x(c * 1e-6, KA, i_in, _gamma_davies) * 1e6))
            max_in = max(max_in, gap(cx, shortcut(p)))
        for i_out in (0.08, 0.11, 0.14):
            p = {"c": c, "i": i_out}
            cx = correct(p)
            max_err = max(max_err, gap(cx, _bisect_x(c * 1e-6, KA, i_out, _gamma_davies) * 1e6))
            min_out = min(min_out, gap(cx, shortcut(p)))
            spread = gap(cx, correct(p, gamma_fn=_gamma_guntelberg))
            max_spread = max(max_spread, spread)
    assert max_err < 1e-6 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    assert max_spread < 0.05, max_spread
    anchor = _gamma_davies(0.1)
    assert abs(anchor - 0.778) / 0.778 < 0.02, anchor   # NaCl, I=0.1 M, 25 C (Robinson & Stokes)
    smallest = correct({"c": 100, "i": 0.08})
    assert smallest > 1.0 and abs(round(smallest, 2) / smallest - 1) < 0.01, smallest
    return (f"fixed-point vs bisection {max_err:.1e}; inside max gap {max_in:.4f}; outside min gap {min_out:.3f}; "
            f"Davies/Guentelberg spread {max_spread:.4f} (<0.05); gamma_pm(I=0.1) {anchor:.3f} "
            f"(anchor 0.778); smallest key {smallest:.2f} umol/L")


TEMPLATE = Template("ionic_strength_activity", "atmospheric_chemistry",
                    "concentration-based dissociation constant -> activity-corrected equilibrium in saline water",
                    "umol/L", base, trigger, text, correct, shortcut)
