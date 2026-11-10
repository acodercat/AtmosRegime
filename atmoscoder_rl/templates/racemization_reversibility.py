"""Irreversible first-order loss of the L form -> reversible racemization toward D/L = 1 (extended,
chemical kinetics / geochemistry, 2026-10-05).

Parent's method: amino-acid racemization dating treated as an irreversible first-order reaction. The L
enantiomer "decays" into the D form with rate constant k, so L/L_0 = e^{-kt}; with no D at death,
L_0 = L + D and 1 + D/L = e^{kt}, i.e. t = ln(1 + D/L)/k. It is the reflex for any "A converts to B
with first-order rate constant k" statement.
Where it fails: racemization is reversible. The D form converts back to the L form with the same rate
constant (enantiomers react identically; aspartic acid has one chiral centre, so the equilibrium D/L is
1). Then d(L - D)/dt = -2k(L - D), L - D = L_0 e^{-2kt}, and
ln[(1 + D/L)/(1 - D/L)] = 2kt, t = artanh(D/L)/k (Bada & Schroeder 1975). The irreversible form is right
while the D form is too scarce to matter (D/L << 1) and underestimates the age by a third or more
once D/L >~ 0.6.
Fair from the text: every arm states that the shell's aspartic acid was all L at death (D/L = 0), that it
racemizes, the first-order rate constant for conversion of the L to the D form at the burial
temperature, and the measured D/L. Nothing else is needed for either answer: the reverse rate constant
equals the forward one by symmetry, so no constant is recalled at all -- only the fact that the reaction
runs both ways. The word "racemizes" is the ordinary name of the process and is in all arms; the inside
and outside texts never mention the back reaction, the equilibrium ratio or the 2k.
Inside: D/L = 0.010-0.034 (a few-centuries-old shell); the irreversible age is within 1.8 % of the key.
Outside: D/L = 0.64-0.82 -- an ordinary Pleistocene or warm-site Holocene value (k = 2.0-8.0 x 10^-5
yr^-1, ages 9 500-58 000 yr); the shortcut is 35-48 % short, still a positive, plausible-looking age.
Conventions: the dating equation is sometimes written ln[(1 + D/L)/(1 - D/L)] = 2kt + C, with C the
D/L induced in sample preparation; the text fixes D/L = 0 at death, so C = 0. Some authors quote
k_rac = 2k (the rate constant of the approach to equilibrium); the text defines k explicitly as the
L -> D rate constant, so that reading is excluded, and it would miss the key by 50 % (`check()` records
it as a wrong method, not a convention). Real fossil racemization departs from first-order kinetics at
high D/L ("apparent parabolic kinetics"); the text states first-order kinetics, which is the textbook
model. No published convention for the stated model moves the key.
Second wrong methods (rule 11): the linear estimate t = (D/L)/k (zero-order build-up of D) is within
5 % of the key at D/L <~ 0.4 -- which is why the outside range starts at 0.64, where it is >= 15 % off;
t = artanh(D/L)/(2k) (the k_rac reading) and ln[(1 + D/L)/(1 - D/L)]/k (the 1/2 dropped) are 50 % and
100 % off. All checked in `check()`.
check(): integrates dL/dt = -kL + kD, dD/dt = kL - kD by RK4 to the stated D/L (bisection on t), an
independent route to the closed form.
Reference: Bada & Schroeder (1975), Naturwissenschaften 62, 71-79 (reversible first-order racemization
kinetics, ln[(1 + D/L)/(1 - D/L)] = 2kt); Atkins & de Paula, *Physical Chemistry*, ch. on reactions
approaching equilibrium (A <-> B first-order in both directions; section number not verified);
Schwarzenbach et al., *Environmental Organic Chemistry* (reversible first-order reactions).

Cluster: new -- differs from box_model (the nearest: a relaxation toward a steady value) because the
parent is an irreversible first-order decay, not a steady state reported too early, and the missing
ingredient is the back reaction that symmetry fixes (no sink, source term or time scale is stated for
it); it shares no relation with reactant_in_excess (one species, no second reactant) or
aqueous_equilibria (no mass-action constant enters).
Independent review (2026-10-05): accepted without change to the texts. Cluster ruling: joins `box_model`
(reversible first-order kinetics is dC/dt = S - C/tau with the source S dropped by the shortcut); this
overrides the "new" claim above and is recorded here only (coverage_clusters.py is not edited by this module).
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True


def shortcut(p):
    """Irreversible L -> D: t = ln(1 + D/L)/k, years."""
    return math.log(1 + p["dl"]) / p["k"]


def correct(p):
    """Reversible racemization: t = artanh(D/L)/k, years."""
    return math.atanh(p["dl"]) / p["k"]


def base(rng):
    return {"k": rnd(rng.uniform(2.0e-5, 8.0e-5))}          # yr^-1


def trigger(rng, regime, base):
    if regime == "inside":
        return {"dl": rnd(rng.uniform(0.010, 0.034), 3)}
    return {"dl": rnd(rng.uniform(0.64, 0.82), 3)}


def text(p, variant):
    given = ("The D form converts back to the L form with the same rate constant, so D/L tends to 1 and "
             "ln[(1 + D/L)/(1 − D/L)] = 2kt. " if variant == "given" else "")
    return (f"When a mollusc died, all of the aspartic acid in its shell was in the L form (D/L = 0). Since then the "
            f"aspartic acid has been racemizing at the constant temperature of the burial site, where the "
            f"first-order rate constant for conversion of the L form to the D form is {num(p['k'])} yr^-1. "
            f"{given}The D/L ratio of aspartic acid in the shell is now {num(p['dl'])}. How long ago did the "
            f"mollusc die? Express your answer in years.")


TEMPLATE = Template("racemization_reversibility", "environmental_chemistry",
                    "irreversible first-order L -> D -> reversible racemization (D/L -> 1)",
                    "yr", base, trigger, text, correct, shortcut)


def _rk4_age(p, n=4000):
    """Independent: integrate the two-species kinetics, bisect on t for the stated D/L."""
    k, target = p["k"], p["dl"]

    def dl_at(t):
        L, D = 1.0, 0.0
        h = t / n
        f = lambda L, D: (-k * L + k * D, k * L - k * D)
        for _ in range(n):
            a1 = f(L, D)
            a2 = f(L + h * a1[0] / 2, D + h * a1[1] / 2)
            a3 = f(L + h * a2[0] / 2, D + h * a2[1] / 2)
            a4 = f(L + h * a3[0], D + h * a3[1])
            L += h * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]) / 6
            D += h * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]) / 6
        return D / L

    lo, hi = 0.0, 5.0 / k
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dl_at(mid) < target else (lo, mid)
    return (lo + hi) / 2


def check() -> str:
    max_in, min_out, max_out, max_err = 0.0, math.inf, 0.0, 0.0
    min_wrong, min_key, min_short = math.inf, math.inf, math.inf
    for k in (2.0e-5, 5.0e-5, 8.0e-5):
        for dl, inside in ((0.010, 1), (0.034, 1), (0.64, 0), (0.73, 0), (0.82, 0)):
            p = {"k": k, "dl": dl}
            c, s = correct(p), shortcut(p)
            max_err = max(max_err, gap(c, _rk4_age(p)))
            min_key = min(min_key, c)
            g = gap(c, s)
            if inside:
                max_in = max(max_in, g)
                continue
            min_out, max_out = min(min_out, g), max(max_out, g)
            min_short = min(min_short, s)
            # second wrong methods: linear build-up, k_rac = 2k reading, 1/2 dropped
            for w in (dl / k, math.atanh(dl) / (2 * k), math.log((1 + dl) / (1 - dl)) / k):
                min_wrong = min(min_wrong, gap(c, w))
    # rendered (rounded) draws, as pair() would produce them
    rng = random.Random("racemization-check")
    for _ in range(3000):
        b = base(rng)
        pin, pout = {**b, **trigger(rng, "inside", b)}, {**b, **trigger(rng, "outside", b)}
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        min_out = min(min_out, gap(correct(pout), shortcut(pout)))
        min_wrong = min(min_wrong, gap(correct(pout), pout["dl"] / pout["k"]))
    assert max_err < 1e-6, max_err
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert min_wrong > 0.10, min_wrong
    assert min_key > 100 and min_short > 0, (min_key, min_short)    # years: clear of round(x, 2)
    # Anchor: at D/L = 1/3, ln[(1 + D/L)/(1 - D/L)] = ln 2, so 2kt = ln 2 -- the racemization half-life
    # t = ln2/(2k) (Bada's definition of the half-life of the approach to equilibrium).
    k = 4.0e-5
    assert abs(correct({"k": k, "dl": 1 / 3}) - math.log(2) / (2 * k)) < 1e-6
    return (f"RK4 two-species kinetics vs closed form {max_err:.1e}; inside max gap {max_in:.4f}; outside gap "
            f"{min_out:.3f}-{max_out:.3f}; second wrong methods min miss {min_wrong:.3f}; min key {min_key:.0f} yr; "
            f"anchor D/L = 1/3 at t = ln2/(2k)")
