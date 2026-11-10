"""Physical Henry's law -> effective Henry's law for a dissociating acid (2026-09-26 expansion).

Parent's method: the dissolved concentration of a gas is H p.
Where it fails: formic acid dissociates in solution (HCOOH <=> H+ + HCOO-). When the pH is not
well below pK_a = 3.74, formate carries much of the dissolved acid, and the total dissolved
concentration is H p (1 + K_a/[H+]) (the effective Henry's law constant H* = H(1 + K_a/[H+])).
Fair from the text: every variant states H, the gas-phase mixing ratio at 1 atm, the pH, and
that the gas phase is replenished; K_a is a handbook constant stated only in the given sentence
(2026-09-27 note) (an open system, so depletion of the gas does not enter). The
question asks for formic acid "in all its forms"; without that phrase the undissociated
concentration alone would be a defensible reading, so it is kept in all arms although it points
at dissociation.
Inside: pH 1.7-2.0 (acid fog), K_a/[H+] <= 0.018, so formate is under 2 % of the total.
Outside: pH 3.8-4.6, typical cloud water; K_a/[H+] = 1.1-7.2, the shortcut misses 53-88 %.
Conventions: activity coefficients are ignored in every textbook treatment at these ionic
strengths; no other convention moves the answer. check() solves the speciation numerically (see
the review note below).
Anchor: K_a(HCOOH, 298 K) = 1.8e-4 M (pK_a 3.75, the standard handbook value); H = 3.6e3
M atm^-1 lies within the compiled range (Sander 2015). Values are given in the text, so the key
does not depend on which compilation a solver remembers.

Review (2026-09-26, independent reviewer): (1) check() used to "recompute" through
alpha_0 = [H+]/([H+] + K_a), which is the key rearranged; it now solves the speciation by nested
bisection -- for a trial total, bisect for [HCOOH] from the mass balance [HCOOH] + K_a[HCOOH]/[H+]
= total, then bisect the total until [HCOOH] equals H p -- without the closed form. (2) pH 1.5 is
at the extreme edge of what cloud water shows; the droplets are now "fog droplets" in all arms and
the inside pH is 1.7-2.0 (Los Angeles acid fogs reached pH ~1.7). The outside range is unchanged.

Review (2026-09-27, domain + measurement-design review): (1) the given sentence's premise was false
-- it said the pH is "not far below the pK_a", but the outside pH (3.8-4.6) is always above
pK_a = 3.74; it now says "close to or above". (2) K_a is removed from the inside/outside texts and
stated only in the given sentence (the new rule: a handbook constant used only by the replacement
appears only there). Values a solver may recall, 1.77-1.80 x 10^-4 M (pK_a 3.74-3.75), move the
outside key by < 1.5 % (checked below). "In all its forms" stays: without it the undissociated
concentration alone would be a defensible reading.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

H, KA = 3.6e3, 1.8e-4               # M atm^-1, M


def base(rng):
    return {"xi": rnd(rng.uniform(0.2, 2.0))}


def trigger(rng, regime, base):
    return {"ph": rnd(rng.uniform(1.7, 2.0) if regime == "inside" else rng.uniform(3.8, 4.6))}


def shortcut(p):
    return H * p["xi"] * 1e-9 * 1e6                               # umol/L


def correct(p):
    return shortcut(p) * (1 + KA / 10 ** -p["ph"])


def text(p, variant):
    given = ("Because the pH is close to or above the pK_a of formic acid, much of the dissolved acid is present "
             "as formate: with the acid dissociation constant K_a = 1.8 × 10^-4 M, the total dissolved "
             "concentration is H p (1 + K_a/[H^+]). " if variant == "given" else "")
    return (f"Fog droplets are in equilibrium with air at 1 atm that contains formic acid (HCOOH) at a mixing "
            f"ratio of {num(p['xi'])} ppb; the air is continually replenished, so this gas-phase level stays fixed. "
            f"The fog water has a pH of {num(p['ph'])}, set by other dissolved species. For formic acid, the "
            f"Henry's law constant is 3.6 × 10^3 M atm^-1. "
            f"{given}What is the total concentration of formic acid dissolved in the fog water, in all its "
            f"forms? Express your answer in μmol L^-1.")


def _bisect(lo, hi, f, n=200):
    """Root of an increasing f on [lo, hi]."""
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def _speciate(xi, ph):
    """Total dissolved acid (umol/L) from mass action alone, without the closed form."""
    h = 10 ** -ph
    target = H * xi * 1e-9                                       # [HCOOH] fixed by Henry's law, M

    def undissociated(total):                                    # [HCOOH] for a given total, M
        return _bisect(0.0, total, lambda ha: ha + KA * ha / h - total)

    return _bisect(0.0, 1.0, lambda total: undissociated(total) - target) * 1e6


def check() -> str:
    max_in, min_out, max_err, max_recall = 0.0, math.inf, 0.0, 0.0
    for xi in (0.2, 1.0, 2.0):
        for ph in [1.7 + i / 10 for i in range(4)] + [3.8 + i / 10 for i in range(9)]:
            p = {"xi": xi, "ph": ph}
            max_err = max(max_err, gap(_speciate(xi, ph), correct(p)))
            g = gap(correct(p), shortcut(p))
            if ph <= 2.0:
                max_in = max(max_in, g)
            else:
                min_out = min(min_out, g)
                for ka in (1.77e-4, 1.78e-4, 1.80e-4):                   # K_a no longer stated
                    recalled = shortcut(p) * (1 + ka / 10 ** -ph)
                    max_recall = max(max_recall, gap(recalled, correct(p)))
    assert max_err < 1e-9 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    assert max_recall < .05, max_recall
    assert abs(-math.log10(KA) - 3.75) < 0.01
    return (f"nested-bisection speciation vs closed form {max_err:.1e}; inside max gap {max_in:.4f}; "
            f"outside min gap {min_out:.3f}; recalled K_a 1.77-1.80e-4 {max_recall:.4f}; pK_a {-math.log10(KA):.3f}")


TEMPLATE = Template("effective_henry", "atmospheric_chemistry",
                    "physical Henry's law -> effective Henry's law when a dissolved acid dissociates",
                    "umol/L", base, trigger, text, correct, shortcut)
