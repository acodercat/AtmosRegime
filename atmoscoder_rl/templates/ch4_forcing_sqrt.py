"""Linear local-slope approximation -> square-root CH4 radiative forcing, damped by N2O overlap
(extended, 2026-10-01; sibling of co2_forcing_log).

Parent's method: scale the top-of-atmosphere forcing change linearly, using the per-ppb rate
measured at the reference concentration: dF = S (M - M0).
Where it fails: CH4 forcing grows with the square root of the concentration and is damped by
absorption-band overlap with background N2O (Myhre, Highwood, Shine & Stordal 1998):
dF = alpha(sqrt(M) - sqrt(M0)) - [f(M, N0) - f(M0, N0)],
f(M, N) = 0.47 ln(1 + 2.01e-5 (MN)^0.75 + 5.31e-15 M(MN)^1.52), alpha = 0.036 W m^-2 ppb^-0.5. S is
the exact numerical tangent of this full expression at M0 (analogous to K = S*C0 in
co2_forcing_log, but here the tangent has to be taken numerically, both because the overlap term f
has no closed-form derivative as clean as the log's, and because M0 itself is drawn per problem --
see the Design note). Away from M0 the square-root-minus-overlap curve bends below the tangent
line fast enough that an 1100-1250 ppb increase -- the ordinary gap between pre-industrial and
present-day CH4 -- misses by 29-38 %.
Fair from the text: every arm states the reference M0 (700-720 ppb) and the per-ppb rate S at that
same M0 (0.59-0.60 mW m^-2 ppb^-1), phrased only as "at M0 ppb, +1 ppb gives +S mW m^-2 ppb^-1", a
plain data point, not flagged as local or approximate. The companion-gas concentration N0 is
needed only by the replacement (the overlap term f), so it is stated only in the "given" sentence,
not in inside or outside -- fair under the reworded rule 3, because N0's sensitivity is checked
below: sweeping it over 270-330 ppb (pre-industrial to present-day N2O) moves the outside key by
at most 1.63 %, far under the 5 % rule-5 bound, so a solver who assumes a slightly different but
plausible N0 is not penalised.
Inside: M - M0 = 2-10 ppb; the tangent line misses the full expression by at most 0.44 %.
Outside: M - M0 = 1100-1250 ppb, i.e. M = 1800-1970 ppb (ordinary present-day-to-near-future
concentrations); the tangent line misses by 29-38 %.
Conventions: none beyond the Myhre expression itself and the fixed N0 -- there is no competing
"standard" form published close enough to serve as a convention-spread check (a pure square-root
form without the N2O overlap differs from the full expression by about 14 % at these M, which is
why the overlap term is kept in `correct` rather than used as the convention probe).
Anchor: at M = 1745 ppb (~1998 concentration, N0 = 275 ppb -- the IPCC TAR's own reference values),
`correct` with M0 = 700 gives 483 mW m^-2 (0.483 W m^-2), matching the commonly quoted ~0.48 W m^-2
of CH4 radiative forcing accumulated since the pre-industrial era.
Related mechanism: co2_forcing_log (same tangent-line-vs-curved-forcing trap, log instead of sqrt).

Design notes (2026-10-01):
1. The answer is expressed in mW m^-2, not W m^-2. The inside key -- a few ppb at a per-ppb rate of
   order 0.0006 W m^-2 -- is of order 1e-3 W m^-2 in the natural unit, which fails a solver's
   round(x, 2) survival check (rule 11: keys must sit at order 1-1000 in the unit used). Scaling by
   1000 is a presentation choice only; `correct` and `shortcut` are evaluated in W m^-2 internally
   and multiplied by 1000 once, so the physics and the gaps are unaffected.
2. The reference M0 is drawn per problem from 700-720 ppb (the task's own "approx 700-720 ppb"
   range), not held at a single fixed value. This is a text-diversity fix, not a physics one: the
   outside trigger dm = M - M0 is a 4-digit number (1100-1250), and `check_templates.check_texts`
   requires every rendered given to survive 3-significant-figure rounding unchanged, which forces
   dm onto multiples of 10 -- only 16 distinct outside values, far short of the 100 distinct rows
   `check_templates` builds. Letting M0 range over 21 integer ppb values (each exactly
   3-sig-fig-representable, being 3-digit numbers) multiplies the distinct (M0, dm) combinations to
   336. S is recomputed exactly for whatever M0 is drawn (not a single global constant), so the
   tangent match is exact in every case, not just at M0 = 700; check() sweeps the whole M0 range.

Independent review (2026-10-01): the claim above that there is no competing standard form is wrong. Etminan et al.
(2016) (the basis of AR6, via Meinshausen 2020), used with its own absolute coefficients, is +23 to +24 % above the key
over the outside grid. The item stays fair because the stated per-ppb rate at M0 pins the Myhre/TAR form: absolute
Etminan contradicts that rate (its tangent at 713 ppb is 0.746 vs the stated 0.596 mW m^-2 ppb^-1), and any square-root
form calibrated to the stated rate is within 1 % of the key (Etminan shape rescaled: -1 %; 2S sqrt(M0)(sqrt(M)-sqrt(M0)):
+1 %). Pure sqrt without the overlap: +14 %; log calibrated to S: -21 % (both fail). The forcing is a
tropopause/stratosphere-adjusted forcing (Myhre 1998).
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

N0 = 275.0     # ppb, fixed N2O used in the overlap term (given only in the "given" text)
ALPHA = 0.036  # W m^-2 ppb^-0.5


def _f_overlap(M, N):
    return 0.47 * math.log(1 + 2.01e-5 * (M * N) ** 0.75 + 5.31e-15 * M * (M * N) ** 1.52)


def _correct_w(M, m0):
    """Myhre et al. (1998) sqrt-minus-overlap forcing relative to m0, in W m^-2."""
    return ALPHA * (math.sqrt(M) - math.sqrt(m0)) - (_f_overlap(M, N0) - _f_overlap(m0, N0))


def _s_display(m0, eps=1e-4):
    """The per-ppb tangent slope of `_correct_w` at m0, in mW m^-2 ppb^-1, rounded to 3 sig figs --
    the number the text states and `shortcut` uses, recomputed exactly for the drawn m0."""
    slope_w = (_correct_w(m0 + eps, m0) - _correct_w(m0 - eps, m0)) / (2 * eps)
    return rnd(slope_w * 1000, 3)


def base(rng):
    return {"m0": float(rng.randint(700, 720))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"dm": rnd(rng.uniform(2.0, 10.0), 3)}
    return {"dm": rnd(rng.uniform(1100.0, 1250.0), 3)}   # 3 sig figs: multiples of 10 here


def shortcut(p):
    return _s_display(p["m0"]) * p["dm"]


def correct(p):
    return 1000 * _correct_w(p["m0"] + p["dm"], p["m0"])


def text(p, variant):
    given = ("Because this forcing increases with the square root of the concentration, damped by "
             "absorption-band overlap with background N2O, rather than increasing linearly, use "
             "ΔF = 1000[0.036(√M − √M0) − (f(M, N0) − f(M0, N0))] mW m^-2, where f(M, N) = "
             "0.47 ln(1 + 2.01×10^-5(MN)^0.75 + 5.31×10^-15 M(MN)^1.52) W m^-2 and the background N2O "
             "concentration is N0 = 275 ppb. " if variant == "given" else "")
    return (f"At a background CH4 concentration of {num(p['m0'])} ppb, increasing the concentration by 1 ppb "
            f"raises the top-of-atmosphere radiative forcing by about {num(_s_display(p['m0']))} mW m^-2. "
            f"{given}If the CH4 concentration rises by {num(p['dm'])} ppb above this {num(p['m0'])} ppb "
            f"reference, what is the increase in radiative forcing? Express your answer in mW m^-2.")


TEMPLATE = Template("ch4_forcing_sqrt", "climate_dynamics",
                    "linear local-slope CH4 forcing -> square-root forcing damped by N2O overlap",
                    "mW/m^2", base, trigger, text, correct, shortcut)


def _gprime(M, n):
    a, b = 2.01e-5, 5.31e-15
    t1 = a * 0.75 * n * (M * n) ** (-0.25)
    t2 = b * (M * n) ** 1.52 + b * M * 1.52 * n * (M * n) ** 0.52
    return t1 + t2


def _g(M, n):
    a, b = 2.01e-5, 5.31e-15
    return 1 + a * (M * n) ** 0.75 + b * M * (M * n) ** 1.52


def _dcorrect_dM_w(M):
    """Analytic d(correct)/dM in W m^-2 per ppb: alpha/(2 sqrt(M)) minus the overlap derivative."""
    return ALPHA / (2 * math.sqrt(M)) - 0.47 * _gprime(M, N0) / _g(M, N0)


def _simpson_correct_mw(m0, dm, n=4000):
    """Independent: Simpson-integrate the analytic derivative from m0 to m0+dm (different algebra
    from the closed-form sqrt-minus-overlap expression `_correct_w` evaluates directly)."""
    M = m0 + dm
    h = (M - m0) / n
    s = _dcorrect_dM_w(m0) + _dcorrect_dM_w(M)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * _dcorrect_dM_w(m0 + i * h)
    return 1000 * s * h / 3


def check() -> str:
    m0_grid = list(range(700, 721))

    max_in = 0.0
    for m0 in m0_grid:
        for dm10 in range(20, 101):
            p = {"m0": float(m0), "dm": dm10 / 10}
            max_in = max(max_in, gap(correct(p), shortcut(p)))

    min_out, max_out = 1.0, 0.0
    for m0 in m0_grid:
        for dm in range(1100, 1251, 10):
            p = {"m0": float(m0), "dm": float(dm)}
            g = gap(correct(p), shortcut(p))
            min_out, max_out = min(min_out, g), max(max_out, g)

    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)

    max_indep = 0.0
    for m0 in (700.0, 710.0, 720.0):
        for dm in (5.0, 500.0, 1100.0, 1250.0):
            c = correct({"m0": m0, "dm": dm})
            s = _simpson_correct_mw(m0, dm)
            max_indep = max(max_indep, abs(c - s) / abs(c))
    assert max_indep < 1e-6, max_indep

    # Rule 5 (reworded rule 3): N0 is given only in the "given" text; check that plausible
    # alternative N2O values do not move the outside key by more than 5 %.
    def correct_with_n(m0, dm, n):
        M = m0 + dm
        return 1000 * (ALPHA * (math.sqrt(M) - math.sqrt(m0)) - (_f_overlap(M, n) - _f_overlap(m0, n)))

    n_spread = 0.0
    for m0 in (700.0, 710.0, 720.0):
        for dm in (1100.0, 1175.0, 1250.0):
            base_val = correct_with_n(m0, dm, N0)
            for n in (270.0, 290.0, 310.0, 330.0):
                n_spread = max(n_spread, abs(correct_with_n(m0, dm, n) - base_val) / base_val)
    assert n_spread < 0.05, n_spread

    anchor = correct({"m0": 700.0, "dm": 1745.0 - 700.0})
    assert 400.0 < anchor < 550.0, anchor   # ~0.48 W m^-2, the commonly quoted CH4 ERF since 1750

    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f} (max {max_out:.3f}), "
            f"Simpson check {max_indep:.1e}, N0 spread {n_spread:.4f}, "
            f"anchor at M0=700, M=1745 ppb: {anchor:.1f} mW/m^2")
