"""Cooper-Jacob straight-line approximation -> full Theis well function (extended, hydrology,
2026-10-01).

Parent's method: drawdown at distance r from a well pumping a confined aquifer at rate Q since time
t is the Theis solution s = (Q/4 pi T) W(u), u = r^2 S/(4 T t), T transmissivity, S storativity, and
W(u) = E1(u) = -Ei(-u) the well function (the exponential integral). W(u) has no elementary closed
form, so the method every introductory groundwater course teaches for hand calculation is Cooper &
Jacob's (1946) straight-line approximation: for small u, W(u) ~ -gamma - ln(u) (gamma the
Euler-Mascheroni constant, 0.5772...), dropping every term of W(u)'s convergent series but the
leading log. This is THE standard field method (plot drawdown against log t, read T and S off the
slope and intercept) and is routinely applied without rechecking u first.
Where it fails: the log truncation is only the small-u limit of W(u)'s series; published validity
criteria (Kruseman & de Ridder, *Analysis and Evaluation of Pumping Test Data*, 1994, sec. 3.2.3) call
for u <~ 0.01-0.05 for the approximation to be trustworthy. At larger u -- early in a pumping test, or
far from the well -- the log form is not a small correction away from W(u), it is a different number.
Fair from the text: every arm states the pumping rate Q, the observation-well distance r, the
transmissivity T and the storativity S -- everything both the log approximation and the full Theis
sum need. Only the elapsed time since pumping began changes between the arms (the one trigger).
Inside: u is sampled in [0.001, 0.008], inside the stricter 0.01 criterion with a 20-80x margin; the
log approximation misses the key by at most 0.19 % there (`check()`).
Outside: u is sampled in [0.22, 0.40] -- an ordinary early-time reading (minutes after pumping
starts) at an ordinary observation-well distance (40-120 m). The log approximation misses the key by
18-52 % there, and its drawdown stays positive (>= 0.34 Q/4 pi T, asserted in `check()`), so nothing
absurd in the shortcut's own number gives the trap away.
Truncating W(u)'s series one term later (-gamma - ln u + u) lands within 1-5 % of the key at this u.
That is the correct series computed by hand, not a distinct shortcut, so a pass with it is legitimate.
No single closed-form alternative to the log approximation is in common field use at this u, so no
second genuinely different wrong formula is checked here. Thiem's steady-state equation is not a
competing method either: it needs a radius of influence, which no arm states.
Independent review (2026-10-02): the first draft sampled outside u in [0.5, 0.8], where
-gamma - ln u <= 0 for u >= 0.5615 -- 81 % of outside rows gave a zero or negative Cooper-Jacob
drawdown, which tells a solver the method failed. Moved to [0.22, 0.40]; dropped the old assert that
the one-extra-term refinement also misses (it should not); renamed from `theis_vs_thiem` (the shortcut
is Cooper-Jacob, not Thiem). Elapsed time stays in minutes in every arm, so the minute-to-day
conversion is the same ordinary step on every arm.
Reference: Theis, C.V. (1935), Trans. AGU 16, 519 (the well-function solution); Cooper, H.H. and
Jacob, C.E. (1946), Trans. AGU 27, 526 (the straight-line approximation and its name); Kruseman,
G.P. and de Ridder, N.A. (1994), ILRI Publication 47, sec. 3.2.3, for the u-based validity criterion;
the well-function values used as an anchor below (W(0.1) = 1.823, W(1.0) = 0.2194) are the standard
tabulated values reproduced in every pumping-test text, e.g. Freeze, R.A. and Cherry, J.A. (1979),
*Groundwater*, table 8.2.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

GAMMA = 0.5772156649015329   # Euler-Mascheroni constant


def well_function(u, terms=80):
    """Theis well function W(u) = E1(u), via its convergent series (Abramowitz & Stegun 5.1.11):
    E1(u) = -gamma - ln(u) + sum_{k=1..inf} (-1)^(k+1) u^k/(k k!)."""
    s = 0.0
    fact = 1.0
    upow = 1.0
    for k in range(1, terms + 1):
        upow *= u
        fact *= k
        s += ((-1) ** (k + 1)) * upow / (k * fact)
    return -GAMMA - math.log(u) + s


def _u(p):
    t_days = p["t"] / 1440.0
    return p["r"] ** 2 * p["S"] / (4.0 * p["T"] * t_days)


def correct(p):
    """Full Theis drawdown, m."""
    return p["Q"] / (4 * math.pi * p["T"]) * well_function(_u(p))


def shortcut(p):
    """Cooper-Jacob straight-line approximation: W(u) ~ -gamma - ln(u)."""
    u = _u(p)
    return p["Q"] / (4 * math.pi * p["T"]) * (-GAMMA - math.log(u))


def base(rng):
    return {"r": rnd(rng.uniform(40.0, 120.0), 3), "T": rnd(rng.uniform(100.0, 250.0), 3),
            "S": rnd(rng.uniform(2e-4, 1e-3), 3), "Q": rnd(rng.uniform(1800.0, 3500.0), 3)}


def trigger(rng, regime, base):
    r, T, S = base["r"], base["T"], base["S"]
    u = rng.uniform(0.001, 0.008) if regime == "inside" else rng.uniform(0.22, 0.40)
    t_days = r ** 2 * S / (4.0 * T * u)
    return {"t": rnd(t_days * 1440.0, 3)}   # minutes since pumping began


def text(p, variant):
    given = ("Because u is not small, the Cooper–Jacob straight-line approximation does not hold "
             "here; use the full Theis equation instead: s = (Q / 4πT) W(u), where u = r²S/(4Tt) "
             "and W(u) is the well function (the exponential integral E1(u)). "
             if variant == "given" else "")
    return (f"A fully penetrating well in a confined aquifer is pumped at a constant rate of "
            f"{num(p['Q'])} m^3/day. An observation well is {num(p['r'])} m from the pumping well. "
            f"The aquifer's transmissivity is {num(p['T'])} m^2/day and its storativity is "
            f"{num(p['S'])}. Pumping began {num(p['t'])} minutes ago. {given}What is the drawdown "
            f"in the observation well? Express your answer in m.")


TEMPLATE = Template("cooper_jacob_early_time", "hydrology",
                    "Cooper-Jacob straight-line approximation -> full Theis well function",
                    "m", base, trigger, text, correct, shortcut)


def _well_function_as(x):
    """Independent recomputation: Abramowitz & Stegun 5.1.53 rational-polynomial approximation of
    E1(x) for 0 < x <= 1 (|error| < 2e-7), a different algorithm from the series summed in
    `well_function` -- not a truncation of the same series."""
    a0, a1, a2, a3, a4, a5 = (-0.57721566, 0.99999193, -0.24991055,
                               0.05519968, -0.00976004, 0.00107857)
    return -math.log(x) + a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5


def check() -> str:
    max_in = 0.0
    min_out = math.inf
    min_cj = math.inf
    max_poly_dev = 0.0
    min_s, max_s = math.inf, 0.0

    for r in (40.0, 80.0, 120.0):
        for T in (100.0, 175.0, 250.0):
            for S in (2e-4, 6e-4, 1e-3):
                for Q in (1800.0, 2650.0, 3500.0):
                    base_p = {"r": r, "T": T, "S": S, "Q": Q}
                    for u in (0.001, 0.0045, 0.008):
                        t = r ** 2 * S / (4 * T * u) * 1440.0
                        p = {**base_p, "t": t}
                        c, s = correct(p), shortcut(p)
                        max_in = max(max_in, gap(c, s))
                        max_poly_dev = max(max_poly_dev, gap(well_function(u), _well_function_as(u)))
                    for u in (0.22, 0.31, 0.40):
                        t = r ** 2 * S / (4 * T * u) * 1440.0
                        p = {**base_p, "t": t}
                        c, s = correct(p), shortcut(p)
                        min_out = min(min_out, gap(c, s))
                        min_s, max_s = min(min_s, c), max(max_s, c)
                        min_cj = min(min_cj, s / (p["Q"] / (4 * math.pi * p["T"])))
                        max_poly_dev = max(max_poly_dev, gap(well_function(u), _well_function_as(u)))

    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_poly_dev < 1e-5, max_poly_dev                  # independent rational-approx cross-check
    assert min_cj >= 0.25, min_cj                             # shortcut drawdown positive, not absurd
    assert min_s > 0.15 and max_s < 50.0                      # clear of the round(x, 2) grading floor

    # Anchor: the standard tabulated well-function values reproduced in every pumping-test textbook.
    assert abs(well_function(0.1) - 1.8229) < 2e-4
    assert abs(well_function(1.0) - 0.2194) < 2e-4

    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, "
            f"min shortcut drawdown {min_cj:.3f} Q/4piT, "
            f"A&S rational-approx deviation {max_poly_dev:.1e}, "
            f"key range {min_s:.2f}-{max_s:.2f} m; W(0.1)={well_function(0.1):.4f} (lit. 1.8229), "
            f"W(1.0)={well_function(1.0):.4f} (lit. 0.2194)")
