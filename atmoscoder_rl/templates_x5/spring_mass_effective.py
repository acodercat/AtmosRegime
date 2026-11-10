"""Massless-spring oscillator -> spring whose own mass is comparable with the load (X5, mechanics).

Parent's method: T = 2 pi sqrt(m / k), the mass-spring period.
Where it fails: it treats the spring as massless. A uniform spring of mass m_s carrying an end mass m oscillates
(fundamental mode) at omega with beta tan(beta) = m_s / m, beta = omega sqrt(m_s / k) (the uniform elastic rod with
an end mass); Rayleigh's m + m_s/3 is the familiar approximation. With m_s / m = 1.2-1.8 the period is 18-29 % longer.
Fair from the text: every arm states m, the spring's own mass and its stiffness; the trigger is the spring's mass,
ordinary data. Nothing is recalled.
Inside: m_s / m = 0.005-0.02 (<= 0.4 %). Outside: m_s / m = 1.2-1.8.
Conventions: Rayleigh's effective mass m + m_s/3 is within 1.7 % of the exact fundamental on the outside grid
(check()), so either answer is accepted at 5 %.
Asked quantity: period of the fundamental vertical oscillation, s.
check(): finite-element chain (200 point masses and springs) eigenfrequency by inverse iteration, independent of the
transcendental equation.
Reference: Rayleigh, Theory of Sound, sec. 156; Thomson, Theory of Vibration with Applications, ch. 2.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd


def base(rng):
    return {"m": rnd(rng.uniform(0.2, 1.0)), "k": float(rng.choice(range(20, 201, 10)))}


def trigger(rng, regime, b):
    f = rng.uniform(0.005, 0.02) if regime == "inside" else rng.uniform(1.2, 1.8)
    return {"ms": rnd(b["m"] * f)}


def _beta(ratio):
    lo, hi = 1e-9, math.pi / 2 - 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid * math.tan(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def correct(p):
    omega = _beta(p["ms"] / p["m"]) * math.sqrt(p["k"] / p["ms"])
    return 2 * math.pi / omega


def shortcut(p):
    return 2 * math.pi * math.sqrt(p["m"] / p["k"])


def text(p, variant):
    given = ("The spring's own mass is not negligible: for the fundamental mode, omega satisfies beta tan(beta) = m_s / m "
             "with beta = omega sqrt(m_s / k). " if variant == "given" else "")
    return (f"A block of mass {num(p['m'])} kg hangs from a uniform helical spring of stiffness {num(p['k'])} N m^-1 "
            f"whose upper end is fixed. The spring itself has a mass of {num(p['ms'])} kg. The block is pulled down a "
            f"little and released. {given}What is the period of the resulting vertical oscillation? "
            f"Express your answer in s.")


TEMPLATE = Template("spring_mass_effective", "mechanics",
                    "massless spring -> distributed spring mass comparable with the load",
                    "s", base, trigger, text, correct, shortcut)


def _chain(p, n=200):
    """n segments: masses ms/n at the nodes (half at the fixed end ignored), end mass m; inverse iteration."""
    kseg, mseg = p["k"] * n, p["ms"] / n
    masses = [mseg] * (n - 1) + [mseg / 2 + p["m"]]
    # K tridiagonal for nodes 1..n (node 0 fixed)
    diag = [2 * kseg] * (n - 1) + [kseg]
    off = [-kseg] * (n - 1)
    x = [1.0] * n
    lam = 0.0
    for _ in range(200):
        rhs = [masses[i] * x[i] for i in range(n)]                    # solve K y = M x
        b = diag[:]; d = rhs[:]
        for i in range(1, n):
            w = off[i - 1] / b[i - 1]
            b[i] -= w * off[i - 1]; d[i] -= w * d[i - 1]
        y = [0.0] * n; y[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            y[i] = (d[i] - off[i] * y[i + 1]) / b[i]
        num_ = sum(y[i] * masses[i] * x[i] for i in range(n)); den = sum(y[i] * masses[i] * y[i] for i in range(n))
        lam = num_ / den                                              # Rayleigh quotient estimate of omega^2
        s = math.sqrt(den); x = [v / s for v in y]
    return 2 * math.pi / math.sqrt(lam)


def check() -> str:
    import random
    rng = random.Random("x5-check:spring_mass_effective")
    worst = 0.0; rayleigh = 0.0; gi, go = [], []
    for _ in range(20):
        pin, pout = pair(TEMPLATE, rng)
        worst = max(worst, abs(_chain(pout) / correct(pout) - 1))
        rayleigh = max(rayleigh, gap(correct(pout), 2 * math.pi * math.sqrt((pout["m"] + pout["ms"] / 3) / pout["k"])))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert abs(_beta(1.0) - 0.8603) < 1e-4                               # beta tan beta = 1
    assert worst < 2e-3 and rayleigh < 0.05 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP
    return (f"FE chain vs exact {worst:.1e}, Rayleigh m + m_s/3 within {rayleigh:.3f}, "
            f"inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}")
