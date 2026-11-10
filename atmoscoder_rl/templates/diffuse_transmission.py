"""Beam (Beer's-law) transmittance applied to a diffuse flux -> the hemispheric diffusivity
approximation (extended, 2026-10-01).

Parent's method: Beer's law for a single beam at normal incidence, applied directly to the whole
flux: F = F0 exp(-tau).
Where it fails: the incident radiation here is isotropic over the hemisphere, not a single beam --
every direction mu = cos(theta) in [0, 1] carries its own path length tau/mu through the layer, and
the flux transmittance is the mu-weighted average over all of them,
T(tau) = 2 integral_0^1 mu exp(-tau/mu) dmu = 2 E3(tau)
(E3 the exponential integral of order 3). Textbooks do not ask a solver to evaluate E3 by hand;
the standard practical substitute is the diffusivity approximation, T(tau) = exp(-D tau) with
D = 1.66 (Elsasser 1942), which reproduces the exact hemispheric integral to within 2 % for
tau <~ 0.5. Nothing about the stated optical depth looks extreme -- 0.3-0.5 is an ordinary thin
haze or trace-gas layer -- but the beam formula (D = 1) applied to this diffuse flux misses the
true transmittance by 22-39 %.
Fair from the text: both inside and outside state the incident flux F0 and the optical depth tau,
everything both formulas need; both also state that the radiation is isotropic over the hemisphere
-- not a hint toward the correction, but the geometry the problem is actually posed in (the trap is
that a solver who knows the diffuse geometry still reaches for Beer's law, not that the geometry is
concealed). Neither inside nor outside names the diffusivity approximation or gives D = 1.66; that
appears only in the "given" text.
Inside: tau = 0.001-0.02; the diffusivity approximation and the beam formula agree to within 1.33 %
(both close to 1, since the layer barely absorbs).
Outside: tau = 0.3-0.5; the beam formula misses the diffusivity approximation by 22-39 %.
Conventions: against the exact hemispheric integral 2 E3(tau) (computed two independent ways in
check(), below), the diffusivity approximation is within 1.64 % over the outside grid; against the
range of diffusivity factors found in the literature, D = 1.6-1.75 (Liou 2002 cites 1.66 = 5/3 as
the standard value; narrower than the sometimes-quoted 1.5-2.0 full range, which exceeds 5 % at the
top of this template's outside grid and is not used here), the key varies by at most 4.4 %.
Reference: Elsasser, W.M. (1942), the diffusivity approximation; Liou, K.N. (2002), An
Introduction to Atmospheric Radiation (2nd ed.), sec. 6.4.4 (D = 1.66, the "diffusivity factor");
Abramowitz, M. & Stegun, I.A. (1964), Handbook of Mathematical Functions, formulas 5.1.11 and
5.1.14 (the E1 series and the E_n recurrence used as the independent check).
Related mechanism: optically_thin_emission (same thin-layer radiative-transfer family, a different
failure: self-absorption rather than beam-vs-diffuse geometry).

Doubt, recorded per the task brief: stating "isotropic over the hemisphere" in every arm is
necessary to pose the problem (the geometry cannot be silently assumed), but it is also the one
piece of information a solver would need to recall the diffusivity approximation from, so it is
unavoidably doing double duty as setup and as partial cue. This is flagged here rather than argued
away; whether it still fools a strong model is for screening, not for this module.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

DIFFUSIVITY = 1.66   # Elsasser's diffusivity factor, D


def base(rng):
    return {"f0": rnd(rng.uniform(150.0, 350.0), 3)}


def trigger(rng, regime, base):
    tau = rng.uniform(0.001, 0.02) if regime == "inside" else rng.uniform(0.3, 0.5)
    return {"tau": rnd(tau, 3)}


def shortcut(p):
    return p["f0"] * math.exp(-p["tau"])


def correct(p):
    return p["f0"] * math.exp(-DIFFUSIVITY * p["tau"])


def text(p, variant):
    given = ("Because the incident radiation is diffuse (isotropic over the hemisphere) rather than a single "
             "beam, use the diffusivity approximation for the flux transmittance, T = exp(−1.66τ), so the "
             "transmitted flux is F = F0 exp(−1.66τ) rather than the beam transmittance F0 exp(−τ). "
             if variant == "given" else "")
    return (f"Infrared radiance leaving the base of a cloud deck is isotropic over the upward hemisphere, "
            f"with an upward flux of {num(p['f0'])} W m^-2 at the cloud base. A thin layer of absorbing gas "
            f"directly above the cloud has an optical depth of {num(p['tau'])} at this wavelength and neither "
            f"emits nor scatters radiation. {given}What flux reaches the top of the absorbing layer? Express "
            f"your answer in W m^-2.")


TEMPLATE = Template("diffuse_transmission", "atmospheric_radiation",
                    "beam Beer's-law transmittance -> hemispheric diffusivity approximation for diffuse flux",
                    "W m^-2", base, trigger, text, correct, shortcut)

_GAMMA = 0.5772156649015329   # Euler-Mascheroni constant


def _e1_series(x, terms=60):
    """Abramowitz & Stegun 5.1.11: E1(x) = -gamma - ln(x) + sum (-1)^(k+1) x^k / (k k!)."""
    s = 0.0
    for k in range(1, terms + 1):
        s += (-1) ** (k + 1) * x ** k / (k * math.factorial(k))
    return -_GAMMA - math.log(x) + s


def _e3_via_recurrence(x):
    """Abramowitz & Stegun 5.1.14: E_(n+1)(x) = [e^-x - x E_n(x)] / n. A different mathematical
    route (series + recurrence for the exponential integral) from the mu-integral below."""
    e1 = _e1_series(x)
    e2 = math.exp(-x) - x * e1
    e3 = (math.exp(-x) - x * e2) / 2
    return e3


def _hemispheric_simpson(tau, n=20000):
    """Independent: direct Simpson quadrature of 2 integral_0^1 mu exp(-tau/mu) dmu (the physical
    definition), cross-checked against the series/recurrence above."""
    def f(mu):
        return 0.0 if mu <= 0 else mu * math.exp(-tau / mu)
    h = 1.0 / n
    s = f(0.0) + f(1.0)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(i * h)
    return 2 * s * h / 3


def check() -> str:
    max_in = 0.0
    for tau1000 in range(1, 21):
        tau = tau1000 / 1000
        p = {"f0": 1.0, "tau": tau}
        max_in = max(max_in, gap(correct(p), shortcut(p)))

    min_out, max_out = 1.0, 0.0
    for tau100 in range(30, 51):
        tau = tau100 / 100
        p = {"f0": 1.0, "tau": tau}
        g = gap(correct(p), shortcut(p))
        min_out, max_out = min(min_out, g), max(max_out, g)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)

    max_cross, max_exact_spread, max_d_spread = 0.0, 0.0, 0.0
    for tau100 in range(30, 51, 2):
        tau = tau100 / 100
        simpson = _hemispheric_simpson(tau)
        series = 2 * _e3_via_recurrence(tau)
        max_cross = max(max_cross, abs(simpson - series) / simpson)
        key = math.exp(-DIFFUSIVITY * tau)
        max_exact_spread = max(max_exact_spread, abs(key - simpson) / simpson)
        d_spread = max(abs(key - math.exp(-d * tau)) / key for d in (1.6, 1.75))
        max_d_spread = max(max_d_spread, d_spread)
    assert max_cross < 1e-8, max_cross
    assert max_exact_spread < 0.05 and max_d_spread < 0.05, (max_exact_spread, max_d_spread)

    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f} (max {max_out:.3f}), "
            f"exact-integral cross-check {max_cross:.1e}, diffusivity vs exact {max_exact_spread:.4f}, "
            f"D=1.6-1.75 spread {max_d_spread:.4f}")
