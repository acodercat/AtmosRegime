"""Flat-surface saturation vapour pressure -> Kelvin-enhanced vapour pressure over a small drop
(spec: MECHANISMS.md).

Parent's method: the air in equilibrium with a pure-water drop holds the saturation vapour
pressure e_s(T) of a plane water surface.
Where it fails: over a strongly curved surface the equilibrium vapour pressure is raised by the
Kelvin factor, e = e_s exp[2 sigma / (rho_w R_v T r)]; for r of a few nanometres the excess is
15-35 %.
Fair from the text: every arm gives the temperature and the e_s formula; the surface tension of
water, its density and R_v are handbook constants (see the 2026-09-27 note), stated only in the
given sentence. Only the radius changes between the arms.
Inside: r = 500-5000 nm; the Kelvin exponent is below 0.0024, so e differs from e_s by < 0.25 %.
Outside: r = 4.0-6.5 nm, Kelvin exponent 0.17-0.30 at 0-20 degC (the gap is 1 - exp(-a/r),
so 15 % needs a/r >= 0.163, i.e. r <= 6.8 nm at 20 degC).

Conventions: the linearised Kelvin factor 1 + a/r (Rogers & Yau eq. 6.4 form) stays within 3.7 %
of the exponential key across the outside grid, and writing the exponent with the molar form
2 sigma M_w/(rho_w R* T r) (R* = 8.314, M_w = 0.018015) changes it by < 0.01 %; both are checked
below. The outside range stops at 4.0 nm precisely so the linearised form stays inside the 5 %
tolerance. Using a temperature-dependent surface tension (0.0756-0.0728 N/m at 0-20 degC) instead of
the stated 0.075 moves the answer by < 0.9 %.
Anchor: Rogers & Yau (1989) give a = 3.3e-5/T cm for the Kelvin coefficient; with the stated
constants a = 3.25e-5/T cm. Wallace & Hobbs (2006, sec. 6.1.1): a 0.01-um pure-water drop needs about
12 % supersaturation; the formula here gives 12.4 % at 5 degC.

Review (2026-09-26, independent reviewer): sigma and rho_w serve no purpose in the problem except
the Kelvin term, so their presence in every arm all but forces the Kelvin computation; inside it
changes nothing, so the inside/outside contrast measures "knows the Kelvin effect" more than
"recognises the regime". Kept for fairness (rule 3), but the expected lure is weak, not medium.
The "molar form" check below rescales the same exponential and is not independent; the Rogers &
Yau and Wallace & Hobbs anchors carry the independent part.

Review (2026-09-27, domain + measurement-design review): sigma, rho_w and R_v are removed from the
inside/outside texts and stated only in the given sentence (the new rule: a universal or handbook
constant used only by the replacement appears only there). With sigma given, the inside problem was
itself a Kelvin exercise, so the shortcut was not what a textbook would have a student use. Any
sigma a solver may recall, 0.072-0.076 N/m, moves the outside key by < 1.3 % (checked below);
rho_w and R_v are universal. Radii are now written in nm (inside 500-5000, outside 4.0-6.5): the
old "0.0056 um" was an odd notation that only the outside arm used, an avoidable cue.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, num, rnd

SIGMA, RHO_W, RV = 0.075, 1000.0, 461.5


def es(t):
    return 6.112 * math.exp(17.67 * t / (t + 243.5))                    # hPa, t in degC


def base(rng):
    return {"t": float(rng.randint(0, 20))}


def trigger(rng, regime, base):
    return {"r": rnd(rng.uniform(500, 5000)) if regime == "inside" else rnd(rng.uniform(4.0, 6.5))}   # nm


def shortcut(p):
    return es(p["t"])


def correct(p):
    return es(p["t"]) * math.exp(2 * SIGMA / (RHO_W * RV * (p["t"] + 273.15) * p["r"] * 1e-9))


def text(p, variant):
    given = ("Because the droplet is so small, its curvature raises the equilibrium vapour pressure above the "
             "flat-surface value: multiply e_s by the Kelvin factor exp[2σ/(ρ_w R_v T r)], with T in kelvin, "
             "r the droplet radius, surface tension σ = 0.075 N m^-1, water density ρ_w = 1000 kg m^-3 and "
             "R_v = 461.5 J kg^-1 K^-1. " if variant == "given" else "")
    return (f"A droplet of pure water of radius {num(p['r'])} nm is at a temperature of {num(p['t'])} °C. "
            f"Take the saturation vapour pressure over a plane water surface "
            f"e_s = 6.112 exp[17.67 T/(T + 243.5)] hPa with T in °C. {given}What vapour pressure must the "
            f"surrounding air have for the droplet to be in equilibrium, neither growing nor evaporating? "
            f"Express your answer in hPa.")


def check() -> str:
    max_in, min_out, max_lin, max_molar, max_sigma, max_recall = 0., math.inf, 0., 0., 0., 0.
    for t in range(0, 21):
        for r in [100 * i for i in range(5, 51)] + [i / 10 for i in range(40, 66)]:   # nm
            p = {"t": t, "r": r}
            key = correct(p)
            tk, rm = t + 273.15, r * 1e-9
            # Independent: chemical-potential form with molar constants.
            molar = es(t) * math.exp(2 * SIGMA * 0.018015 / (RHO_W * 8.314 * tk * rm))
            max_molar = max(max_molar, abs(molar / key - 1))
            g = abs(shortcut(p) - key) / key
            if r >= 500:
                max_in = max(max_in, g)
                continue
            min_out = min(min_out, g)
            a = 2 * SIGMA / (RHO_W * RV * tk)
            max_lin = max(max_lin, abs(es(t) * (1 + a / rm) / key - 1))
            sig_t = 0.0756 - 0.000145 * t                                    # ~0.0756 at 0 C, 0.0727 at 20 C
            max_sigma = max(max_sigma, abs(math.exp(2 * sig_t / (RHO_W * RV * tk * rm))
                                           / math.exp(a / rm) - 1))
            for recalled in (0.072, 0.073, 0.074, 0.076):                    # sigma no longer stated
                max_recall = max(max_recall, abs(math.exp(2 * recalled / (RHO_W * RV * tk * rm))
                                                 / math.exp(a / rm) - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP
    assert max(max_lin, max_molar, max_sigma, max_recall) < .05
    for t in (0, 10, 20):                                                    # Rogers & Yau a = 3.3e-5/T cm
        tk = t + 273.15
        assert abs(2 * SIGMA / (RHO_W * RV * tk) * 100 / (3.3e-5 / tk) - 1) < .02
    super_ = math.exp(2 * SIGMA / (RHO_W * RV * 278.15 * 1e-8)) - 1           # W&H: ~12 % at 0.01 um
    assert .11 < super_ < .13
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; linearised Kelvin {max_lin:.4f}, "
            f"molar form {max_molar:.1e}, T-dependent sigma {max_sigma:.4f}, recalled sigma 0.072-0.076 "
            f"{max_recall:.4f}; 0.01 um anchor {super_:.3f}")


TEMPLATE = Template("kelvin_curvature", "cloud_physics",
                    "flat-surface saturation vapour pressure -> Kelvin curvature enhancement for nanometre drops",
                    "hPa", base, trigger, text, correct, shortcut)
