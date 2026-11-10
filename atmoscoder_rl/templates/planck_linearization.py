"""Linearised Planck response -> exact Stefan-Boltzmann balance (2026-09-26).

Parent's method: a small change in absorbed flux dF changes the emission temperature by
dT = dF / (4 sigma T^3), the black-body Planck response.
Where it fails: the response is the first term of a Taylor series of T = (F/sigma)^(1/4); when dF is
a large fraction of the emitted flux sigma T^4 the new equilibrium must be solved exactly:
dT = (T^4 + dF/sigma)^(1/4) - T.
Fair from the text: all arms give T, sigma, that the planet emits as a black body, and dF, so the
exact balance is fully specified.
Outside dF = 200-400 W m^-2 on sigma T^4 = 221-401 W m^-2, the linear response is 17-53 % high.
Inside dF = 1-5 W m^-2, dF/(sigma T^4) < 0.023 and the linear response is within 0.9 %.
Conventions: none beyond sigma, which is stated.
Anchor: 4 sigma T^3 at 255 K = 3.76 W m^-2 K^-1, the black-body Planck response quoted for
Earth's effective emission temperature (e.g. Hartmann 2016, sec. 10.3, "about 3.8").
The answer is a temperature change in kelvin, not an absolute temperature.

Review (2026-09-26, independent reviewer): the outside gap range above said 17-45 %; sampling gives
up to 53 % (T = 250 K, dF = 400 W m^-2: key 73.6 K, linear 113 K). Docstring corrected; no code
changed.
"""
from __future__ import annotations

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

SIGMA = 5.67e-8


def base(rng):
    return {"t": float(rng.randint(250, 290))}


def trigger(rng, regime, base):
    return {"df": rnd(rng.uniform(1, 5)) if regime == "inside" else rnd(rng.uniform(200, 400))}


def shortcut(p):
    return p["df"] / (4 * SIGMA * p["t"] ** 3)


def correct(p):
    return (p["t"] ** 4 + p["df"] / SIGMA) ** 0.25 - p["t"]


def text(p, variant):
    given = ("Because the change in flux is not small compared with the emitted flux, do not linearise: solve "
             "σT_new^4 = σT^4 + ΔF for the new emission temperature T_new. " if variant == "given" else "")
    return (f"A planet with no internal heat source radiates to space as a black body with an emission temperature "
            f"of {num(p['t'])} K, in equilibrium with the stellar radiation it absorbs. The absorbed stellar flux, "
            f"averaged over the planet's surface, then increases by {num(p['df'])} W m^-2 and stays at the new "
            f"value. Take σ = 5.67 × 10^-8 W m^-2 K^-4. {given}Once the planet has reached its new equilibrium, "
            f"by how much has its emission temperature increased? Express your answer in K.")


TEMPLATE = Template("planck_linearization", "climate_dynamics",
                    "linearised Planck response dF/(4 sigma T^3) -> exact Stefan-Boltzmann balance for large dF",
                    "K", base, trigger, text, correct, shortcut)


def check() -> str:
    max_in, min_out = 0.0, 1.0
    for t in range(250, 291):
        for df in [1, 2, 3, 5, 200, 250, 300, 350, 400]:
            p = {"t": t, "df": df}
            # Independent: bisection on sigma T'^4 = sigma T^4 + dF.
            lo, hi = float(t), 2.0 * t
            for _ in range(100):
                mid = (lo + hi) / 2
                if SIGMA * mid ** 4 < SIGMA * t ** 4 + df:
                    lo = mid
                else:
                    hi = mid
            assert abs(correct(p) - (lo - t)) < 1e-9 * t
            g_ = gap(correct(p), shortcut(p))
            if df <= 5:
                max_in = max(max_in, g_)
            else:
                min_out = min(min_out, g_)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP
    assert abs(4 * SIGMA * 255 ** 3 - 3.76) < .01
    return f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, Planck response at 255 K {4*SIGMA*255**3:.3f}"
