"""Equilibrium warming F/lambda -> transient slab-ocean response (2026-09-26).

Parent's method: a sustained forcing F warms the surface by F/lambda, the equilibrium response set
by the climate feedback parameter lambda.
Where it fails: a slab of heat capacity C = rho c h per unit area approaches equilibrium with
e-folding time tau = C/lambda (3.2-13 years here); before several tau have passed the response is
F/lambda (1 - exp(-t/tau)).
Fair from the text: all arms give the slab depth, sea-water density and specific heat, that the slab
exchanges no heat with the deep ocean, lambda, F and the elapsed time, so C dT/dt = F - lambda T
is fully specified.
Outside t = 1-5 years, t/tau < 1.6 and the equilibrium value overshoots by at least 27 %.
Inside t = 60-150 years, t/tau > 4.6 and the equilibrium value is within 1 %.
Conventions: a 365- versus 365.25-day year changes nothing at this tolerance (so the text does not
fix it); land and atmosphere heat capacities are excluded by the slab covering the planet.
Anchor: h = 70 m, lambda = 1.2 W m^-2 K^-1 gives tau = 7.6 years, within the "several years"
mixed-layer response time of textbook energy-balance models (Hartmann 2016, sec. 10.6).
The answer is a temperature change in kelvin.

Review (2026-09-26, independent reviewer): the docstring gave tau = 5-13 years; over the base grid
it is 3.24-12.96 years (h = 50 m, lambda = 2.0 gives 3.24). The claims that matter hold: outside
t/tau < 1.6 with the equilibrium value at least 27 % high, inside t/tau > 4.6 with a gap < 1 %.
Docstring only; no code change.

Review (2026-09-27, domain + measurement-design review): sea-water density and specific heat were
candidates for moving to the given sentence (they serve only the replacement). They stay stated in
all arms, deliberately: recalled values span 1000-1030 kg m^-3 and 3900-4186 J kg^-1 K^-1, and at
t = 1 yr, h = 100 m, lambda = 1.0 the pair 1030 / 4186 (sea-water density with the pure-water
specific heat, a plausible mix) moves the key by 4.96 %, and by 5.02 % with a 365-day year -- over
the 5 % tolerance, so an unstated value would fail a correct answer. check() records that spread.
The depth h stays stated in any case.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

RHO, C, YEAR = 1025.0, 3990.0, 3.15576e7


def base(rng):
    return {"h": float(rng.choice(range(50, 110, 10))), "lam": rng.randint(10, 20) / 10,
            "f": rng.randint(25, 45) / 10}


def trigger(rng, regime, base):
    return {"yr": rnd(rng.uniform(60, 150)) if regime == "inside" else rnd(rng.uniform(1, 5))}


def tau(p):
    return RHO * C * p["h"] / p["lam"] / YEAR                     # years


def shortcut(p):
    return p["f"] / p["lam"]


def correct(p):
    return p["f"] / p["lam"] * (1 - math.exp(-p["yr"] / tau(p)))


def text(p, variant):
    given = ("Because the slab has a finite heat capacity C = ρ c h per unit area, integrate "
             "C dT/dt = F - λT from T = 0: T(t) = (F/λ)(1 - exp(-λt/C)). " if variant == "given" else "")
    return (f"In a simple climate model, the surface is an ocean mixed layer {num(p['h'])} m deep covering the "
            f"whole planet, with sea-water density 1025 kg m^-3 and specific heat 3990 J kg^-1 K^-1; it exchanges "
            f"no heat with the deep ocean. The climate feedback parameter is {num(p['lam'])} W m^-2 K^-1. "
            f"Starting from equilibrium, a constant radiative forcing of {num(p['f'])} W m^-2 is switched on. "
            f"{given}How much has the surface temperature risen {num(p['yr'])} years after the forcing was "
            f"switched on? Express your answer in K.")


TEMPLATE = Template("mixed_layer_response", "climate_dynamics",
                    "equilibrium warming F/lambda -> transient slab-ocean response before several e-folding times",
                    "K", base, trigger, text, correct, shortcut)


def check() -> str:
    max_in, min_out, max_365, unstated = 0.0, 1.0, 0.0, 0.0
    for h in range(50, 110, 10):
        for lam10 in range(10, 21):
            for yr in [60, 100, 150, 1, 2, 3.5, 5]:
                p = {"h": h, "lam": lam10 / 10, "f": 3.7, "yr": yr}
                # Independent: RK4 integration of C dT/dt = F - lambda T in seconds.
                cap, n, temp = RHO * C * h, 4000, 0.0
                dt = yr * YEAR / n
                f = lambda x: (p["f"] - p["lam"] * x) / cap
                for _ in range(n):
                    k1 = f(temp); k2 = f(temp + dt * k1 / 2); k3 = f(temp + dt * k2 / 2); k4 = f(temp + dt * k3)
                    temp += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                assert math.isclose(correct(p), temp, rel_tol=1e-8)
                g_ = gap(correct(p), shortcut(p))
                if yr >= 60:
                    max_in = max(max_in, g_)
                else:
                    min_out = min(min_out, g_)
                    alt = p["f"] / p["lam"] * (1 - math.exp(-yr * 365 * 86400 / (cap / p["lam"])))
                    max_365 = max(max_365, gap(correct(p), alt))
                    for rho in (1000, 1025, 1030):                       # why rho and c stay stated
                        for c in (3900, 3990, 4000, 4186):
                            recalled = p["f"] / p["lam"] * (1 - math.exp(-yr * 365 * 86400 * p["lam"] / (rho * c * h)))
                            unstated = max(unstated, gap(correct(p), recalled))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and max_365 < .05
    assert abs(tau({"h": 70, "lam": 1.2}) - 7.56) < .01
    assert unstated >= .05, "if recalled rho, c stayed within tolerance they could move to the given arm"
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, 365-day spread {max_365:.5f}; "
            f"spread if rho, c were unstated {unstated:.4f} (> 5 %, so they stay stated)")
