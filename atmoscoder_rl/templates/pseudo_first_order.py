"""Pseudo-first-order NO + O3 titration -> mixed second-order kinetics (2026-09-26 expansion).

Parent's method: with ozone in large excess, [O3] is constant and NO decays exponentially at the
rate k[O3]_0, so its half-life is ln 2 / (k [O3]_0).
Where it fails: when the initial NO is comparable with the ozone, ozone is consumed as well and
the pseudo-first-order rate slows down. For A + B -> products with [A]_0 < [B]_0,
ln([B][A]_0 / ([A][B]_0)) = ([B]_0 - [A]_0) k t, so A reaches half its initial value at
t = ln(2 - [A]_0/[B]_0) / (k ([B]_0 - [A]_0)).
Fair from the text: both variants state both initial mixing ratios, the rate constant, the air
number density and that this is the only reaction (dark chamber, so no NO2 photolysis regenerates
NO); outside, [O3]_0/[NO]_0 = 1.2-1.5.
Inside: NO = 0.05-0.3 ppb against O3 = 30-60 ppb ([NO]_0/[O3]_0 <= 0.01), so ozone changes by under
0.5 % and the pseudo-first-order half-life is within 0.4 % of the exact one.
Conventions: none; the rate constant is given. Outside the shortcut is 18.6-25 % short
(check() integrates the rate equations numerically, independently of the closed form).
Anchor: JPL 19-5, k(NO + O3) = 3.0e-12 exp(-1500/T) = 1.95e-14 cm^3 molecule^-1 s^-1 at 298 K;
the text uses 1.9e-14 (3 % lower; the key follows the text).
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

K, NAIR = 1.9e-14, 2.46e19          # cm^3 molecule^-1 s^-1; molecules cm^-3 at 25 degC, 1 atm
KPPB = K * NAIR * 1e-9              # ppb^-1 s^-1


def base(rng):
    return {"o3": float(rng.randint(30, 60))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"no": rnd(rng.uniform(0.05, 0.3))}
    return {"no": rnd(base["o3"] / rng.uniform(1.2, 1.5))}


def shortcut(p):
    return math.log(2) / (KPPB * p["o3"])


def correct(p):
    a, b = p["no"], p["o3"]
    return math.log(2 - a / b) / (KPPB * (b - a))


def text(p, variant):
    given = ("Because the two reactants start at comparable mixing ratios, ozone is consumed along with NO and "
             "cannot be treated as constant: use the integrated rate law for A + B → products with unequal "
             "initial concentrations, ln([B][A]_0/([A][B]_0)) = ([B]_0 − [A]_0)kt. " if variant == "given" else "")
    return (f"In a dark reaction chamber at 25 °C and 1 atm, where the number density of air is "
            f"2.46 × 10^19 molecules cm^-3, the air initially contains {num(p['o3'])} ppb of ozone and {num(p['no'])} ppb of "
            f"nitric oxide. The only reaction is NO + O3 → NO2 + O2, with a rate constant of "
            f"1.9 × 10^-14 cm^3 molecule^-1 s^-1. {given}How long does it take for the NO mixing ratio to fall to "
            f"half its initial value? Express your answer in s.")


def _integrated_half_life(a0, b0):
    """RK4 on d[NO]/dt = -k[NO][O3], [O3] = [O3]_0 - ([NO]_0 - [NO]); linear interpolation at a0/2."""
    dt = math.log(2) / (KPPB * b0) / 4000
    f = lambda a: -KPPB * a * (b0 - a0 + a)
    t, a = 0.0, a0
    while True:
        k1 = f(a); k2 = f(a + dt * k1 / 2); k3 = f(a + dt * k2 / 2); k4 = f(a + dt * k3)
        new = a + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if new <= a0 / 2:
            return t + dt * (a - a0 / 2) / (a - new)
        t, a = t + dt, new


def check() -> str:
    max_in, min_out, max_err = 0.0, math.inf, 0.0
    for o3 in range(30, 65, 5):
        for no in (0.05, 0.1, 0.2, 0.3):
            p = {"o3": o3, "no": no}
            max_err = max(max_err, gap(_integrated_half_life(no, o3), correct(p)))
            max_in = max(max_in, gap(correct(p), shortcut(p)))
        for ratio in (1.2, 1.3, 1.4, 1.5):
            for no in {rnd(o3 / ratio), rnd(o3 / (ratio + 0.049))}:
                p = {"o3": o3, "no": no}
                max_err = max(max_err, gap(_integrated_half_life(no, o3), correct(p)))
                min_out = min(min_out, gap(correct(p), shortcut(p)))
    assert max_err < 1e-4 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    k_jpl = 3.0e-12 * math.exp(-1500 / 298.15)
    assert abs(k_jpl / K - 1) < 0.05, k_jpl   # 1.96e-14 from the JPL expression; 3 % from the text value
    return (f"closed form vs RK4 max rel err {max_err:.1e}; inside max gap {max_in:.4f}; outside min gap "
            f"{min_out:.3f}; JPL k(298) {k_jpl:.3g}")


TEMPLATE = Template("pseudo_first_order", "atmospheric_chemistry",
                    "pseudo-first-order decay -> mixed second-order kinetics when the partner is not in excess",
                    "s", base, trigger, text, correct, shortcut)
