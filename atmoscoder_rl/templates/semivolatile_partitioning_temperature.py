"""Room-temperature saturation concentration used as-is -> Clausius-Clapeyron-adjusted saturation
concentration at the ambient temperature (absorptive gas-particle partitioning)
(spec: MECHANISMS.md; extended set, 2026-10-01).

Parent's method: a semivolatile organic compound partitions between the gas and particle phases
over an absorbing organic-aerosol phase of mass concentration C_OA; the particle-phase fraction is
xi = 1/(1 + C*/C_OA) (Pankow 1994 absorptive partitioning), using the compound's effective
saturation concentration C* as stated, measured at a reference temperature of 298 K.
Where it fails: C* is itself strongly temperature-dependent. At the ambient temperature T it must be
adjusted from the 298 K reference value by the Clausius-Clapeyron relation
C*(T) = C*(298 K) (298/T) exp[(Delta H_vap/R)(1/298 - 1/T)]
(Donahue, Robinson, Stanier & Pandis 2006, "Coupled partitioning, dilution, and chemical aging of
semivolatile organics", Environ. Sci. Technol. 40, 2635, eq. 6). Using the as-stated 298 K value of
C* regardless of the ambient temperature silently drops this adjustment.
Fair from the text: every arm states C_OA, the total (gas + particle) concentration C_total, C* at
298 K and the enthalpy of vaporization Delta H_vap; R = 8.314 J mol^-1 K^-1 is universal and used
only by the replacement, so it is given only in the "given" sentence. Only the ambient air
temperature changes between the arms.
Ratio design: the inside/outside gap depends only on Delta H_vap, on T, and on the ratio
C*(298 K)/C_OA -- C_OA only rescales C*, and C_total only rescales the answer, so neither affects
the gap. Delta H_vap = 55 kJ/mol, an ordinary value for semivolatile organics, is fixed; C* is then
sampled as C*(298 K) = ratio x C_OA with ratio = 0.27-0.36 (checked below).
Note on range: letting Delta H_vap itself vary 50-70 kJ/mol (the spec's suggestion) was tried first,
sampling C* and C_OA independently; no ratio range satisfies the inside bound at Delta H_vap = 70
kJ/mol and the outside bound at Delta H_vap = 50 kJ/mol simultaneously (the admissible ratio window
shrinks and shifts with Delta H_vap: [0.222, 0.265] at 70 kJ/mol vs [0.269, 0.432] at 50 kJ/mol, no
overlap-safe intersection across the full range -- MECHANISMS.md rule 2, do not rely on rejection to
carve the regime). Fixing Delta H_vap removes that conflict.
Inside: T = 297.5-298.5 K, within half a kelvin of the 298 K reference; at ratio = 0.27-0.36 the
particle-phase mass concentration from the as-stated C* is within 0.96 % of the properly adjusted
one.
Outside: T = 260-272 K, an ordinary cold/winter air temperature; C* at that temperature is well
below its 298 K value (d ln C*/dT ~ Delta H_vap/(R T^2) ~ 7-8 %/K here), so the particle-phase mass
concentration is underestimated by 18-25 % at every sampled ratio and temperature.

Scenario design (2026-10-01 review): the compound is kept a trace species, C_total = 0.1-0.5 ug/m^3
against C_OA = 5-15 ug/m^3, so the self-consistent partitioning equation
C_p = C_total/(1 + C*/(C_OA + C_p)) -- which accounts for the compound's own condensed mass C_p
diluting C_OA, rather than assuming it is negligible -- agrees with the as-used formula
xi = 1/(1 + C*/C_OA) to within 0.40 % across the outside grid (checked below). C_p itself is not
always tiny next to C_OA at the smallest sampled C_OA (up to ~10 % at C_OA = 5 ug/m^3, C_total =
0.5 ug/m^3), but by the time T has dropped enough to be in the outside window, C*/C_OA is itself
small, so the second-order correction from adding C_p to the denominator stays well under the trap
gap. An earlier draft sampled C_total = 10-40 ug/m^3, comparable to or larger than C_OA; there the
two formulas disagree by enough to eat most of the trap gap, so the scenario was not internally
self-consistent.

Conventions: dropping the (298/T) prefactor that precedes the exponential (a common textbook
simplification, since the exponential term dominates) changes the particle-phase mass concentration
by under 0.40 % across the outside grid -- checked below.
Answer: the particle-phase mass concentration in ng/m^3 (not ug/m^3: at C_total = 0.1-0.5 ug/m^3 the
smallest keyed value would be under 0.1 ug/m^3, too close to the grader's round(x, 2) floor -- as in
`drop_ventilation`, the unit is scaled down so the smallest key, ~73 ng/m^3, survives rounding
comfortably). No CONVERTED entry exists for ng/m^3, so the commensurate-unit check is skipped for
this template, same as for any other unit without one (see check_templates.py).

Independent review (2026-10-01): C_total (10-40 ug/m^3) was comparable to or larger than C_OA
(5-15 ug/m^3), so the compound's own particle-phase mass was not negligible next to the absorbing
phase and the standard formula xi = 1/(1 + C*/C_OA) -- which assumes it is -- was not actually the
correct physics for the stated scenario; the self-consistent reading collapses most of the trap gap.
Fixed by making the compound trace (C_total = 0.1-0.5 ug/m^3), re-deriving the answer unit as ng/m^3
so the smallest key survives round(x, 2), adding the self-consistent solve to `check()`'s convention
spread (0.40 % across the outside grid, well inside the gap), and widening both margins by narrowing
the inside T window to 297.5-298.5 K and shifting the outside T window colder to 260-272 K (new
bounds: max inside gap 0.96 % vs. the previous 1.94 %, min outside gap 18.47 % vs. the previous
15.8 %).
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

R = 8.314       # J mol^-1 K^-1
T_REF = 298.0   # K
DH = 55.0       # kJ/mol, enthalpy of vaporization (fixed -- see "Ratio design" above)


def base(rng):
    coa = rnd(rng.uniform(5, 15), 2)
    cstar298 = rnd(coa * rng.uniform(0.27, 0.36), 2)
    return {"coa": coa, "ctot": rnd(rng.uniform(0.1, 0.5), 2), "cstar298": cstar298}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"t": rnd(rng.uniform(297.5, 298.5), 3)}
    return {"t": rnd(rng.uniform(260, 272), 3)}


def _cstar(cstar298, t):
    dh = DH * 1000.0
    return cstar298 * (T_REF / t) * math.exp((dh / R) * (1 / T_REF - 1 / t))


def _xi(cstar, coa):
    return 1 / (1 + cstar / coa)


def _self_consistent_cp(cstar298, coa, ctot, t):
    """The particle-phase mass concentration that solves the self-consistent absorptive
    partitioning equation C_p = C_total / (1 + C*/(C_OA + C_p)) -- i.e. without assuming the
    compound's own condensed mass is negligible next to C_OA -- as the positive root of the
    equivalent quadratic C_p^2 + C_p (C_OA + C* - C_total) - C_total C_OA = 0."""
    cstar_t = _cstar(cstar298, t)
    b = coa + cstar_t - ctot
    return (-b + math.sqrt(b * b + 4 * ctot * coa)) / 2


def shortcut(p):
    return _xi(p["cstar298"], p["coa"]) * p["ctot"] * 1000.0            # ng/m^3


def correct(p):
    return _xi(_cstar(p["cstar298"], p["t"]), p["coa"]) * p["ctot"] * 1000.0   # ng/m^3


def text(p, variant):
    given = ("Because the saturation concentration depends on temperature, adjust it from the "
             "reference value using the Clausius-Clapeyron relation: "
             "C*(T) = C*(298 K) x (298/T) x exp[(Delta H_vap/R)(1/298 - 1/T)], with T in kelvin and "
             "R = 8.314 J mol^-1 K^-1. " if variant == "given" else "")
    return (f"A semivolatile organic compound partitions between the gas and particle phases over an "
            f"organic aerosol phase of mass concentration C_OA = {num(p['coa'])} ug m^-3, by absorptive "
            f"gas-particle partitioning. At a reference temperature of 298 K the compound's effective "
            f"saturation concentration is C* = {num(p['cstar298'])} ug m^-3, and its enthalpy of "
            f"vaporization is Delta H_vap = {DH:.0f} kJ mol^-1. The total (gas + particle) "
            f"concentration of the compound is C_total = {num(p['ctot'])} ug m^-3, and the ambient air "
            f"is at a temperature of {num(p['t'])} K. {given}What is the particle-phase mass "
            f"concentration of the compound? Express your answer in ng m^-3.")


TEMPLATE = Template("semivolatile_partitioning_temperature", "atmospheric_aerosols",
                    "298 K saturation concentration used as-is -> Clausius-Clapeyron adjustment to "
                    "the ambient temperature",
                    "ng/m^3", base, trigger, text, correct, shortcut)


def check() -> str:
    """Independent xi recomputation, gap bounds over the sampled ranges, self-consistent-partitioning
    and dropped-prefactor conventions, trace-species sanity, rounding safety."""
    max_in, min_out, max_prefactor, max_self_consistent = 0.0, math.inf, 0.0, 0.0
    max_cp_over_coa, smallest_key = 0.0, math.inf
    for coa in (5.0, 10.0, 15.0):
        for ctot in (0.1, 0.3, 0.5):
            for ratio in (0.27, 0.30, 0.33, 0.36):
                cstar298 = ratio * coa
                p0 = {"coa": coa, "ctot": ctot, "cstar298": cstar298}
                for t in (297.5, 298.0, 298.5):
                    p = {**p0, "t": t}
                    cst = _cstar(cstar298, t)
                    # Independent: recompute xi from the log form directly.
                    xi_indep = 1 / (1 + math.exp(math.log(cst) - math.log(coa)))
                    assert abs(xi_indep / _xi(cst, coa) - 1) < 1e-9
                    max_in = max(max_in, gap(correct(p), shortcut(p)))
                    smallest_key = min(smallest_key, correct(p))
                for t in (260.0, 264.0, 268.0, 272.0):
                    p = {**p0, "t": t}
                    key = correct(p)
                    g = gap(key, shortcut(p))
                    min_out = min(min_out, g)
                    smallest_key = min(smallest_key, key)
                    no_pref = cstar298 * math.exp((DH * 1000 / R) * (1 / T_REF - 1 / t))
                    max_prefactor = max(max_prefactor,
                                         abs(_xi(no_pref, coa) / _xi(_cstar(cstar298, t), coa) - 1))
                    cp_self = _self_consistent_cp(cstar298, coa, ctot, t)
                    max_self_consistent = max(max_self_consistent, abs(cp_self * 1000.0 / key - 1))
                    max_cp_over_coa = max(max_cp_over_coa, cp_self / coa)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP
    assert max_prefactor < 0.05
    assert max_self_consistent < 0.05   # what actually has to be small: the correction to the key
    assert smallest_key > 10.0 and abs(round(smallest_key, 2) / smallest_key - 1) < 0.01
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; "
            f"dropped-prefactor convention {max_prefactor:.4f}; "
            f"self-consistent-partitioning convention {max_self_consistent:.4f}; "
            f"max C_p/C_OA {max_cp_over_coa:.4f} (informational only); "
            f"smallest key {smallest_key:.2f} ng/m^3")
