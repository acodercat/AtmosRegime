"""Melt = absorbed energy / L_f -> the cold content must be satisfied first (extended, hydrology,
2026-10-05).

Parent's method: the snow-water equivalent melted by an energy input Q (per unit area) is
M = Q / (rho_w L_f) -- the energy-balance melt equation, applied with every joule going into melt.
Where it fails: that equation holds for a ripe (isothermal 0 degC) pack. A pack below 0 degC first
has to be warmed to 0 degC; the energy this takes is its cold content, CC = rho_w SWE c_i |T_s|
(Dingman's definition, with T_s the mean snowpack temperature). Only Q - CC is left for melting:
M = (Q - CC)/(rho_w L_f). Meltwater produced at the surface of a cold pack refreezes inside it and
releases its latent heat there, so the bookkeeping is the same however the warming is distributed.
Fair from the text: every arm states the snow-water equivalent, that all of it is frozen, the mean
pack temperature, the total energy absorbed over the warm spell, L_f and the mm <-> kg m^-2
equivalence -- everything the melt equation needs. The specific heat of ice is a handbook constant
used only by the replacement, so it is stated only in the given sentence (rule 3 as reworded
2026-09-27); its spread is checked below.
Inside: T_s = -0.1 to -0.3 degC (a nearly ripe spring pack); CC is at most 1.6 % of Q, so the melt
equation is within 1.6 % of the key.
Outside: T_s is set so that CC/Q = 0.2-0.4, which for SWE = 180-300 mm and Q = 12-18 MJ m^-2 is
T_s = -3.8 to -19 degC -- an ordinary mid-winter pack temperature. The shortcut melt is 25-67 % above
the key and is still a modest fraction of the pack (<= 30 %), so nothing about it looks absurd.
Conventions: c_i = 2.1 kJ kg^-1 K^-1 (the value in the given sentence and in Dingman) vs 2.05 or a
temperature-dependent c_i(T) = 0.185 + 0.00689 T kJ kg^-1 K^-1 (T in K; Fukusako 1990), integrated
from T_s to 0 degC, move the key by < 3 % across the outside grid (checked). L_f = 334 kJ kg^-1 is
stated in all arms. Liquid-water retention affects outflow, not the melt asked for. The heat capacity
of the air in the pores is negligible (< 0.1 % of the ice term).
check(): an independent stepwise simulation adds the energy in small increments, warming the ice
until it reaches 0 degC and melting afterwards.
Reference: Dingman, *Physical Hydrology* (3rd ed.), sec. 5.4 (cold content, ripening, and the
energy-balance melt equation; worked examples of the three melt phases); DeWalle & Rango, *Principles
of Snow Hydrology* (2008), ch. 7. Related: freezing_latent_heat (a latent-heat term omitted from an
atmospheric warming budget) -- different relation and different omitted term (sensible heat of the
solid before a phase change); new cluster "cold content / sensible heat before a phase change".

Independent review (2026-10-05): accepted without change.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

LF, CI = 334.0, 2.1           # kJ/kg, kJ/kg/K


def shortcut(p):
    """Melt equation, all energy into melt: mm w.e."""
    return p["q"] * 1e3 / LF


def correct(p):
    """Satisfy the cold content first, then melt: mm w.e."""
    return (p["q"] * 1e3 - p["swe"] * CI * abs(p["t"])) / LF


def base(rng):
    return {"swe": rnd(rng.uniform(180.0, 300.0)), "q": rnd(rng.uniform(12.0, 18.0))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"t": -rnd(rng.uniform(0.1, 0.3))}
    ratio = rng.uniform(0.2, 0.4)                       # cold content / energy absorbed
    return {"t": -rnd(ratio * base["q"] * 1e3 / (base["swe"] * CI))}


def text(p, variant):
    given = ("Because the pack is below 0 °C, part of the energy must first warm the ice to 0 °C before any net "
             "melt occurs (its cold content); take the specific heat of ice as 2.1 kJ kg^-1 K^-1. "
             if variant == "given" else "")
    return (f"A seasonal snowpack holds {num(p['swe'])} mm of snow-water equivalent, all of it frozen, at a mean "
            f"temperature of {num(p['t'])} °C. During a warm spell the snowpack absorbs a net {num(p['q'])} MJ m^-2 "
            f"of energy in total. Take the latent heat of fusion as 334 kJ kg^-1, and note that 1 mm of water "
            f"equivalent is 1 kg of water per square metre. {given}How much of the snowpack's water equivalent "
            f"has melted by the end of the warm spell? Express your answer in mm.")


TEMPLATE = Template("snowpack_cold_content", "hydrology",
                    "energy-balance melt Q/L_f -> cold content satisfied before melt",
                    "mm", base, trigger, text, correct, shortcut)


def _stepwise(p, ci=lambda t: CI, n=20000):
    """Independent: add the energy in n increments; warm the ice while below 0 degC, then melt."""
    de = p["q"] * 1e3 / n                  # kJ m^-2 per step
    t, ice, melted = p["t"], p["swe"], 0.0  # degC, kg m^-2, kg m^-2
    for _ in range(n):
        e = de
        if t < 0:
            dt = e / (ice * ci(t))
            if t + dt <= 0:
                t += dt
                continue
            e -= -t * ice * ci(t)
            t = 0.0
        melted += e / LF
    return melted


def _ci_fukusako(t_c):
    return 0.185 + 0.00689 * (t_c + 273.15)


def _key_ci(p, ci_mean):
    return (p["q"] * 1e3 - p["swe"] * ci_mean * abs(p["t"])) / LF


def check() -> str:
    max_in, min_out, max_out = 0.0, math.inf, 0.0
    max_err = spread = 0.0
    t_min, t_max, max_frac, min_key = 0.0, -math.inf, 0.0, math.inf
    for swe in (180.0, 240.0, 300.0):
        for q in (12.0, 15.0, 18.0):
            b = {"swe": swe, "q": q}
            cases = [(-0.1, 1), (-0.3, 1)] + [(-r * q * 1e3 / (swe * CI), 0) for r in (0.2, 0.3, 0.4)]
            for t, inside in cases:
                p = {**b, "t": t}
                c, s = correct(p), shortcut(p)
                max_err = max(max_err, gap(c, _stepwise(p)))
                g = gap(c, s)
                min_key = min(min_key, c)
                if inside:
                    max_in = max(max_in, g)
                    continue
                min_out, max_out = min(min_out, g), max(max_out, g)
                t_min, t_max = min(t_min, t), max(t_max, t)
                max_frac = max(max_frac, s / swe)
                # rule 5: other c_i conventions (constant 2.05; Fukusako's c_i(T), integrated)
                fuku_mean = sum(_ci_fukusako(t * (k + 0.5) / 200) for k in range(200)) / 200
                for alt in (_key_ci(p, 2.05), _key_ci(p, fuku_mean)):
                    spread = max(spread, gap(c, alt))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_err < 1e-6, max_err
    assert spread < 0.05, spread
    assert max_frac < 0.35 and min_key > 10, (max_frac, min_key)   # shortcut a modest share; no rounding issue
    # Anchor: Fukusako's c_i at 0 degC is 2.067 kJ/kg/K and at -20 degC 1.929 (tabulated ice ~2.05, ~1.94).
    assert abs(_ci_fukusako(0) - 2.067) < 0.005 and abs(_ci_fukusako(-20) - 1.93) < 0.01
    return (f"stepwise simulation vs closed form {max_err:.1e}; inside max gap {max_in:.4f}; outside gap "
            f"{min_out:.3f}-{max_out:.3f}; outside T {t_min:.1f} to {t_max:.1f} degC; c_i convention spread "
            f"{spread:.3f}; shortcut <= {max_frac:.2f} SWE; min key {min_key:.1f} mm")
