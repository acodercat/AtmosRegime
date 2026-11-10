"""All the rain evaporates -> evaporation stops at saturation (new mechanism, no source trap).

Added 2026-09-27, after the A/B registration of 2026-09-26: no A/B label was predicted for it
before screening. New cluster, "saturation limit" (no other template caps a process at
saturation).

Parent's method: rain supplying M kilograms of water per kilogram of air evaporates completely;
the isobaric energy balance gives a cooling Delta T = L_v M / c_p.
Where it fails: the air can take up only as much vapour as brings it to saturation. When M exceeds
that uptake, evaporation stops once the air is saturated, and it cools only to its isobaric
wet-bulb temperature T_w: c_p (T - T_w) = L_v [r_s(T_w, p) - r]. The cooling is therefore
min(L_v M / c_p, T - T_w).
Fair from the text: every arm states p, T, relative humidity, M, L_v and c_p; the only other
ingredient is a saturation-vapour-pressure formula, which only the replacement uses and which is
left to the solver (rule 3, reworded 2026-09-27): Bolton, Alduchov-Eskridge, Tetens and
Goff-Gratch move the key by at most 0.11 %. Outside, the stated M would cool the air by 19-30 K,
far below the dew point; a solver checking the vapour budget sees that it cannot all evaporate.
Inside: M = 0.5-1.8 g/kg is at most 0.67 of the uptake anywhere on the grid (uptake 2.7-4.9 g/kg),
so all of it evaporates and the parent's answer is exact.
Conventions (outside grid, all within 5 %, measured in check()): relative humidity as r/r_s
instead of e/e_s, specific humidity instead of mixing ratio, moist c_p, the heat capacity of the
unevaporated liquid (drops starting at the air temperature) all within 2.4 %, and Stull's (2011) empirical
sea-level wet-bulb formula, which ignores the stated p, L_v and c_p. Stull's fit is why the grid is
22-26 degC and 1000-1020 hPa: at 30-35 degC or below ~950 hPa it departs from the energy balance by
5-10 %, and at 28 degC and 980 hPa by 4.7 %, too close to the tolerance; on this grid it stays
within 3.5 % (narrowed 2026-09-27 after the blind solve, as was ice_saturation). L_v = 2.45e6 J/kg (the value at ~20 degC) rather than 2.5e6, for the same reason.
Two-sided lure, recorded (review 2026-09-27): "evaporating rain cools the air to its wet-bulb
temperature" is itself textbook lore, so a solver who answers the wet-bulb depression by habit
passes the outside arm without recognising anything — and fails the inside arm (depression >= 6.6 K
against inside keys <= 4.4 K); screening criterion (a) and the conditioning of (b) on inside-correct
pairs are what keep such a solver from reading as a recogniser. A solver who checks the vapour
budget at the *initial* temperature (r_s(T) - r >= M in most outside items) concludes that all the
rain evaporates and returns the shortcut exactly; that counts as a capture — the check ignored the
cooling, which is the point of the template.
Wording: "falls into a layer ... Over the shower, the rain makes M available" rather than a closed
parcel holding M of rain (8-12 g/kg held at once would be 5-10x a heavy-rain water content); "warm,
dry" was dropped.
Not a convention but a numerics error, recorded: a single linearised (one Newton step from T)
psychrometric solve under-predicts the depression by 12-22 %; it fails in the direction
opposite to the shortcut.
Answer: a cooling in K (a temperature difference, rule 6), 1.2-11.9 K. The grader converts degC to K with
the offset, which is wrong for a difference, so every arm asks for K (as mixed_layer_response).
References: Rogers & Yau, A Short Course in Cloud Physics, ch. 2 (isobaric wet-bulb temperature);
Stull (2011), J. Appl. Meteor. Climatol. 50, 2267, doi:10.1175/JAMC-D-11-0143.1 (20 degC, 50 % ->
T_w = 13.7 degC at 1013.25 hPa, the anchor).
"""
from __future__ import annotations

import math
from itertools import product

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

LV, CP, EPS = 2.45e6, 1005.0, 0.622


def base(rng):
    return {"p": float(rng.choice(range(1000, 1030, 10))), "t": float(rng.randint(22, 26)),
            "rh": float(rng.choice(range(25, 55, 5)))}


def trigger(rng, regime, base):
    return {"m": rnd(rng.uniform(0.5, 1.8)) if regime == "inside" else rnd(rng.uniform(8, 12))}


def _es(t):
    """Saturation vapour pressure over water, hPa (Bolton 1980), t in degC."""
    return 6.112 * math.exp(17.67 * t / (t + 243.5))


def _depression(p):
    """T - T_w, K: Newton on c_p x = L_v [r_s(T - x) - r]."""
    e = p["rh"] / 100 * _es(p["t"])
    r = EPS * e / (p["p"] - e)
    x = 0.0
    for _ in range(50):
        tw = p["t"] - x
        es = _es(tw)
        rs = EPS * es / (p["p"] - es)
        drs = EPS * p["p"] / (p["p"] - es) ** 2 * es * 17.67 * 243.5 / (tw + 243.5) ** 2   # d r_s / d T_w
        f = CP * x - LV * (rs - r)
        x -= f / (CP + LV * drs)
    return x


def shortcut(p):
    return LV * p["m"] * 1e-3 / CP                                  # K


def correct(p):
    return min(shortcut(p), _depression(p))


def text(p, variant):
    given = ("Because the air saturates before all of the rain has evaporated, evaporation stops at "
             "saturation and the rest of the water stays liquid: the air cools to the temperature T_w at "
             "which c_p(T − T_w) = L_v[r_s(T_w) − r], where r is the initial water-vapour mixing ratio "
             "and r_s(T_w) the saturation mixing ratio at T_w and the given pressure. "
             if variant == "given" else "")
    return (f"Rain from a passing shower falls into a layer of air near the ground. The air is "
            f"at a pressure of {num(p['p'])} hPa, a temperature of {num(p['t'])} °C and a relative humidity "
            f"of {num(p['rh'])}%. Over the shower, the rain makes {num(p['m'])} g of liquid water per "
            f"kilogram of air available to evaporate into the layer. Treat the air and the water it "
            f"receives as one system at constant pressure that exchanges no heat with its surroundings, "
            f"and take "
            f"L_v = 2.45 × 10^6 J kg^-1 and c_p = 1005 J kg^-1 K^-1. {given}By how much has the air "
            f"temperature fallen once evaporation has stopped? Express your answer in K.")


TEMPLATE = Template("evaporative_cooling_limit", "atmospheric_thermodynamics",
                    "complete evaporation of the supplied rain -> evaporation limited by saturation "
                    "(isobaric wet-bulb cooling)",
                    "K", base, trigger, text, correct, shortcut)


def _es_gg(t):
    """Goff-Gratch over water, hPa."""
    tk, ts = t + 273.15, 373.15
    log = (-7.90298 * (ts / tk - 1) + 5.02808 * math.log10(ts / tk)
           - 1.3816e-7 * (10 ** (11.344 * (1 - tk / ts)) - 1)
           + 8.1328e-3 * (10 ** (-3.49149 * (ts / tk - 1)) - 1) + math.log10(1013.246))
    return 10 ** log


def _bisect(p, m, es=_es, cp=CP, rh_by_r=False, specific=False, cw=0.0):
    """Cooling by bisection on the vapour budget: cool by x only if the air can still take up the
    water that x needs and the rain still has it. Independent of `correct`'s Newton on T_w."""
    def sat(t):
        e_s = es(t)
        r_s = EPS * e_s / (p["p"] - e_s)
        return r_s / (1 + r_s) if specific else r_s
    if rh_by_r:
        q0 = p["rh"] / 100 * sat(p["t"])
    else:
        e = p["rh"] / 100 * es(p["t"])
        r = EPS * e / (p["p"] - e)
        q0 = r / (1 + r) if specific else r
    lo, hi = 0.0, 40.0
    for _ in range(80):
        x = (lo + hi) / 2
        need = (cp + cw * m) * x / LV
        lo, hi = (x, hi) if need < min(m, sat(p["t"] - x) - q0) else (lo, x)
    return (lo + hi) / 2


def _stull(t, rh):
    """Stull (2011) empirical wet-bulb temperature at 1013.25 hPa, degC."""
    return (t * math.atan(0.151977 * (rh + 8.313659) ** 0.5) + math.atan(t + rh) - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035)


def check() -> str:
    """Bisection on the vapour budget, gap bounds over the grid, convention spread, Stull's anchor."""
    worst_indep = worst_es = worst_conv = worst_stull = 0.0
    uptake_min, uptake_max, keys = math.inf, 0.0, []
    for press, temp, rh in product(range(1000, 1030, 10), range(22, 27), range(25, 55, 5)):
        q = {"p": press, "t": temp, "rh": rh}
        for m in (0.5, 1.8):                                   # inside corners: all evaporates
            pin = {**q, "m": m}
            assert gap(correct(pin), shortcut(pin)) <= INSIDE_MAX_GAP
            worst_indep = max(worst_indep, gap(_bisect(pin, m * 1e-3), correct(pin)))
            keys.append(correct(pin))
        for m in (7.5, 12.0):                                  # outside corners (rnd bounds)
            pout = {**q, "m": m}
            key = correct(pout)
            keys.append(key)
            assert gap(key, shortcut(pout)) >= OUTSIDE_MIN_GAP
            worst_indep = max(worst_indep, gap(_bisect(pout, m * 1e-3), key))
            for es in (_es_gg, lambda t: 6.1094 * math.exp(17.625 * t / (t + 243.04)),
                       lambda t: 6.1078 * math.exp(17.27 * t / (t + 237.3))):
                worst_es = max(worst_es, gap(_bisect(pout, m * 1e-3, es=es), key))
            for kw in ({"rh_by_r": True}, {"specific": True}, {"cp": CP * (1 + 0.87 * 0.015)},
                       {"cw": 4186.0}, {"rh_by_r": True, "specific": True}):
                worst_conv = max(worst_conv, gap(_bisect(pout, m * 1e-3, **kw), key))
            worst_stull = max(worst_stull, gap(temp - _stull(temp, rh), key))
        uptake = _depression(q) * CP / LV * 1e3                     # g/kg the air can take up
        uptake_min, uptake_max = min(uptake_min, uptake), max(uptake_max, uptake)
    assert worst_indep < 1e-6 and worst_es < .01 and worst_conv < .05 and worst_stull < .035
    assert 1.8 / uptake_min < .75 and 7.5 / uptake_max > 1.3
    anchor = 20 - _depression({"p": 1013.25, "t": 20, "rh": 50})
    assert abs(anchor - 13.7) < .2, anchor
    return (f"independent {worst_indep:.1e}, e_s formulas {worst_es:.4f}, conventions {worst_conv:.4f}, "
            f"Stull {worst_stull:.4f}; uptake {uptake_min:.2f}-{uptake_max:.2f} g/kg; "
            f"keys {min(keys):.2f}-{max(keys):.2f} K; T_w(20 degC, 50 %) = {anchor:.2f} degC")
