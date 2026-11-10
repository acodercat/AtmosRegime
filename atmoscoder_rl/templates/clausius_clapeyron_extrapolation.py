"""Linear local-rate extrapolation -> exponential Clausius-Clapeyron growth of saturation vapour
pressure (extended, 2026-10-01).

Parent's method: given the saturation vapour pressure e_s0 at a reference temperature T0 and its
local rate of increase there, extrapolate linearly, e_s(T0+dT) = e_s0 + rate * dT.
Where it fails: saturation vapour pressure grows exponentially with temperature (the
Clausius-Clapeyron relation), not linearly; the stated rate is only the tangent to that curve at
T0, and the tangent line pulls away from the true exponential as dT grows. The true value is taken
here to be the standard Magnus/Bolton formula e_s(T) = 6.112 exp[17.67 T/(T+243.5)] hPa (T in °C) --
the same formula already used as the exact saturation-vapour-pressure reference in `kelvin_curvature`
-- evaluated at T0+dT.
Fair from the text: every arm states the reference temperature T0, the saturation vapour pressure
there (e_s0) and the local rate (both derived from the Bolton formula and rounded, so they are
exactly what a solver would read off a real table or chart at T0), and the temperature rise dT; only
dT differs between arms. The water-specific constants the *replacement* needs beyond e_s0, T0 and
rate -- the latent-heat-to-gas-constant ratio L/R_v -- are universal/handbook (not tied to this
problem's data) and so, following the kelvin_curvature precedent, are stated only in the given
sentence, not in inside/outside (stating them there would name the mechanism). T0 is needed by the
correct method (through 1/T0 - 1/T) as well as by the narrative, so it is given in every arm.
Inside: dT = 1-2 K, T0 = -5 to 20 °C. The tangent line and the true curve agree to within 1.0 %.
Outside: dT = 14-20 K (an ordinary morning-to-afternoon warming, e.g. 10 °C to a 28 °C afternoon is
dT = 18 K, inside this range), same T0 range; the linear extrapolation misses the true e_s by
18-38 % (worst case, smallest gap, at the warmest T0 and the smallest dT in this range, where the
curve's own curvature has had the least room to act -- see Conventions).
Keying choice: the key is e_s itself (hPa), not the increase above e_s0 or a multiplicative factor.
An earlier sibling template (`cc_scaling`, MECHANISMS.md, dropped 2026-09-26) keyed a *factor*
(close to 1 inside), which made the 5 % tolerance on the factor equivalent to 50-85 % on the actual
increase and graded the control arm far more leniently than the trap arm. Keying the absolute e_s
(4.6-74.0 hPa over the sampled grid, never close to zero) avoids that: both arms are graded by the
same 5 % standard on the same kind of quantity.
Conventions: the true standard Clausius-Clapeyron form with constant L, e_s(T) = e_s0 exp[(L/R_v)
(1/T0 - 1/T)] (T, T0 in kelvin), is the form given to the solver; with the standard L/R_v = 5417 K
(L = 2.5e6 J/kg, R_v = 461.5 J/(kg K)) it agrees with the Bolton key to within 2.9 % over the whole
outside grid (checked below) -- comfortably inside the 5 % tolerance, so a solver who recalls the
textbook CC form is credited alongside one who recalls Bolton/Magnus directly. The naive tangent-
matched exponential e_s0 exp[(rate/e_s0) dT] (replacing 1/T0-1/T by its own linearisation in dT) is
*not* a safe convention -- it drifts up to 12 % from the key over the same grid -- which is exactly
why the given sentence states the physical 1/T0-1/T form and not that shortcut-looking exponential.
The Tetens (1930) and WMO formulas for e_s agree with Bolton to within 0.4 % over the outside grid.
Anchor: e_s(0 °C) = 6.11 hPa is the standard reference value quoted in every textbook treatment of
Clausius-Clapeyron (e.g. Wallace & Hobbs, *Atmospheric Science*, 2nd ed., sec. 3.3); checked below.
Reference: Bolton, D. (1980), *Mon. Wea. Rev.* 108, 1046 (the e_s formula used throughout this
codebase, see `kelvin_curvature`); Rogers & Yau, *A Short Course in Cloud Physics* (3rd ed.), sec.
2.5 (Clausius-Clapeyron with constant L). Related: `kelvin_curvature` (shares the Bolton e_s formula
for a different purpose), `co2_forcing_log` (the same tangent-vs-exact design, opposite curvature).

Independent review (2026-10-01): accepted. Spread over the outside grid: Goff-Gratch -0.2 to 0 %, Tetens -0.3 to
+0.1 %, constant L/Rv from e_s0 +0.2 to +3.0 %, L/Rv inferred from the stated rate then exact CC +0.4 to +1.3 %; the
stated rate is the Bolton derivative. Linearised exponent e_s0 exp(rate dT/e_s0): +4.5 to +12 %. Dropped "at that
temperature" after the rate (a locality cue that does not move the answer, rule 4).
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

L_OVER_RV = 5417.0   # K; L = 2.5e6 J/kg, R_v = 461.5 J/(kg K) -- the standard textbook ratio


def es(t):
    """Bolton (1980) saturation vapour pressure, hPa; t in degC."""
    return 6.112 * math.exp(17.67 * t / (t + 243.5))


def des_dt(t):
    """Analytic derivative of es() at t, hPa/K."""
    return es(t) * 17.67 * 243.5 / (t + 243.5) ** 2


def base(rng):
    t0 = float(rng.randint(-5, 20))
    return {"t0": t0, "es0": rnd(es(t0), 3), "rate": rnd(des_dt(t0), 3)}


def trigger(rng, regime, base):
    dt = rng.uniform(1, 2) if regime == "inside" else rng.uniform(14, 20)
    return {"dt": rnd(dt)}


def shortcut(p):
    """Linear extrapolation from the stated reference value and local rate. hPa."""
    return p["es0"] + p["rate"] * p["dt"]


def correct(p):
    """True (Bolton) saturation vapour pressure at T0 + dT. hPa."""
    return es(p["t0"] + p["dt"])


def text(p, variant):
    given = ("Because this quantity grows exponentially with temperature rather than linearly (the "
             "Clausius–Clapeyron relation), use e_s(T) = e_s0 · exp[(L/R_v)(1/T0 − 1/T)], with T0 and T in "
             "kelvin and L/R_v ≈ 5417 K. " if variant == "given" else "")
    return (f"At a temperature of {num(p['t0'])} °C, the saturation vapour pressure of water is "
            f"{num(p['es0'])} hPa, increasing by about {num(p['rate'])} hPa per additional kelvin. "
            f"{given}If the temperature rises by {num(p['dt'])} K, what is the saturation vapour "
            f"pressure then? Express your answer in hPa.")


TEMPLATE = Template("clausius_clapeyron_extrapolation", "atmospheric_thermodynamics",
                    "linear local-rate extrapolation -> exponential Clausius-Clapeyron growth of e_s",
                    "hPa", base, trigger, text, correct, shortcut)


def check() -> str:
    max_in = min_out = None
    max_cc_spread = max_tetens_spread = max_wmo_spread = 0.
    min_key, max_key = 1e9, 0.
    for t0 in range(-5, 21):
        es0, rate = rnd(es(t0), 3), rnd(des_dt(t0), 3)
        # Independent: central finite-difference derivative, not the analytic formula in des_dt.
        h = 1e-4
        fd = (es(t0 + h) - es(t0 - h)) / (2 * h)
        assert math.isclose(fd, des_dt(t0), rel_tol=1e-5)
        for dt_raw in (1.0, 1.3, 1.6, 2.0):
            dt = rnd(dt_raw)
            p = {"t0": t0, "es0": es0, "rate": rate, "dt": dt}
            g = gap(correct(p), shortcut(p))
            max_in = g if max_in is None else max(max_in, g)
        for dt_raw in (14.0, 15.5, 17.0, 18.5, 20.0):
            dt = rnd(dt_raw)
            p = {"t0": t0, "es0": es0, "rate": rate, "dt": dt}
            key = correct(p)
            min_key, max_key = min(min_key, key), max(max_key, key)
            g = gap(key, shortcut(p))
            min_out = g if min_out is None else min(min_out, g)
            t0k, tk = t0 + 273.15, t0 + dt + 273.15
            cc = es0 * math.exp(L_OVER_RV * (1 / t0k - 1 / tk))
            max_cc_spread = max(max_cc_spread, gap(key, cc))
            tet = 6.1078 * math.exp(17.27 * (t0 + dt) / (t0 + dt + 237.3))
            max_tetens_spread = max(max_tetens_spread, gap(key, tet))
            wmo = 6.112 * math.exp(17.62 * (t0 + dt) / (t0 + dt + 243.12))
            max_wmo_spread = max(max_wmo_spread, gap(key, wmo))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP
    assert max_cc_spread < 0.05 and max_tetens_spread < 0.05 and max_wmo_spread < 0.05
    assert min_key > 1.0
    assert abs(es(0) - 6.11) < 0.01   # standard freezing-point anchor
    return (f"finite-difference derivative agrees; max inside gap {max_in:.4f}, min outside gap "
            f"{min_out:.3f}; CC(L/Rv) spread {max_cc_spread:.3f}, Tetens spread {max_tetens_spread:.4f}, "
            f"WMO spread {max_wmo_spread:.4f}; key range {min_key:.1f}-{max_key:.1f} hPa; "
            f"es(0C) = {es(0):.3f} hPa anchor")
