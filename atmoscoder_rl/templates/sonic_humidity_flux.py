"""Sensible heat flux from the sonic-temperature covariance as if it were air temperature -> with the
humidity term of sonic thermometry (extended, observation, 2026-10-05).

Parent's method: the eddy-covariance sensible heat flux is H = rho c_p w'T', and the sonic anemometer's
temperature signal supplies T, so H = rho c_p w'T_s' with rho = p/(R_d T).
Where it fails: a sonic thermometer measures the speed of sound, which depends on humidity as well as on
temperature: T_s = T(1 + 0.51 q) (Kaimal & Gaynor 1991; 0.51 = 0.608 + 0.87 - 0.97 from the gas constant
and the heat capacities of moist air; 0.50-0.51 depending on the vapour heat capacities used). The covariance it reports is therefore
w'T_s' = w'T' + 0.51 T w'q', and H = rho c_p (w'T_s' - 0.51 T w'q') (Schotanus et al. 1983), with
rho w'q' = E the evaporation flux. Over a dry surface the humidity term is negligible; over a well-watered
crop at midday (E = 0.50-0.53 mm/h, LE ~ 340-360 W m^-2) it is 21-23 W m^-2, 17-23 % of the true H.
Fair from the text: every arm states air temperature and pressure (the shortcut needs both for rho), R_d
and c_p, the sonic covariance and the evaporation rate from the field. The evaporation rate is the trigger,
the only difference between arms, and is ordinary station output in mm/h; the text calls the instrument a
sonic anemometer-thermometer and its output "its temperature signal" -- what it is -- without saying that
humidity affects it. Instrument-internal crosswind correction is not raised (modern sonics apply it).
Inside: E = 0.010-0.030 mm/h (dry surface): humidity term <= 1.3 W m^-2, gap <= 1.1 %.
Outside: E = 0.50-0.53 mm/h (air 25-28 deg C): gap 17-23 %; the shortcut (120-145 W m^-2) is an ordinary midday flux.
The window is narrow on purpose: see Conventions.
Conventions (rule 5): "sonic temperature is virtual temperature" (coefficient 0.61 instead of 0.51) is a
common textbook simplification; it shifts the key by 0.1/0.51 of the humidity term, <= 4.5 % on this grid
(asserted; this caps the outside gap at ~23 %). The Kaimal-Gaynor form 0.32 e/p (= 0.514 q) and using the dry-air
density for w'q' move the key < 1 %.
Second wrong method (rule 11): applying the humidity term with the wrong sign is >= 34 % off (asserted
> 10 %).
check(): an independent thermodynamic route -- T_s defined from the speed of sound in moist air,
c^2 = gamma_m R_m T, with R_m, c_pm and c_vm from the gas constants and heat capacities of dry air and vapour,
linearised by finite differences about a mean q of 0.015, and w'T' solved from w'T_s' -- agrees with the
Schotanus key within 1.5 % (it also keeps the small (1 + 0.51 q) factor the textbook form drops); anchor:
the moist-air coefficient from the gas constants and heat capacities, 0.50-0.51.
Reference: Schotanus, Nieuwstadt & de Bruin (1983), Boundary-Layer Meteorol. 26, 81; Kaimal & Gaynor
(1991), Boundary-Layer Meteorol. 56, 401; Kaimal & Finnigan, *Atmospheric Boundary Layer Flows* (1994),
ch. 6 (sonic thermometry); Foken, *Micrometeorology* (2nd ed., 2017), ch. 4 (corrections of eddy-covariance
fluxes). Section numbers not verified.
Cluster: new -- differs from condensate_loading (nearest: a humidity / water term in a temperature-like
quantity) because there the parent is parcel buoyancy from virtual temperature and the missing ingredient
is the weight of liquid water, whereas here the parent is the eddy-covariance heat flux and the missing
ingredient is the humidity dependence of the speed of sound the instrument converts to temperature; no
other template corrects a measured covariance for its sensor's physics.
Independent review (2026-10-05): the 0.61 (T_s = T_v) route was 4.2-4.8 % off, too close to the tolerance;
air temperature narrowed to 25-28 deg C and outside E capped at 0.53 mm/h, and check() now asserts that route
stays <= 4.5 %. Cluster: new.
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

EXTENDED = True

R_D, CP = 287.0, 1005.0
SONIC = 0.51


def _rho(p):
    return p["p"] * 100 / (R_D * (p["t"] + 273.15))


def shortcut(p):
    return _rho(p) * CP * p["cov"]


def _humidity_term(p, coef=SONIC, rho_for_q=None):
    flux = p["e"] / 3600                                    # mm/h of water -> kg m^-2 s^-1
    wq = flux / (rho_for_q if rho_for_q is not None else _rho(p))
    return _rho(p) * CP * coef * (p["t"] + 273.15) * wq    # W m^-2


def correct(p):
    return shortcut(p) - _humidity_term(p)


def base(rng):
    return {"t": float(rng.randint(25, 28)), "p": float(rng.choice(range(980, 1030, 10))),
            "cov": rnd(rng.uniform(0.108, 0.120), 3)}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"e": rnd(rng.uniform(0.010, 0.030))}
    return {"e": rnd(rng.uniform(0.50, 0.53))}


def text(p, variant):
    given = ("A sonic thermometer infers temperature from the speed of sound, which also rises with humidity: its "
             "signal is T_s = T(1 + 0.51 q), q the specific humidity, so w'T' = w'T_s' − 0.51 T w'q', where w'q' is "
             "the evaporation flux (kg m^-2 s^-1) divided by the air density. " if variant == "given" else "")
    return (f"At a surface energy-balance station over a crop field, the air temperature is {num(p['t'])} °C and the "
            f"pressure is {num(p['p'])} hPa. The station's sonic anemometer–thermometer reports a covariance between "
            f"the vertical wind and its temperature signal of {num(p['cov'])} K m s^-1, and its hygrometer gives an "
            f"evaporation rate from the field of {num(p['e'])} mm of water per hour. Take R_d = 287 J kg^-1 K^-1 and "
            f"c_p = 1005 J kg^-1 K^-1 for the air. {given}What is the sensible heat flux from the field? Express "
            f"your answer in W m^-2.")


TEMPLATE = Template("sonic_humidity_flux", "observation_and_modeling",
                    "sensible heat flux from the sonic-temperature covariance -> minus its humidity term (Schotanus)",
                    "W/m^2", base, trigger, text, correct, shortcut)


# ---- check --------------------------------------------------------------------------------------------
def _sonic_t(t_k, q):
    """Sonic temperature from the speed of sound in moist air, c^2 = gamma_m R_m T, expressed as the dry-air
    temperature with the same speed of sound: T_s = gamma_m R_m T / (gamma_d R_d)."""
    rv, cpv, cvv = 461.5, 1875.0, 1414.0
    cvd = CP - R_D
    rm = (1 - q) * R_D + q * rv
    cpm = (1 - q) * CP + q * cpv
    cvm = (1 - q) * cvd + q * cvv
    return (cpm / cvm) * rm * t_k / ((CP / cvd) * R_D)


def _thermo_route(p, q=0.015):
    t_k = p["t"] + 273.15
    dt, dq = 1e-3, 1e-6
    dts_dt = (_sonic_t(t_k + dt, q) - _sonic_t(t_k - dt, q)) / (2 * dt)
    dts_dq = (_sonic_t(t_k, q + dq) - _sonic_t(t_k, q - dq)) / (2 * dq)
    wq = p["e"] / 3600 / _rho(p)
    wt = (p["cov"] - dts_dq * wq) / dts_dt
    return _rho(p) * CP * wt


def _virtual(p):
    return shortcut(p) - _humidity_term(p, coef=0.61)                       # T_s taken as virtual temperature


def _conventions(p):
    return (shortcut(p) - _humidity_term(p, coef=0.61),                     # T_s taken as virtual temperature
            shortcut(p) - _humidity_term(p, coef=0.32 / 0.622),             # Kaimal-Gaynor 0.32 e/p
            shortcut(p) - _humidity_term(p, rho_for_q=_rho(p) * 0.985))     # dry-air density for w'q'


def check() -> str:
    max_in = max_err = spread = virt = 0.0
    min_out, max_out, min_wrong = math.inf, 0.0, math.inf
    keys = [math.inf, 0.0]
    for t in (25.0, 28.0):
        for pr in (980.0, 1020.0):
            for cov in (0.108, 0.120):
                for e in (0.010, 0.030):
                    p = {"t": t, "p": pr, "cov": cov, "e": e}
                    max_in = max(max_in, gap(correct(p), shortcut(p)))
                    max_err = max(max_err, gap(correct(p), _thermo_route(p)))
                for e in (0.50, 0.53):
                    p = {"t": t, "p": pr, "cov": cov, "e": e}
                    c = correct(p)
                    g = gap(c, shortcut(p))
                    min_out, max_out = min(min_out, g), max(max_out, g)
                    max_err = max(max_err, gap(c, _thermo_route(p)))
                    spread = max(spread, *(gap(c, v) for v in _conventions(p)))
                    virt = max(virt, gap(c, _virtual(p)))
                    min_wrong = min(min_wrong, gap(c, shortcut(p) + _humidity_term(p)))
                    keys = [min(keys[0], c), max(keys[1], c)]
    rng = random.Random("sonic_humidity_flux:rounded")
    for _ in range(2000):
        pin, pout = pair(TEMPLATE, rng)
        c = correct(pout)
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        g = gap(c, shortcut(pout))
        min_out, max_out = min(min_out, g), max(max_out, g)
        spread = max(spread, *(gap(c, v) for v in _conventions(pout)))
        virt = max(virt, gap(c, _virtual(pout)))
        min_wrong = min(min_wrong, gap(c, shortcut(pout) + _humidity_term(pout)))
        keys = [min(keys[0], c), max(keys[1], c, correct(pin))]
    assert max_err < 0.015, max_err
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert spread < 0.05, spread
    assert virt <= 0.045, virt                            # review margin for the T_s = T_v convention
    assert min_wrong > 0.10, min_wrong
    assert keys[0] > 50, keys
    coef = (_sonic_t(300.0, 1e-4) / 300.0 - 1) / 1e-4       # anchor: T_s = T (1 + 0.51 q)
    assert abs(coef - 0.51) < 0.01, coef
    return (f"speed-of-sound route vs Schotanus {max_err:.4f}; inside max gap {max_in:.4f}; outside gap "
            f"{min_out:.3f}-{max_out:.3f}; 0.61 / 0.514 / dry-density spread {spread:.3f} (0.61: {virt:.3f}); sign-error route >= "
            f"{min_wrong:.2f} off; key {keys[0]:.0f}-{keys[1]:.0f} W m^-2; dT_s/dq / T = {coef:.3f} (0.51)")
