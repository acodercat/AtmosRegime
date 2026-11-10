"""Freezing point of sea water from salinity alone -> with the pressure depression at depth (extended,
oceanography, 2026-10-05).

Parent's method: the freezing point of sea water is set by its salinity, T_f = -0.0575 S (deg C, S in
g/kg) -- the "sea water freezes at about -1.9 deg C" every oceanography course teaches -- so water of
temperature T is T - T_f = T + 0.0575 S kelvin above freezing (the "thermal driving" of ice-shelf melt).
Where it fails: pressure lowers the freezing point too, by the Clausius-Clapeyron slope T dv/L of the
ice-water transition, 7.40e-4 K per dbar from the stated densities and latent heat (Millero 1978: 7.53e-4). At the sea
surface it is negligible; at the base of an ice shelf or a deep iceberg keel (700-1150 m) the freezing
point is 0.5-0.9 K lower, a sizeable share of a thermal driving of 2-3.5 K. The key is
T + 0.0575 S + 7.40e-4 p, with p the hydrostatic pressure in dbar (rho = 1028 kg m^-3, g = 9.81).
Fair from the text: every arm states the water temperature, the salinity, the depth of the contact with
the ice, and the salinity depression 0.0575 deg C per g/kg "relative to that of fresh water" (true at any
pressure, so nothing stated is contradicted at depth). The fresh-water freezing point and its fall with
pressure are recalled; the 0 deg C of fresh water at the surface is the habit the trap relies on. The
text never mentions pressure. Depth is the only difference between the arms and is ordinary data.
Inside: 2-12 m below sea level (an iceberg's flank, the underside of a thin ice tongue): the pressure
term is <= 0.009 K on keys >= 1.6 K, gap <= 0.6 %.
Outside: 700-1150 m (ice-shelf base, deep keel; Pine Island and Thwaites bases lie at 500-1200 m with
water at -0.3 to +1 deg C at those depths): gap 17-35 %, shortcut an ordinary 1.6-2.5 K (key 2.2-3.4 K).
Ranges: water temperature -0.3 to +0.5 deg C (both arms; water above its freezing point at every
depth on the grid, so ice melts and nothing is supercooled), salinity 33.8-34.8 g/kg.
Conventions (rule 5, checked on the outside grid and on rendered draws): pressure slope 7.61e-4 K/dbar
(TEOS-10 / Jenkins ice-shelf models) or the pure-ice Clausius-Clapeyron slope; 1 dbar per metre instead of
the hydrostatic pressure; the nonlinear UNESCO salinity term (Millero 1978) or the linearised ice-shelf
form T_f = -0.0573 S + 0.0832 - 7.61e-4 z (Holland & Jenkins 1999) in place of the stated 0.0575 S: all
within 5 % of the key (the last two contradict the stated salinity slope and are checked only to show the
key is not convention-fragile).
Second wrong method (rule 11): ignoring salinity but keeping pressure (fresh-water freezing point at
depth) is >= 45 % off; checked > 10 %.
check(): the freezing point recomputed by bisection on the equilibrium condition written as a
Clausius-Clapeyron pressure integral plus the colligative salinity term (a different route from the
closed form), the gap bounds over the grid and 2000 rendered draws, the convention spread, and the
anchor T_f(35 g/kg, 0 dbar) = -1.92 deg C (UNESCO) within 5 %.
Reference: Millero (1978), UNESCO Tech. Pap. Mar. Sci. 28, freezing point of sea water; Talley et al.,
*Descriptive Physical Oceanography* (6th ed., 2011), ch. 3 (freezing point vs salinity and pressure);
Holland & Jenkins (1999), J. Phys. Oceanogr. 29, 1787 (ice-shelf thermal driving, linearised T_f).
Cluster: new -- differs from ice_saturation / clausius_clapeyron_extrapolation (nearest: phase
equilibrium, Clausius-Clapeyron) because their parent is a saturation vapour pressure and their
correction is its temperature dependence or the ice surface, while here the parent is a salinity-only
freezing point and the missing ingredient is the pressure dependence of the solid-liquid transition,
which no other template uses.
Independent review (2026-10-05), must-fix (rules 3/10): the pressure coefficient was stated only in the given
arm, so the outside arm could not be solved from the text. Every arm now states the densities of ice (917),
fresh water (1000) and sea water (1028 kg m^-3) and the latent heat of fusion (3.34e5 J/kg); the key uses the
Clausius-Clapeyron slope these give, 7.40e-4 K/dbar (was Millero 7.53e-4; key moves < 0.5 %), and Millero's
value joins the convention check. Recorded hazard, not a convention: putting the sea-water density into dv
(1/917 - 1/1028) gives 9.6e-4 K/dbar, 5-10 % above the key (check() reports the range). Cluster: new.
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

EXTENDED = True

DTF_DS = 0.0575          # K per g/kg (stated)
DTF_DP = 273.15 * (1 / 917.0 - 1 / 1000.0) / 3.34e5 * 1e4   # K per dbar: Clausius-Clapeyron from the stated data (7.40e-4)
RHO_SW, G = 1028.0, 9.81


def _pressure_dbar(z):
    return RHO_SW * G * z / 1e4


def shortcut(p):
    return p["t"] + DTF_DS * p["s"]


def correct(p):
    return p["t"] + DTF_DS * p["s"] + DTF_DP * _pressure_dbar(p["z"])


def base(rng):
    return {"t": float(rng.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5])),
            "s": rnd(rng.uniform(33.8, 34.8), 3)}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"z": float(rng.randint(2, 12))}
    return {"z": rnd(rng.uniform(700.0, 1150.0), 3)}


def text(p, variant):
    given = ("Pressure lowers the freezing point as well, by T Δv/L ≈ 7.40 × 10^-4 K per decibar (about one decibar per "
             "metre of depth), so at depth the water freezes at −0.0575 S − 7.40 × 10^-4 p °C, with p in dbar. "
             if variant == "given" else "")
    return (f"Ocean water with a temperature of {num(p['t'])} °C and a salinity of {num(p['s'])} g/kg is in "
            f"contact with glacial ice at a depth of {num(p['z'])} m below sea level. Dissolved salt lowers the "
            f"freezing point of sea water by 0.0575 °C for each g/kg of salinity, relative to that of fresh "
            f"water. Take the density of ice as 917 kg m^-3, its latent heat of fusion as 3.34 × 10^5 J kg^-1, the "
            f"density of fresh water as 1000 kg m^-3 and the density of sea water as 1028 kg m^-3. {given}By how many kelvin does the temperature of this water exceed its freezing point? "
            f"Express your answer in K.")


TEMPLATE = Template("freezing_point_depth", "oceanography",
                    "salinity-only freezing point of sea water -> with the pressure depression at depth",
                    "K", base, trigger, text, correct, shortcut)


# ---- check --------------------------------------------------------------------------------------------
def _tf_bisect(s, z):
    """Independent route: the fresh-water melting curve integrated from Clausius-Clapeyron for ice
    (dT/dp = T dv / L, with the stated-free handbook slope pinned to 7.53e-4 K/dbar at 0 dbar through
    dv/L) plus the colligative salinity depression, solved for the T at which the melt condition is met."""
    dv_over_l = DTF_DP / 1e4 / 273.15                   # (m^3/kg) / (J/kg) such that T dv/L = 7.53e-4 K/dbar
    p_pa = RHO_SW * G * z
    # integrate dT/dp = -T dv/L from (273.15 K, 0) to p: T(p) = 273.15 exp(-dv/L p)
    t_fresh = 273.15 * math.exp(-dv_over_l * p_pa) - 273.15
    lo, hi = -10.0, 5.0
    for _ in range(200):                                  # f(T) = T - (fresh melting point - salt term)
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mid < t_fresh - DTF_DS * s else (lo, mid)
    return (lo + hi) / 2


def _unesco_salt(s):
    return -0.0575 * s + 1.710523e-3 * s ** 1.5 - 2.154996e-4 * s ** 2


def _conventions(p):
    pd = _pressure_dbar(p["z"])
    return (p["t"] + DTF_DS * p["s"] + 7.61e-4 * pd,
            p["t"] + DTF_DS * p["s"] + 7.53e-4 * pd,                    # Millero 1978
            p["t"] + DTF_DS * p["s"] + DTF_DP * p["z"],                 # 1 dbar per metre
            p["t"] - _unesco_salt(p["s"]) + 7.53e-4 * pd,
            p["t"] - (-0.0573 * p["s"] + 0.0832 - 7.61e-4 * p["z"]))      # Holland & Jenkins linearised


def _sea_water_dv():
    """Hazard route: sea-water density (1028) instead of fresh water in the Clausius-Clapeyron dv: range of
    its miss on the outside grid."""
    slope = 273.15 * (1 / 917.0 - 1 / 1028.0) / 3.34e5 * 1e4
    out = [gap(correct(p), p["t"] + DTF_DS * p["s"] + slope * _pressure_dbar(p["z"]))
           for p in ({"t": t, "s": s, "z": z} for t in (-0.3, 0.5) for s in (33.8, 34.8) for z in (700.0, 1150.0))]
    return min(out), max(out)


def check() -> str:
    max_in = max_err = spread = 0.0
    min_out, min_wrong = math.inf, math.inf
    keys = [math.inf, 0.0]
    t_grid = (-0.3, 0.1, 0.5)
    for t in t_grid:
        for s in (33.8, 34.3, 34.8):
            for z in (2.0, 7.0, 12.0):
                p = {"t": t, "s": s, "z": z}
                max_in = max(max_in, gap(correct(p), shortcut(p)))
                max_err = max(max_err, gap(correct(p), p["t"] - _tf_bisect(s, z)))
            for z in (700.0, 900.0, 1150.0):
                p = {"t": t, "s": s, "z": z}
                c = correct(p)
                keys = [min(keys[0], c), max(keys[1], c)]
                min_out = min(min_out, gap(c, shortcut(p)))
                max_err = max(max_err, gap(c, p["t"] - _tf_bisect(s, z)))
                spread = max(spread, *(gap(c, v) for v in _conventions(p)))
                min_wrong = min(min_wrong, gap(c, p["t"] + DTF_DP * _pressure_dbar(z)))   # no salt
                assert p["t"] > -DTF_DS * s - DTF_DP * _pressure_dbar(z)      # water above freezing
    rng = random.Random("freezing_point_depth:rounded")
    for _ in range(2000):
        pin, pout = pair(TEMPLATE, rng)
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        c = correct(pout)
        min_out = min(min_out, gap(c, shortcut(pout)))
        spread = max(spread, *(gap(c, v) for v in _conventions(pout)))
        min_wrong = min(min_wrong, gap(c, pout["t"] + DTF_DP * _pressure_dbar(pout["z"])))
        keys = [min(keys[0], c, correct(pin)), max(keys[1], c)]
    assert max_err < 1e-3, max_err
    sw = _sea_water_dv()
    assert sw[0] > 0.04, sw                              # recorded below, not a convention                       # bisection / exp integral vs linear closed form
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert spread < 0.05, spread
    assert min_wrong > 0.10, min_wrong
    assert keys[0] > 1.0, keys                           # K: round(x, 2) moves it < 0.5 %
    anchor = _unesco_salt(35.0)                          # UNESCO: -1.922 deg C at S = 35, p = 0
    assert abs(anchor + 1.922) < 0.002 and gap(anchor, -DTF_DS * 35.0) < 0.05, anchor
    return (f"CC-integral bisection vs closed form {max_err:.1e}; inside max gap {max_in:.4f}; outside min gap "
            f"{min_out:.3f}; convention spread {spread:.3f}; no-salt route >= {min_wrong:.2f} off; key "
            f"{keys[0]:.2f}-{keys[1]:.2f} K; sea-water-dv route {sw[0]:.3f}-{sw[1]:.3f} off; T_f(35, 0) UNESCO {anchor:.3f} vs stated linear {-DTF_DS * 35:.3f}")
