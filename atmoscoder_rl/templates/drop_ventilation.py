"""Stationary-drop diffusional growth equation -> ventilation-enhanced evaporation of a falling drop
(spec: MECHANISMS.md; extended set, 2026-10-01).

Parent's method: a water drop's radius changes by vapor diffusion alone, r dr/dt = (S - 1)/(F_k + F_d),
the classic stationary-drop growth/evaporation equation (Rogers & Yau, A Short Course in Cloud
Physics, ch. 7), with S = RH as a fraction (S - 1 negative, subsaturated air), F_k the heat-conduction term and
F_d the vapor-diffusion term.
Where it fails: a drop that is actually falling has air flowing past it, which ventilates away the
vapor (and heat) it exchanges with its surroundings faster than still-air diffusion alone. The
correct rate multiplies the stationary-drop equation by the ventilation factor f (Pruppacher &
Klett 1997, sec. 13.2): with X = N_Sc^(1/3) N_Re^(1/2) (N_Sc = nu_air/D_v, N_Re = v(2r)/nu_air, v the
drop's fall speed),
    f = 1 + 0.108 X^2            (X <= 1.4)
    f = 0.78 + 0.308 X           (X > 1.4),
which correctly tends to f -> 1 as the drop (and its fall speed) shrinks to zero, and exceeds 1 by a
wide margin once the drop is large enough to fall fast.
Fair from the text: every arm states the ambient temperature and relative humidity (which set S,
F_k and F_d) and the drop's fall speed; the thermal conductivity, vapor diffusivity, latent heat,
gas constant and density of liquid water needed for F_k and F_d are given as handbook numbers in
every arm. The fall speed
itself is the terminal velocity implied by the stated radius (Stokes' law below 30 um; the standard
two-regime empirical fit, v ~ r and v ~ sqrt(r), above it -- Rogers & Yau Table 2.1) -- it is reported
as a plain physical fact about the drop, not derived in the text, so stating it is not itself a cue.
Only the radius (and the fall speed it implies) changes between the arms.
Inside: r = 2-20 um, small cloud droplets. Re is of order 1e-4 to 0.1 there, so X << 1 and f is
within 1.0 % of unity at the largest sampled radius (20 um); the stationary-drop equation is correct.
Outside: r = 0.3-1 mm, ordinary drizzle and light-rain drops -- nothing about the stated radius looks
extreme. Re is 90-900 there, X is comfortably above the 1.4 crossover, and f = 3.4-8.5: the
stationary-drop rate underestimates the true evaporation rate by 70-88 % across the grid.
Answer in nm/s (the radius shrink rate -dr/dt): outside, large slowly evaporating drops give small
rates (tens of nm/s); a um/s answer would put the smallest outside key below 0.1, risking corruption
under round(x, 2) (as in `cunningham_slip`), so the unit is scaled down instead.

Conventions: using the T-dependence of the latent heat (2.45-2.50 x 10^6 J/kg over 0-20 degC) instead
of the single stated value, or a +-3 % handbook spread in the thermal conductivity or vapor
diffusivity, each move the key by under 3 %, checked below. The piecewise ventilation formula is
also checked for continuity at its X = 1.4 boundary (both branches agree to 0.04 %). The Ranz-Marshall
correlation f = 1 + 0.3 X (a common simpler alternative to the keyed Pruppacher & Klett piecewise
form, same X) and a separate heat-ventilation variant -- r dr/dt = S / (F_k/f_h + F_d/f_v), with f_v
the keyed vapour-ventilation factor (Sc-based X) and f_h the same functional form evaluated at the
Prandtl-number X_h = Pr^(1/3) N_Re^(1/2), Pr = 0.71, instead of lumping heat and vapour under one f --
are both checked against the key across the outside grid below.
Reference: Pruppacher, H. R. and Klett, J. D. (1997), Microphysics of Clouds and Precipitation, 2nd
ed., sec. 13.2 (the ventilation coefficient) and ch. 10 (the terminal-velocity regimes used for the
fall speed); Rogers, R. R. and Yau, M. K. (1989), A Short Course in Cloud Physics, 3rd ed., ch. 7
(the stationary-drop growth equation) and Table 2.1 (fall-speed regimes); Ranz, W. E. and Marshall,
W. R. (1952), "Evaporation from drops", Chem. Eng. Prog. 48, for the simpler ventilation correlation.

Independent review (2026-10-01): at the previous outside lower bound (r = 200 um) the Ranz-Marshall
ventilation correlation disagrees with the keyed Pruppacher & Klett form by 7.0 %, over the 5 %
convention-spread bound. Raised the outside minimum radius from 200 um to 300 um, where the
disagreement is 4.5 % (still decreasing with r, so 300 um is the worst case over the new outside
grid); the outside gap bound only strengthens from this change (min outside gap now 70.3 %, up from
~60 % at the old lower bound). Added the Ranz-Marshall correlation (4.54 % deviation at the worst
corner) and a separate heat/vapour ventilation-factor variant (3.14 % at the worst corner) to
`check()`'s convention spread, both comfortably under 5 %.

Lead review (2026-10-01): a screening run (deepseek, inside arm) scored 3/20 passed, 1/20 given --
far below the other extended templates. Reading the transcripts showed two distinct failure modes,
neither a physics error: (a) most misses returned the correct magnitude with the wrong sign -- the
model computed dr/dt itself (negative, the radius decreasing) and submitted that, where the key is
the positive shrinking rate -dr/dt; the old question, "At what rate is the drop's radius
shrinking?", does not say the answer must be positive, so a model that reports a signed derivative
is answering a reasonable reading of the words and still failing the grader. (b) several more misses
were off by a clean power of 10 (1e3 or 1e6), consistent with the model solving correctly in SI
units and then mis-converting to nm/s, or reporting um/s or m/s while labelling it nm/s. Fixed both
in the text, not the physics or the key (`correct`/`shortcut` were already magnitude-only via
`abs(s)`, so neither `check()` bound changes): the question now states explicitly that dr/dt < 0
and asks for -dr/dt as a positive number, and the nm/s unit is spelled out with its SI and um/s
equivalents so a model cannot mistake which decade it is reporting in. Also added the density of
liquid water rho_w = 1.00 x 10^3 kg m^-3 to the stated constants -- it is needed by both F_k and F_d
(`_fk`, `_fd`) and was the one handbook number `F_k`/`F_d` depend on that the text never gave; a
careful expert re-deriving the key from the stated constants alone could not have matched it
without assuming a value. `check()` now also covers a +-1 % rho_w spread (water's density varies by
less than 0.2 % over 0-20 degC) alongside the existing D_v/K/L spreads, confirming the omission
never mattered for the gap bounds (rate scales as 1/rho_w, so the spread is by construction a small
fraction of the 5 % convention bound) -- it mattered only for whether the inside arm was fully
solvable from the text as stated. Re-ran `check()` and the full `check_templates --extended --only
drop_ventilation` suite after the edit; both pass.

Lead review 2 (2026-10-02): the full re-screen after that fix still scored inside 9/20 and given
3/20. Transcripts: the model writes dm/dt = -4 pi r D_v (rho_vs - rho_v) f -- vapour diffusion with
the drop at air temperature, dropping the heat-conduction term F_k -- so the given arm's "multiply the
stationary-drop rate by f" multiplies a rate the text never defined, and lands about 2.6x off the key
for a reason unrelated to ventilation (the inside misses were the same omission, or unit slips). Fix:
every arm now states the diffusional growth equation r dr/dt = (S - 1)/(F_k + F_d) with F_k, F_d and
e_s written out -- the parent's method, not the replacement -- so the only thing left to recall on the
outside arm is the ventilation factor. Independent review: stating the equation as *the* rate law
for this drop made the outside key contradict the text, so the text now gives it only as the source
of the F_k and F_d terms (cited to Rogers & Yau ch. 7), with no mention of air flow or a stationary
drop (rule 10); nu printed in the same x 10^ notation as the other constants. Text only; `correct`,
`shortcut` and the `check()` bounds are unchanged.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

RHO_W, G = 1000.0, 9.81                   # kg/m^3, m/s^2
MU_AIR, RHO_AIR = 1.81e-5, 1.225          # Pa s, kg/m^3 (sea level)
NU_AIR = MU_AIR / RHO_AIR                 # m^2/s
DV, K_AIR = 2.4e-5, 2.4e-2                # m^2/s, W/m/K
LV, RV = 2.5e6, 461.5                     # J/kg, J/kg/K
SC = NU_AIR / DV
SC13 = SC ** (1 / 3)
PR = 0.71                                 # Prandtl number of air (heat-ventilation convention)
PR13 = PR ** (1 / 3)


def _es(t_c):
    """Saturation vapour pressure over water, Pa (Bolton 1980 form), t in degC."""
    return 611.2 * math.exp(17.67 * t_c / (t_c + 243.5))


def _fk(t_c):
    tk = t_c + 273.15
    return (LV / (RV * tk) - 1) * LV * RHO_W / (K_AIR * tk)


def _fd(t_c):
    tk = t_c + 273.15
    return RHO_W * RV * tk / (DV * _es(t_c))


def _v_fall(r_um):
    """Terminal fall speed implied by the radius, m/s (Rogers & Yau Table 2.1 regimes, the same
    formula stated in `text` for every variant)."""
    r_cm = r_um * 1e-4
    if r_um <= 30:
        v_cmps = 1.19e6 * r_cm ** 2
    elif r_um <= 600:
        v_cmps = 8.0e3 * r_cm
    else:
        v_cmps = 2.01e3 * math.sqrt(r_cm)
    return v_cmps / 100.0


def _reynolds(r_um):
    v = _v_fall(r_um)
    return v * 2 * (r_um * 1e-6) / NU_AIR


def _fvent_of_x(x):
    """Pruppacher & Klett's piecewise ventilation factor, as a function of X."""
    return 1.0 + 0.108 * x ** 2 if x <= 1.4 else 0.78 + 0.308 * x


def _fvent(r_um):
    x = SC13 * math.sqrt(_reynolds(r_um))
    return _fvent_of_x(x)


def _rate(p, vent: bool):
    """Magnitude of dr/dt, m/s."""
    s = p["rh"] / 100.0 - 1.0
    f = _fk(p["t"]) + _fd(p["t"])
    r_m = p["r"] * 1e-6
    base = abs(s) / f / r_m
    return base * (_fvent(p["r"]) if vent else 1.0)


def base(rng):
    return {"t": rnd(rng.uniform(0, 20), 3), "rh": rnd(rng.uniform(50, 90), 3)}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"r": rnd(rng.uniform(2, 20), 3)}
    return {"r": rnd(rng.uniform(300, 1000), 3)}


def shortcut(p):
    return _rate(p, vent=False) * 1e9    # nm/s


def correct(p):
    return _rate(p, vent=True) * 1e9     # nm/s


def text(p, variant):
    given = ("Because the drop is falling rather than stationary, the air flowing past it "
             "ventilates away heat and vapour faster than diffusion alone: multiply the rate "
             "from the diffusional growth equation by the ventilation factor f = 1 + 0.108 X^2 (X <= 1.4) or "
             "f = 0.78 + 0.308 X (X > 1.4), where X = N_Sc^(1/3) N_Re^(1/2), with N_Sc = nu/D_v and "
             "N_Re = v(2r)/nu using the drop's fall speed v and the kinematic viscosity of air. "
             if variant == "given" else "")
    return (f"A spherical water drop of radius {num(p['r'])} um falls at its terminal velocity "
            f"through air at a temperature of {num(p['t'])} degC and a relative humidity of "
            f"{num(p['rh'])}%. Take the drop's terminal fall speed, with r in cm and v in cm/s, to be "
            f"v = 1.19 x 10^6 r^2 for r <= 30 um, v = 8.0 x 10^3 r for 30 um < r <= 0.6 mm, and "
            f"v = 2.01 x 10^3 sqrt(r) for 0.6 mm < r <= 2 mm (Rogers & Yau's standard approximation). "
            f"Take the thermal conductivity of air K = 2.4 x 10^-2 W m^-1 K^-1, the diffusivity of "
            f"water vapour in air D_v = 2.4 x 10^-5 m^2 s^-1, the kinematic viscosity of air "
            f"nu = {NU_AIR * 1e5:.2f} x 10^-5 m^2 s^-1, the latent heat of vaporization L = 2.5 x 10^6 J kg^-1, "
            f"the gas constant for water vapour R_v = 461.5 J kg^-1 K^-1, and the density of liquid "
            f"water rho_w = 1.00 x 10^3 kg m^-3. Take the heat-conduction and vapour-diffusion terms of "
            f"the diffusional growth equation (Rogers & Yau ch. 7, r dr/dt = (S - 1)/(F_k + F_d), S the "
            f"saturation ratio of the air, i.e. the relative humidity as a fraction) to be "
            f"F_k = (L/(R_v T) - 1) L rho_w/(K T) and F_d = rho_w R_v T/(D_v e_s(T)), with T the air "
            f"temperature in K and e_s(T) = 611.2 exp[17.67 T_c/(T_c + 243.5)] Pa the saturation vapour "
            f"pressure over water (T_c in degC). {given}The drop is evaporating, so its radius is "
            f"decreasing with time: dr/dt < 0. By how many nanometres per second is the radius "
            f"decreasing -- i.e. what is the positive quantity -dr/dt? Give a positive number, in "
            f"nm s^-1 (1 nm s^-1 = 10^-9 m s^-1 = 10^-3 um s^-1).")


TEMPLATE = Template("drop_ventilation", "cloud_physics",
                    "stationary-drop diffusional evaporation -> ventilation-enhanced evaporation of "
                    "a falling drop",
                    "nm/s", base, trigger, text, correct, shortcut)


def _rate_ranz_marshall(p):
    """Rate using the simpler Ranz-Marshall ventilation correlation f = 1 + 0.3 X in place of
    the keyed Pruppacher & Klett piecewise form (same X), m/s."""
    s = p["rh"] / 100.0 - 1.0
    f = _fk(p["t"]) + _fd(p["t"])
    r_m = p["r"] * 1e-6
    x = SC13 * math.sqrt(_reynolds(p["r"]))
    f_rm = 1.0 + 0.3 * x
    return abs(s) / f / r_m * f_rm


def _rate_heat_vent(p):
    """Rate using separate heat and vapour ventilation factors, r dr/dt = S/(F_k/f_h + F_d/f_v),
    instead of lumping both terms under one Sc-based f: f_v is the keyed factor, f_h is the same
    functional form evaluated at the Prandtl-number X_h = Pr^(1/3) N_Re^(1/2), m/s."""
    s = p["rh"] / 100.0 - 1.0
    fk, fd = _fk(p["t"]), _fd(p["t"])
    r_m = p["r"] * 1e-6
    re = _reynolds(p["r"])
    f_v = _fvent_of_x(SC13 * math.sqrt(re))
    f_h = _fvent_of_x(PR13 * math.sqrt(re))
    return abs(s) / (fk / f_h + fd / f_v) / r_m


def check() -> str:
    """Gap bounds over the grid, continuity of the piecewise f, Ranz-Marshall and separate
    heat/vapour-ventilation conventions, and constant-spread conventions."""
    max_in, min_out = 0.0, math.inf
    for t in (0.0, 10.0, 20.0):
        for rh in (50.0, 70.0, 90.0):
            for r in (2.0, 10.0, 20.0):
                p = {"t": t, "rh": rh, "r": r}
                max_in = max(max_in, gap(correct(p), shortcut(p)))
            for r in (300.0, 500.0, 1000.0):
                p = {"t": t, "rh": rh, "r": r}
                min_out = min(min_out, gap(correct(p), shortcut(p)))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP

    f_lo = 1.0 + 0.108 * 1.4 ** 2
    f_hi = 0.78 + 0.308 * 1.4
    assert abs(f_lo - f_hi) / f_lo < 0.01

    def rate_with(p, dv=DV, k=K_AIR, lv=LV, rho_w=RHO_W):
        tk = p["t"] + 273.15
        fk = (lv / (RV * tk) - 1) * lv * rho_w / (k * tk)
        fd = rho_w * RV * tk / (dv * _es(p["t"]))
        s = p["rh"] / 100.0 - 1.0
        r_m = p["r"] * 1e-6
        return abs(s) / (fk + fd) / r_m * _fvent(p["r"])

    max_conv, max_rm, max_heat_vent = 0.0, 0.0, 0.0
    for t in (0.0, 10.0, 20.0):
        for rh in (50.0, 70.0, 90.0):
            for r in (300.0, 500.0, 1000.0):
                p = {"t": t, "rh": rh, "r": r}
                key = correct(p) / 1e9
                for dv in (DV * 0.97, DV * 1.03):
                    max_conv = max(max_conv, abs(rate_with(p, dv=dv) / key - 1))
                for k in (K_AIR * 0.97, K_AIR * 1.03):
                    max_conv = max(max_conv, abs(rate_with(p, k=k) / key - 1))
                for lv in (2.45e6, 2.50e6):
                    max_conv = max(max_conv, abs(rate_with(p, lv=lv) / key - 1))
                for rho_w in (RHO_W * 0.99, RHO_W * 1.01):   # water's density varies <0.2% over 0-20 degC
                    max_conv = max(max_conv, abs(rate_with(p, rho_w=rho_w) / key - 1))
                max_rm = max(max_rm, abs(_rate_ranz_marshall(p) / key - 1))
                max_heat_vent = max(max_heat_vent, abs(_rate_heat_vent(p) / key - 1))
    assert max_conv < 0.05
    assert max_rm < 0.05
    assert max_heat_vent < 0.05
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; "
            f"f continuity at X=1.4: {abs(f_lo - f_hi) / f_lo:.4f}; "
            f"constant-spread conventions {max_conv:.4f}; "
            f"Ranz-Marshall convention {max_rm:.4f}; heat-ventilation convention {max_heat_vent:.4f}")
