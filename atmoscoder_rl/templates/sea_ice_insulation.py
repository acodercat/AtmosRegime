"""Stefan growth from open water added to the existing ice -> growth insulated by the existing ice
(extended, oceanography / sea ice, 2026-10-05).

Parent's method: Stefan's law, the textbook thickness of sea ice grown by conduction in time t under a
surface held Delta T below the freezing point, s = sqrt(2 k Delta T t/(rho L)) ("about 3.4 cm per square
root of a freezing-degree-day"); the solver reports the ice already present plus this growth, h0 + s.
Where it fails: Stefan's law is the solution for ice that starts from zero thickness. The growth rate is
set by the heat conducted through the whole ice sheet, rho L dh/dt = k Delta T / h, so ice that is already
thick grows slowly: integrating from h0 gives h^2 = h0^2 + 2 k Delta T t/(rho L). Adding the open-water
growth to thick ice overstates the final thickness by 22-41 % on the outside grid.
Fair from the text: every arm states the initial thickness, the surface temperature, the freezing point of
the water beneath (-1.8 deg C), the duration, k, rho and L, and says the latent heat is conducted up
through the ice, with snow, ocean heat flux and heat storage in the ice neglected (the Stefan assumptions,
the same in all arms). The initial thickness is the only difference between arms and is ordinary data
(new ice a few millimetres thick vs a first-year floe a metre or more thick). Nothing says that growth
slows with thickness.
Inside: h0 = 0.20-0.40 cm (newly formed nilas): gap <= 1.0 % (the open-water growth is >= 41 cm).
Outside: h0 = 80-160 cm (first-year ice in mid-winter): gap 22-41 %; the shortcut (1.2-2.8 m) is an
ordinary sea-ice thickness, not absurd. Surface temperature -12 to -30 deg C, 15-45 days.
Conventions (rule 5): all constants are stated, so none enter the spread; the sea-ice values a solver
might prefer (k = 2.03-2.2, L = 3.0-3.34 x 10^5, rho = 900-920) contradict the stated ones and are not
counted. A linear surface-to-base temperature profile is the Stefan assumption stated in every arm.
Second wrong method (rule 11): reporting the open-water growth alone (ignoring the initial ice) is
off (>= 17 %, asserted > 10 %). Freezing at the constant initial rate k Delta T/(rho L h0) -- the tangent of the
key -- is within 5 % where growth is small compared with h0; that is a valid approximation there, not a
second wrong method, and is not asserted.
check(): the growth ODE integrated by RK4 from h0, against the closed form; anchor: the Stefan coefficient
sqrt(2 k x 86400/(rho L)) = 3.36 cm per sqrt(K day) for these constants (Leppaeranta 1993 quotes ~3.3).
Reference: Stefan (1891); Leppaeranta (1993), Atmosphere-Ocean 31, 123 ("A review of analytical models of
sea-ice growth"); Talley et al., *Descriptive Physical Oceanography* (6th ed., 2011), ch. 3 (sea ice);
Thomas (ed.), *Sea Ice* (3rd ed., 2017), ch. 2 (thermodynamic growth). Section numbers not verified.
Cluster: new -- differs from latent_heating (freezing_latent_heat, snowpack_cold_content; nearest: latent
heat of fusion) because their parent is an energy budget Delta T = L m/c_p or melt = Q/L_f with a term
left out, while here the parent is a conduction-limited growth law and the missing step is that the
existing ice is part of the conducting layer; also differs from unconfined_dupuit (the other h^2 law),
whose parent is confined Thiem drawdown with a constant transmissivity and whose missing step is that the
saturated thickness falls with the drawdown -- different parent, different text frame.
Independent review (2026-10-05): accepted, no change to the physics. Cluster ruling: joins unconfined_dupuit
(flux through the unknown thickness, integrate h dh), not a new cluster as argued above.
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

EXTENDED = True

K_ICE, RHO_ICE, L_F, T_FREEZE = 2.0, 917.0, 3.34e5, -1.8


def _s2(p):
    """Square of the open-water Stefan growth, m^2."""
    return 2 * K_ICE * (T_FREEZE - p["ts"]) * p["days"] * 86400 / (RHO_ICE * L_F)


def shortcut(p):
    return p["h0"] / 100 + math.sqrt(_s2(p))


def correct(p):
    return math.sqrt((p["h0"] / 100) ** 2 + _s2(p))


def base(rng):
    return {"ts": float(-rng.randint(12, 30)), "days": float(rng.randint(15, 45))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"h0": rnd(rng.uniform(0.20, 0.40))}
    return {"h0": rnd(rng.uniform(80.0, 160.0))}


def text(p, variant):
    given = ("Because the latent heat must be conducted through the whole thickness, growth slows as the ice "
             "thickens: integrating ρ L h dh/dt = k ΔT from the initial thickness h0 gives "
             "h² = h0² + 2 k ΔT t/(ρ L). " if variant == "given" else "")
    return (f"A layer of sea ice {num(p['h0'])} cm thick covers sea water that is at its freezing point, −1.8 °C. "
            f"For the next {num(p['days'])} days the upper surface of the ice is held at {num(p['ts'])} °C, and "
            f"the ice thickens as sea water freezes onto its base, the latent heat released there being conducted "
            f"up through the ice. Take the thermal conductivity of the ice as 2.0 W m^-1 K^-1, its density as "
            f"917 kg m^-3 and the latent heat of fusion as 3.34 × 10^5 J kg^-1; neglect snow, heat supplied by the "
            f"ocean and the heat stored in the ice. {given}How thick is the ice at the end of the {num(p['days'])} "
            f"days? Express your answer in m.")


TEMPLATE = Template("sea_ice_insulation", "oceanography",
                    "Stefan growth from open water added to existing ice -> growth insulated by the existing ice",
                    "m", base, trigger, text, correct, shortcut)


# ---- check --------------------------------------------------------------------------------------------
def _rk4(p, steps=4000):
    """Integrate dh/dt = k dT/(rho L h) from h0 (a tiny seed for near-zero h0 is unnecessary: h0 > 0)."""
    h, dt = p["h0"] / 100, p["days"] * 86400 / steps
    rate = K_ICE * (T_FREEZE - p["ts"]) / (RHO_ICE * L_F)
    f = lambda x: rate / x
    for _ in range(steps):
        k1 = f(h); k2 = f(h + dt * k1 / 2); k3 = f(h + dt * k2 / 2); k4 = f(h + dt * k3)
        h += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return h


def check() -> str:
    max_in = max_err = 0.0
    min_out, max_out, min_wrong = math.inf, 0.0, math.inf
    keys = [math.inf, 0.0]
    for ts in (-12.0, -30.0):
        for days in (15.0, 45.0):
            for h0 in (0.20, 0.40):
                p = {"ts": ts, "days": days, "h0": h0}
                max_in = max(max_in, gap(correct(p), shortcut(p)))
                max_err = max(max_err, gap(correct(p), _rk4(p, steps=200000)))
                keys = [min(keys[0], correct(p)), max(keys[1], correct(p))]
            for h0 in (80.0, 120.0, 160.0):
                p = {"ts": ts, "days": days, "h0": h0}
                c = correct(p)
                g = gap(c, shortcut(p))
                min_out, max_out = min(min_out, g), max(max_out, g)
                max_err = max(max_err, gap(c, _rk4(p)))
                min_wrong = min(min_wrong, gap(c, math.sqrt(_s2(p))))
                keys = [min(keys[0], c), max(keys[1], c)]
                assert 1.0 < shortcut(p) < 3.0                   # an ordinary thickness
    rng = random.Random("sea_ice_insulation:rounded")
    for _ in range(2000):
        pin, pout = pair(TEMPLATE, rng)
        c = correct(pout)
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        g = gap(c, shortcut(pout))
        min_out, max_out = min(min_out, g), max(max_out, g)
        min_wrong = min(min_wrong, gap(c, math.sqrt(_s2(pout))))
        keys = [min(keys[0], c, correct(pin)), max(keys[1], c)]
    assert max_err < 1e-5, max_err
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert min_wrong > 0.10, min_wrong
    assert keys[0] > 0.3, keys                                   # m: round(x, 2) moves it < 2 %
    stefan = math.sqrt(2 * K_ICE * 86400 / (RHO_ICE * L_F)) * 100  # cm per sqrt(K day)
    assert 3.2 < stefan < 3.5, stefan
    return (f"RK4 growth ODE vs closed form {max_err:.1e}; inside max gap {max_in:.4f}; outside gap {min_out:.3f}-"
            f"{max_out:.3f}; open-water-only route >= {min_wrong:.2f} off; key {keys[0]:.2f}-{keys[1]:.2f} m; "
            f"Stefan coefficient {stefan:.2f} cm/sqrt(K day)")
