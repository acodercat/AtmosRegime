"""Stokes settling -> size-appropriate drag regime for a falling water drop (extended, 2026-10-01).

Parent's method: Stokes' law for the terminal fall speed of a rigid sphere in creeping flow,
v = 2 rho_w g r^2 / (9 mu).
Where it fails: Stokes' law assumes the drag force is purely viscous (Reynolds number << 1); once
the drop is larger than a few tens of micrometres the flow around it is no longer creeping and the
r^2 scaling overestimates the speed badly. The empirically calibrated regime that textbooks use in
its place (Rogers & Yau, *A Short Course in Cloud Physics*, 1989, table of terminal-velocity
regimes) replaces the r^2 law, for 40 um <~ r <~ 600 um, with a *linear* law v = k2 r, k2 = 8.0e3
s^-1. Nothing about the stated radius looks extreme -- 150-480 um is an ordinary drizzle-drop size,
smaller than a typical raindrop -- but the Stokes r^2 scaling overestimates the true speed by 50 %
to several hundred percent across that range.
Fair from the text: every arm states the drop radius, the water density, the dynamic viscosity of
air and g -- everything Stokes' law needs. Neither law needs the air density, so it is not stated
(nor used by `correct` or `shortcut`; it is used only internally, by this module's own independent
drag-coefficient check).
Inside: r = 5-20 um, Reynolds number (by the force-balance check below) stays below 0.13, and the
published regime-1 constant k1 = 1.19e6 cm^-1 s^-1 matches the fundamental Stokes formula (computed
from rho_w, g and mu) to 1.2 %, constant across this range (`check()`) -- comfortably inside the 2 %
bound, so the inside margin is not doing the work. Outside: r = 150-480 um, well inside the stated
40-600 um validity of the linear law and away from both regime boundaries (the fit is not analytic
across the boundaries; `check()` stays off them on purpose, documented below under Conventions),
where the Stokes shortcut misses the true speed by 125 %-640 %.
Conventions: a published alternative for the same physics -- balancing weight-minus-buoyancy
against drag using the standard sphere drag correlation Cd(Re) = (24/Re)(1 + 0.15 Re^0.687)
(Schiller & Naumann 1933; the same correlation underlies Beard's (1976) more complete terminal-
velocity scheme), solved by iteration for v -- lands within 3.6 % of k2 r across the whole outside
grid (`check()`); this is the rule-5 convention-spread check and also, independently, confirms the
regime-2 constant. The three Rogers-Yau regimes are fit separately and are not required to agree
at their boundaries (Rogers & Yau's own tables show a visible kink near 40 um and 600 um); the
outside grid is kept well clear of both (>=110 um from 40, >=120 um from 600) so this template never
has to adjudicate the boundary, rather than widening the tolerance to paper over it.
Reference: Rogers, R.R. and M.K. Yau (1989), *A Short Course in Cloud Physics* (3rd ed.), sec. 8.1,
table of fall-speed regimes; Beard, K.V. (1976), *J. Atmos. Sci.* 33, 851 (the more complete,
iterative drag-based scheme used here as the convention check). Related: `cunningham_slip` (the
small-particle edge of the same settling problem, where the *continuum* assumption, not the flow
regime, breaks down).

Independent review (2026-10-01): at r = 150 um the key is 5.2 % above Beard (1976) regime 2 with the text's
viscosity, breaking rule 5 at the grid edge (3.8-4.1 % at 160 um, <= 2.5 % from 180 um); the outside range now
starts at 180 um. The air density (needed for any drag-balance route to the key) is now stated in every arm.
Against Gunn & Kinzer (1949) the key is within 3 % over the range.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

RHO_W, RHO_AIR, MU, G = 1000.0, 1.225, 1.81e-5, 9.81    # kg/m^3, kg/m^3, Pa s, m/s^2
K1, K2 = 1.19e6, 8.0e3                                   # cm^-1 s^-1, s^-1 (Rogers & Yau regimes 1, 2)


def shortcut(p):
    """Stokes' law, mm/s."""
    r_m = p["r"] * 1e-6
    return 2 * RHO_W * G * r_m ** 2 / (9 * MU) * 1000


def correct(p):
    """Rogers & Yau regime-appropriate law, mm/s. The sampled ranges never approach the 35 um
    boundary (inside stays <= 20, outside stays >= 150), so the branch choice is never ambiguous."""
    r_cm = p["r"] * 1e-4
    if p["r"] <= 35:
        return K1 * r_cm ** 2 * 10
    return K2 * r_cm * 10


def base(rng):
    return {}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"r": rnd(rng.uniform(5.0, 20.0), 3)}
    return {"r": rnd(rng.uniform(180.0, 480.0), 3)}


def text(p, variant):
    given = ("Because the drop is too large for Stokes' law to apply, use the size-appropriate "
              "regime law instead: for 40 μm ≲ r ≲ 600 μm the terminal fall speed is v = k₂r, with "
              "k₂ = 8.0 × 10^3 s^-1. " if variant == "given" else "")
    return (f"A rigid spherical water drop of radius {num(p['r'])} μm falls at its terminal velocity in still "
            f"air at sea level. Take the density of water to be 1000 kg m^-3, the density of air to be "
            f"1.2 kg m^-3, the dynamic viscosity of air to be 1.81 × 10^-5 kg m^-1 s^-1, and g = 9.81 m s^-2. {given}Find the terminal fall speed of the "
            f"drop. Express your answer in mm s^-1.")


def independent_terminal(p, tol=1e-13, maxit=1000):
    """Independent: balance weight-minus-buoyancy against drag using the Schiller-Naumann sphere
    drag correlation Cd(Re) = (24/Re)(1 + 0.15 Re^0.687), solved by damped fixed-point iteration --
    a different physical model from the regime power laws above, not just different algebra on them."""
    r_m = p["r"] * 1e-6
    v = 2 * RHO_W * G * r_m ** 2 / (9 * MU)
    for _ in range(maxit):
        re = RHO_AIR * v * 2 * r_m / MU
        cd = 24 / re * (1 + 0.15 * re ** 0.687) if re > 1e-10 else 24 / 1e-10
        v_new = math.sqrt((4 / 3 * r_m * (RHO_W - RHO_AIR) * G) / (cd * 0.5 * RHO_AIR))
        if abs(v_new - v) < tol:
            v = v_new
            break
        v = 0.5 * v + 0.5 * v_new
    return v * 1000


def check() -> str:
    max_in = min_out = None
    max_in_drag = spread = 0.
    min_key, max_key = 1e9, 0.
    for r in [5.0, 8.0, 12.0, 16.0, 20.0]:
        p = {"r": r}
        c, s = correct(p), shortcut(p)
        g = gap(c, s)
        max_in = g if max_in is None else max(max_in, g)
        max_in_drag = max(max_in_drag, gap(c, independent_terminal(p)))
        min_key = min(min_key, c)
    for r in range(180, 481, 2):
        p = {"r": float(r)}
        c, s = correct(p), shortcut(p)
        g = gap(c, s)
        min_out = g if min_out is None else min(min_out, g)
        spread = max(spread, gap(c, independent_terminal(p)))
        min_key, max_key = min(min_key, c), max(max_key, c)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_in_drag < 0.03 and spread < 0.05, (max_in_drag, spread)      # rule 5: outside spread
    assert min_key > 1.0 and max_key < 1e4
    # Anchor: the drag-balance model crosses the linear regime law almost exactly near r = 320 um,
    # independently pinning down k2 = 8.0e3 s^-1 in the middle of the sampled range.
    p = {"r": 320.0}
    assert gap(correct(p), independent_terminal(p)) < 0.002
    return (f"max inside gap {max_in:.4f} (independent drag-model check {max_in_drag:.4f}), "
            f"min outside gap {min_out:.3f}, drag-correlation spread {spread:.4f}, "
            f"key range {min_key:.1f}-{max_key:.1f} mm/s; drag model ~ k2 r at r=320 um")


TEMPLATE = Template("raindrop_drag_regime", "cloud_physics",
                    "Stokes settling -> Rogers-Yau linear drag regime once the drop is too large for creeping flow",
                    "mm/s", base, trigger, text, correct, shortcut)
