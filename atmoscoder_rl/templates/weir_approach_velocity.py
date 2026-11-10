"""Broad-crested weir formula on the measured head -> energy head including the velocity of approach
(extended, hydrology / open-channel hydraulics, 2026-10-05).

Parent's method: the discharge over a broad-crested weir from the head H of the upstream water surface
above the crest, Q = b sqrt(g) (2H/3)^{3/2} ~ 1.705 b H^{3/2} (SI): critical depth 2H/3 on the crest,
critical velocity sqrt(g y_c). It is the weir formula of every hydrology and hydraulics text.
Where it fails: critical depth on the crest is 2/3 of the upstream *energy* head E = H + V^2/(2g), where
V = Q/(b (H + P)) is the velocity of the water approaching in the channel of depth H + P (P = crest height
above the bed). Over a high weir the approach flow is slow and E = H; over a low sill the approach
velocity head is a large share of H, and Q = b sqrt(g) (2E/3)^{3/2} must be solved together with
E = H + Q^2/(2 g b^2 (H + P)^2) (Henderson 1966; Chow 1959). The measured-head formula underestimates Q by
16-18 % at P/H = 0.25-0.30.
Fair from the text: every arm states the channel width (the weir spans it), the crest height above the
bed, the head of the upstream water surface above the crest, g, that the flow over the crest is free and
passes through critical depth, and that energy losses are neglected -- everything both answers need;
nothing is recalled. The crest height is ordinary geometry; the inside and outside texts never mention
the approach velocity or the energy head.
Inside: P/H = 3.0-5.0 (a tall weir; the approach velocity head is <= 1 % of H); the weir formula is within
1.5 % of the key.
Outside: P/H = 0.25-0.30, a low weir (H = 0.75-1.00 m over a 0.19-0.30 m crest, at or above the
P >= 0.15 m of ISO 3846 / Bos); the upstream Froude
number stays below 0.6, so the approach flow is subcritical and the critical control on the crest is
physically consistent on the whole grid (check() asserts Fr < 0.75 and that the subcritical root exists).
The shortcut discharge is 16-18 % low and looks ordinary.
Conventions: the ideal (loss-free) relation is what the text asks for; real broad-crested weirs carry a
discharge coefficient of 0.85-1.0, excluded by "neglect energy losses". Writing the critical-flow
condition as Bernoulli between the upstream section and the crest, H + P + V^2/2g = P + 3 y_c/2, is the same
physics (independent check). One fixed-point step (E from the velocity implied by the shortcut Q) is a
truncated version of the key, reported in check() (its miss over the outside grid is in the check line; see the review note below).
Second wrong method (rule 11): taking the approach area as b H (forgetting the crest height) puts the
upstream flow at or near critical and gives a discharge >= 40 % above the key (`check()`).
check(): bisection on q of Bernoulli with the critical-depth condition (q^2 = g y_c^3), independent of the
fixed-point iteration in correct().
Reference: Henderson, *Open Channel Flow* (1966), sec. 6.6 (broad-crested weirs; velocity of approach);
Chow, *Open-Channel Hydraulics* (1959), ch. 3 and 14 (critical flow, the approach-velocity head);
Chanson, *The Hydraulics of Open Channel Flow* (2nd ed., 2004), ch. 19 (broad-crested weir, total head
above crest). Section numbers not verified.

Cluster: new -- differs from surface_gravity_wave (the only other open-channel cluster: long-wave speed
sqrt(gH) and its dispersion correction) because the parent is the critical-flow weir discharge and the
missing ingredient is the kinetic head of the approach flow in the energy budget; no existing template
uses Bernoulli or a specific-energy balance.
Independent review (2026-10-05): (a) the outside draw P/H = 0.15-0.30 with H = 0.50-1.00 m gave crests of
0.075-0.30 m, below the ISO 3846 / Bos calibration range (P >= 0.15 m); (b) a single fixed-point correction
(E from the velocity implied by the shortcut Q) missed the key by 4.9-9.4 %. Now H = 0.75-1.00 m in all arms
and outside P/H = 0.25-0.30, so P >= 0.19 m; check() asserts the outside gap on rendered draws. The one-step
miss on the new outside grid is 4.9-6.0 % -- recorded, not asserted. Cluster ruling: new.
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

G = 9.81


def shortcut(p):
    """Weir formula on the measured head, m^3/s."""
    return p["b"] * math.sqrt(G) * (2 * p["H"] / 3) ** 1.5


def _energy_head(H, depth):
    """Subcritical root of E = H + q(E)^2/(2 g depth^2) with q = sqrt(g)(2E/3)^1.5, i.e. E = H + (4/27) E^3/depth^2."""
    c = (4.0 / 27.0) / depth ** 2                          # q^2/(2 g depth^2) = c E^3
    e = H
    for _ in range(500):                                   # fixed point converges to the subcritical root
        e_new = H + c * e ** 3
        if abs(e_new - e) < 1e-14:
            break
        e = e_new
    return e


def correct(p):
    """Critical depth on the crest from the energy head including the approach velocity, m^3/s."""
    e = _energy_head(p["H"], p["H"] + p["P"])
    return p["b"] * math.sqrt(G) * (2 * e / 3) ** 1.5


def base(rng):
    return {"b": rnd(rng.uniform(2.0, 10.0)), "H": rnd(rng.uniform(0.75, 1.00))}


def trigger(rng, regime, base):
    ratio = rng.uniform(3.0, 5.0) if regime == "inside" else rng.uniform(0.25, 0.30)
    return {"P": rnd(base["H"] * ratio)}                   # m


def text(p, variant):
    given = ("The water approaches the weir with an appreciable speed, so critical depth on the crest is 2/3 of "
             "the upstream energy head E = H + V²/(2g), where V = Q/[b(H + P)] is the approach velocity in the "
             "channel; solve Q = b√g (2E/3)^(3/2) together with it. " if variant == "given" else "")
    return (f"A broad-crested weir spans the full width of a horizontal rectangular channel {num(p['b'])} m wide; "
            f"its crest is {num(p['P'])} m above the channel bed. Upstream of the weir, the water surface stands "
            f"{num(p['H'])} m above the crest. The flow over the crest is free (unaffected by the tailwater) and "
            f"passes through critical depth there; neglect energy losses and take g = 9.81 m s^-2. {given}What is "
            f"the discharge over the weir? Express your answer in m^3/s.")


TEMPLATE = Template("weir_approach_velocity", "hydrology",
                    "broad-crested weir formula on the measured head -> energy head with the velocity of approach",
                    "m^3/s", base, trigger, text, correct, shortcut)


def _bernoulli(p):
    """Independent: bisection on q of H + P + q^2/(2 g y1^2) = P + 1.5 (q^2/g)^(1/3), subcritical branch."""
    y1 = p["H"] + p["P"]
    f = lambda q: p["H"] + q * q / (2 * G * y1 * y1) - 1.5 * (q * q / G) ** (1 / 3)
    lo, hi = 1e-9, math.sqrt(G * y1 ** 3) * 0.999        # below the upstream critical discharge
    # f(lo) > 0; on the subcritical branch f falls until the root
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return p["b"] * (lo + hi) / 2


def _no_crest_wrong(p):
    """Wrong: approach area b H (crest height forgotten); smallest root of E = H + (4/27) E^3/H^2 on [H, 1.5H]."""
    H = p["H"]
    f = lambda e: H + (4.0 / 27.0) * e ** 3 / H ** 2 - e
    lo, hi = H, 1.5 * H
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return p["b"] * math.sqrt(G) * (2 * lo / 3) ** 1.5


def _one_step(p):
    q0 = shortcut(p) / p["b"]
    e = p["H"] + q0 ** 2 / (2 * G * (p["H"] + p["P"]) ** 2)
    return p["b"] * math.sqrt(G) * (2 * e / 3) ** 1.5


def check() -> str:
    max_in, min_out, max_out, max_err = 0.0, math.inf, 0.0, 0.0
    min_wrong, max_fr, min_key, max_step = math.inf, 0.0, math.inf, 0.0
    for b in (2.0, 10.0):
        for H in (0.75, 0.875, 1.00):
            for ratio, inside in ((3.0, 1), (5.0, 1), (0.25, 0), (0.275, 0), (0.30, 0)):
                p = {"b": b, "H": H, "P": H * ratio}
                c, s = correct(p), shortcut(p)
                max_err = max(max_err, gap(c, _bernoulli(p)))
                min_key = min(min_key, c)
                g = gap(c, s)
                if inside:
                    max_in = max(max_in, g)
                    continue
                min_out, max_out = min(min_out, g), max(max_out, g)
                y1 = H + p["P"]
                max_fr = max(max_fr, (c / b / y1) / math.sqrt(G * y1))      # upstream Froude number
                min_wrong = min(min_wrong, gap(c, _no_crest_wrong(p)))
                max_step = max(max_step, gap(c, _one_step(p)))
    rng = random.Random("weir-check")
    for _ in range(3000):
        bb = base(rng)
        pin, pout = {**bb, **trigger(rng, "inside", bb)}, {**bb, **trigger(rng, "outside", bb)}
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        min_out = min(min_out, gap(correct(pout), shortcut(pout)))
        y1 = pout["H"] + pout["P"]
        max_fr = max(max_fr, (correct(pout) / pout["b"] / y1) / math.sqrt(G * y1))
        max_err = max(max_err, gap(correct(pout), _bernoulli(pout)))
    assert max_err < 1e-6, max_err
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert min_wrong > 0.10, min_wrong
    assert max_fr < 0.75, max_fr                          # upstream flow subcritical: crest control holds
    assert min_key > 0.5, min_key                         # m^3/s: clear of round(x, 2)
    # Anchor: ideal broad-crested weir coefficient sqrt(g)(2/3)^1.5 = 1.705 m^0.5/s.
    coeff = shortcut({"b": 1.0, "H": 1.0, "P": 10.0})
    assert abs(coeff - 1.705) < 0.001, coeff
    return (f"Bernoulli bisection vs fixed point {max_err:.1e}; inside max gap {max_in:.4f}; outside gap "
            f"{min_out:.3f}-{max_out:.3f}; max upstream Froude {max_fr:.2f}; crest-forgotten wrong method min miss "
            f"{min_wrong:.2f}; one-step iteration off by <= {max_step:.3f}; min key {min_key:.2f} m3/s; "
            f"anchor C = {coeff:.3f}")
