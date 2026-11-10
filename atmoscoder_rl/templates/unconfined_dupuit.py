"""Thiem drawdown with a fixed transmissivity -> Dupuit-Thiem for a water-table aquifer (extended,
hydrology, 2026-10-05).

Parent's method: steady drawdown at distance r from a well pumped at rate Q, with a radius of
influence R, is Thiem's s = Q ln(R/r) / (2 pi T). For a water-table (unconfined) aquifer the
transmissivity is taken as T = K b, with b the saturated thickness before pumping -- the routine
hand calculation, and the form every pumping-test worksheet uses.
Where it fails: in a water-table aquifer the saturated thickness is not fixed; it is the head h
itself, so the flow cross-section shrinks as the water table is drawn down. Darcy's law through a
cylinder of height h, Q = 2 pi r K h dh/dr, integrates to the Dupuit-Thiem equation
h_0^2 - h^2 = Q ln(R/r) / (pi K), s = h_0 - h. Writing x = s_Thiem / h_0, the true drawdown is
h_0 (1 - sqrt(1 - 2x)), larger than s_Thiem by about x/2 for small x and by 20-50 % once the
drawdown is a third or more of the saturated thickness.
Fair from the text: every arm states that the well fully penetrates a water-table aquifer on a
horizontal impermeable base, the height of the water table above that base before pumping (h_0),
the hydraulic conductivity K, the steady pumping rate Q, the radius r of the pumping well and the
radius of influence R -- everything Thiem with T = K h_0 needs, and everything Dupuit-Thiem needs. The
"water-table aquifer" wording is the ordinary description of the aquifer (without it a confined
reading would make Thiem correct, an ambiguity, not a trap); it is in all arms. Nothing in the
inside or outside text mentions the shrinking saturated thickness.
Inside: Q is set so that x = s_Thiem / h_0 = 0.012-0.035; Dupuit-Thiem exceeds Thiem by <= 1.8 %.
Outside: x = 0.27-0.34 -- an ordinary production rate (about 280 to 5300 m^3/day, K = 5-20 m/day,
h_0 = 17-30 m, well radius 0.15-0.3 m, R = 400-1000 m); the water level in the well is drawn down by
32-43 % of h_0 (h_w^2 = h_0^2 (1 - 2x) > 0: the well is never dry) and Thiem misses it by 16-22 %. The
shortcut drawdown (27-34 % of h_0) is positive and well below h_0, so nothing in its own number looks
absurd.
Conventions: Jacob's (1944) correction s' = s - s^2/(2 h_0), which converts an unconfined drawdown
to an equivalent confined one, is algebraically the same relation (checked as an identity); so is
Thiem with T = K (h_0 + h)/2, the mean saturated thickness. No other closed form is in use for
steady radial flow to a well in a water-table aquifer; the Dupuit assumption itself (horizontal
flow) is the textbook model and is what every arm's wording implies. The key is the water level
in the pumping well: by Charny's (1951) proof the Dupuit-Thiem discharge formula is exact for the
well water level even though a seepage face forms above it, so the seepage face does not move the key.
check(): integrates the radial Darcy equation dh/dr = Q/(2 pi K r h) numerically (RK4 in ln r)
from R inward to r -- independent of the closed form.
Reference: Fetter, *Applied Hydrogeology* (4th ed.), sec. 5.4 (Thiem and the unconfined
Dupuit-Thiem form); Todd & Mays, *Groundwater Hydrology* (3rd ed.), sec. 4.2 (steady radial flow
to a well, confined vs unconfined); Jacob (1944), Trans. AGU 25, for the drawdown correction.
Related: cooper_jacob_early_time (also a well, but a transient-series truncation in a confined
aquifer; a different relation and validity condition) -- new cluster "geometry of the flow
cross-section changes with the solution".

Independent review (2026-10-05): the first draft asked for the drawdown at an observation well 5-25 m
away with x = 0.29-0.42 there; on every outside draw Dupuit-Thiem then gave h^2 < 0 at the pumping well
(a dry well, so the stated steady Q was impossible). The question is now the drawdown in the pumping
well itself (radius 0.15-0.3 m, stated in all arms), outside x = 0.27-0.34, and check() asserts
h_w^2 > 0 over the whole outside grid.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True


def _lnrr(p):
    return math.log(p["R"] / p["r"])


def shortcut(p):
    """Thiem with T = K h_0, m."""
    return p["Q"] * _lnrr(p) / (2 * math.pi * p["K"] * p["h0"])


def correct(p):
    """Dupuit-Thiem, m."""
    return p["h0"] - math.sqrt(p["h0"] ** 2 - p["Q"] * _lnrr(p) / (math.pi * p["K"]))


def base(rng):
    return {"K": rnd(rng.uniform(5.0, 20.0)), "h0": rnd(rng.uniform(17.0, 30.0)),
            "r": rnd(rng.uniform(0.15, 0.3)), "R": rnd(rng.uniform(400.0, 1000.0))}


def trigger(rng, regime, base):
    x = rng.uniform(0.012, 0.035) if regime == "inside" else rng.uniform(0.27, 0.34)
    q = 2 * math.pi * base["K"] * base["h0"] ** 2 * x / math.log(base["R"] / base["r"])
    return {"Q": rnd(q, 3)}          # m^3/day


def text(p, variant):
    given = ("Because the aquifer is unconfined, its saturated thickness is the water-table height h itself, "
             "which falls toward the well, so Darcy's law gives Q = 2πrKh dh/dr, which integrates to the "
             "Dupuit–Thiem equation h_0² − h² = Q ln(R/r)/(πK), with drawdown s = h_0 − h. "
             if variant == "given" else "")
    return (f"A well fully penetrates a water-table aquifer that rests on a horizontal impermeable base. Before "
            f"pumping, the water table stands {num(p['h0'])} m above the base. The hydraulic conductivity is "
            f"{num(p['K'])} m/day. The well is pumped at a steady rate of {num(p['Q'])} m^3/day, and the flow has "
            f"reached steady state with a radius of influence of {num(p['R'])} m. {given}The pumping well "
            f"has a radius of {num(p['r'])} m. What is the drawdown of the water level in the pumping well? "
            f"Express your answer in m.")


TEMPLATE = Template("unconfined_dupuit", "hydrology",
                    "Thiem drawdown with transmissivity K h_0 -> Dupuit-Thiem for a water-table aquifer",
                    "m", base, trigger, text, correct, shortcut)


def _rk4(p, n=4000):
    """Independent: integrate dh/d(ln r) = Q/(2 pi K h) from (ln R, h_0) inward to ln r."""
    a = p["Q"] / (2 * math.pi * p["K"])
    f = lambda h: a / h
    h = p["h0"]
    step = -_lnrr(p) / n
    for _ in range(n):
        k1 = f(h)
        k2 = f(h + step * k1 / 2)
        k3 = f(h + step * k2 / 2)
        k4 = f(h + step * k3)
        h += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return p["h0"] - h


def check() -> str:
    max_in, min_out, max_out = 0.0, math.inf, 0.0
    max_err = max_jacob = 0.0
    min_key, max_frac, min_q, max_q, min_hw2 = math.inf, 0.0, math.inf, 0.0, math.inf
    for K in (5.0, 12.0, 20.0):
        for h0 in (17.0, 23.0, 30.0):
            for r in (0.15, 0.22, 0.3):
                for R in (400.0, 700.0, 1000.0):
                    b = {"K": K, "h0": h0, "r": r, "R": R}
                    for x, inside in ((0.012, 1), (0.035, 1), (0.27, 0), (0.305, 0), (0.34, 0)):
                        q = 2 * math.pi * K * h0 ** 2 * x / math.log(R / r)
                        p = {**b, "Q": q}
                        c, s = correct(p), shortcut(p)
                        max_err = max(max_err, gap(c, _rk4(p)))
                        # Jacob's correction of the true drawdown returns the Thiem value (identity).
                        max_jacob = max(max_jacob, gap(s, c - c ** 2 / (2 * h0)))
                        g = gap(c, s)
                        min_key = min(min_key, c)
                        if inside:
                            max_in = max(max_in, g)
                        else:
                            min_out, max_out = min(min_out, g), max(max_out, g)
                            max_frac = max(max_frac, s / h0)
                            min_q, max_q = min(min_q, q), max(max_q, q)
                            hw2 = h0 ** 2 - q * _lnrr(p) / (math.pi * K)
                            min_hw2 = min(min_hw2, hw2 / h0 ** 2)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_err < 1e-8 and max_jacob < 1e-12, (max_err, max_jacob)
    assert max_frac < 0.45 and max_q < 2e4                      # shortcut drawdown well below h_0: not absurd
    assert min_hw2 > 0.25, min_hw2              # the well is never dry: h_w^2 > 0
    assert min_key > 0.2                         # survives round(x, 2) within 2.5 %
    # Anchor (hand-worked): K = 20 m/day, h0 = 30 m, Q = 3000 m^3/day, r = 20 m, R = 500 m:
    # h0^2 - h^2 = 3000 ln 25/(20 pi) = 153.7 m^2 -> h = 27.32 m, s = 2.68 m (Thiem: 2.56 m).
    anchor = correct({"K": 20.0, "h0": 30.0, "Q": 3000.0, "r": 20.0, "R": 500.0})
    assert abs(anchor - 2.681) < 0.002, anchor
    return (f"RK4 Darcy integration vs closed form {max_err:.1e}; Jacob identity {max_jacob:.1e}; "
            f"inside max gap {max_in:.4f}; outside gap {min_out:.3f}-{max_out:.3f}; outside Q "
            f"{min_q:.0f}-{max_q:.0f} m3/day; shortcut <= {max_frac:.2f} h0; min key {min_key:.2f} m; "
            f"min h_w^2/h0^2 {min_hw2:.2f}; anchor s = {anchor:.2f} m")
