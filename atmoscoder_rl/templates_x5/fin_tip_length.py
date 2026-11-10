"""Infinitely long pin fin -> fin of finite length with an insulated tip (X5, heat transfer).

Parent's method: the heat a pin fin conducts from its base is q = sqrt(h P k A) theta_b (the infinite-fin result,
P = pi D, A = pi D^2 / 4), the closed form most textbooks lead with.
Where it fails: it assumes the fin is long enough that its tip reaches the fluid temperature (m L > ~2.65,
m = sqrt(4 h / (k D))). A short fin with an insulated tip carries q = sqrt(h P k A) theta_b tanh(m L)
(Incropera, Table 3.4, case B).
Fair from the text: every arm states D, L, k, h, the base excess and that heat loss from the tip face is negligible
(the boundary condition that fixes the key; stated in all arms, it does not name the long-fin assumption).
The trigger is the fin length, ordinary geometry.
Inside: m L = 3.2-4.0 (tanh >= 0.997). Outside: m L = 0.45-0.70 (tanh 0.42-0.60): the infinite-fin answer is 1.6-2.4x the key.
Conventions: with the tip insulated the result is exact for the 1-D fin; the corrected-length convective tip does not apply.
Asked quantity: heat rate from the base, W.
check(): finite-difference solution of theta'' = m^2 theta with theta(0) = theta_b, theta'(L) = 0, independent of tanh.
Reference: Incropera & DeWitt, sec. 3.6.2, Table 3.4.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd


def base(rng):
    return {"d": rnd(rng.uniform(3, 8)), "k": float(rng.choice((15, 20, 25, 45, 60))),
            "h": rnd(rng.uniform(40, 120)), "tb": float(rng.choice(range(40, 101, 5)))}


def _m(p):
    return math.sqrt(4 * p["h"] / (p["k"] * p["d"] * 1e-3))


def trigger(rng, regime, b):
    ml = rng.uniform(3.2, 4.0) if regime == "inside" else rng.uniform(0.45, 0.70)
    return {"l": rnd(ml / _m(b) * 1000)}                                 # mm


def _qinf(p):
    d = p["d"] * 1e-3
    return math.sqrt(p["h"] * math.pi * d * p["k"] * math.pi * d * d / 4) * p["tb"]


def correct(p):
    return _qinf(p) * math.tanh(_m(p) * p["l"] * 1e-3)


def shortcut(p):
    return _qinf(p)


def text(p, variant):
    given = ("The fin is not long enough for its tip to reach the air temperature: with an insulated tip the heat rate is "
             "sqrt(h P k A) theta_b tanh(m L), where m = sqrt(h P / (k A)). " if variant == "given" else "")
    return (f"A cylindrical pin fin of diameter {num(p['d'])} mm and length {num(p['l'])} mm projects from a wall whose "
            f"temperature is {num(p['tb'])} K above that of the surrounding air. The fin's thermal conductivity is "
            f"{num(p['k'])} W m^-1 K^-1 and the heat-transfer coefficient on its surface is {num(p['h'])} W m^-2 K^-1; "
            f"heat loss from the end face at the tip is negligible. {given}How much heat does the fin conduct away "
            f"from the wall? Express your answer in W.")


TEMPLATE = Template("fin_tip_length", "heat_transfer",
                    "infinitely long fin -> finite fin with insulated tip when m L is of order 1",
                    "W", base, trigger, text, correct, shortcut)


def _fd(p, n=4000):
    """theta'' = m^2 theta on [0, L], theta(0) = tb, theta'(L) = 0; q = -k A theta'(0)."""
    L, m = p["l"] * 1e-3, _m(p)
    dx = L / n
    a = [1.0] * (n + 1); b = [-(2 + (m * dx) ** 2)] * (n + 1); c = [1.0] * (n + 1); d = [0.0] * (n + 1)
    b[0], c[0], d[0] = 1.0, 0.0, p["tb"]
    a[n], b[n] = 2.0, -(2 + (m * dx) ** 2)                               # ghost node for theta'(L) = 0
    for i in range(1, n + 1):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]; d[i] -= w * d[i - 1]
    th = [0.0] * (n + 1); th[n] = d[n] / b[n]
    for i in range(n - 1, -1, -1):
        th[i] = (d[i] - c[i] * th[i + 1]) / b[i]
    area = math.pi * (p["d"] * 1e-3) ** 2 / 4
    return -p["k"] * area * (-3 * th[0] + 4 * th[1] - th[2]) / (2 * dx)


def check() -> str:
    import random
    rng = random.Random("x5-check:fin_tip_length")
    worst = 0.0; gi, go = [], []
    for _ in range(20):
        pin, pout = pair(TEMPLATE, rng)
        worst = max(worst, abs(_fd(pout) / correct(pout) - 1))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert worst < 1e-3 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP
    return f"FD vs tanh {worst:.2e}, inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}"
