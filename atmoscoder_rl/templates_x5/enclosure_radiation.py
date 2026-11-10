"""Small body in a large enclosure -> radiation between concentric spheres of comparable size (X5, heat transfer).

Parent's method: net radiation from a body to its surroundings q = eps_1 sigma A_1 (T_1^4 - T_2^4), the
"small object in a large enclosure" result.
Where it fails: it drops the enclosure's reflection, valid only when A_1 / A_2 -> 0. For concentric spheres the
exact two-surface grey-diffuse result is q = sigma A_1 (T_1^4 - T_2^4) / (1/eps_1 + (A_1/A_2)(1/eps_2 - 1))
(Incropera, Table 13.3). With a polished (low-emissivity) shell close around the sphere the small-body
answer is 4.5-13x the key.
Fair from the text: every arm states both radii, both emissivities and both temperatures. The trigger is the
shell's inner radius, ordinary geometry.
Inside: r_2 / r_1 = 25-40 (correction 0.4-2 %). Outside: r_2 / r_1 = 1.10-1.30 with eps_2 = 0.05-0.10.
Conventions: grey, diffuse, isothermal surfaces is the textbook model and the text states it; there is no
competing convention for the key.
Asked quantity: net radiative heat transfer from the sphere, W.
check(): radiosity network (two nodes) solved as a 2x2 linear system, independent of the closed form.
Reference: Incropera & DeWitt, sec. 13.3, Table 13.3 (concentric spheres).
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd

SIGMA = 5.67e-8


def base(rng):
    return {"r1": rnd(rng.uniform(5, 15)), "e1": rnd(rng.uniform(0.6, 0.9)), "e2": rnd(rng.uniform(0.05, 0.10)),
            "t1": float(rng.choice(range(500, 801, 10))), "t2": float(rng.choice(range(290, 331, 5)))}


def trigger(rng, regime, b):
    f = rng.uniform(25, 40) if regime == "inside" else rng.uniform(1.10, 1.30)
    return {"r2": rnd(b["r1"] * f, 3)}


def _a(r_cm):
    return 4 * math.pi * (r_cm / 100) ** 2


def correct(p):
    ratio = (p["r1"] / p["r2"]) ** 2
    return SIGMA * _a(p["r1"]) * (p["t1"] ** 4 - p["t2"] ** 4) / (1 / p["e1"] + ratio * (1 / p["e2"] - 1))


def shortcut(p):
    return p["e1"] * SIGMA * _a(p["r1"]) * (p["t1"] ** 4 - p["t2"] ** 4)


def text(p, variant):
    given = ("Because the shell is close to the sphere and reflective, its reflection cannot be neglected: "
             "q = sigma A_1 (T_1^4 - T_2^4) / (1/eps_1 + (A_1/A_2)(1/eps_2 - 1)). " if variant == "given" else "")
    return (f"A sphere of radius {num(p['r1'])} cm with emissivity {num(p['e1'])} is held at {num(p['t1'])} K at the "
            f"centre of an evacuated spherical shell of inner radius {num(p['r2'])} cm, whose inner surface has "
            f"emissivity {num(p['e2'])} and is held at {num(p['t2'])} K. Both surfaces are grey and diffuse. Take "
            f"sigma = 5.67 × 10^-8 W m^-2 K^-4. {given}What is the net rate of heat transfer by radiation from the "
            f"sphere to the shell? Express your answer in W.")


TEMPLATE = Template("enclosure_radiation", "heat_transfer",
                    "small body in large enclosure -> two-surface grey exchange when the enclosure is close and reflective",
                    "W", base, trigger, text, correct, shortcut)


def _network(p):
    """Radiosities J1, J2 from the two-node network; q = (Eb1 - J1) / R1."""
    a1, a2 = _a(p["r1"]), _a(p["r2"])
    eb1, eb2 = SIGMA * p["t1"] ** 4, SIGMA * p["t2"] ** 4
    r1 = (1 - p["e1"]) / (p["e1"] * a1); r12 = 1 / a1; r2 = (1 - p["e2"]) / (p["e2"] * a2)
    # (Eb1 - J1)/r1 = (J1 - J2)/r12 ; (J1 - J2)/r12 = (J2 - Eb2)/r2
    m = [[1 / r1 + 1 / r12, -1 / r12], [-1 / r12, 1 / r12 + 1 / r2]]
    v = [eb1 / r1, eb2 / r2]
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    j1 = (v[0] * m[1][1] - m[0][1] * v[1]) / det
    return (eb1 - j1) / r1


def check() -> str:
    import random
    rng = random.Random("x5-check:enclosure_radiation")
    worst = 0.0; gi, go = [], []
    for _ in range(50):
        pin, pout = pair(TEMPLATE, rng)
        worst = max(worst, abs(_network(pout) / correct(pout) - 1), abs(_network(pin) / correct(pin) - 1))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert worst < 1e-9 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP
    return f"network vs closed form {worst:.1e}, inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}"
