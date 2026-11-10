"""Thin-lens focal length -> thick-lens focal length of a near-ball lens (X5, optics).

Parent's method: the lensmaker's equation 1/f = (n - 1)(1/R_1 - 1/R_2), which treats the lens as thin.
Where it fails: when the centre thickness d is comparable with the radii the paraxial focal length is
1/f = (n - 1)[1/R_1 - 1/R_2 + (n - 1) d / (n R_1 R_2)] (Hecht, sec. 6.2). For a symmetric biconvex lens with
d / R = 1.1-1.5 the thin formula is 17-32 % short (n = 1.45-1.75).
Fair from the text: every arm states both radii (equal, symmetric biconvex), the refractive index and the centre
thickness (the trigger, ordinary geometry). Nothing is recalled.
Inside: d / R = 0.02-0.04 (<= 1.2 %). Outside: d / R = 1.1-1.5.
Conventions: "focal length" of a thick lens is the effective (equivalent) focal length, measured from the principal
plane; the back focal length is a different quantity (check() reports how far it is from the key).
Asked quantity: focal length, mm.
check(): paraxial ray trace (y, u) through both surfaces for a ray parallel to the axis; f = -y_in / u_out.
Reference: Hecht, Optics, sec. 6.2; Born & Wolf, Principles of Optics, sec. 4.4.
"""
from __future__ import annotations

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd


def base(rng):
    return {"r": float(rng.choice(range(20, 81, 2))), "n": rnd(rng.uniform(1.45, 1.75), 3)}


def trigger(rng, regime, b):
    f = rng.uniform(0.02, 0.04) if regime == "inside" else rng.uniform(1.1, 1.5)
    return {"d": rnd(b["r"] * f)}


def correct(p):
    r1, r2, n, d = p["r"], -p["r"], p["n"], p["d"]
    return 1 / ((n - 1) * (1 / r1 - 1 / r2 + (n - 1) * d / (n * r1 * r2)))


def shortcut(p):
    return 1 / ((p["n"] - 1) * 2 / p["r"])


def text(p, variant):
    given = ("The lens is not thin: its focal length is 1/f = (n - 1)[1/R_1 - 1/R_2 + (n - 1) d / (n R_1 R_2)], with "
             "R_2 = -R_1 for this lens. " if variant == "given" else "")
    return (f"A symmetric biconvex glass lens has both surfaces with radius of curvature {num(p['r'])} mm, refractive "
            f"index {num(p['n'])} and a centre thickness of {num(p['d'])} mm. It is used in air with paraxial light. "
            f"{given}What is its focal length? Express your answer in mm.")


TEMPLATE = Template("thick_lens", "optics",
                    "thin-lens lensmaker equation -> thick-lens focal length when thickness is comparable with the radii",
                    "mm", base, trigger, text, correct, shortcut)


def _trace(p):
    n, r, d = p["n"], p["r"], p["d"]
    y, u = 1.0, 0.0                                                     # unit height, parallel
    u = (1 * u - y * (n - 1) / r) / n                                   # refraction at R1 = +r (n u' = u - y (n-1)/R)
    y = y + d * u                                                       # transfer
    u = (n * u - y * (1 - n) / (-r)) / 1                                # refraction at R2 = -r
    return -1.0 / u


def check() -> str:
    import random
    rng = random.Random("x5-check:thick_lens")
    worst = 0.0; bfl = []; gi, go = [], []
    for _ in range(100):
        pin, pout = pair(TEMPLATE, rng)
        worst = max(worst, abs(_trace(pout) / correct(pout) - 1), abs(_trace(pin) / correct(pin) - 1))
        f = correct(pout)
        bfl.append(gap(f, f * (1 - (pout["n"] - 1) * pout["d"] / (pout["n"] * pout["r"]))))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert worst < 1e-9 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP
    return (f"ray trace vs formula {worst:.1e}, back focal length differs by {min(bfl):.2f}-{max(bfl):.2f}, "
            f"inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}")
