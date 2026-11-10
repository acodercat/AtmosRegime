"""Thin-wall hoop stress -> Lame stress in a thick-walled cylinder (X5, structures).

Parent's method: hoop stress in a pressurised cylinder sigma = p r / t (p d / (2t)), the thin-wall formula.
Where it fails: it assumes the stress is uniform through the wall, t / r << 1 (< 0.1). For a thick wall the hoop
stress peaks at the bore: sigma_max = p (r_o^2 + r_i^2) / (r_o^2 - r_i^2) (Lame, open or closed ends, no external pressure).
Fair from the text: every arm states the bore radius, wall thickness and internal pressure. The trigger is the wall
thickness, ordinary geometry; nothing is recalled.
Inside: t / r_i = 0.025-0.035 (Lame within 1.8 % of p r_i / t; pressure 2-6 MPa keeps the thin-wall stress at 60-240 MPa). Outside: t / r_i = 1.0-1.5 (thin formula 40-52 % low).
Conventions: the thin formula with the mean radius, p (r_i + t/2) / t, is 9-16 % low on the outside grid (a second
wrong method, rule 11; check() reports it). Lame is exact for linear elasticity.
Asked quantity: largest hoop stress in the wall, MPa.
check(): radial equilibrium d(r sigma_r)/dr = sigma_theta with compatibility, solved by finite differences on
u'' + u'/r - u/r^2 = 0 with sigma_r(r_i) = -p, sigma_r(r_o) = 0 (plane stress), independent of the closed form.
Reference: Timoshenko & Goodier, Theory of Elasticity, sec. 28; Budynas, Shigley's Mechanical Engineering Design, sec. 3-16.
"""
from __future__ import annotations

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd


def base(rng):
    return {"ri": float(rng.choice(range(20, 101, 5))), "p": rnd(rng.uniform(2, 6))}


def trigger(rng, regime, b):
    f = rng.uniform(0.025, 0.035) if regime == "inside" else rng.uniform(1.0, 1.5)
    return {"t": rnd(b["ri"] * f)}


def correct(p):
    ro = p["ri"] + p["t"]
    return p["p"] * (ro ** 2 + p["ri"] ** 2) / (ro ** 2 - p["ri"] ** 2)


def shortcut(p):
    return p["p"] * p["ri"] / p["t"]


def text(p, variant):
    given = ("The wall is not thin compared with the bore, so the hoop stress is not uniform through it: at the bore "
             "sigma = p (r_o^2 + r_i^2) / (r_o^2 - r_i^2). " if variant == "given" else "")
    return (f"A steel cylinder with an internal radius of {num(p['ri'])} mm and a wall thickness of {num(p['t'])} mm "
            f"carries an internal pressure of {num(p['p'])} MPa; the outside is at zero gauge pressure."
            f" {given}What is the largest circumferential (hoop) stress in the wall? Express "
            f"your answer in MPa.")


TEMPLATE = Template("thick_wall_hoop", "structures",
                    "thin-wall hoop stress -> Lame thick-wall stress at wall thickness comparable to the bore",
                    "MPa", base, trigger, text, correct, shortcut)


def _fd(p, n=2000):
    """u'' + u'/r - u/r^2 = 0 on [ri, ro]; plane stress sigma_r = E/(1-v^2)(u' + v u/r); find sigma_theta(ri)."""
    nu = 0.3
    ri, ro = p["ri"], p["ri"] + p["t"]
    h = (ro - ri) / n
    rs = [ri + i * h for i in range(n + 1)]
    # Unknowns u_0..u_n; interior: central differences; boundaries: sigma_r with one-sided second-order derivative.
    size = n + 1
    rows = []
    for i in range(1, n):
        r = rs[i]
        rows.append({i - 1: 1 / h ** 2 - 1 / (2 * h * r), i: -2 / h ** 2 - 1 / r ** 2, i + 1: 1 / h ** 2 + 1 / (2 * h * r), "rhs": 0.0})
    # E/(1-v^2) scaled to 1: u' + nu u / r = sigma_r
    rows.insert(0, {0: -3 / (2 * h) + nu / rs[0], 1: 4 / (2 * h), 2: -1 / (2 * h), "rhs": -p["p"]})
    rows.append({n: 3 / (2 * h) + nu / rs[n], n - 1: -4 / (2 * h), n - 2: 1 / (2 * h), "rhs": 0.0})
    # banded Gaussian elimination (bandwidth 2)
    a = [dict(rw) for rw in rows]
    for i in range(size):
        piv = a[i][i]
        for j in range(i + 1, min(i + 3, size)):
            if i in a[j]:
                f = a[j][i] / piv
                for k, v in a[i].items():
                    if k != "rhs":
                        a[j][k] = a[j].get(k, 0.0) - f * v
                a[j]["rhs"] -= f * a[i]["rhs"]
    u = [0.0] * size
    for i in range(size - 1, -1, -1):
        s = a[i]["rhs"] - sum(v * u[k] for k, v in a[i].items() if k not in ("rhs", i) and k > i)
        u[i] = s / a[i][i]
    du0 = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * h)
    return u[0] / rs[0] + nu * du0                                      # sigma_theta = (u/r + nu u') in the same scale


def check() -> str:
    import random
    rng = random.Random("x5-check:thick_wall_hoop")
    worst = 0.0; mean_r = []; gi, go = [], []
    for _ in range(30):
        pin, pout = pair(TEMPLATE, rng)
        worst = max(worst, abs(_fd(pout) / correct(pout) - 1))
        mean_r.append(gap(correct(pout), pout["p"] * (pout["ri"] + pout["t"] / 2) / pout["t"]))
        gi.append(gap(correct(pin), shortcut(pin))); go.append(gap(correct(pout), shortcut(pout)))
    assert worst < 1e-3 and max(gi) <= INSIDE_MAX_GAP and min(go) >= OUTSIDE_MIN_GAP and min(mean_r) > 0.05
    return (f"FD vs Lame {worst:.1e}, mean-radius thin formula miss {min(mean_r):.2f}-{max(mean_r):.2f}, "
            f"inside gap <= {max(gi):.4f}, outside gap >= {min(go):.2f}")
