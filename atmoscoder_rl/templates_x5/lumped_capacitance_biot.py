"""Lumped-capacitance cooling of a sphere -> transient conduction with internal gradients (X5, heat transfer).

Parent's method: the sphere's temperature is uniform and decays as theta = theta_i exp(-t / tau),
tau = rho c R / (3 h) (V/A = R/3) -- the first transient model of every heat-transfer course.
Where it fails: the lumped model needs Bi = h (V/A) / k < 0.1. When conduction inside is slow (low k), the
centre lags the surface; the centre excess is the exact series theta_c / theta_i = sum C_n exp(-z_n^2 Fo),
C_n = 4 (sin z_n - z_n cos z_n) / (2 z_n - sin 2 z_n), 1 - z_n cot z_n = Bi_R, Bi_R = h R / k, Fo = k t / (rho c R^2)
(Incropera, sec. 5.6).
Fair from the text: every arm states R, rho, c, k, h, the initial and fluid temperatures and t; nothing is recalled.
The trigger is the conductivity k, ordinary material data.
Inside: h R / k = 0.01-0.02 (k ~ 70-290, aluminium / AlN class): centre within 1.1 % of the lumped value.
Outside: h R / k = 1.0-2.0 (k ~ 0.7-2.9: glass, granite, ceramic) with t / tau = 1.2-2.0, so Fo = (t/tau) / (3 h R / k)
= 0.2-0.67 and the centre excess is 1.6-2.8x the lumped value. (Ranges revised after review 2026-10-12: earlier draws
paired rho 2200-3000 with k < 0.7, which no dense solid has.)
Conventions: the one-term approximation (Heisler chart) is within 2.5 % of the series for Fo >= 0.2 (check()).
Asked quantity: the centre's excess over the fluid temperature, in K (a difference, rule 6).
Second wrong method: the lumped model with the Biot-corrected effective h (h / (1 + Bi/5)) -- check() reports its miss.
check(): implicit finite differences in r (400 shells), independent of the series.
Reference: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, sec. 5.2-5.6.
"""
from __future__ import annotations

import math

from ..templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, pair, rnd


def base(rng):
    r = rnd(rng.uniform(0.030, 0.045))
    rho, c, h = rnd(rng.uniform(2200, 3000)), rnd(rng.uniform(800, 950)), rnd(rng.uniform(45, 65))
    tau = rho * c * r / (3 * h)
    return {"r": r, "rho": rho, "c": c, "h": h, "dt0": float(rng.choice(range(60, 101, 5))),
            "t": rnd(tau * rng.uniform(1.2, 2.0))}


def trigger(rng, regime, b):
    hr = b["h"] * b["r"]
    return {"k": rnd(hr / rng.uniform(0.01, 0.02)) if regime == "inside" else rnd(hr / rng.uniform(1.0, 2.0))}


def _roots(bi, n=60):
    out = []
    for i in range(1, n + 1):
        lo, hi = (i - 1) * math.pi + 1e-9, i * math.pi - 1e-9
        f = lambda z: 1 - z / math.tan(z) - bi
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        out.append((lo + hi) / 2)
    return out


def correct(p):
    bi = p["h"] * p["r"] / p["k"]
    fo = p["k"] * p["t"] / (p["rho"] * p["c"] * p["r"] ** 2)
    s = sum(4 * (math.sin(z) - z * math.cos(z)) / (2 * z - math.sin(2 * z)) * math.exp(-z * z * fo) for z in _roots(bi))
    return p["dt0"] * s


def shortcut(p):
    return p["dt0"] * math.exp(-p["t"] * 3 * p["h"] / (p["rho"] * p["c"] * p["r"]))


def text(p, variant):
    given = ("Conduction inside the sphere is not fast compared with the surface heat transfer, so the sphere is not at "
             "a uniform temperature: the centre excess is theta_i * sum_n C_n exp(-z_n^2 Fo), with Fo = k t / (rho c R^2), "
             "C_n = 4 (sin z_n - z_n cos z_n) / (2 z_n - sin 2z_n) and z_n the positive roots of 1 - z cot z = h R / k. "
             if variant == "given" else "")
    return (f"A solid sphere of radius {num(p['r'] * 100)} cm, density {num(p['rho'])} kg m^-3, specific heat "
            f"{num(p['c'])} J kg^-1 K^-1 and thermal conductivity {num(p['k'])} W m^-1 K^-1 is initially at a uniform "
            f"temperature {num(p['dt0'])} K above that of the surrounding fluid. At t = 0 it is placed in the fluid, which "
            f"stays at constant temperature; the heat-transfer coefficient at the surface is {num(p['h'])} W m^-2 K^-1. "
            f"{given}By how many kelvin does the temperature at the centre of the sphere exceed the fluid temperature "
            f"after {num(p['t'])} s? Express your answer in K.")


TEMPLATE = Template("lumped_capacitance_biot", "heat_transfer",
                    "lumped-capacitance cooling -> transient conduction with internal gradient at Biot number of order 1",
                    "K", base, trigger, text, correct, shortcut)


def _fd_centre(p, n=400, steps=4000):
    """Implicit finite volumes in r; returns the centre excess."""
    r_out, alpha = p["r"], p["k"] / (p["rho"] * p["c"])
    dr, dt = r_out / n, p["t"] / steps
    rc = [(i + 0.5) * dr for i in range(n)]
    vol = [4 / 3 * math.pi * ((i + 1) ** 3 - i ** 3) * dr ** 3 for i in range(n)]
    face = [4 * math.pi * ((i + 1) * dr) ** 2 for i in range(n)]
    theta = [p["dt0"]] * n
    for _ in range(steps):
        a, b, cc, d = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
        for i in range(n):
            cap = p["rho"] * p["c"] * vol[i] / dt
            b[i], d[i] = cap, cap * theta[i]
            if i > 0:
                g = p["k"] * face[i - 1] / dr
                a[i] = -g; b[i] += g
            if i < n - 1:
                g = p["k"] * face[i] / dr
                cc[i] = -g; b[i] += g
            else:
                g = 1 / (dr / 2 / (p["k"] * face[i]) + 1 / (p["h"] * face[i]))
                b[i] += g
        for i in range(1, n):                      # Thomas algorithm
            w = a[i] / b[i - 1]
            b[i] -= w * cc[i - 1]; d[i] -= w * d[i - 1]
        theta[n - 1] = d[n - 1] / b[n - 1]
        for i in range(n - 2, -1, -1):
            theta[i] = (d[i] - cc[i] * theta[i + 1]) / b[i]
    # centre by extrapolating the two innermost cells (theta is even in r)
    return theta[0] + (theta[0] - theta[1]) * (rc[0] ** 2) / (rc[1] ** 2 - rc[0] ** 2)


def check() -> str:
    import random
    rng = random.Random("x5-check:lumped_capacitance_biot")
    worst_fd = worst_one = 0.0
    gaps_in, gaps_out = [], []
    for _ in range(6):
        pin, pout = pair(TEMPLATE, rng)
        gaps_in.append(gap(correct(pin), shortcut(pin))); gaps_out.append(gap(correct(pout), shortcut(pout)))
        fd = _fd_centre(pout, n=200, steps=2000)
        worst_fd = max(worst_fd, abs(fd / correct(pout) - 1))
        bi = pout["h"] * pout["r"] / pout["k"]; z = _roots(bi, 1)[0]
        fo = pout["k"] * pout["t"] / (pout["rho"] * pout["c"] * pout["r"] ** 2)
        one = pout["dt0"] * 4 * (math.sin(z) - z * math.cos(z)) / (2 * z - math.sin(2 * z)) * math.exp(-z * z * fo)
        worst_one = max(worst_one, abs(one / correct(pout) - 1))
    # Incropera Table 5.1 anchor: sphere, Bi = 1.0 -> zeta_1 = 1.5708, C_1 = 1.2732
    z = _roots(1.0, 1)[0]
    c1 = 4 * (math.sin(z) - z * math.cos(z)) / (2 * z - math.sin(2 * z))
    assert abs(z - 1.5708) < 1e-3 and abs(c1 - 1.2732) < 1e-3
    assert worst_fd < 0.01 and worst_one < 0.05 and max(gaps_in) <= INSIDE_MAX_GAP and min(gaps_out) >= OUTSIDE_MIN_GAP
    return (f"FD vs series {worst_fd:.4f}, one-term vs series {worst_one:.4f}, inside gap <= {max(gaps_in):.4f}, "
            f"outside gap >= {min(gaps_out):.2f}")
