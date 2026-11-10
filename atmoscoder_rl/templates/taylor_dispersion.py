"""Near-field linear plume growth -> Taylor's statistical diffusion at long travel times.

Parent's method: a particle keeps its initial lateral velocity, so sigma_y = sigma_v t, t = x/U.
Where it fails: once the travel time is long compared with the Lagrangian integral time scale T_L,
velocities decorrelate and the spread grows diffusively. Taylor (1921) with an exponential
autocorrelation gives sigma_y^2 = 2 sigma_v^2 T_L^2 (t/T_L - 1 + exp(-t/T_L)).
Fair from the text: both variants state sigma_v, T_L (named as the Lagrangian integral time scale)
and U, so t/T_L is computable. The autocorrelation shape is NOT stated, on purpose (rule 4): the
outside arm keeps t/T_L >= 14, where every autocorrelation with integral scale T_L gives
sigma_y^2 = 2 sigma_v^2 T_L (t - c T_L) with c of order 1, so the far-field law
sigma_y = sqrt(2 sigma_v^2 T_L t), the exponential form and a Gaussian autocorrelation agree
within 5 % (`check` reports the spread).
Inside: t/T_L = 0.05-0.08, where sigma_y = sigma_v t (1 - t/(6 T_L) + ...) differs by < 1.5 %
for any smooth autocorrelation.
Reference: Taylor (1921) Proc. London Math. Soc. 20, 196; Seinfeld & Pandis, Atmospheric
Chemistry and Physics, ch. 18 (statistical theory of turbulent diffusion).
Related mechanism: none in this set.

Review (2026-09-26, independent reviewer): inside t/T_L started at 0.02, so the inside distance
could be 4 m (0.02 x 100 s x 2 m/s) and the key about 1 m -- "sigma_y at 4 m downwind" reads
oddly. The inside floor is now t/T_L = 0.05 (distance >= 10 m, typically 20-120 m); the
inside gap bound (< 1.5 %) is unchanged, since it is largest at the upper end.
"""
from __future__ import annotations

import math

from .base import OUTSIDE_MIN_GAP, Template, gap, num, rnd


def base(rng):
    return {"sv": rnd(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5])),
            "tl": float(rng.choice(range(100, 310, 10))),
            "u": rng.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])}


def trigger(rng, regime, base):
    ratio = rng.uniform(0.05, 0.08) if regime == "inside" else rng.uniform(15, 30)
    return {"x": rnd(ratio * base["tl"] * base["u"])}


def shortcut(p):
    return p["sv"] * p["x"] / p["u"]


def correct(p):
    r = p["x"] / p["u"] / p["tl"]
    return p["sv"] * p["tl"] * math.sqrt(2 * (r + math.expm1(-r)))


def text(p, variant):
    given = ("Because the travel time t = x/U is long compared with the Lagrangian time scale T_L, use Taylor's "
             "result for an exponential velocity autocorrelation: σ_y^2 = 2σ_v^2 T_L^2 (t/T_L - 1 + exp(-t/T_L)). "
             if variant == "given" else "")
    return (f"Particles are released continuously from a point source into horizontally homogeneous, stationary "
            f"turbulence. The standard deviation of the crosswind velocity fluctuations is {num(p['sv'])} m s^-1 "
            f"and their Lagrangian integral time scale is {num(p['tl'])} s. The particles are carried downwind by a "
            f"mean wind of {num(p['u'])} m s^-1. {given}Estimate the standard deviation σ_y of the crosswind "
            f"spread of the particles at a distance of {num(p['x'])} m downwind of the source. "
            f"Express your answer in m.")


def _numeric(p, autocorr):
    """sigma^2 = 2 sigma_v^2 int_0^t (t - tau) R(tau) dtau, Simpson in tau."""
    t = p["x"] / p["u"]
    n = 20000
    h = t / n
    f = [(t - i * h) * autocorr(i * h, p["tl"]) for i in range(n + 1)]
    s = f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2])
    return p["sv"] * math.sqrt(2 * s * h / 3)


def check() -> str:
    expo = lambda tau, tl: math.exp(-tau / tl)
    gauss = lambda tau, tl: math.exp(-math.pi * tau ** 2 / (4 * tl ** 2))   # integral scale tl
    max_in, min_out, far, gs = 0.0, math.inf, 0.0, 0.0
    for sv in [0.5, 1.5]:
        for tl in [100.0, 200.0, 300.0]:
            for u in [2.0, 3.5, 5.0]:
                for ratio in [0.048, 0.065, 0.084, 14.0, 20.0, 31.5]:   # rounding moves x by <= 5 %
                    p = {"sv": sv, "tl": tl, "u": u, "x": ratio * tl * u}
                    key = correct(p)
                    assert math.isclose(_numeric(p, expo), key, rel_tol=1e-7), p
                    g = gap(key, shortcut(p))
                    if ratio < 1:
                        max_in = max(max_in, g)
                        continue
                    min_out = min(min_out, g)
                    far = max(far, gap(key, sv * math.sqrt(2 * tl * p["x"] / u)))
                    gs = max(gs, gap(key, _numeric(p, gauss)))
    assert max_in <= 0.015 and min_out >= OUTSIDE_MIN_GAP and far < 0.05 and gs < 0.05, (max_in, min_out, far, gs)
    # Anchor: exponential autocorrelation at t = T_L gives sigma^2/(sigma_v T_L)^2 = 2/e.
    assert math.isclose(correct({"sv": 1.0, "tl": 1.0, "u": 1.0, "x": 1.0}) ** 2, 2 / math.e, rel_tol=1e-12)
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.2f}, outside spread vs key: "
            f"far-field law {far:.4f}, Gaussian autocorrelation {gs:.4f}; t=T_L anchor 2/e")


TEMPLATE = Template("taylor_dispersion", "boundary_layer",
                    "near-field sigma_y = sigma_v t -> Taylor diffusion when travel time >> Lagrangian time scale",
                    "m", base, trigger, text, correct, shortcut)
