"""Gaussian plume with ground reflection -> plume trapped under a mixing lid.

Parent's method: the ground-level centreline concentration of an elevated point source with total
reflection at the ground, C = Q/(pi u sigma_y sigma_z) exp(-H^2/(2 sigma_z^2)).
Where it fails: once sigma_z is comparable to the mixing height h, the plume is also reflected by
the capping inversion. Summing the image sources (at +-H + 2nh) gives the key; when
sigma_z >= 1.5 h the sum is uniform in the vertical to better than 1e-4, i.e. the well-mixed
formula C = Q/(sqrt(2 pi) u sigma_y h) (the sigma_z >= 1.6 h criterion of regulatory ISC-type models;
see the 2026-09-27 note).
Fair from the text: both variants state the inversion height, H, Q, u and the sigma power laws,
so sigma_z/h is computable at the stated distance.
Inside: sigma_z = 0.1-0.25 h; the first lid image is >= 6.6 sigma_z away, so the parent's
formula is exact to better than 1e-9 (and H/sigma_z <= 2.9 keeps the concentration finite).
Outside: sigma_z = 1.5-3 h; the parent's formula is at least 44 % below the key.
Conventions: image sum and well-mixed formula agree within 1e-4 outside (`check`); the
dispersion parameters are given, so no Pasquill-Gifford curve choice enters.
Reference: Turner (1970), Workbook of Atmospheric Dispersion Estimates, eqs. 5.1-5.6;
Seinfeld & Pandis, ch. 18 (multiple reflections).
Related mechanism: finite_line_source (same Gaussian-plume family, different failure).

Review (2026-09-26, independent reviewer): inside keys reached about 8 mg m^-3 of SO2 (q = 300 g/s,
u = 3 m/s, x ~ 550 m), implausibly high for a ground-level concentration. Emission is now
50-150 g/s and the wind 4-8 m/s; only the prefactor changes, so the gaps are unchanged.

Review (2026-09-27, domain + measurement-design review): the docstring attributed the sigma_z >= 1.6 h
criterion to Turner (1970). Turner's workbook instead applies uniform mixing beyond 2 x_L, where
sigma_z(x_L) = 0.47 L (roughly sigma_z of 0.85 h at 2 x_L); the 1.6 h criterion comes from regulatory
ISC-type models. Attribution corrected here and in check(); the key and gaps are unchanged.
"""
from __future__ import annotations

import math

from .base import OUTSIDE_MIN_GAP, Template, gap, num, rnd

A, P, C, Q_EXP = 0.16, 0.9, 0.14, 0.85      # sigma_y = A x^P, sigma_z = C x^Q_EXP (m)


def base(rng):
    return {"q": float(rng.choice(range(50, 160, 10))), "u": rng.choice([4.0, 4.5, 5.0, 6.0, 7.0, 8.0]),
            "hs": float(rng.choice(range(40, 90, 5))), "h": float(rng.choice(range(300, 650, 50)))}


def trigger(rng, regime, base):
    ratio = rng.uniform(0.1, 0.25) if regime == "inside" else rng.uniform(1.5, 3.0)
    return {"x": rnd((ratio * base["h"] / C) ** (1 / Q_EXP))}


def _sig(p):
    return A * p["x"] ** P, C * p["x"] ** Q_EXP


def shortcut(p):
    sy, sz = _sig(p)
    return p["q"] * 1e6 / (math.pi * p["u"] * sy * sz) * math.exp(-p["hs"] ** 2 / (2 * sz ** 2))


def correct(p):
    sy, sz = _sig(p)
    s = sum(math.exp(-(2 * n * p["h"] - p["hs"]) ** 2 / (2 * sz ** 2)) +
            math.exp(-(2 * n * p["h"] + p["hs"]) ** 2 / (2 * sz ** 2)) for n in range(-60, 61))
    return p["q"] * 1e6 / (2 * math.pi * p["u"] * sy * sz) * s


def text(p, variant):
    given = ("Because σ_z at this distance exceeds the mixing height, the inversion traps the plume, which is "
             "then uniformly mixed between the ground and the inversion: use C = Q/(√(2π) u σ_y h), where h is "
             "the inversion height. " if variant == "given" else "")
    return (f"A stack emits sulfur dioxide at {num(p['q'])} g s^-1 from an effective height of {num(p['hs'])} m into "
            f"a mean wind of {num(p['u'])} m s^-1. The mixed layer is capped by an elevated inversion at "
            f"{num(p['h'])} m above the ground. For the prevailing stability the dispersion parameters are "
            f"σ_y = 0.16x^0.9 and σ_z = 0.14x^0.85, with σ and x in metres. {given}Estimate the ground-level "
            f"concentration on the plume centreline {num(p['x'])} m downwind of the stack. "
            f"Express your answer in μg m^-3.")


def _fourier(p):
    """Independent: Poisson-summed image series, sum_n g(y + 2nh) = (sqrt(2pi) s/2h) sum_k ..."""
    sy, sz = _sig(p)
    k_terms = sum(math.exp(-(math.pi * k * sz / p["h"]) ** 2 / 2) * math.cos(math.pi * k * p["hs"] / p["h"])
                  for k in range(1, 4000))
    s = 2 * (math.sqrt(2 * math.pi) * sz / (2 * p["h"])) * (1 + 2 * k_terms)
    return p["q"] * 1e6 / (2 * math.pi * p["u"] * sy * sz) * s


def check() -> str:
    max_in, min_out, mixed = 0.0, math.inf, 0.0
    for q in [50.0, 150.0]:
        for u in [4.0, 8.0]:
            for hs in [40.0, 85.0]:
                for h in [300.0, 450.0, 600.0]:
                    for ratio in [0.095, 0.17, 0.26, 1.42, 2.0, 3.15]:   # rounding moves sigma_z by < 5 %
                        p = {"q": q, "u": u, "hs": hs, "h": h, "x": (ratio * h / C) ** (1 / Q_EXP)}
                        key = correct(p)
                        assert math.isclose(_fourier(p), key, rel_tol=1e-8), p
                        g = gap(key, shortcut(p))
                        if ratio < 1:
                            max_in = max(max_in, g)
                            continue
                        min_out = min(min_out, g)
                        sy, _ = _sig(p)
                        mixed = max(mixed, gap(key, q * 1e6 / (math.sqrt(2 * math.pi) * u * sy * h)))
    assert max_in <= 1e-6 and min_out >= OUTSIDE_MIN_GAP and mixed < 0.05, (max_in, min_out, mixed)
    # Anchor: the ISC-type well-mixed criterion sigma_z = 1.6 h -- image sum is uniform to < 1e-4 there.
    p = {"q": 100.0, "u": 5.0, "hs": 60.0, "h": 400.0, "x": (1.6 * 400 / C) ** (1 / Q_EXP)}
    sy, _ = _sig(p)
    assert gap(correct(p), 100e6 / (math.sqrt(2 * math.pi) * 5 * sy * 400)) < 1e-4
    return (f"max inside gap {max_in:.1e}, min outside gap {min_out:.3f}, well-mixed formula vs image sum "
            f"outside {mixed:.1e}; ISC sigma_z=1.6h anchor")


TEMPLATE = Template("plume_mixing_lid", "air_quality",
                    "Gaussian plume with ground reflection -> trapping under an inversion once sigma_z ~ h",
                    "ug/m^3", base, trigger, text, correct, shortcut)
