"""Gravitational settling alone -> settling plus Brownian diffusion across a surface boundary layer
(spec: MECHANISMS.md; extended set, 2026-10-01).

Parent's method: a particle's deposition velocity onto a nearby surface equals its slip-corrected
gravitational settling speed, v_s = rho_p g d^2 C(d) / (18 mu), C(d) the Cunningham slip correction.
Where it fails: close to a surface, the air is no longer turbulently mixed right down to the wall;
transport across a thin stagnant layer of stated thickness delta next to the surface is by settling
and Brownian diffusion together, not settling alone. For steady drift v_s plus diffusivity D across
a layer of thickness delta with a perfectly absorbing surface (the thin-layer limit of Lai & Nazaroff
2000), the exact deposition velocity is v_d = v_s / (1 - exp(-v_s delta / D)), D = k_B T C(d)/(3 pi
mu d) the slip-corrected Stokes-Einstein diffusivity. In terms of the layer Peclet number
Pe = v_s delta / D, the shortcut-to-key gap works out to exactly gap = exp(-Pe): small Pe (diffusion
comparable to settling) gives a large gap, large Pe (settling swamps diffusion) collapses it.
Fair from the text: every arm states the particle density, the air viscosity and mean free path (for
the slip correction), the temperature and the layer thickness delta -- delta and T are used only by
the replacement but are given in every arm regardless (rule 3). Only the particle diameter changes
between the arms.
Inside: d = 5-10 um. Pe is in the thousands to tens of thousands there (it scales roughly as d^3), so
exp(-Pe) is indistinguishable from zero across the sampled density range (1000-2000 kg/m^3): settling
totally dominates, as the parent's method assumes.
Outside: d = 0.3-0.4 um, ordinary accumulation-mode sizes (not an outlying number); Pe is 0.34-1.63
across the grid, and gap = exp(-Pe) is 20-71 %, worst (smallest) at the largest diameter and highest
density (d=0.4 um, rho=2000 kg/m^3, Pe=1.63, gap=19.7 %). The layer thickness delta = 10 um is an
ordinary value for a diffusive sublayer next to a surface in a ventilated room or chamber (Lai &
Nazaroff's own fitted sublayer thicknesses span a few to several tens of um).
Note on range (2026-10-01 review): the gap is exp(-Pe) exactly, so the regime boundary is set by Pe,
not by diameter directly -- the previous 0.3-0.6 um outside window let Pe run up to ~5.5 at its
upper corner (d=0.6 um, rho=2000 kg/m^3), well past the gap >= 15 % line (Pe <= ln(1/0.15) = 1.90).
The window was narrowed to 0.3-0.4 um so the worst corner (Pe=1.63 there) still clears the bound with
margin; the inside window (Pe >= ln(1/0.02) = 3.91 needed) was already far inside it at 5-10 um and
did not need to move.

Conventions: the Cunningham correction itself is checked (as in `cunningham_slip`) against the
Allen-Raabe and Kim et al. (2005) coefficient sets, both within 3 % of the key on the outside grid.
Reference: Lai, A. C. K. and Nazaroff, W. W. (2000), "Modeling indoor particle deposition from
turbulent flow onto smooth surfaces", J. Aerosol Sci. 31, 463, for the drift-diffusion-across-a-layer
picture (thin-layer limit, their eq. for v_d given a stagnant sublayer); Hinds (1999), Aerosol
Technology, ch. 7, for the Stokes-Einstein-Cunningham diffusivity.
Answer in um/s (as `cunningham_slip`): the smallest outside key is well above 1, so round(x, 2)
is safe.

Independent review (2026-10-01): the previous key, v_d = v_s + D/delta, was not the solution of
steady drift plus diffusion across a stagnant layer with an absorbing surface -- it is only the
small-Pe (diffusion-dominated) limit of the correct result and overstates v_d everywhere Pe is not
tiny. Replaced `correct` with the exact flux solution v_d = v_s/(1 - exp(-Pe)), Pe = v_s delta/D, and
renarrowed the outside diameter window from 0.3-0.6 um to 0.3-0.4 um so the worst corner still clears
the 15 % floor (new min outside gap 19.7 %, vs. a window that under the exact formula would otherwise
fall as low as exp(-5.5) = 0.4 % at the old 0.6 um corner). Reworded "still air" to "slowly moving air
that is well mixed outside a stagnant layer" and removed "diffusive"/"diffusion" from the inside and
outside text (now only in the `given` sentence, which also now states the exact formula rather than
the additive one). `check()` now cross-checks the exact formula against an independently coded
Scharfetter-Gummel (exponentially fitted) finite-volume solve of the same steady drift-diffusion
boundary-value problem -- stable at any Peclet number, unlike a naive central-difference discretization
-- in addition to the Cunningham coefficient-set spread.
"""
from __future__ import annotations

import math
from itertools import product

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

MU, LAM, G = 1.81e-5, 0.066, 9.81     # Pa s, um, m s^-2
KB = 1.380649e-23                     # J/K
T, DELTA = 293.0, 10.0                # K, um (stagnant-layer thickness)


def _cunningham(d_um):
    r = d_um / 2
    kn = LAM / r
    return 1 + kn * (1.257 + 0.4 * math.exp(-1.1 / kn))


def _vs(d_um, rho):
    """Slip-corrected settling speed, m/s."""
    d = d_um * 1e-6
    return rho * G * d ** 2 * _cunningham(d_um) / (18 * MU)


def _diffusivity(d_um):
    """Slip-corrected Stokes-Einstein diffusivity, m^2/s."""
    d = d_um * 1e-6
    return KB * T * _cunningham(d_um) / (3 * math.pi * MU * d)


def _vd(d_um, rho):
    """Exact steady drift-diffusion deposition velocity across the stagnant layer, m/s:
    v_d = v_s / (1 - exp(-Pe)), Pe = v_s delta / D (Lai & Nazaroff 2000, thin-layer limit)."""
    v_s = _vs(d_um, rho)
    pe = v_s * (DELTA * 1e-6) / _diffusivity(d_um)
    return v_s / (1 - math.exp(-pe))


def base(rng):
    return {"rho": rnd(rng.uniform(1000, 2000), 3)}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"d": rnd(rng.uniform(5, 10), 3)}
    return {"d": rnd(rng.uniform(0.3, 0.4), 3)}


def shortcut(p):
    return _vs(p["d"], p["rho"]) * 1e6                        # um/s


def correct(p):
    return _vd(p["d"], p["rho"]) * 1e6                        # um/s


def text(p, variant):
    given = ("Within the stagnant layer the particle also moves by Brownian diffusion, so its "
             "deposition velocity is the exact drift-diffusion result v_d = v_s / (1 - exp(-v_s "
             "delta / D)), where v_s = rho g d^2 C/(18 mu) is the slip-corrected settling speed, "
             "D = k_B T C/(3 pi mu d) is the slip-corrected diffusivity (C the Cunningham factor, "
             "k_B = 1.381 x 10^-23 J/K), and delta is the stagnant-layer thickness. "
             if variant == "given" else "")
    return (f"A rigid spherical particle of diameter {num(p['d'])} um and density {num(p['rho'])} "
            f"kg m^-3 deposits onto a horizontal surface from slowly moving air that is well mixed "
            f"outside a stagnant layer of thickness {DELTA:.0f} um next to the surface, at a "
            f"temperature of {T:.0f} K. The dynamic viscosity of air is 1.81 x 10^-5 kg m^-1 s^-1 "
            f"and the mean free path of air molecules is 0.066 um. Take g = 9.81 m s^-2. {given}"
            f"Find the particle's deposition velocity onto the surface. Express your answer in "
            f"um s^-1.")


TEMPLATE = Template("deposition_brownian", "atmospheric_aerosols",
                    "gravitational settling alone -> settling plus Brownian diffusion across a "
                    "surface boundary layer",
                    "um/s", base, trigger, text, correct, shortcut)


def _bernoulli(x):
    """x / (exp(x) - 1), stable near x = 0 via a short Taylor series."""
    if abs(x) < 1e-7:
        return 1 - x / 2 + x * x / 12
    return x / math.expm1(x)


def _vd_scharfetter_gummel(d_um, rho, n=40):
    """Independent numeric solve of D C'' + v_s C' = 0 across the layer (C=0 at the surface,
    C=1 at the top): a Scharfetter-Gummel (exponentially fitted) finite-volume discretization,
    exact and stable for any Peclet number, coded separately from `_vd`'s closed form. Flux
    continuity at every interior node gives a tridiagonal system for the node concentrations;
    the surface flux divided by the top concentration is v_d."""
    v_s = _vs(d_um, rho)
    d_coef = _diffusivity(d_um)
    h = (DELTA * 1e-6) / n
    x = v_s * h / d_coef
    bp, bm = _bernoulli(x), _bernoulli(-x)
    m = n - 1
    a, b, c = bp, -(bp + bm), bm
    rhs = [0.0] * m
    rhs[-1] -= c * 1.0                     # C_n = 1 (top of the layer)
    cp, dp = [0.0] * m, [0.0] * m
    cp[0], dp[0] = c / b, rhs[0] / b
    for k in range(1, m):
        denom = b - a * cp[k - 1]
        cp[k] = c / denom
        dp[k] = (rhs[k] - a * dp[k - 1]) / denom
    conc = [0.0] * m
    conc[-1] = dp[-1]
    for k in range(m - 2, -1, -1):
        conc[k] = dp[k] - cp[k] * conc[k + 1]
    c0, c1 = 0.0, conc[0]                  # C_0 = 0 (the absorbing surface)
    flux = (d_coef / h) * (bp * c0 - bm * c1)
    return abs(flux)


def check() -> str:
    """Gap bounds over the grid, an independent Scharfetter-Gummel BVP solve of the same
    drift-diffusion problem, Cunningham coefficient-set spread, rounding safety."""
    max_in, min_out = 0.0, math.inf
    alt_max = [0.0, 0.0]
    max_numeric_dev = 0.0
    for rho, d in product((1000.0, 1500.0, 2000.0), (5.0, 7.0, 10.0)):
        p = {"rho": rho, "d": d}
        max_in = max(max_in, gap(correct(p), shortcut(p)))
        num_vd = _vd_scharfetter_gummel(d, rho) * 1e6
        max_numeric_dev = max(max_numeric_dev, abs(num_vd / correct(p) - 1))
    smallest_out = math.inf
    for rho, d in product((1000.0, 1500.0, 2000.0), (0.3, 0.33, 0.36, 0.4)):
        p = {"rho": rho, "d": d}
        key = correct(p)
        min_out = min(min_out, gap(key, shortcut(p)))
        smallest_out = min(smallest_out, key)
        num_vd = _vd_scharfetter_gummel(d, rho) * 1e6
        max_numeric_dev = max(max_numeric_dev, abs(num_vd / key - 1))
        kn = LAM / (d / 2)
        for j, (a, b, c) in enumerate(((1.142, .558, .999), (1.165, .483, .997))):
            c_alt = 1 + kn * (a + b * math.exp(-c / kn))
            d_m = d * 1e-6
            vs_alt = rho * G * d_m ** 2 * c_alt / (18 * MU)
            diff_alt = KB * T * c_alt / (3 * math.pi * MU * d_m)
            pe_alt = vs_alt * (DELTA * 1e-6) / diff_alt
            vd_alt = vs_alt / (1 - math.exp(-pe_alt)) * 1e6
            alt_max[j] = max(alt_max[j], abs(vd_alt / key - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP
    assert max(alt_max) < 0.05
    assert max_numeric_dev < 1e-6
    assert smallest_out > 0.5 and abs(round(smallest_out, 2) / smallest_out - 1) < 0.01
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; "
            f"Scharfetter-Gummel BVP deviation {max_numeric_dev:.2e}; "
            f"Cunningham coefficient-set deviations {alt_max[0]:.4f} / {alt_max[1]:.4f}; "
            f"smallest outside key {smallest_out:.2f} um/s")
