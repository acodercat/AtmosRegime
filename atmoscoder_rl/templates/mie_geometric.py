"""Rayleigh lambda^-4 extinction ratio -> Mie ratio for a large sphere (spec: MECHANISMS.md).

Parent's method: a particle small compared with the wavelength scatters as lambda^-4, so the
ratio of its extinction cross sections at a blue and a red wavelength is (lambda_red/lambda_blue)^4.
Where it fails: when the particle is many wavelengths across, the extinction efficiency approaches
its large-particle limit Q_ext -> 2 at both wavelengths and the ratio approaches 1 (why clouds are
white). The key is the full Mie ratio Q_ext(x_blue)/Q_ext(x_red), x = 2 pi r / lambda.
Fair from the text: both variants give the radius, both wavelengths and the refractive index
(1.33, non-absorbing), which fully specify Mie extinction; outside, r = 25-60 um, so x = 210-940.
Inside: r = 0.005-0.025 um (x <= 0.39); the Rayleigh ratio is within 1.1 % of the Mie ratio.

Conventions: a solver who answers the geometric-optics limit (ratio exactly 1, whether Q = 2 or
Q = 1 is assumed) is within 3.8 % of the Mie key over the whole outside grid (checked below);
below r ~ 20 um the Mie interference ripple pushes that above 5 %, which is why the outside range
starts at 25 um. Anomalous-diffraction theory (van de Hulst) is closer still.
The Mie series (Bohren & Huffman 1983, BHMIE: downward log-derivative recursion, upward
Riccati-Bessel recursion) reproduces the published Wiscombe (1979, NCAR/TN-140) test values
Q_ext = 2.8820 (m = 1.5, x = 10), 2.0944 (x = 100), 2.0133 (x = 1000), 2.2323 (m = 0.75, x = 10).

Review (2026-09-26, independent reviewer): the Q_ext = Q_sca identity runs through the same
recurrences as the key, so it is not an independent check. check() now also computes the ratio
from __future__ import annotations

from anomalous-diffraction theory (van de Hulst 1957, non-absorbing sphere:
Q_ext = 2 - (4/rho) sin(rho) + (4/rho^2)(1 - cos(rho)), rho = 2x(m - 1)), a closed form that shares
no code with the Mie series, over the whole outside grid.

Review (2026-09-27, domain + measurement-design review): the given sentence called Q_ext ≈ 2 the
"geometric-optics limit". Pure geometric optics gives Q_ext = 1; the value 2 is the large-particle
limit that includes diffraction (the extinction paradox). The sentence now says so. No key changes.
"""
import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, num, rnd

M = 1.33


def q_ext(x, m=M):
    """Mie extinction efficiency of a homogeneous sphere with real refractive index m."""
    q, _ = _mie(x, m)
    return q


def _mie(x, m):
    nstop = int(x + 4 * x ** (1 / 3) + 2)
    mx = m * x
    nmx = int(max(nstop, abs(mx))) + 16
    d = [0.0] * (nmx + 1)
    for n in range(nmx, 0, -1):                      # downward: stable for any x
        d[n - 1] = n / mx - 1 / (d[n] + n / mx)
    psi0, psi1, chi0, chi1 = math.cos(x), math.sin(x), -math.sin(x), math.cos(x)
    xi1 = complex(psi1, -chi1)
    qe = qs = 0.0
    for n in range(1, nstop + 1):
        psi = (2 * n - 1) * psi1 / x - psi0
        chi = (2 * n - 1) * chi1 / x - chi0
        xi = complex(psi, -chi)
        da, db = d[n] / m + n / x, m * d[n] + n / x
        a = (da * psi - psi1) / (da * xi - xi1)
        b = (db * psi - psi1) / (db * xi - xi1)
        qe += (2 * n + 1) * (a.real + b.real)
        qs += (2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)
        psi0, psi1, chi0, chi1, xi1 = psi1, psi, chi1, chi, xi
    return 2 * qe / x ** 2, 2 * qs / x ** 2


def base(rng):
    return {"l1": float(rng.choice(range(400, 490, 10))), "l2": float(rng.choice(range(650, 760, 10)))}


def trigger(rng, regime, base):
    return {"r": rnd(rng.uniform(0.005, 0.025)) if regime == "inside" else rnd(rng.uniform(25, 60))}


def correct(p):
    return q_ext(2 * math.pi * p["r"] * 1000 / p["l1"]) / q_ext(2 * math.pi * p["r"] * 1000 / p["l2"])


def shortcut(p):
    return (p["l2"] / p["l1"]) ** 4


def text(p, variant):
    given = ("Because the particle is many wavelengths across, Rayleigh scattering does not apply: use Mie "
             "theory, in which the extinction efficiency of a particle this large is close to its "
             "large-particle (extinction-paradox) limit Q_ext ≈ 2 at both wavelengths. " if variant == "given" else "")
    return (f"A spherical particle of radius {num(p['r'])} μm has a refractive index of 1.33 at visible "
            f"wavelengths and absorbs no light. It is illuminated in turn by blue light of wavelength "
            f"{num(p['l1'])} nm and by red light of wavelength {num(p['l2'])} nm. {given}Find the ratio of the "
            f"particle's extinction cross section at {num(p['l1'])} nm to its extinction cross section at "
            f"{num(p['l2'])} nm.")


def _adt(x, m=M):
    rho = 2 * x * (m - 1)
    return 2 - 4 / rho * math.sin(rho) + 4 / rho ** 2 * (1 - math.cos(rho))


def check() -> str:
    # 1. Published test values (Wiscombe 1979).
    for x, m, ref in ((10, 1.5, 2.8820), (100, 1.5, 2.0944), (1000, 1.5, 2.0133), (10, .75, 2.2323)):
        assert abs(q_ext(x, m) - ref) < 1e-3, (x, m, q_ext(x, m))
    # 2. Independent identities: Q_ext == Q_sca for a real index (optical theorem vs |a|^2+|b|^2);
    #    small-x limit = 8/3 x^4 ((m^2-1)/(m^2+2))^2.
    for x in (0.05, 1, 7.3, 55, 400):
        qe, qs = _mie(x, M)
        assert abs(qe / qs - 1) < 1e-9, x
    k = (M * M - 1) / (M * M + 2)
    assert abs(q_ext(0.01) / (8 / 3 * 0.01 ** 4 * k * k) - 1) < 1e-4
    # 3. Grid over the sampling ranges (radii at the 2-significant-figure values drawn).
    max_in = max_out_geo = max_adt = 0.
    min_out = math.inf
    for l1 in range(400, 490, 10):
        for l2 in range(650, 760, 10):
            for r in [i / 1000 for i in range(5, 26)]:
                p = {"l1": l1, "l2": l2, "r": r}
                max_in = max(max_in, abs(shortcut(p) - correct(p)) / correct(p))
            for r in range(25, 61):
                p = {"l1": l1, "l2": l2, "r": r}
                c = correct(p)
                min_out = min(min_out, abs(shortcut(p) - c) / c)
                max_out_geo = max(max_out_geo, abs(1 / c - 1))      # geometric-limit answer = 1
                adt = _adt(2 * math.pi * r * 1000 / l1) / _adt(2 * math.pi * r * 1000 / l2)
                max_adt = max(max_adt, abs(adt / c - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and max_out_geo < .05 and max_adt < .03, (max_adt,)
    return (f"Wiscombe test values reproduced; max inside gap {max_in:.4f}, min outside gap {min_out:.2f}, "
            f"geometric-limit answer (ratio 1) within {max_out_geo:.4f} of Mie outside; anomalous-diffraction ratio "
            f"within {max_adt:.4f}")


TEMPLATE = Template("mie_geometric", "atmospheric_radiation",
                    "Rayleigh lambda^-4 extinction -> Mie extinction (Q_ext -> 2) for particles many wavelengths across",
                    "", base, trigger, text, correct, shortcut)
