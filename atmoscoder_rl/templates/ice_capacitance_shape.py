"""Diffusional growth of a sphere -> growth through the electrostatic capacitance of a flat crystal
(extended, cloud physics, 2026-10-05).

Parent's method: the diffusion-limited growth rate of a sphere, dm/dt = 4 pi r D (rho_v,inf - rho_v,s)
(Maxwell), with r taken as half the particle's largest dimension.
Where it fails: the vapour field around a non-spherical particle obeys Laplace's equation like the
electric potential around a charged conductor of the same shape, so r is replaced by the particle's
capacitance C (in length units). For an oblate spheroid with semi-axes a > c,
C = a e / arcsin(e), e = sqrt(1 - c^2/a^2); a thin plate (c/a -> 0) has C -> 2a/pi = 0.64 a. A plate
crystal therefore grows 33-35 % slower than the sphere of the same diameter.
Fair from the text: every arm states the particle's equatorial diameter and axial thickness (it is
described as an oblate spheroid, which fixes the shape), the vapour-density excess, the diffusivity of
water vapour, and that growth is diffusion-limited (latent heating neglected -- without this the answer
would need the thermal term and is not unique). The word "capacitance" never appears outside the given
sentence.
Inside: thickness 0.96-0.99 of the diameter (a frozen drop): C/a >= 0.986, so the sphere formula is
within 1.4 % of the key. Outside: thickness 0.025-0.06 of the diameter (D 0.6-1.6 mm, thickness
15-96 um, the shape of a real hexagonal plate): C/a = 0.647-0.661, the shortcut 51-55 % high, an
ordinary growth rate (4-44 ng/s; key 2.7-29 ng/s) with nothing absurd about it.
Conventions (rule 5): the thin-disk value 2a/pi, the standard textbook stand-in for a plate, is within
3.7 % of the spheroid key on the outside grid (`check()`); so a solver who models the plate as a disk
passes. An equal-volume sphere (r = a (c/a)^(1/3)) is a wrong method far off (>= 40 %). The closest
second wrong method is the equal-surface-area sphere (r ~ a/sqrt(2)): >= 7.7 % above the key on the
outside grid -- it does not pass, but only by a few percent; recorded.
Reference: Rogers & Yau, *A Short Course in Cloud Physics* (3rd ed., 1989), ch. 9 (growth of ice
crystals by deposition, electrostatic analogy, capacitance of spheroids and disks); Pruppacher & Klett,
*Microphysics of Clouds and Precipitation* (2nd ed., 1997), sec. 13.3; Lamb & Verlinde, *Physics and
Chemistry of Clouds* (2011), ch. 8 (section numbers not re-verified). Related: `drop_ventilation`
(the same Maxwell relation, but its validity condition is a *stationary* particle; here it is a
*spherical* one -- a different ingredient) and `ice_saturation` (the ice surface vapour pressure, not the
geometry of the flux).
Independent review (2026-10-05): the inside thickness could equal the diameter ("1.1 mm across and
1.1 mm thick", contradictory for an oblate spheroid); the ratio is now capped at 0.99. Nearest wrong
method recorded: the equal-surface-area sphere, >= 7.7 % above the key on the outside grid; `check()`
asserts it stays > 5 % off.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

DV = 2.2e-5          # m^2 s^-1


def _cap_ratio(c_over_a):
    """C/a of an oblate spheroid (prolate never sampled)."""
    if c_over_a >= 1.0:
        return 1.0
    e = math.sqrt(1 - c_over_a ** 2)
    return e / math.asin(e)


def _rate(radius_m, p):
    return 4 * math.pi * radius_m * DV * p["drho"] * 1e-3 * 1e12        # ng s^-1 (drho in g m^-3)


def shortcut(p):
    return _rate(p["d"] / 2 * 1e-3, p)


def correct(p):
    return _rate(p["d"] / 2 * 1e-3 * _cap_ratio(p["t"] / p["d"]), p)


def base(rng):
    return {"d": rnd(rng.uniform(0.6, 1.6)), "drho": rnd(rng.uniform(0.05, 0.20))}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"t": min(rnd(base["d"] * 0.99, 3), rnd(base["d"] * rng.uniform(0.96, 0.99), 3))}
    return {"t": rnd(base["d"] * rng.uniform(0.025, 0.06))}


def text(p, variant):
    given = ("For a non-spherical particle the radius in the sphere's growth law is replaced by the particle's "
             "electrostatic capacitance C (in length units): for an oblate spheroid with semi-axes a > c, "
             "C = a e/arcsin(e), where e = √(1 − c²/a²). " if variant == "given" else "")
    return (f"In a cloud at −15 °C an ice particle grows by vapour deposition. It is an oblate spheroid "
            f"{num(p['d'])} mm across its equator and {num(p['t'])} mm thick along its axis of symmetry. The "
            f"ambient vapour density exceeds the vapour density at the particle's surface by {num(p['drho'])} "
            f"g m^-3, and the diffusivity of water vapour in air is 2.2 × 10^-5 m^2 s^-1. Treat the growth as "
            f"limited by vapour diffusion alone (neglect latent heating). {given}What is the particle's rate of "
            f"mass increase? Express your answer in ng s^-1.")


def _cap_integral(a, c, n=4000):
    """Independent: capacitance of an ellipsoid from its defining integral
    1/C = (1/2) int_0^inf dl / sqrt((a^2 + l)(b^2 + l)(c^2 + l)) with b = a, evaluated by Simpson's rule.
    The substitution u = sqrt(c^2 + l) turns it into int_c^inf du / (a^2 - c^2 + u^2), and
    u = c + a x/(1 - x), x in [0, 1), maps it onto a finite interval with a smooth integrand."""
    def g(x):
        if x >= 1.0:
            return 1.0 / a                                   # limit of the mapped integrand
        u = c + a * x / (1 - x)
        return (a / (1 - x) ** 2) / (a * a - c * c + u * u)
    h = 1.0 / n
    tot = g(0.0) + g(1.0) + sum((4 if i % 2 else 2) * g(i * h) for i in range(1, n))
    return 1 / (tot * h / 3)


def check() -> str:
    max_in, min_out, max_err, spread, min_wrong, near_wrong = 0.0, math.inf, 0.0, 0.0, math.inf, math.inf
    min_key, max_key, max_short = math.inf, 0.0, 0.0
    for d in (0.6, 1.1, 1.6):
        for drho in (0.05, 0.20):
            for ratio in (0.96, 0.975, 0.99):
                p = {"d": d, "drho": drho, "t": d * ratio}
                max_in = max(max_in, gap(correct(p), shortcut(p)))
                min_key = min(min_key, correct(p))
            for ratio in (0.025, 0.04, 0.06):
                p = {"d": d, "drho": drho, "t": d * ratio}
                c, s = correct(p), shortcut(p)
                a = d / 2 * 1e-3
                max_err = max(max_err, gap(c, _rate(_cap_integral(a, a * ratio), p)))
                min_out = min(min_out, gap(c, s))
                spread = max(spread, gap(c, _rate(2 * a / math.pi, p)))            # thin-disk convention
                min_wrong = min(min_wrong, gap(c, _rate(a * ratio ** (1 / 3), p)))  # equal-volume sphere
                e = math.sqrt(1 - ratio ** 2)
                area = 2 * math.pi * a * a * (1 + (1 - e * e) / e * math.atanh(e))
                near_wrong = min(near_wrong, gap(c, _rate(math.sqrt(area / (4 * math.pi)), p)))
                min_key, max_key, max_short = min(min_key, c), max(max_key, c), max(max_short, s)
    assert max_err < 1e-4 and max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_err, max_in, min_out)
    assert spread < 0.05, spread
    assert min_wrong > 0.15 and near_wrong > 0.05, (min_wrong, near_wrong)
    assert min_key > 2 and max_short < 100, (min_key, max_short)
    # Anchors: C(sphere) = a; thin-disk limit 2a/pi (Rogers & Yau ch. 9).
    assert abs(_cap_ratio(1e-6) - 2 / math.pi) < 1e-5 and abs(_cap_integral(1.0, 0.999999) - 1) < 1e-4
    return (f"capacitance integral check {max_err:.1e}; inside max gap {max_in:.4f}; outside min gap {min_out:.3f}; "
            f"thin-disk spread {spread:.4f}; equal-volume miss {min_wrong:.2f}, equal-area miss {near_wrong:.3f}; "
            f"key {min_key:.1f}-{max_key:.1f} ng/s; C/a -> 2/pi at c/a -> 0")


TEMPLATE = Template("ice_capacitance_shape", "cloud_physics",
                    "diffusional growth of a sphere of the particle's radius -> capacitance of a flat (oblate) crystal",
                    "ng/s", base, trigger, text, correct, shortcut)
