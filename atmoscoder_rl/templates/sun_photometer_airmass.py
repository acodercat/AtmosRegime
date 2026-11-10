"""Plane-parallel air mass sec(theta) -> Kasten & Young (1989) air mass (extended, 2026-10-01).

Parent's method: a sun photometer relates the measured direct-beam irradiance I to the known
extraterrestrial irradiance I0 at the same wavelength through Beer's law, I = I0 exp(-tau m), with
the plane-parallel air mass m = 1/cos(theta) (theta = solar zenith angle); solving for the vertical
optical depth gives tau = ln(I0/I) / m.
Where it fails: 1/cos(theta) diverges as theta -> 90 degrees, while the real atmosphere is a curved,
refracting shell, so the true air mass stays finite near the horizon. The standard empirical
replacement (Kasten & Young 1989) is m = 1/[cos(theta) + 0.50572(96.07995 - theta)^-1.6364], fit to
radiosonde ray-tracing and in routine use for sun-photometer and solar-radiation air-mass
calculations.
Fair from the text: every arm states the measured-to-extraterrestrial irradiance ratio (the
transmittance T = I/I0) and the solar zenith angle theta -- exactly what both the plane-parallel
and the Kasten-Young air mass need (the parent's shortcut, m = 1/cos(theta), uses neither of the
Kasten-Young fit constants at all). Those three fit constants (0.50572, 96.07995, 1.6364) are
therefore given only in the `given` sentence (rule 3): check() shows that two other published
air-mass fits (Kasten 1965, Rozenberg 1966) stay within 1 % of Kasten-Young 1989 over the whole
outside grid, so there is no meaningful "convention spread" being hidden by singling one fit out;
Young (1994)'s polynomial fit stays within 4.7 % over the sampled outside grid -- inside the 5 %
rule-5 bound, so a solver who recalls that fit instead is also credited.
Inside: theta = 45-60 degrees, where the plane-parallel and Kasten-Young air masses agree to within
0.3 %.
Outside: theta = 86.5-87.0 degrees, an ordinary near-sunrise/near-sunset sun-photometer measurement
(AERONET-style instruments routinely record well past 80 degrees), not an extreme grazing angle.
The plane-parallel air mass overstates the true air mass here by 17-29 %, carrying the same error
into the retrieved optical depth.
Judgement call, recorded plainly: because the air mass ratio m_KY/m_sec depends only on theta (not
on T), the gap bound is independent of how deep the attenuation is; but the *absolute* magnitude of
tau is tau = -ln(T)/m, and m_outside (~13-15) is roughly an order of magnitude larger than m_inside
(~1.4-2.0) over these ranges, so the same T that keeps tau_outside comfortably clear of the grader's
round(x, 2) floor (0.15-0.30 here) pushes tau_inside to 1.4-2.8 -- optical depths typical of a hazy
or smoke/dust-laden sky, not a pristine one. The text does not claim clear air; both arms are left as
plain transmittance measurements, consistent with heavy but real-world aerosol loading (daily-mean
AOD above 2 is routinely reported during wildfire-smoke and dust events).
Conventions: see Fair-from-the-text above (Kasten 1965 and Rozenberg 1966 within ~1 %, Young 1994
within 4.7 %, all checked in check()).
Reference: Kasten, F. and Young, A.T. (1989), Revised optical air mass tables and approximation
formula, Applied Optics 28(22), 4735-4738 (m(90 degrees) = 37.92, matched in check()); Young, A.T.
(1994), Air mass and refraction, Applied Optics 33(6), 1108-1110 (the polynomial convention check).
Related mechanism: spherical_slant_path (also a plane-parallel-path trap near the horizon, there for
a uniform absorbing layer rather than air mass via an empirical fit).

Independent review (2026-10-01): further spreads over the outside grid — Kasten 1965 +0.5 %, Rozenberg -0.6 %,
Young 1994 -4.7 % (edge), refraction-free exponential atmosphere H = 7-8.4 km -0.5 to -3.7 %; the homogeneous
spherical-shell formula is +4.6 to +5.9 % (fails on about half the grid: a grading-noise risk); reading the zenith
angle as apparent rather than true shifts m by about -4.7 % (inside tolerance, just). The single air mass assumes the
extinction is distributed like air; a constituent-by-constituent air mass (aerosol +16 to +22 %, ozone -28 to -33 %)
cannot be keyed without the wavelength and composition, which are not given. Inside optical depths 1.4-2.8 (heavy
smoke or dust) follow from the shared transmittance (rule 1).
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

A, B, C = 0.50572, 96.07995, 1.6364   # Kasten & Young (1989) fit constants


def m_sec(theta_deg: float) -> float:
    return 1.0 / math.cos(math.radians(theta_deg))


def m_ky(theta_deg: float) -> float:
    return 1.0 / (math.cos(math.radians(theta_deg)) + A * (B - theta_deg) ** (-C))


def base(rng):
    return {"t": rnd(rng.uniform(0.02, 0.06), 3)}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"theta": rnd(rng.uniform(45.0, 60.0), 3)}
    return {"theta": rnd(rng.uniform(86.5, 87.0), 3)}


def shortcut(p):
    """Plane-parallel air mass."""
    return -math.log(p["t"]) / m_sec(p["theta"])


def correct(p):
    """Kasten & Young (1989) air mass."""
    return -math.log(p["t"]) / m_ky(p["theta"])


def text(p, variant):
    given = ("Because the Sun is low, the plane-parallel air mass 1/cos(θ) is not accurate; use the Kasten "
             "& Young (1989) air mass formula instead: m = 1/[cos(θ) + 0.50572(96.07995 − θ)^−1.6364], with "
             "θ in degrees. " if variant == "given" else "")
    return (f"A sun photometer at the surface measures a direct-beam solar irradiance equal to "
            f"{num(p['t'])} times the known extraterrestrial (top-of-the-atmosphere) irradiance at this "
            f"wavelength, at a solar zenith angle of {num(p['theta'])}°. {given}What is the atmospheric "
            f"optical depth along the vertical path at this wavelength? Express your answer as a "
            f"dimensionless number.")


TEMPLATE = Template("sun_photometer_airmass", "observation_and_modeling",
                    "plane-parallel air mass 1/cos(theta) -> Kasten-Young (1989) air mass",
                    "", base, trigger, text, correct, shortcut)


def _m_kasten1965(t):
    return 1.0 / (math.cos(math.radians(t)) + 0.15 * (93.885 - t) ** (-1.253))


def _m_rozenberg(t):
    c = math.cos(math.radians(t))
    return 1.0 / (c + 0.025 * math.exp(-11 * c))


def _m_young1994(t):
    c = math.cos(math.radians(t))
    num_ = 1.002432 * c ** 2 + 0.148386 * c + 0.0096467
    den_ = c ** 3 + 0.149864 * c ** 2 + 0.0102963 * c + 0.000303978
    return num_ / den_


def check() -> str:
    thetas_in = [45.0, 48.75, 52.5, 56.25, 60.0]
    thetas_out = [86.5, 86.625, 86.75, 86.875, 87.0]
    ts = [0.02, 0.03, 0.04, 0.05, 0.06]

    max_in = 0.0
    min_out = 1.0
    max_beer = 0.0
    max_kasten = max_rozenberg = max_young = 0.0
    tau_in_lo, tau_in_hi = 1e9, -1e9
    tau_out_lo, tau_out_hi = 1e9, -1e9

    for t in ts:
        for th in thetas_in:
            p = {"t": t, "theta": th}
            c, s = correct(p), shortcut(p)
            max_in = max(max_in, gap(c, s))
            tau_in_lo, tau_in_hi = min(tau_in_lo, c), max(tau_in_hi, c)
            # Independent: Beer's law closes the loop -- exp(-tau * m) must reconstruct T.
            max_beer = max(max_beer, abs(math.exp(-c * m_ky(th)) - t) / t)
        for th in thetas_out:
            p = {"t": t, "theta": th}
            c, s = correct(p), shortcut(p)
            min_out = min(min_out, gap(c, s))
            tau_out_lo, tau_out_hi = min(tau_out_lo, c), max(tau_out_hi, c)
            max_beer = max(max_beer, abs(math.exp(-c * m_ky(th)) - t) / t)
            max_kasten = max(max_kasten, gap(m_ky(th), _m_kasten1965(th)))
            max_rozenberg = max(max_rozenberg, gap(m_ky(th), _m_rozenberg(th)))
            max_young = max(max_young, gap(m_ky(th), _m_young1994(th)))

    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_beer < 1e-9, max_beer
    assert max_kasten < 0.02 and max_rozenberg < 0.02 and max_young < 0.05, (max_kasten, max_rozenberg, max_young)
    assert tau_out_lo > 0.1 and tau_in_lo > 0.5                  # clear of the round(x, 2) grading floor
    # Anchor: Kasten & Young (1989) report m(90 degrees) = 37.92.
    assert abs(m_ky(90.0) - 37.92) / 37.92 < 2e-3
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}, Beer's-law closure "
            f"{max_beer:.1e}, convention spread Kasten65 {max_kasten:.3f} Rozenberg {max_rozenberg:.3f} "
            f"Young94 {max_young:.3f}, key range in [{tau_in_lo:.2f},{tau_in_hi:.2f}] / out "
            f"[{tau_out_lo:.2f},{tau_out_hi:.2f}]; m(90°)={m_ky(90.0):.2f} (lit. 37.92)")
