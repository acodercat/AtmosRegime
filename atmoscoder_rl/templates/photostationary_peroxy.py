"""Leighton photostationary-state ozone -> peroxy radicals also convert NO to NO2
(extended set, 2026-10-01).

Parent's method: at the photostationary state of the NO-NO2-O3 triad, NO2 photolysis is balanced
by NO's reaction with O3 alone: j[NO2] = k1[NO][O3] (k1 the NO+O3 rate constant), so the ozone
implied by measured NO2, NO and j is [O3] = j[NO2] / (k1[NO]) (Leighton 1961).
Where it fails: peroxy radicals (HO2 and organic RO2) also oxidise NO to NO2, in parallel with O3,
so the NO2 balance is really j[NO2] = (k1[O3] + k_RO2[RO2])[NO] (k_RO2 a single rate constant
representing both HO2+NO and the generic RO2+NO reaction, which are close in magnitude). Solving
for the ozone this implies: [O3] = j[NO2]/(k1[NO]) - k_RO2[RO2]/k1. The peroxy term is a pure
subtraction, independent of [NO] and [NO2] -- the shortcut drops it and over-predicts [O3].
Fair from the text: every arm states [NO2], [NO], j, k1, k_RO2, the peroxy radical number density
[RO2] (the trigger), and the number density of air used to convert the mixing ratios; no arm states
the extended photostationary-state balance itself (given only in the given arm).
Inside: [RO2] = 1-8 x 10^5 cm^-3, a clean / night-like background level; the implied ozone deficit
k_RO2[RO2]/k1 is at most ~0.02 ppb against keys of 11-13 ppb, gap <= 0.12 %.
Outside: [RO2] = 1.1-5 x 10^8 cm^-3, an ordinary polluted midday total peroxy-radical level
(HOx/ROx of this order are routinely reported in urban photochemistry, e.g. Cantrell et al. 1993,
1996); the implied deficit is 1.9-8.6 ppb against the same 11-13 ppb baseline, gap >= 15 % everywhere
on a 200 000-draw stress test (min observed 17.2 %; see check()).
Narrowing: to keep gap >= 15 % at the bottom of the suggested [RO2] range (1 x 10^8 cm^-3) while
keeping the implied ozone positive at the top of it (5 x 10^8 cm^-3), the Leighton-only baseline
[O3]_shortcut must sit in a narrow band (~11-13 ppb); it is fixed there by holding j constant at a
typical midday value and keeping [NO2]/[NO] within a narrow ratio (0.73-0.83), which also keeps
[NO2] and [NO] themselves (3.6-6.6 and 5.0-8.0 ppb) ordinary polluted midday figures. The resulting
implied ozone (inside and outside both land at 2.7-13 ppb) is itself an ordinary, NOx-rich,
somewhat O3-suppressed photostationary value, not an extreme one.
Conventions: k1 = 1.9 x 10^-14 cm^3 molecule^-1 s^-1 (NO+O3, used identically in `pseudo_first_order`);
the JPL Arrhenius value at 298 K is 1.96 x 10^-14 (3.1 % higher); check() shows this convention moves
the key by under 5 %. k_RO2 =
8.0 x 10^-12 cm^3 molecule^-1 s^-1 lumps HO2+NO (~8.5 x 10^-12) and a generic RO2+NO (~7.5 x 10^-12,
e.g. CH3O2+NO); check() shows splitting it into those two literature values instead of the lumped
constant moves the key by under 2 %.
Reference: Leighton, P.A. (1961), Photochemistry of Air Pollution (the unperturbed photostationary
relation); Cantrell, C.A., et al. (1993), J. Geophys. Res., 98, 2897, and (1996), J. Geophys. Res.,
101, 14653 (photostationary-state deviations attributed to peroxy radicals, the basis for the
extended balance used here). Related mechanism: `pseudo_first_order` (same NO+O3 rate constant,
a different reaction regime).

Independent review (2026-10-01): "the photostationary state" conventionally means the Leighton NO-NO2-O3 triad,
with peroxy chemistry treated as a deviation from it, so the shortcut was a defensible reading of the old
question; it now asks for NO2 in steady state (production balances photolysis). "Midday" removed: the inside
peroxy levels (1e5-8e5 cm^-3) are unrealistic at midday. Expect a weak trap: the peroxy term is stated.
"""
from __future__ import annotations

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True

NAIR = 2.46e19          # molecules cm^-3, 25 degC and 1 atm
K1 = 1.9e-14             # cm^3 molecule^-1 s^-1, NO + O3
KRO2 = 8.0e-12           # cm^3 molecule^-1 s^-1, lumped HO2/RO2 + NO
J = 7.2e-3               # s^-1, NO2 photolysis frequency, typical clear midday sky


def base(rng):
    no = rnd(rng.uniform(5.0, 8.0), 3)
    ratio = rnd(rng.uniform(0.73, 0.83), 3)
    return {"no": no, "no2": rnd(ratio * no, 3)}


def trigger(rng, regime, base):
    if regime == "inside":
        return {"ro2": rnd(rng.uniform(1e5, 8e5), 2)}
    return {"ro2": rnd(rng.uniform(1.1e8, 5e8), 2)}


def shortcut(p):
    """Leighton relation: [O3] implied by NO2 photolysis balanced by NO + O3 alone."""
    no2_n, no_n = p["no2"] * 1e-9 * NAIR, p["no"] * 1e-9 * NAIR
    o3_n = J * no2_n / (K1 * no_n)
    return o3_n / (NAIR * 1e-9)                          # ppb


def correct(p):
    no2_n, no_n = p["no2"] * 1e-9 * NAIR, p["no"] * 1e-9 * NAIR
    o3_n = J * no2_n / (K1 * no_n) - KRO2 * p["ro2"] / K1
    return o3_n / (NAIR * 1e-9)                          # ppb


def text(p, variant):
    given = ("Because the peroxy radicals also convert NO to NO2, the photostationary NO2 balance is "
             "j[NO2] = (k1[O3] + k_RO2[RO2])[NO], so [O3] = j[NO2]/(k1[NO]) - k_RO2[RO2]/k1. "
             if variant == "given" else "")
    return (f"At a roadside monitoring site, the mixing ratios are {num(p['no2'])} ppb of NO2 and "
            f"{num(p['no'])} ppb of NO, and the photolysis frequency of NO2 is j = 7.2 × 10^-3 s^-1. "
            f"The air also contains peroxy radicals (HO2 and organic RO2 combined) at a number density "
            f"of {num(p['ro2'])} cm^-3, which react with NO with a rate constant k_RO2 = 8.0 × 10^-12 "
            f"cm^3 molecule^-1 s^-1. NO reacts with O3 with a rate constant k1 = 1.9 × 10^-14 "
            f"cm^3 molecule^-1 s^-1. The number density of air is 2.46 × 10^19 cm^-3 (25 °C, 1 atm). "
            f"{given}Assuming NO2 is in steady state (its production balances its photolysis), what ozone mixing ratio is consistent with these observations? "
            f"Express your answer in ppb.")


def _bisection_o3(p, lo=-50.0, hi=2000.0, iters=200):
    """Independent: solve j[NO2] = (k1 O3 + k_RO2 RO2)[NO] for O3 by bisection, not algebra."""
    no2_n, no_n = p["no2"] * 1e-9 * NAIR, p["no"] * 1e-9 * NAIR

    def residual(o3_ppb):
        o3_n = o3_ppb * 1e-9 * NAIR
        return J * no2_n - (K1 * o3_n + KRO2 * p["ro2"]) * no_n

    for _ in range(iters):
        mid = (lo + hi) / 2
        if residual(lo) * residual(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def check() -> str:
    max_in, min_out, min_key, max_key = 0.0, 1.0, 1e18, 0.0
    spread = 0.0
    for no in (5.0, 6.5, 8.0):
        for ratio in (0.73, 0.78, 0.83):
            no2 = round(ratio * no, 3)
            for ro2 in (1e5, 4e5, 8e5):
                p = {"no": no, "no2": no2, "ro2": ro2}
                c = correct(p)
                assert abs(c - _bisection_o3(p)) < 1e-6
                max_in = max(max_in, gap(c, shortcut(p)))
            for ro2 in (1.1e8, 3e8, 5e8):
                p = {"no": no, "no2": no2, "ro2": ro2}
                c = correct(p)
                assert c > 0, p
                assert abs(c - _bisection_o3(p)) < 1e-6
                g_ = gap(c, shortcut(p))
                min_out = min(min_out, g_)
                min_key, max_key = min(min_key, c), max(max_key, c)
                k1_jpl = 3.0e-12 * 2.718281828459045 ** (-1500 / 298.15)
                alt1 = (J * (no2 * 1e-9 * NAIR) / (k1_jpl * (no * 1e-9 * NAIR))
                        - KRO2 * ro2 / k1_jpl) / (NAIR * 1e-9)
                spread = max(spread, gap(c, alt1))
                # split RO2 evenly between HO2 and generic RO2 instead of the lumped constant
                k_ho2, k_ro2_generic = 8.5e-12, 7.5e-12
                half = ro2 / 2
                o3_split_n = J * (no2 * 1e-9 * NAIR) / (K1 * (no * 1e-9 * NAIR)) \
                    - (k_ho2 * half + k_ro2_generic * half) / K1
                alt2 = o3_split_n / (NAIR * 1e-9)
                spread = max(spread, gap(c, alt2))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP and spread < 0.05, (max_in, min_out, spread)
    assert min_key > 0.1                                  # well away from zero
    k1_jpl_anchor = 3.0e-12 * 2.718281828459045 ** (-1500 / 298.15)
    assert abs(k1_jpl_anchor / K1 - 1) < 0.05              # JPL 298 K value within 5 % of the text's k1
    return (f"bisection solve agrees; max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; "
            f"outside key range {min_key:.2f}-{max_key:.2f} ppb; k1/k_RO2 convention spread {spread:.4f}; "
            f"JPL k1(298 K) = {k1_jpl_anchor:.3g}")


TEMPLATE = Template("photostationary_peroxy", "atmospheric_chemistry",
                    "Leighton photostationary-state ozone -> peroxy radicals also convert NO to NO2",
                    "ppb", base, trigger, text, correct, shortcut)
