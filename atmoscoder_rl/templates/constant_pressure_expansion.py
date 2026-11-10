"""Constant-volume first-order decay of a partial pressure -> constant-pressure reactor whose gas expands
as the reaction makes moles (extended, chemical kinetics, 2026-10-05).

Parent's method: for a first-order gas-phase reaction the reactant's concentration, and so its partial
pressure, falls as e^{-kt}: p_A = y_0 P e^{-kt}, with y_0 the initial mole fraction of A and P the total
pressure. That is the integrated rate law every kinetics course drills, and it is exact at constant
volume.
Where it fails: at constant total pressure a reaction A -> 2B adds one mole of gas per mole of A
consumed, so the gas expands. The moles of A still fall as e^{-kt} (the first-order rate per volume
k C_A times the volume is k n_A, whatever the volume does), but they are now a share of a growing total:
n_tot = n_0[1 + y_0(1 - e^{-kt})], so p_A = P y_0 e^{-kt}/[1 + y_0(1 - e^{-kt})] (Levenspiel's / Fogler's
variable-volume batch reactor, expansion factor eps = y_0 delta with delta = 1). The parent is right when A
is a trace in an inert diluent (y_0 << 1) and misses by y_0(1 - e^{-kt}) otherwise.
Fair from the text: every arm states the stoichiometry A -> 2B, that each molecule of A decomposes
independently with first-order rate constant k, the initial mole fraction of A in nitrogen (inert), the constant temperature and the
constant total pressure P, and the elapsed time -- everything either answer needs; no constant has to be
recalled. "Constant total pressure" is the ordinary description of the reactor and is in all arms
(without it the constant-volume reading would make the parent correct -- an ambiguity, not a trap).
Nothing in the inside or outside text mentions the expansion, the volume or the total number of moles.
Inside: y_0 = 0.008-0.020 (A diluted in nitrogen); the expansion moves p_A by y_0 c <= 1.6 %.
Outside: y_0 = 0.40-0.60 with kt = 0.7-1.6 (50-80 % converted): the shortcut is 20-48 % high, an
ordinary-looking partial pressure well below the initial one.
Conventions: none -- ideal-gas mixture, mole-based first-order rate (the text says first order in A with
a rate constant in min^-1; for a first-order law the mole and concentration forms coincide). Using
concentrations C_A = n_A/V with V(t) = V_0[1 + y_0(1 - e^{-kt})] and p_A = C_A R T gives the same key
(an identity, checked).
Second wrong method (rule 11): treating the feed as pure A (y_0 = 1 in the expansion term),
p_A = P y_0 e^{-kt}/(2 - e^{-kt}), misses the key by c(1 - y_0)/(1 + c), c = 1 - e^{-kt}; it is >= 13 % off on the outside grid, which is why
the outside y_0 stops at 0.60 and kt starts at 0.7 (`check()`).
check(): integrates dn_A/dt = -k n_A, dn_B/dt = 2k n_A with an inert pool by RK4 and forms
p_A = P n_A/n_tot -- independent of the closed form.
Reference: Levenspiel, *Chemical Reaction Engineering* (3rd ed., 1999), ch. 3 ("varying-volume batch
reactor", the expansion factor eps_A); Fogler, *Elements of Chemical Reaction Engineering* (4th ed.),
ch. 3 (gas-phase variable-volume stoichiometry); Atkins & de Paula, *Physical Chemistry*, ch. on
integrated rate laws (constant-volume form). Section numbers not verified.

Cluster: new -- differs from density_with_height (ppm_conversion_pressure, the nearest ideal-gas
template: the volume follows a stated pressure) because here the pressure is fixed and the missing step
is the change in the number of gas moles produced by the reaction's stoichiometry; it differs from
reactant_in_excess (a second reactant's concentration assumed constant) because there is one reactant
and the rate law itself is exact.
Independent review (2026-10-05): "first order in A with a rate constant of ... min^-1" could be read as
d[A]/dt = -k[A] at variable volume, which reproduces the shortcut exactly (a rule-5 ambiguity). All arms now
say "each molecule of A decomposes independently with a first-order rate constant", which fixes
dn_A/dt = -k n_A. Cluster ruling: joins `reactant_in_excess` (the reviewer's ruling overrides the "new" claim
above; recorded here only -- coverage_clusters.py is not edited by this module).
"""
from __future__ import annotations

import math
import random

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True


def _e(p):
    return math.exp(-p["k"] * p["t"])


def shortcut(p):
    """Constant-volume integrated rate law: p_A = y0 P e^{-kt}, hPa."""
    return p["y"] * p["P"] * _e(p)


def correct(p):
    """Constant pressure, A -> 2B: p_A = P y0 e^{-kt}/(1 + y0 (1 - e^{-kt})), hPa."""
    e = _e(p)
    return p["P"] * p["y"] * e / (1 + p["y"] * (1 - e))


def base(rng):
    k = rnd(rng.uniform(0.020, 0.10))                       # min^-1
    return {"P": float(rng.choice(range(1000, 3001, 50))), "k": k,
            "t": rnd(rng.uniform(0.7, 1.6) / k)}              # min


def trigger(rng, regime, base):
    if regime == "inside":
        return {"y": rnd(rng.uniform(0.008, 0.020))}
    return {"y": rnd(rng.uniform(0.40, 0.60))}


def text(p, variant):
    given = ("Because each mole of A that reacts makes two moles of gas and the pressure is held fixed, the gas "
             "expands: the moles of A still fall as e^(−kt), but the partial pressure of A is P times the mole "
             "fraction of A in the growing total. " if variant == "given" else "")
    return (f"A gas mixture of compound A in nitrogen, with A at an initial mole fraction of {num(p['y'])}, is held "
            f"at constant temperature and at a constant total pressure of {num(p['P'])} hPa. A decomposes by the "
            f"gas-phase reaction A → 2B, in which each molecule of A decomposes independently with a first-order rate constant of {num(p['k'])} min^-1; "
            f"nitrogen does not react. {given}What is the partial pressure of A {num(p['t'])} min after the "
            f"reaction starts? Express your answer in hPa.")


TEMPLATE = Template("constant_pressure_expansion", "environmental_chemistry",
                    "constant-volume first-order decay of p_A -> constant-pressure reactor expanding as A -> 2B",
                    "hPa", base, trigger, text, correct, shortcut)


def _rk4(p, n=2000):
    """Independent: mole balances with an inert pool, p_A = P n_A / n_tot."""
    k = p["k"]
    na, nb, ni = p["y"], 0.0, 1 - p["y"]
    h = p["t"] / n
    for _ in range(n):
        f = lambda a: -k * a
        a1 = f(na)
        a2 = f(na + h * a1 / 2)
        a3 = f(na + h * a2 / 2)
        a4 = f(na + h * a3)
        dna = h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        na += dna
        nb -= 2 * dna
    return p["P"] * na / (na + nb + ni)


def _concentration_route(p, R=8.314, T=300.0):
    """Identity check: C_A = n_A / V(t) with the expanded volume, p_A = C_A R T."""
    n0 = p["P"] * 100 / (R * T)          # mol per m^3 of the initial volume
    v = 1 + p["y"] * (1 - _e(p))         # V/V0
    return n0 * p["y"] * _e(p) / v * R * T / 100


def check() -> str:
    max_in, min_out, max_out, max_err, max_id = 0.0, math.inf, 0.0, 0.0, 0.0
    min_wrong, min_key, max_ratio = math.inf, math.inf, 0.0
    for P in (1000.0, 3000.0):
        for k in (0.02, 0.1):
            for kt in (0.7, 1.0, 1.6):
                for y, inside in ((0.008, 1), (0.020, 1), (0.40, 0), (0.50, 0), (0.60, 0)):
                    p = {"P": P, "k": k, "t": kt / k, "y": y}
                    c, s = correct(p), shortcut(p)
                    max_err = max(max_err, gap(c, _rk4(p)))
                    max_id = max(max_id, gap(c, _concentration_route(p)))
                    min_key = min(min_key, c)
                    g = gap(c, s)
                    if inside:
                        max_in = max(max_in, g)
                        continue
                    min_out, max_out = min(min_out, g), max(max_out, g)
                    max_ratio = max(max_ratio, s / (P * y))      # shortcut below the initial p_A
                    e = _e(p)
                    min_wrong = min(min_wrong, gap(c, P * y * e / (2 - e)))   # pure-A expansion
    rng = random.Random("constant-pressure-check")
    for _ in range(3000):
        b = base(rng)
        pin, pout = {**b, **trigger(rng, "inside", b)}, {**b, **trigger(rng, "outside", b)}
        max_in = max(max_in, gap(correct(pin), shortcut(pin)))
        min_out = min(min_out, gap(correct(pout), shortcut(pout)))
        e = _e(pout)
        min_wrong = min(min_wrong, gap(correct(pout), pout["P"] * pout["y"] * e / (2 - e)))
        min_key = min(min_key, correct(pin))
    assert max_err < 1e-9 and max_id < 1e-12, (max_err, max_id)
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert min_wrong > 0.10, min_wrong
    assert min_key > 1.0 and max_ratio < 0.61, (min_key, max_ratio)   # hPa: clear of round(x, 2)
    # Anchor: pure A (y0 = 1) at half conversion: n_A = 0.5, n_tot = 1.5, p_A = P/3.
    anchor = correct({"P": 1200.0, "k": 1.0, "t": math.log(2), "y": 1.0})
    assert abs(anchor - 400.0) < 1e-9, anchor
    return (f"RK4 mole balance vs closed form {max_err:.1e}; concentration route {max_id:.1e}; inside max gap "
            f"{max_in:.4f}; outside gap {min_out:.3f}-{max_out:.3f}; pure-A expansion min miss {min_wrong:.3f}; "
            f"min key {min_key:.2f} hPa; anchor pure A at half conversion p_A = P/3")
