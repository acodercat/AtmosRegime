"""Plug-flow exponential removal -> well-mixed (CSTR) removal in a continuously flushed chamber.

Added 2026-10-01 (extended set, docs/PROPOSAL.md E0d / E4'): a mechanism where the standard
formula silently omits a physical process (back-mixing of already-cleaned air into the inlet
stream), and the trigger (a rate constant) is an ordinary value in both arms. "The chamber is
well mixed" is stated in *every* arm, including inside, specifically so it cannot work as a cue
that only appears when it matters (rule 4): a solver has to notice that the well-mixed statement
and the exponential decay law it habitually reaches for are in tension, not spot a new sentence.

Parent's method: a pollutant entering a reactor and destroyed by a first-order process with rate
constant k, after a residence time tau, is reduced by the plug-flow law C/C0 = exp(-k tau) --
every parcel of air spends exactly tau in the reactor and decays independently, so the removed
fraction is 1 - exp(-k tau). This is the formula taught as "the" first-order removal law in most
intro treatments and is correct for a plug-flow reactor or a single parcel's residence time.
Where it fails: the chamber is stated to be well mixed, so the concentration inside is uniform and
equal to the outlet concentration -- freshly entering, dirty air is instantly diluted into
already-cleaned air, and some of what leaves has spent far less than tau inside while some has
spent far longer. A steady-state mass balance over the whole chamber (Q C0 = Q C + k V C, with
tau = V/Q) gives C/C0 = 1/(1 + k tau), i.e. the removed fraction is the CSTR expression
k tau/(1 + k tau) (Levenspiel, Chemical Reaction Engineering, ch. 5; Seinfeld & Pandis, Atmospheric
Chemistry and Physics, ch. 18, box/well-mixed indoor and chamber models). The two formulas share
the same leading-order behaviour as k tau -> 0 (both -> k tau) but diverge once k tau is not small.
Fair from the text: every arm states the chamber volume V, the flow rate Q and the rate constant k
(so tau = V/Q and the Damkohler number Da = k tau are both computable), and every arm states that
the chamber is well mixed. Nothing about the replacement formula or the word "CSTR"/"back-mixing"
appears outside the given sentence.
Inside: Da = k tau = 0.01-0.03, i.e. a lightly loaded or fast-flushing chamber; the plug-flow and
CSTR formulas agree to within 1.6 % there (the first point at which they could possibly diverge:
both expand as Da - Da^2/2 + ... vs Da - Da^2 + ..., so the gap is second order in Da).
Outside: Da = k tau = 1.0-3.0, an ordinary operating point for an air-cleaning device (neither the
rate constant nor the Damkohler number look extreme on their own); the formulas disagree by at
least 26 % there. V and Q are drawn (30-60 m^3, 20-50 m^3/min) so that tau = 0.6-3.0 min and the
resulting k values (0.003-5 min^-1 across both arms) stay ordinary.
Conventions: the CSTR mass balance is the unique textbook result for a single well-mixed stage with
first-order loss at steady state -- it is not a case of competing empirical fits the way a vapour-
pressure or emission-factor formula is. check()'s "convention" entry instead measures robustness
to (a) rounding k to two significant figures before rendering the text, against the exact
pre-rounding Damkohler number, and (b) the order of multiplication (k*(V/Q) vs (k*V)/Q).
Reference: Levenspiel (1999), Chemical Reaction Engineering, 3rd ed., ch. 5 (CSTR vs. PFR
conversion); Seinfeld & Pandis (2016), ch. 18 (well-mixed box models for indoor/chamber air
quality). Related mechanism: plume_mixing_lid (original 30; a different Gaussian-plume mixing
assumption), ppm_conversion_pressure (same air-quality family, different omitted term).

Judgement call (flagged): this mechanism has no competing physical "convention" the way most other
templates in this set do (there is no alternative textbook formula for a single CSTR at steady
state), so check()'s convention check is a robustness/numerics check rather than a formula-spread
check; this is noted rather than silently reused as if it were the same kind of evidence.


Lead review (2026-10-01): dropped the clause explaining what 'well mixed' implies for the outlet
concentration (it spelled out the replacement model); 'well mixed' alone is kept in every arm.
"""
from __future__ import annotations

import math

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True   # extended set (templates/__init__.py); not one of the original 30


def base(rng):
    return {"v": float(rng.choice(range(30, 61, 5))), "q": float(rng.choice(range(20, 51, 5)))}


def trigger(rng, regime, base):
    da = rng.uniform(0.01, 0.03) if regime == "inside" else rng.uniform(1.0, 3.0)
    tau = base["v"] / base["q"]
    return {"k": rnd(da / tau, 2)}


def _da(p):
    return p["k"] * p["v"] / p["q"]


def shortcut(p):
    return 100 * (1 - math.exp(-_da(p)))                              # % removed, plug flow


def correct(p):
    da = _da(p)
    return 100 * da / (1 + da)                                        # % removed, well-mixed CSTR


def text(p, variant):
    given = ("Because the chamber is well mixed, a steady-state mass balance gives "
             "C = C0/(1 + k τ) rather than the plug-flow decay law, where τ = V/Q is the mean "
             "residence time, so the fraction removed is k τ/(1 + k τ). " if variant == "given" else "")
    return (f"A chamber of volume V = {num(p['v'])} m^3 is continuously flushed by contaminated air "
            f"at a volumetric flow rate Q = {num(p['q'])} m^3 min^-1. The air inside the chamber is "
            f"well mixed. The pollutant is destroyed by a first-order process with rate constant "
            f"k = {num(p['k'])} min^-1. {given}In steady state, what percentage of the pollutant "
            f"entering the chamber is removed before the air leaves? Express your answer as a "
            f"percentage.")


TEMPLATE = Template("cstr_vs_pfr", "air_quality",
                    "plug-flow exponential removal -> well-mixed (CSTR) removal once the Damköhler "
                    "number is not small",
                    "%", base, trigger, text, correct, shortcut)


def _steady_state(p, steps_per_tau=4000, taus=25):
    """Independent: integrate dC/dt = (Q/V)(C0 - C) - kC from C=0 with explicit Euler over many
    residence times to approximate steady state, instead of using the closed form."""
    tau = p["v"] / p["q"]
    dt = tau / steps_per_tau
    c, c0, k = 0.0, 1.0, p["k"]
    for _ in range(int(steps_per_tau * taus)):
        c += dt * ((c0 - c) / tau - k * c)
    return 100 * (1 - c / c0)


def check() -> str:
    max_in, min_out, max_ss = 0.0, float("inf"), 0.0
    keys = []
    for v in (30.0, 60.0):
        for q in (20.0, 50.0):
            for da in (0.01, 0.03):
                pin = {"v": v, "q": q, "k": da * q / v}
                max_in = max(max_in, gap(correct(pin), shortcut(pin)))
                max_ss = max(max_ss, abs(_steady_state(pin) - correct(pin)) / correct(pin))
            for da in (1.0, 3.0):
                pout = {"v": v, "q": q, "k": da * q / v}
                g_ = gap(correct(pout), shortcut(pout))
                min_out = min(min_out, g_)
                keys.append(correct(pout))
                max_ss = max(max_ss, abs(_steady_state(pout) - correct(pout)) / correct(pout))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_ss < 0.01

    # Robustness: order of operations, and rounding k to 2 sig figs before rendering.
    p = {"v": 45.0, "q": 35.0, "k": 1.2345}
    assert math.isclose(p["k"] * (p["v"] / p["q"]), (p["k"] * p["v"]) / p["q"], rel_tol=1e-12)
    tau = p["v"] / p["q"]
    da_exact = 2.0
    k_exact = da_exact / tau
    k_round = rnd(k_exact, 2)
    round_err = abs(correct({**p, "k": k_round}) / correct({**p, "k": k_exact}) - 1)
    assert round_err < 0.02, round_err

    # Hand-checkable anchor: Da = 1 -> plug flow 1 - 1/e = 63.21 %, CSTR 1/2 = 50.00 %.
    anchor = {"v": 1.0, "q": 1.0, "k": 1.0}
    assert abs(shortcut(anchor) - 100 * (1 - 1 / math.e)) < 1e-9
    assert abs(correct(anchor) - 50.0) < 1e-9

    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; steady-state ODE vs "
            f"closed form {max_ss:.1e}; k-rounding robustness {round_err:.4f}; Da=1 anchor "
            f"{shortcut(anchor):.2f}% vs {correct(anchor):.2f}%; outside keys "
            f"{min(keys):.1f}-{max(keys):.1f}%")
