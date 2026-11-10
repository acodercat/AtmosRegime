"""Warming from condensation alone -> warming from condensation plus freezing of supercooled liquid.

Added 2026-10-01 (extended set, docs/PROPOSAL.md E0d / E4'): a mechanism where the standard
formula silently omits a physical process (the heat of fusion), and the trigger (mass of liquid
that freezes) is an ordinary in-cloud value in both arms -- nothing in the outside problem looks
numerically extreme.

Parent's method: during an ascent step, m_v grams of water vapour condense per kilogram of air,
releasing latent heat of vaporisation and warming the air isobarically: Delta T = L_v m_v / c_p.
Where it fails: the cloud also carries supercooled liquid water, and during the same step m_f
grams per kilogram freeze. Freezing releases the latent heat of fusion in addition to whatever
condensation is happening, and the two contributions simply add (both are isobaric phase changes
releasing their heat into the same parcel): Delta T = (L_v m_v + L_f m_f) / c_p. A solver who reads
"m_v condenses" and "m_f freezes" but treats the freezing as a bookkeeping fact about the cloud's
state -- not a separate heat source -- reproduces only the first term; nothing about m_f on its own
looks unusual (1-3 g/kg of newly frozen cloud water is ordinary), so there is no numeric cue that a
term is missing.
Fair from the text: every arm states m_v, m_f, L_v, L_f and c_p; the inside arm carries the same
m_f given even though it barely matters there (rule 3). Neither arm says that freezing releases
heat or writes the two-term formula -- only the given sentence does.
Inside: m_f = 0.01-0.03 g/kg (a trace of supercooled liquid, consistent with a cloud that has only
just become supercooled, or with most of the drop population not yet glaciated); the fusion term
is below 1.4 % of the condensation term anywhere on the grid.
Outside: m_f = 1.0-3.0 g/kg (an ordinary amount of liquid water freezing in a cold cumulus or during
riming onset), while m_v (the condensation happening in this same step) is kept small, 0.3-0.55
g/kg -- deliberately a modest ascent step, so that the pre-existing supercooled liquid's freezing is
not swamped by fresh condensation. The fusion term is then 20-73 % of the condensation term, giving
a key at least 19.5 % above the shortcut everywhere on the grid.
Conventions: L_v = 2.5e6 J/kg and L_f = 3.34e5 J/kg (0 degC values, Rogers & Yau / Wallace & Hobbs
standard table) vs. L_v = 2.501e6, L_f = 3.337e5 (slightly different compilations), and dry c_p =
1005 J/kg/K vs. a moist c_p correction c_p(1+0.87 q) for the total water present, all move the key
by well under 2 % (checked below).
Rule 6: the key is a temperature RISE (a difference, K), never an absolute temperature.
Reference: Rogers & Yau, A Short Course in Cloud Physics, ch. 2-3 (latent heats, mixed-phase
isobaric warming); the same energetics underlie riming and the Wegener-Bergeron-Findeisen process.
Related mechanism: evaporative_cooling_limit (same family, opposite sign and a saturation cap
instead of a missing term); ice_saturation (also a phase-change mechanism, but a vapour-pressure
offset rather than a missing heat term).

Judgement call (flagged): the condensation amount m_v is deliberately kept small (0.3-0.55 g/kg,
below the typical several-g/kg of a vigorous ascent) so the fusion term from an ordinary-looking
m_f can clear the 15 % gap without needing an implausibly large m_f. This is a modest, not
vigorous, ascent step -- stated as such in the text -- rather than a tuning artefact hidden from
the reader.

Independent review (2026-10-01): "ascends" contradicted "constant pressure" (an ascending parcel cools by
expansion); the text now describes one step of the parcel's evolution and says to consider only the latent heating.
"""
from __future__ import annotations

from .base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, Template, gap, num, rnd

EXTENDED = True   # extended set (templates/__init__.py); not one of the original 30

LV, LF, CP = 2.5e6, 3.34e5, 1005.0


def base(rng):
    return {"mv": rnd(rng.uniform(0.3, 0.55), 2)}


def trigger(rng, regime, base):
    return {"mf": rnd(rng.uniform(0.01, 0.03) if regime == "inside" else rng.uniform(1.0, 3.0), 2)}


def shortcut(p):
    return LV * p["mv"] * 1e-3 / CP                                  # K, fusion ignored


def correct(p):
    return (LV * p["mv"] + LF * p["mf"]) * 1e-3 / CP                 # K


def text(p, variant):
    given = ("Freezing of the supercooled liquid releases additional latent heat on top of the heat "
             "of condensation, and the two add: ΔT = (L_v m_v + L_f m_f)/c_p, where m_f is the mass "
             "of water that freezes per kilogram of air. " if variant == "given" else "")
    return (f"In one step of a cloud parcel's evolution, {num(p['mv'])} g of water vapour "
            f"condenses per kilogram of air. The cloud also contains supercooled liquid water, and "
            f"during this same step {num(p['mf'])} g of that liquid freezes per kilogram of air. "
            f"Treat the parcel and the water it carries as a closed system that exchanges no heat "
            f"with its surroundings and stays at constant pressure, consider only the latent heating in this step, "
            f"and take L_v = 2.5 × 10^6 "
            f"J kg^-1, L_f = 3.34 × 10^5 J kg^-1 and c_p = 1005 J kg^-1 K^-1. {given}By how much "
            f"does the parcel's temperature rise over this step? Express your answer in K.")


TEMPLATE = Template("freezing_latent_heat", "cloud_physics",
                    "isobaric warming from condensation alone -> warming from condensation plus "
                    "freezing of supercooled liquid",
                    "K", base, trigger, text, correct, shortcut)


def _stepwise(p, n=200):
    """Independent: accumulate the two heat releases in n small interleaved steps rather than one
    closed-form sum, confirming the result is path-independent (order of release does not matter)."""
    t = 0.0
    dmv, dmf = p["mv"] / n, p["mf"] / n
    for _ in range(n):
        t += LV * dmv * 1e-3 / CP
        t += LF * dmf * 1e-3 / CP
    return t


def check() -> str:
    max_in, min_out, max_step, max_conv = 0.0, float("inf"), 0.0, 0.0
    keys = []
    for mv in (0.3, 0.55):
        for mf in (0.01, 0.03):
            pin = {"mv": mv, "mf": mf}
            max_in = max(max_in, gap(correct(pin), shortcut(pin)))
            max_step = max(max_step, abs(_stepwise(pin) - correct(pin)) / correct(pin))
        for mf in (1.0, 3.0):
            pout = {"mv": mv, "mf": mf}
            g_ = gap(correct(pout), shortcut(pout))
            min_out = min(min_out, g_)
            keys.append(correct(pout))
            max_step = max(max_step, abs(_stepwise(pout) - correct(pout)) / correct(pout))
            for lv, lf, cp in ((2.501e6, 3.337e5, 1005.0), (2.5e6, 3.34e5, 1005.0 * (1 + 0.87 * (mv + mf) * 1e-3))):
                alt = (lv * mv + lf * mf) * 1e-3 / cp
                max_conv = max(max_conv, abs(alt / correct(pout) - 1))
    assert max_in <= INSIDE_MAX_GAP and min_out >= OUTSIDE_MIN_GAP, (max_in, min_out)
    assert max_step < 1e-9 and max_conv < 0.02
    ratio = LF / LV
    assert abs(ratio - 0.1336) < 0.002, ratio
    return (f"max inside gap {max_in:.4f}, min outside gap {min_out:.3f}; stepwise release vs "
            f"closed form {max_step:.1e}; constant/c_p conventions {max_conv:.4f}; "
            f"L_f/L_v = {ratio:.4f}; outside keys {min(keys):.2f}-{max(keys):.2f} K")
