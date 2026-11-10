"""Reviewed wording for regime templates; physics and numerical keys stay unchanged.

Shared by the historical E0 rewording and the T inputs. Other templates retain their
original text. A replacement must match exactly once, so template edits fail visibly.
"""
from __future__ import annotations

from .templates import ALL_TEMPLATES
from .templates.base import num


def _given_sentence(name: str, p: dict) -> str:
    """The sentence the template inserts in its given arm (the given text minus the outside text)."""
    t = ALL_TEMPLATES[name]
    g, o = t.text(p, "given"), t.text(p, "outside")
    i = 0
    while i < len(o) and g[i] == o[i]:
        i += 1
    return g[i:i + len(g) - len(o)]


def _line(p: dict, variant: str) -> str:
    given = _given_sentence("line_broadening", p) if variant == "given" else ""
    return (f"A carbon dioxide absorption line is centred at 667 cm^-1. At a temperature of {num(p['t'])} K and a "
            f"pressure of 1013 hPa, its pressure-broadened (Lorentz) half-width at half maximum is 2100 MHz, and the "
            f"pressure-broadened width is proportional to pressure. {given}At a level in the atmosphere where the "
            f"pressure is {num(p['p'])} hPa and the temperature is {num(p['t'])} K, what is the half-width at half "
            f"maximum of the line's absorption profile, as a spectrometer of unlimited resolution would measure it "
            f"there? Express your answer in MHz.")


REPLACE = {
    "drop_ventilation": [
        ("terms of the diffusional growth equation (Rogers & Yau ch. 7,",
         "terms of the growth equation for a drop at rest relative to the air (Rogers & Yau ch. 7,"),
        ("By how many nanometres per second is the radius decreasing",
         "By how many nanometres per second is the falling drop's radius decreasing"),
    ],
    "clausius_clapeyron_extrapolation": [
        ("increasing by about", "currently increasing at about"),
        (" hPa per additional kelvin.", " hPa per kelvin."),
    ],
}

def text(name: str, params: dict, variant: str) -> str:
    if name == "line_broadening":
        return _line(params, variant)
    result = ALL_TEMPLATES[name].text(params, variant)
    for old, new in REPLACE.get(name, ()):
        if result.count(old) != 1:
            raise ValueError(f"{name}/{variant}: {old!r} found {result.count(old)} times")
        result = result.replace(old, new)
    return result
