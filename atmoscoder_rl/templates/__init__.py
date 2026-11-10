"""Regime templates: one module per mechanism, each generating matched problems (see `base.py`).

A template module defines `TEMPLATE` (a `base.Template`) and `check()` — the offline physics
check `atmoscoder_rl.check_templates` runs (independent recomputation, gap bounds over the
sampled ranges, spread of published conventions, a literature anchor). Every module here that
defines `TEMPLATE` is registered, in file-name order, so a template is added by adding its
file. The design rules and the status of each template are in `MECHANISMS.md`.

`PILOT` names the pilot templates (2026-09-24) still in the set; the rest were added 2026-09-26.

A module that sets `EXTENDED = True` belongs to the **extended set** (2026-10-01, E0d / E4' in
docs/PROPOSAL.md) and is registered in `EXTENDED_MODULES` / `EXTENDED_TEMPLATES`, not in `MODULES` /
`TEMPLATES`: the original 30 stay exactly the set every earlier run was built from.
"""
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from .base import Template

#: The three pilot templates still in the set (the pilot's other two, vorticity_stretching and
#: nonhydrostatic_wave, were removed 2026-09-27: their shortcut is not the method textbooks teach).
PILOT = ("cunningham_slip", "spherical_slant_path", "saturated_lapse")

MODULES: dict[str, ModuleType] = {}
EXTENDED_MODULES: dict[str, ModuleType] = {}
for _info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
    _module = importlib.import_module(f"{__name__}.{_info.name}")
    if not hasattr(_module, "TEMPLATE"):
        continue
    if not callable(getattr(_module, "check", None)):
        raise ImportError(f"template module {_info.name} defines TEMPLATE but no check()")
    (EXTENDED_MODULES if getattr(_module, "EXTENDED", False) else MODULES)[_module.TEMPLATE.name] = _module

#: The original 30 by name. A new module that forgets `EXTENDED = True` would otherwise join this set
#: silently and change every build of "all templates" (it happened once, 2026-10-01).
ORIGINAL_30 = frozenset({
    "anelastic_continuity", "coriolis_neglect", "cunningham_slip", "doppler_aliasing", "effective_henry",
    "evaporative_cooling_limit", "finite_line_source", "gradient_wind_curvature", "henry_depletion",
    "hydrostatic_constant_density", "ice_saturation", "inertia_gravity_wave", "kelvin_curvature",
    "line_broadening", "log_wind_stability", "lognormal_mass", "mie_geometric", "mixed_layer_response",
    "optically_thin_emission", "planck_linearization", "plume_mixing_lid", "ppm_conversion_pressure",
    "pseudo_first_order", "radar_beam_height", "rossby_deformation", "saturated_lapse", "solid_angle",
    "spherical_slant_path", "taylor_dispersion", "weak_acid_dissociation"})
if set(MODULES) != ORIGINAL_30:
    raise ImportError(f"the original template set changed: extra {sorted(set(MODULES) - ORIGINAL_30)}, "
                      f"missing {sorted(ORIGINAL_30 - set(MODULES))} (a new module needs EXTENDED = True)")

TEMPLATES: dict[str, Template] = {name: module.TEMPLATE for name, module in MODULES.items()}
EXTENDED_TEMPLATES: dict[str, Template] = {name: module.TEMPLATE for name, module in EXTENDED_MODULES.items()}
ALL_TEMPLATES: dict[str, Template] = {**TEMPLATES, **EXTENDED_TEMPLATES}
