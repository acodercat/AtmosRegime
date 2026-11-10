"""X5 (docs/PROPOSAL.md section 5.6, PREREG_X amendment X-3): regime templates outside the environmental sciences —
heat transfer, fluid mechanics, structures, mechanics, optics, electromagnetism, chemical engineering.

Same contract as `atmoscoder_rl.templates` (`base.Template`, one trigger, inside / outside / given, rules 1-11 of
`templates/MECHANISMS.md`), kept in a separate registry so that no build of the atmospheric set changes.
Evaluation only; never trained on. Every module defines `TEMPLATE` and `check()`.
"""
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from ..templates.base import Template

MODULES: dict[str, ModuleType] = {}
for _info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
    _module = importlib.import_module(f"{__name__}.{_info.name}")
    if not hasattr(_module, "TEMPLATE"):
        continue
    if not callable(getattr(_module, "check", None)):
        raise ImportError(f"X5 template module {_info.name} defines TEMPLATE but no check()")
    MODULES[_module.TEMPLATE.name] = _module

TEMPLATES: dict[str, Template] = {name: module.TEMPLATE for name, module in MODULES.items()}

#: Domain of each template (reported per domain; never pooled with the atmospheric set).
DOMAIN = {
    "lumped_capacitance_biot": "heat_transfer", "fin_tip_length": "heat_transfer", "enclosure_radiation": "heat_transfer",
    "pipe_friction_regime": "fluid_mechanics", "orifice_choked": "fluid_mechanics",
    "thick_wall_hoop": "structures", "pendulum_amplitude": "mechanics", "spring_mass_effective": "mechanics",
    "gravity_altitude": "mechanics", "thick_lens": "optics", "grating_small_angle": "optics",
    "solenoid_finite": "electromagnetism", "stefan_diffusion": "chemical_engineering",
}
