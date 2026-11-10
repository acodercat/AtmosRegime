"""Mechanism clusters over the whole template library (72 templates, 32 clusters).

Any train / held-out split of templates (docs/PREREG_MAIN.md, experiment T) is made **by
cluster**: templates in one cluster are never split across the two sides, otherwise a model could
"transfer" to a held-out template only because it learned the same mechanism on a sibling. The rule is
therefore CONSERVATIVE. Two templates are in one cluster if they

- share the parent's standard relation (the shortcut is the same law), or
- apply the same kind of correction to the same law, or
- would let a habit learned on one solve the other (same text frame and same missing step).

When in doubt, merge.

This module extends `screening.CLUSTERS` without changing it: every original cluster keeps its name and
all its members, and only extended templates are ever added to an original cluster (an original
singleton may seed a *new* cluster, never join an original one). `screening.py` belongs to a sealed
pre-registration and is not edited. Template-reviewer rulings of 2026-10-05 are adopted as given.
"""
from __future__ import annotations

from .screening import CLUSTERS as SCREENING_CLUSTERS
from .templates import ALL_TEMPLATES, ORIGINAL_30

COVERAGE_CLUSTERS: dict[str, tuple[str, ...]] = {
    # ---- original clusters (screening.CLUSTERS), possibly extended --------------------------------
    "rotation": ("coriolis_neglect", "inertia_gravity_wave",
                 # + same parent as inertia_gravity_wave (omega = N k/m), a frequency correction to it
                 #   (reviewer ruling 2026-10-05)
                 "gravity_wave_doppler_shift"),
    "rotating_balance": ("gradient_wind_curvature", "rossby_deformation",
                         # + parent is the geostrophic wind V_g = a/f, as in gradient_wind_curvature;
                         #   correction = one extra term (friction) in the same horizontal balance
                         "frictional_three_force_balance"),
    "density_with_height": ("hydrostatic_constant_density", "anelastic_continuity", "ppm_conversion_pressure",
                            # + hydrostatic pressure profile integrated with the wrong T(z) above 11 km
                            "isa_above_tropopause"),
    "spherical_geometry": ("radar_beam_height", "solid_angle", "spherical_slant_path",
                           # + plane-parallel sec(theta) path vs curved shell near the horizon: the
                           #   parent and correction of spherical_slant_path
                           "sun_photometer_airmass"),
    "aqueous_equilibria": ("henry_depletion", "effective_henry", "weak_acid_dissociation",
                           # + the same weak-acid mass-action law, corrected for activities
                           "ionic_strength_activity"),
    "dispersion": ("plume_mixing_lid", "finite_line_source", "taylor_dispersion",
                   # + Gaussian plume with an image source at a reflecting boundary, as plume_mixing_lid
                   "plume_ground_reflection"),
    "radiative_linearisation": ("planck_linearization", "optically_thin_emission",
                                # + tangent-line (local-slope) extrapolation used for a large increment,
                                #   same missing step as planck_linearization
                                "co2_forcing_log", "ch4_forcing_sqrt", "clausius_clapeyron_extrapolation",
                                # + Clausius-Clapeyron temperature dependence of a saturation quantity,
                                #   the correction clausius_clapeyron_extrapolation trains
                                "semivolatile_partitioning_temperature"),
    "size_vs_length_scale": ("mie_geometric", "kelvin_curvature", "cunningham_slip", "ice_saturation",
                             # + parent is Stokes settling, as in cunningham_slip; regime correction
                             "raindrop_drag_regime",
                             # + parent is the slip-corrected Stokes settling speed of cunningham_slip
                             "deposition_brownian"),
    # ---- new clusters --------------------------------------------------------------------------------
    # surface gravity wave: water-body length vs wavelength, sqrt(gH) long-wave limit (ruling 2026-10-05)
    "surface_gravity_wave": ("shallow_water_wave_speed", "bay_tidal_resonance"),
    # latent heating Delta T = L m / c_p (or melt = Q / L_f) with a term or limit left out
    # (snowpack_cold_content joins freezing_latent_heat by ruling 2026-10-05; evaporative_cooling_limit's
    # shortcut is exactly freezing_latent_heat's parent relation)
    "latent_heating": ("evaporative_cooling_limit", "freezing_latent_heat", "snowpack_cold_content"),
    # one-box / relaxation model dC/dt = S - C/tau: steady value reported before it is reached, or a
    # loss channel (ventilation, flushing) left out (box_ventilation_loss joins nonsteady_burden by
    # ruling 2026-10-05; mixed_layer_response is the same F/lambda (1 - e^{-t/tau}) step)
    "box_model": ("mixed_layer_response", "nonsteady_burden", "box_ventilation_loss", "cstr_vs_pfr",
                  # + L + D = 1 gives dL/dt = k - 2kL: the same relaxation with the source term dropped
                  #   (ruling 2026-10-05, batch 2)
                  "racemization_reversibility"),
    # second reactant assumed in excess (constant), when it is in fact comparable or limiting
    "reactant_in_excess": ("pseudo_first_order", "limiting_reagent",
                           # + same integrated first-order law of a gas-phase reactant; the correction is
                           #   bookkeeping of the other species (ruling 2026-10-05, batch 2)
                           "constant_pressure_expansion"),
    # neutral log wind profile u = (u*/kappa) ln(z/z0) with a correction (stability, displacement height)
    "log_wind_profile": ("log_wind_stability", "displacement_height_canopy"),
    # Maxwell diffusional growth of a particle, sphere-at-rest form corrected (ruling 2026-10-05)
    "diffusional_growth": ("drop_ventilation", "ice_capacitance_shape"),
    # flux through a layer whose thickness is itself the unknown: write the flux through the current h and
    # integrate h dh (ruling 2026-10-05, batch 2)
    "flux_through_growing_layer": ("unconfined_dupuit", "sea_ice_insulation"),
    # cos(theta) weighting between radiance and flux / path through a slab (ruling 2026-10-05, batch 2)
    "angular_weighting": ("diffuse_transmission", "actinic_flux_horizontal"),
    # integrate N(D) D^k over the stated size distribution exactly as stated (ruling 2026-10-05, batch 2)
    "size_distribution_moment": ("lognormal_mass", "drop_size_truncation"),
}

_CLUSTER_OF: dict[str, str] = {}
for _name, _members in COVERAGE_CLUSTERS.items():
    for _t in _members:
        assert _t not in _CLUSTER_OF, f"{_t} is in two clusters ({_CLUSTER_OF[_t]}, {_name})"
        _CLUSTER_OF[_t] = _name


def unit_of(template: str) -> str:
    """The cluster a template belongs to, or the template's own name if it is a singleton."""
    if template not in ALL_TEMPLATES:
        raise KeyError(template)
    return _CLUSTER_OF.get(template, template)


def units() -> dict[str, tuple[str, ...]]:
    """Every independent unit -> its templates; each template in ALL_TEMPLATES appears exactly once."""
    out: dict[str, tuple[str, ...]] = {name: members for name, members in COVERAGE_CLUSTERS.items()}
    for t in sorted(ALL_TEMPLATES):
        if t not in _CLUSTER_OF:
            out[t] = (t,)
    return out


# ---- import-time checks ------------------------------------------------------------------------------
_unknown = set(_CLUSTER_OF) - set(ALL_TEMPLATES)
assert not _unknown, f"unknown templates in COVERAGE_CLUSTERS: {sorted(_unknown)}"
for _name, _members in SCREENING_CLUSTERS.items():
    # every original cluster still exists, holds all its members, and gained no original template
    assert _name in COVERAGE_CLUSTERS, f"original cluster {_name} missing"
    assert set(_members) <= set(COVERAGE_CLUSTERS[_name]), f"original cluster {_name} was split"
    assert set(COVERAGE_CLUSTERS[_name]) & ORIGINAL_30 == set(_members), \
        f"an original template was moved into original cluster {_name}"
_all = [t for members in units().values() for t in members]
assert len(_all) == len(set(_all)) == len(ALL_TEMPLATES) and set(_all) == set(ALL_TEMPLATES), \
    "every template must appear in exactly one unit"
del _all, _unknown
