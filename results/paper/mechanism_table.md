# Mechanism table (one row per mechanism; `explore/mechanism_table.py`)

Percent. base = Qwen3-8B base outside accuracy; p64 = base pass@64 of each family; A/D/F = Qwen3-8B training-fold outside accuracy after RL (two seeds averaged); A35 = Qwen3.5-9B A arm; P/Pk2/Pshuf/S = OPD arms; DOSE = dose:RL outside accuracy per seed (42 / 43).

| mechanism | category | cluster | fold | base | p64 Q3-8B | p64 Q3.5-9B | p64 Ministral | A | D | F | A35 | P | Pk2 | Pshuf | S | DOSE Q3-8B | DOSE Ministral |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| actinic_flux_horizontal | atmospheric_chemistry | angular_weighting | F1 | 0 | 65 | 100 | 100 | 100 | 100 | 50 | 100 | 100 | 30 | 0 | 0 | - | - |
| anelastic_continuity | atmospheric_dynamics | density_with_height | F2 | 0 | 45 | 100 | 25 | 0 | 0 | 0 | 100 | 20 | 25 | 0 | 5 | - | - |
| bay_tidal_resonance | oceanography | surface_gravity_wave | F1 | 0 | 30 | 100 | 30 | 0 | 0 | 2 | 100 | 0 | 0 | 0 | 0 | - | - |
| box_ventilation_loss | air_quality | box_model | F1 | 0 | 50 | 100 | 100 | 100 | 100 | 100 | 100 | 65 | 25 | 20 | 0 | - | - |
| capillary_rise_gravity | soil | capillary_rise_gravity | F2 | 0 | 10 | 100 | 15 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| ch4_forcing_sqrt | climate_dynamics | radiative_linearisation | F2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | d0:0 / d0:0 | d0:0 / d0:0 |
| clausius_clapeyron_extrapolation | atmospheric_thermodynamics | radiative_linearisation | F2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 85 | 10 | 0 | 0 | d4:100 / d1:0 | d4:100 / d4:100 |
| co2_forcing_log | climate_dynamics | radiative_linearisation | F2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 100 | 100 | 0 | 0 | d2:0 / d4:100 | d4:100 / d1:100 |
| condensate_loading | cloud_physics | condensate_loading | F1 | 55 | 100 | 100 | 75 | 100 | 100 | 100 | 100 | 100 | 100 | 90 | 100 | - | - |
| constant_pressure_expansion | environmental_chemistry | reactant_in_excess | F2 | 0 | 5 | 100 | 75 | 100 | 95 | 98 | 100 | 15 | 0 | 0 | 0 | - | - |
| cooper_jacob_early_time | hydrology | cooper_jacob_early_time | F1 | 0 | 80 | 100 | 80 | 100 | 100 | 100 | 100 | 5 | 0 | 5 | 0 | - | - |
| coriolis_neglect | atmospheric_dynamics | rotation | F2 | 0 | 15 | 100 | 20 | 0 | 0 | 0 | 100 | 30 | 60 | 0 | 0 | - | - |
| cstr_vs_pfr | air_quality | box_model | F1 | 0 | 75 | 100 | 100 | 98 | 100 | 100 | 100 | 100 | 100 | 0 | 5 | - | - |
| cunningham_slip | atmospheric_aerosols | size_vs_length_scale | F2 | 0 | 5 | 100 | 80 | 0 | 0 | 0 | 100 | 55 | 85 | 0 | 0 | d2:100 / d1:95 | - |
| deposition_brownian | atmospheric_aerosols | size_vs_length_scale | F2 | 0 | 5 | 95 | 25 | 0 | 0 | 0 | 60 | 5 | 0 | 0 | 0 | - | - |
| diffuse_transmission | atmospheric_radiation | angular_weighting | F1 | 0 | 0 | 100 | 10 | 0 | 0 | 0 | 100 | 100 | 0 | 0 | 0 | d0:0 / d1:0 | d1:100 / d1:0 |
| displacement_height_canopy | boundary_layer | log_wind_profile | F1 | 0 | 0 | 100 | 100 | 0 | 0 | 0 | 100 | 95 | 100 | 0 | 0 | d2:0 / d1:0 | - |
| doppler_aliasing | observation_and_modeling | doppler_aliasing | F2 | 0 | 10 | 100 | 0 | 0 | 0 | 10 | 100 | 75 | 0 | 0 | 0 | d0:0 / d2:100 | d0:100 / d0:100 |
| drop_size_truncation | observation_and_modeling | size_distribution_moment | F2 | 35 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 45 | 65 | 20 | 45 | - | - |
| drop_ventilation | cloud_physics | diffusional_growth | F2 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| effective_henry | atmospheric_chemistry | aqueous_equilibria | F1 | 5 | 100 | 100 | 90 | 100 | 100 | 95 | 100 | 45 | 80 | 0 | 5 | - | - |
| freezing_latent_heat | cloud_physics | latent_heating | F2 | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 0 | 0 | 5 | - | - |
| freezing_point_depth | oceanography | freezing_point_depth | F1 | 0 | 0 | 100 | 10 | 0 | 0 | 2 | 100 | 95 | 60 | 0 | 0 | d4:100 / d2:100 | d4:100 / d2:100 |
| fresnel_reflectance_grazing | atmospheric_radiation | fresnel_reflectance_grazing | F2 | 5 | 60 | 100 | 100 | 100 | 100 | 60 | 100 | 45 | 80 | 5 | 0 | - | - |
| frictional_three_force_balance | boundary_layer | rotating_balance | F1 | 0 | 45 | 100 | 50 | 50 | 100 | 0 | 100 | 35 | 35 | 0 | 5 | - | - |
| gradient_wind_curvature | atmospheric_dynamics | rotating_balance | F1 | 0 | 35 | 100 | 25 | 100 | 98 | 95 | 100 | 5 | 10 | 0 | 0 | - | - |
| gravity_wave_doppler_shift | atmospheric_dynamics | rotation | F2 | 0 | 10 | 100 | 70 | 95 | 92 | 0 | 100 | 0 | 5 | 0 | 0 | - | - |
| ground_heat_flux_neglect | boundary_layer | ground_heat_flux_neglect | F1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | - | - |
| henry_depletion | atmospheric_chemistry | aqueous_equilibria | F1 | 0 | 5 | 100 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | d2:0 / d2:0 | d0:0 / d0:0 |
| hydrostatic_constant_density | atmospheric_thermodynamics | density_with_height | F2 | 15 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 90 | 85 | 80 | 75 | - | - |
| ice_above_flotation | climate_dynamics | ice_above_flotation | F2 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| ice_capacitance_shape | cloud_physics | diffusional_growth | F2 | 0 | 10 | 45 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| ice_saturation | cloud_physics | size_vs_length_scale | F2 | 5 | 95 | 100 | 100 | 100 | 100 | 100 | 100 | 80 | 25 | 0 | 5 | - | - |
| inertia_gravity_wave | atmospheric_dynamics | rotation | F2 | 0 | 5 | 15 | 5 | 0 | 0 | 0 | 100 | 15 | 10 | 0 | 0 | d0:0 / d1:100 | d1:100 / d1:100 |
| ionic_strength_activity | atmospheric_chemistry | aqueous_equilibria | F1 | 0 | 80 | 100 | 40 | 18 | 8 | 22 | 100 | 5 | 5 | 0 | 10 | - | - |
| isa_above_tropopause | atmospheric_thermodynamics | density_with_height | F2 | 0 | 10 | 100 | 90 | 100 | 100 | 98 | 100 | 5 | 0 | 0 | 0 | - | - |
| isotope_mixing_weighting | environmental_chemistry | isotope_mixing_weighting | F1 | 0 | 15 | 100 | 85 | 100 | 100 | 100 | 100 | 15 | 0 | 0 | 0 | - | - |
| kelvin_curvature | cloud_physics | size_vs_length_scale | F2 | 0 | 60 | 100 | 100 | 50 | 100 | 0 | 100 | 70 | 80 | 0 | 5 | - | - |
| limiting_reagent | atmospheric_chemistry | reactant_in_excess | F2 | 0 | 25 | 100 | 95 | 100 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | - | - |
| line_broadening | atmospheric_radiation | line_broadening | F1 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | d4:100 / d0:0 | d2:100 / d4:100 |
| log_wind_stability | boundary_layer | log_wind_profile | F1 | 15 | 100 | 95 | 70 | 0 | 100 | 48 | 100 | 70 | 40 | 5 | 0 | - | - |
| lognormal_mass | atmospheric_aerosols | size_distribution_moment | F2 | 0 | 15 | 100 | 65 | 0 | 85 | 0 | 100 | 5 | 0 | 0 | 0 | - | - |
| mie_geometric | atmospheric_radiation | size_vs_length_scale | F2 | 5 | 100 | 100 | 90 | 100 | 50 | 0 | 100 | 85 | 30 | 55 | 25 | - | - |
| mixed_layer_response | climate_dynamics | box_model | F1 | 40 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 85 | 55 | - | - |
| nonsteady_burden | atmospheric_chemistry | box_model | F1 | 15 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 60 | 40 | - | - |
| optically_thin_emission | atmospheric_radiation | radiative_linearisation | F2 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 90 | 95 | - | - |
| photostationary_peroxy | atmospheric_chemistry | photostationary_peroxy | F2 | 25 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 45 | 30 | 45 | 40 | - | - |
| planck_linearization | climate_dynamics | radiative_linearisation | F2 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 90 | 100 | 90 | 100 | - | - |
| plume_ground_reflection | air_quality | dispersion | F2 | 0 | 0 | 100 | 70 | 0 | 0 | 0 | 100 | 20 | 0 | 0 | 0 | d2:0 / d4:0 | - |
| plume_mixing_lid | air_quality | dispersion | F2 | 10 | - | - | 10 | - | - | - | - | 25 | 45 | 5 | 5 | - | - |
| ppm_conversion_pressure | air_quality | density_with_height | F2 | 35 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 85 | 70 | 90 | 95 | - | - |
| pseudo_first_order | atmospheric_chemistry | reactant_in_excess | F2 | 0 | 10 | 100 | 25 | 100 | 100 | 48 | 100 | 0 | 5 | 0 | 0 | - | - |
| racemization_reversibility | environmental_chemistry | box_model | F1 | 0 | 65 | 100 | 80 | 95 | 100 | 2 | 100 | 100 | 100 | 35 | 0 | - | - |
| radar_beam_height | observation_and_modeling | spherical_geometry | F2 | 0 | 30 | 100 | 85 | 0 | 0 | 0 | 100 | 85 | 0 | 5 | 0 | - | - |
| raindrop_drag_regime | cloud_physics | size_vs_length_scale | F2 | 0 | 5 | 100 | 10 | 0 | 35 | 0 | 100 | 0 | 5 | 0 | 0 | - | - |
| rossby_deformation | atmospheric_dynamics | rotating_balance | F1 | 0 | 0 | 90 | 0 | 52 | 65 | 0 | 100 | 0 | 0 | 0 | 0 | - | d4:100 / d2:0 |
| saturated_lapse | atmospheric_thermodynamics | saturated_lapse | F1 | 0 | 0 | 65 | 5 | 8 | 0 | 0 | 100 | 15 | 0 | 0 | 0 | - | - |
| sea_ice_insulation | oceanography | flux_through_growing_layer | F1 | 10 | 70 | 100 | 100 | 98 | 100 | 80 | 100 | 90 | 0 | 5 | 10 | - | - |
| semivolatile_partitioning_temperature | atmospheric_aerosols | radiative_linearisation | F2 | 15 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 25 | 15 | 25 | 25 | - | - |
| shallow_water_wave_speed | oceanography | surface_gravity_wave | F1 | 0 | 65 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 100 | 15 | 30 | - | - |
| snowpack_cold_content | hydrology | latent_heating | F2 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 100 | 85 | 85 | 0 | 0 | d4:100 / d2:100 | d2:100 / d2:100 |
| solid_angle | observation_and_modeling | spherical_geometry | F2 | 0 | 65 | 100 | 100 | 98 | 100 | 100 | 100 | 95 | 100 | 0 | 0 | - | - |
| sonic_humidity_flux | observation_and_modeling | sonic_humidity_flux | F2 | 0 | 15 | 0 | 70 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| spherical_slant_path | atmospheric_radiation | spherical_geometry | F2 | 0 | 10 | 100 | 25 | 0 | 0 | 0 | 100 | 100 | 0 | 0 | 0 | d1:0 / d0:0 | - |
| sun_photometer_airmass | observation_and_modeling | spherical_geometry | F2 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 100 | 85 | 100 | 0 | 0 | d1:100 / d4:100 | d2:0 / d4:100 |
| surface_multiple_reflection | atmospheric_radiation | surface_multiple_reflection | F1 | 0 | 10 | 100 | 95 | 0 | 0 | 0 | 100 | 90 | 0 | 0 | 0 | d1:100 / d0:0 | - |
| taylor_dispersion | boundary_layer | dispersion | F2 | 45 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 0 | 60 | 80 | - | - |
| unconfined_dupuit | hydrology | flux_through_growing_layer | F1 | 0 | 5 | 100 | 50 | 0 | 0 | 0 | 100 | 0 | 100 | 0 | 0 | d1:100 / d4:100 | - |
| weak_acid_dissociation | atmospheric_chemistry | aqueous_equilibria | F1 | 10 | 95 | 100 | 100 | 100 | 100 | 98 | 100 | 85 | 100 | 30 | 20 | - | - |
| weir_approach_velocity | hydrology | weir_approach_velocity | F1 | 5 | 95 | 70 | 100 | 100 | 100 | 25 | 100 | 5 | 0 | 0 | 0 | - | - |
