# Adjudication sheet

For each row pick ONE of the two candidate types (A or B). The type is the kind of approximation in the standard method that breaks down.

## templates

| # | template | category | cluster | standard -> replacement | parent's method | where it fails |
|---|---|---|---|---|---|---|
| 2 | `anelastic_continuity` | atmospheric_dynamics | density_with_height | Kinematic vertical velocity with constant density -> with density decreasing with height (added 2026-09-26). | with uniform horizontal convergence C (divergence -C) and w = 0 at the ground, the incompressible continuity equation dw/dz = C gives w = C z. | mass continuity is d(rho w)/dz = rho C. When the layer is not thin compared with the density scale height H = R_d T/g, rho falls with height and w grows faster than linearly: in an isothermal atmosphere rho w = C rho0 H (1 - e^{-z/H}), so w = C H (e^{z/H} - 1). |

- A: linearisation / local-slope extrapolation
- B: reference state / datum / boundary condition

| # | template | category | cluster | standard -> replacement | parent's method | where it fails |
|---|---|---|---|---|---|---|
| 11 | `cooper_jacob_early_time` | hydrology | cooper_jacob_early_time | Cooper-Jacob straight-line approximation -> full Theis well function (extended, hydrology, 2026-10-01). | drawdown at distance r from a well pumping a confined aquifer at rate Q since time t is the Theis solution s = (Q/4 pi T) W(u), u = r^2 S/(4 T t), T transmissivity, S storativity, and W(u) = E1(u) = -Ei(-u) the well function (the exponential integral). W(u) has no elementary closed form, so the method every introductory groundwater course teaches for hand calculation is Cooper & Jacob's (1946) straight-line approximation: for small u, W(u) ~ -gamma - ln(u) (gamma the Euler-Mascheroni constant, 0.5772...), dropping every term of W(u)'s convergent series but the leading log. This is THE standard field method (plot drawdown against log t, read T and S off the slope and intercept) and is routinely applied without rechecking u first. | the log truncation is only the small-u limit of W(u)'s series; published validity criteria (Kruseman & de Ridder, *Analysis and Evaluation of Pumping Test Data*, 1994, sec. 3.2.3) call for u <~ 0.01-0.05 for the approximation to be trustworthy. At larger u -- early in a pumping test, or far from the well -- the log form is not a small correction away from W(u), it is a different number. |

- A: dimensionless-number regime crossing
- B: linearisation / local-slope extrapolation

| # | template | category | cluster | standard -> replacement | parent's method | where it fails |
|---|---|---|---|---|---|---|
| 26 | `fresnel_reflectance_grazing` | atmospheric_radiation | fresnel_reflectance_grazing | Normal-incidence reflectance -> angle-dependent Fresnel reflectance of a calm sea (extended, radiation, 2026-10-05). | the reflectance of a water surface from its refractive index, R0 = ((n - 1)/(n + 1))^2 (about 2 % for sea water), so the fraction of direct sunlight entering the sea is 1 - R0. It is the one reflectance formula everybody remembers and the textbook value quoted for "water reflects ~2 %". | R0 is the normal-incidence limit. For unpolarised light the Fresnel reflectance R = (1/2)[sin^2(i - t)/sin^2(i + t) + tan^2(i - t)/tan^2(i + t)], sin i = n sin t, stays near R0 out to an incidence angle of ~40 deg and then rises steeply toward 1 at grazing incidence: with the sun 5-14 deg above the horizon a calm sea reflects 24-59 % of the direct beam. |

- A: linearisation / local-slope extrapolation
- B: geometric idealisation (plane-parallel, flat, infinite)

| # | template | category | cluster | standard -> replacement | parent's method | where it fails |
|---|---|---|---|---|---|---|
| 32 | `hydrostatic_constant_density` | atmospheric_thermodynamics | density_with_height | Hydrostatic pressure drop with constant density -> isothermal exponential profile (2026-09-26). | over a shallow layer the density is taken as its surface value, so the pressure falls linearly with height: dp = rho_0 g z with rho_0 = p_0 / (R_d T). | over a depth comparable to the scale height H = R_d T / g (about 8 km) the density itself falls with the pressure; the hydrostatic equation with the ideal-gas law gives, for an isothermal layer, p(z) = p_0 exp(-z/H), so the drop is p_0 (1 - exp(-z/H)). |

- A: reference state / datum / boundary condition
- B: linearisation / local-slope extrapolation

| # | template | category | cluster | standard -> replacement | parent's method | where it fails |
|---|---|---|---|---|---|---|
| 37 | `ionic_strength_activity` | atmospheric_chemistry | aqueous_equilibria | Concentration-based dissociation constant -> activity-corrected equilibrium in a saline droplet (2026-10-01 extension). | for a weak acid HA (HA <-> H+ + A-) at total concentration C with dissociation constant K_a, solve the mass-action law treating concentrations as activities: K_a = x^2/(C - x), giving x = [H+] = (-K_a + sqrt(K_a^2 + 4 K_a C)) / 2 (the exact quadratic solution; "exact" here means only that the small-x approximation is not used -- see `weak_acid_dissociation` for that separate trap -- not that activities are accounted for). | K_a is a thermodynamic (activity-based) constant, K_a = {H+}{A-}/{HA}. In a droplet that also carries dissolved ions unrelated to this equilibrium (sea-salt or other aerosol background), the ionic strength is high enough that activity coefficients depart from 1, and treating concentrations as activities under- or over-states [H+]. The correct treatment solves K_a = gamma_pm^2 x^2 / (C - x) self-consistently with the ionic strength I = I_bg + x (I_bg the background, x the acid's own contribution; HA is neutral, so its activity coefficient is taken as 1, the standard approximation for a neutral solute at these ionic strengths). Nothing about the outside background ionic strength looks extreme -- sea-spray and marine cloud/fog water routinely carry ionic strengths from a few hundredths to several tenths of a molar, an entirely ordinary range for atmospheric liquid water, not a contrived edge case. |

- A: neglected process / missing term
- B: reference state / datum / boundary condition

| # | template | category | cluster | standard -> replacement | parent's method | where it fails |
|---|---|---|---|---|---|---|
| 42 | `line_broadening` | atmospheric_radiation | line_broadening | Pressure-scaled Lorentz half-width -> Voigt (Doppler-dominated) half-width aloft (spec: MECHANISMS.md). | the half-width of a pressure-broadened line scales in proportion to pressure, alpha_L = alpha_0 p / p_0. | at low pressure the Lorentz width falls below the Doppler width alpha_D = (nu_0/c) sqrt(2 ln2 R T / M), which does not depend on pressure; the line is then a Voigt profile whose half-width is set by Doppler broadening. For the 667 cm^-1 CO2 line the two widths cross near 8 hPa (about 33 km); at 0.1-0.4 hPa the pressure-scaled width is 19-85 times too small. |

- A: dimensionless-number regime crossing
- B: neglected process / missing term

| # | template | category | cluster | standard -> replacement | parent's method | where it fails |
|---|---|---|---|---|---|---|
| 70 | `unconfined_dupuit` | hydrology | flux_through_growing_layer | Thiem drawdown with a fixed transmissivity -> Dupuit-Thiem for a water-table aquifer (extended, hydrology, 2026-10-05). | steady drawdown at distance r from a well pumped at rate Q, with a radius of influence R, is Thiem's s = Q ln(R/r) / (2 pi T). For a water-table (unconfined) aquifer the transmissivity is taken as T = K b, with b the saturated thickness before pumping -- the routine hand calculation, and the form every pumping-test worksheet uses. | in a water-table aquifer the saturated thickness is not fixed; it is the head h itself, so the flow cross-section shrinks as the water table is drawn down. Darcy's law through a cylinder of height h, Q = 2 pi r K h dh/dr, integrates to the Dupuit-Thiem equation h_0^2 - h^2 = Q ln(R/r) / (pi K), s = h_0 - h. Writing x = s_Thiem / h_0, the true drawdown is h_0 (1 - sqrt(1 - 2x)), larger than s_Thiem by about x/2 for small x and by 20-50 % once the drawdown is a third or more of the saturated thickness. |

- A: fixed reservoir / steady state / reactant in excess
- B: linearisation / local-slope extrapolation

## traps

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 24 | `trap_6.6_gen` | atmospheric_chemistry | activity vs concentration in a concentrated solution |

- A: reference state / datum / boundary condition
- B: neglected process / missing term

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 55 | `trap_air_405_gen` | atmospheric_thermodynamics | hydrostatic integral with a linear temperature profile instead of isothermal |

- A: linearisation / local-slope extrapolation
- B: reference state / datum / boundary condition

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 59 | `trap_ca_5.1_gen` | air_quality | modified Deutsch-Anderson exponent for high-resistivity dust |

- A: neglected process / missing term
- B: dimensionless-number regime crossing

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 63 | `trap_dn_13.4_gen` | air_quality | humid-air composition in combustion stoichiometry |

- A: reference state / datum / boundary condition
- B: neglected process / missing term

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 64 | `trap_dn_15.18_gen` | atmospheric_chemistry | compressibility factor in the real-gas number density |

- A: reference state / datum / boundary condition
- B: neglected process / missing term

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 68 | `trap_fp_2.4_gen` | atmospheric_chemistry | compressibility factor in the real-gas number density |

- A: neglected process / missing term
- B: reference state / datum / boundary condition

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 71 | `trap_holton_38_gen` | atmospheric_dynamics | local height tendency added to the advective altitude change |

- A: fixed reservoir / steady state / reactant in excess
- B: neglected process / missing term

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 86 | `trap_p_70_gen` | atmospheric_chemistry | compressibility factor in the real-gas number density |

- A: neglected process / missing term
- B: reference state / datum / boundary condition

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 103 | `trap_ry_14.2_p11` | cloud_physics | ice-supersaturated parcel driven past ice saturation: deposition reverses to sublimation |

- A: fixed reservoir / steady state / reactant in excess
- B: reference state / datum / boundary condition

| # | trap id | category | mechanism (or first sentence of the problem) |
|---|---|---|---|
| 92 | `trap_snp_4_gen` | atmospheric_chemistry | compressibility factor in the real-gas number density |

- A: reference state / datum / boundary condition
- B: neglected process / missing term

