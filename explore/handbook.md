# Reference notes: standard relations and where they stop holding

Each note gives a standard relation, the conditions under which it holds, and the relation or correction to use outside them.

## Dynamics

### Geostrophic wind V_g = (1/(ρf)) ∂p/∂n = (g/f) ∂Z/∂n
Valid when: the flow is straight or only weakly curved, Rossby number V/(fR) ≪ 1 (R = radius of curvature of the isobars or height contours).
Outside that: gradient-wind balance V²/R + fV = fV_g. Cyclonic flow: V = −fR/2 + √(f²R²/4 + fRV_g) (subgeostrophic); anticyclonic flow: V = fR/2 − √(f²R²/4 − fRV_g) (supergeostrophic) (Holton & Hakim §3.2).

### Frictionless horizontal force balance (geostrophic wind, no drag)
Valid when: the frictional deceleration is small compared with the Coriolis term, k ≪ f for a linear drag −kV.
Outside that: with linear drag the steady three-force balance gives V = V_g f/√(f² + k²), turned toward low pressure by tan α = k/f; the instantaneous rate of change of wind speed is dV/dt = (pressure-gradient force component along V) − kV (Holton & Hakim §5.3; Wallace & Hobbs §7.2).

### Motion under a constant pressure-gradient force with rotation neglected, v = a t
Valid when: the elapsed time is short compared with 1/f (f t ≪ 1), so the path stays nearly straight.
Outside that: on an f-plane the flow performs an inertial oscillation; from rest |V| = (2a/f)|sin(ft/2)| about the geostrophic speed a/f. A free object on a frictionless rotating plane moves on an inertial circle of radius V/f with period 2π/f. After a sudden change of forcing or of a linear friction r, the wind oscillates about the new steady state (computed with the new parameters): initial amplitude = vector difference between the old and new steady winds, decaying as e^(−rt) (Holton & Hakim §3.2; Vallis §2.9).

### Internal gravity-wave frequency ω = N k/m (hydrostatic, non-rotating)
Valid when: horizontal wavelength ≫ vertical wavelength (k ≪ m) and ω ≫ f.
Outside that: full Boussinesq relation ω² = (N²k² + f²m²)/(k² + m²). Non-hydrostatic, non-rotating: ω = N k/√(k² + m²), vertical group velocity magnitude N k m/(k² + m²)^(3/2). Long waves feel rotation: ω² ≈ f² + N²k²/m², and the period cannot exceed 2π/f (Holton & Hakim §7.4; Fritts & Alexander 2003).

### Non-divergent Rossby-wave phase speed c − ū = −β/k²
Valid when: the wavelength is short compared with the Rossby radius of deformation, k²L_d² ≫ 1, with L_d = √(gH)/f₀ for a shallow layer of depth H (NH/f₀ for a stratified layer).
Outside that: with a free surface, c − ū = −β/(k² + l² + L_d⁻²) (Vallis §6.4; Holton & Hakim §7.7).

### Kinematic vertical velocity with constant density, w(z) = −(∇·V_h) z
Valid when: the layer is thin compared with the density scale height H = R_d T/g (z ≪ H).
Outside that: use mass continuity ∂(ρw)/∂z = −ρ ∇·V_h; for uniform convergence C in an isothermal atmosphere w = C H (e^(z/H) − 1). Equivalently in pressure coordinates ω(p) = ω(p₀) − ∫(∇·V)dp, w ≈ −ω/(ρg) (Holton & Hakim §3.5).

### Relative vorticity of a solid-body vortex on a tangent plane, ζ = 2V/R
Valid when: the vortex radius R is small compared with the Earth's radius a.
Outside that: for solid-body rotation about a point on the sphere, with great-circle radius R and tangential speed V, ζ = 2V/(a tan(R/a)) (negative for anticyclonic rotation in the Northern Hemisphere) (rigid rotation at angular speed ω about a surface point: V = ωa sin(R/a), ζ = 2ω cos(R/a); Holton & Hakim §4.2).

### Vorticity amplification by stretching applied to relative vorticity, dζ/dt = −ζ ∇·V
Valid when: |ζ| ≫ |f|.
Outside that: stretching acts on the absolute vorticity: d(ζ + f)/dt = −(ζ + f) ∇·V, so with a constant e-folding time it is (ζ + f), not ζ, that grows exponentially (Holton & Hakim §4.4).

### Ekman-layer depth and spin-down time with molecular viscosity ν
Valid when: the flow is laminar.
Outside that: in turbulent flow replace ν by the eddy viscosity K: depth δ = √(2K/f), spin-down time τ = H √(2/(K f)) for fluid depth H, lateral diffusion time over a distance L ≈ L²/K; in a rotating tank f = 2Ω (Holton & Hakim §5.4–5.5).

### Height change of an object moving along a constant-pressure surface, dz/dt = V·∇z
Valid when: the pressure field is steady (∂z/∂t = 0 at a fixed point).
Outside that: dz/dt = ∂z/∂t + V·∇z (material derivative of the isobaric height); per unit horizontal distance divide by the ground speed; for a geostrophic wind ∇z = (f/g)(v_g, −u_g) (Holton & Hakim §1.3, §3.2).

## Thermodynamics

### Hydrostatic pressure drop with constant density, Δp = ρ₀ g Δz
Valid when: Δz ≪ the scale height H = R_d T/g (about 8 km).
Outside that: combine the hydrostatic equation with the ideal-gas law: isothermal p(z) = p₀ exp(−gz/(R_d T)); constant lapse rate Γ: p = p₀(1 − Γz/T₀)^(g/(R_d Γ)) (Wallace & Hobbs §3.2).

### Constant-lapse-rate (tropospheric) barometric formula
Valid when: below the tropopause (11 km in the International Standard Atmosphere).
Outside that: integrate layer by layer. ISA: 0–11 km lapse rate 6.5 K/km; 11–20 km isothermal at 216.65 K, p = p₁₁ exp[−g(z − 11 km)/(R_d T₁₁)]; 20–32 km temperature increases at 1.0 K/km; each layer starts from p and T at the top of the layer below (ICAO standard atmosphere; Wallace & Hobbs §3.2).

### Potential-temperature lapse rate dθ/dz ≈ dT/dz + Γ_d
Valid when: near the reference pressure p₀, where θ ≈ T.
Outside that: dθ/dz = (θ/T)(dT/dz + g/c_p), with θ/T = (p₀/p)^(R_d/c_p) (Wallace & Hobbs §3.3; Holton & Hakim §2.7).

### Dry adiabatic lapse rate Γ_d = g/c_p
Valid when: the parcel is unsaturated.
Outside that: a saturated parcel cools at Γ_s = g(1 + L_v r_s/(R_d T))/(c_p + L_v² r_s/(R_v T²)). Wherever a parcel lapse rate enters (stability, the adiabatic method for vertical velocity with the factor Γ − γ), use Γ_s for saturated air (Wallace & Hobbs §3.5; Holton & Hakim §2.9).

### Saturation vapour pressure over a plane liquid-water surface
Valid when: the condensed phase or surface is liquid water (including supercooled liquid).
Outside that: over ice use Clausius–Clapeyron with the latent heat of sublimation L_s ≈ 2.83 × 10⁶ J/kg: e_i = e_tp exp[(L_s/R_v)(1/T_tp − 1/T)] with e_tp = 611 Pa at T_tp = 273.16 K, or e_i = 6.112 exp[22.46 T/(272.62 + T)] hPa (T in °C). Use the frost point rather than the dew point, and ice density ≈ 917 kg/m³ (Rogers & Yau ch. 2; Wallace & Hobbs §3.7).

### Saturation vapour pressure (or saturation concentration) taken as constant or linear in temperature
Valid when: the temperature is within a few kelvin of the reference temperature.
Outside that: Clausius–Clapeyron: e_s(T) = e_s(T₀) exp[(L/R_v)(1/T₀ − 1/T)], L_v/R_v ≈ 5420 K. Likewise the effective saturation concentration of a semivolatile compound: C*(T) = C*(T₀)(T₀/T) exp[(ΔH_vap/R)(1/T₀ − 1/T)] (Wallace & Hobbs §3.7; Seinfeld & Pandis ch. 14; Donahue et al. 2006).

### Parcel buoyancy from the virtual-temperature excess, B = g(T_v − T_ve)/T_ve
Valid when: the parcel carries negligible condensate.
Outside that: condensate adds weight without buoyant volume: B = g[(T_v − T_ve)/T_ve − r_l], r_l the condensate mixing ratio in kg/kg (density temperature T_ρ ≈ T_v(1 − r_l)) (Emanuel 1994 §4.3; Markowski & Richardson §2.2).

### Evaporative cooling with all the available water evaporating, ΔT = L_v M/c_p
Valid when: the water supplied, M (kg per kg of air), is less than what the air can take up before it saturates.
Outside that: evaporation stops at saturation and the air cools only to its isobaric wet-bulb temperature T_w, c_p(T − T_w) = L_v[r_s(T_w, p) − r]; cooling = min(L_v M/c_p, T − T_w) (Wallace & Hobbs §3.5; Rogers & Yau ch. 2).

### Latent heating from condensation alone, ΔT = L_v m_v/c_p
Valid when: no liquid freezes and no vapour deposits on or sublimates from ice.
Outside that: freezing adds the latent heat of fusion: ΔT = (L_v m_v + L_f m_f)/c_p. When a cloud glaciates, the vapour also relaxes from water saturation toward ice saturation at the new temperature, adding L_s[w_initial − w_i(T_new)]/c_p; this term is negative (sublimation) if the warming raises w_i above the vapour present, so solve for T_new iteratively (Rogers & Yau ch. 2; Wallace & Hobbs §6.5).

## Cloud physics

### Equilibrium vapour pressure over a drop equal to the flat-surface e_s
Valid when: the Kelvin term 2σ/(ρ_w R_v T r) ≪ 1, i.e. radius larger than about 0.1 µm (excess below about 1 %).
Outside that: e_s(r) = e_s(∞) exp[2σ/(ρ_w R_v T r)], σ ≈ 0.075 N/m for water; for solution drops add the Raoult (solute) term (Köhler theory) (Rogers & Yau ch. 6).

### Stokes terminal velocity v = 2 ρ_w g r²/(9μ) (= k₁ r², k₁ ≈ 1.19 × 10⁶ cm⁻¹ s⁻¹)
Valid when: Reynolds number ≪ 1: water drops in air up to radius about 30–40 µm (and much larger than the mean free path).
Outside that: 40 µm ≲ r ≲ 0.6 mm: v = k₂ r, k₂ ≈ 8 × 10³ s⁻¹; 0.6 mm ≲ r ≲ 2 mm: v = k₃ √r, k₃ ≈ 2.01 × 10³ (ρ₀/ρ)^(1/2) cm^(1/2) s⁻¹, ρ₀ = 1.20 kg m⁻³ (about 9 m/s at r = 2 mm near sea level) (Rogers & Yau ch. 8).

### Diffusional growth or evaporation of a stationary drop, r dr/dt = (S − 1)/(F_k + F_d)
Valid when: the drop is small enough that its fall speed does not ventilate it (ventilation factor ≈ 1).
Outside that: multiply the rate by the ventilation factor f = 1 + 0.108 X² (X ≤ 1.4) or f = 0.78 + 0.308 X (X > 1.4), X = N_Sc^(1/3) N_Re^(1/2), N_Sc = ν/D_v, N_Re = 2 r v/ν with v the fall speed (Pruppacher & Klett §13.2).

## Radiation and climate

### Rayleigh scattering, cross section ∝ λ⁻⁴
Valid when: size parameter x = 2πr/λ ≪ 1.
Outside that: Mie theory; for x ≫ 1 the extinction efficiency approaches Q_ext ≈ 2 at every wavelength, so extinction becomes nearly wavelength-independent (why clouds are white) (Bohren & Huffman; Petty ch. 12).

### Pressure-broadened (Lorentz) half-width α_L = α_L0 (p/p₀)(T₀/T)^n
Valid when: α_L is much larger than the Doppler width — for thermal-infrared lines, pressures well above about 10 hPa.
Outside that: the Doppler half-width (HWHM) α_D = (ν₀/c)√(2 ln2 k_B T/m) does not depend on pressure; the combined Voigt half-width is α_V ≈ 0.5346 α_L + √(0.2166 α_L² + α_D²) (Liou §1.3; Olivero & Longbothum 1977).

### Absorption along a path evaluated at a single pressure (homogeneous path)
Valid when: the path is horizontal, or the layer is thin compared with the pressure scale height H.
Outside that: along a vertical path the pressure varies, dp/dz = −p/H; a layer of thickness Δz spans a pressure difference Δp = p̄ Δz/H exactly when isothermal, where p̄ = Δp/ln(p_bottom/p_top) is the logarithmic-mean pressure (≈ the mean pressure for a thin layer); equivalently Δp = p_bottom(1 − e^(−Δz/H)), its absorber column is q Δp/g, and pressure-dependent quantities such as line widths are weighted by pressure along the path (Curtis–Godson) (Liou ch. 4; Goody & Yung ch. 6).

### Optically thin layer: emissivity = absorptivity ≈ τ (brightness temperature ≈ τT)
Valid when: optical depth τ ≪ 1.
Outside that: an isothermal layer has ε = a = 1 − e^(−τ) and T_b = T(1 − e^(−τ)) (Rayleigh–Jeans, cold background); a small change of optical depth changes it by Δε = e^(−τ) Δτ (Petty §8.2; Jacob ch. 7).

### Skin temperature of a top layer, T = (OLR/(2σ))^(1/4)
Valid when: the top layer is optically thin in the infrared (absorptivity f ≪ 1).
Outside that: a gray layer of absorptivity f above an upwelling flux F_up satisfies f F_up = 2 f σT⁴ and OLR = (1 − f) F_up + f σT⁴, so T = [OLR/((2 − f)σ)]^(1/4); a black layer (f = 1) has σT⁴ = OLR (Jacob ch. 7; Hartmann ch. 3).

### Layer greenhouse models with an atmosphere transparent to sunlight
Valid when: absorption of solar radiation in the atmosphere is negligible.
Outside that: write the energy balance of the surface, of each layer and at the top with the absorbed solar flux in the layer that absorbs it. One layer of infrared absorptivity f absorbing a fraction α of the absorbed solar flux Q: σT_s⁴ = Q(1 − α/2)/(1 − f/2); N black layers with a fraction α of Q absorbed in the top layer: σT_s⁴ = Q[(N + 1) − Nα] (Jacob ch. 7; Hartmann ch. 3).

### Linearised black-body response ΔT = ΔF/(4σT³)
Valid when: ΔF ≪ σT⁴.
Outside that: solve σT_new⁴ = σT⁴ + ΔF exactly (Hartmann §10.3).

### Beam (Beer's-law) transmittance e^(−τ/μ) applied to a flux
Valid when: the radiation is a collimated beam at zenith-angle cosine μ.
Outside that: for radiation incident isotropically over the hemisphere, the flux transmittance is 2E₃(τ) ≈ exp(−1.66 τ) (diffusivity approximation) (Liou ch. 4; Petty ch. 10).

### Surface absorption under a reflecting layer counted on a single pass, T S (1 − α)
Valid when: the product of the surface albedo α and the layer's reflectance R for upwelling light is small.
Outside that: sum the multiple reflections between surface and layer: absorbed flux = T S (1 − α)/(1 − αR) (adding method; Liou ch. 6; Petty ch. 13).

### Plane-parallel slant path H sec θ (air mass m = 1/cos θ)
Valid when: θ ≲ 70–75° and the layer is thin compared with its radius of curvature.
Outside that: spherical shell from radius r to r + H: L = √((r + H)² − r² sin²θ) − r cos θ, where r is the radius of the layer's lower boundary (Earth's radius plus its altitude). A limb (tangent) path through a shell of thickness d at radius r has length 2√((r + d)² − r²) ≈ 2√(2rd). For the whole atmosphere use the Kasten–Young air mass m = 1/[cos θ + 0.50572 (96.07995 − θ)^(−1.6364)] (θ in degrees) (Kasten & Young 1989; Liou ch. 3).

### Solid angle of a sphere Ω = πR²/d²
Valid when: R ≪ d (d = distance to the sphere's centre).
Outside that: the exact cone solid angle Ω = 2π(1 − cos α), sin α = R/d (Petty ch. 2).

### Transmissivity, reflectivity and absorptivity from measured radiation, a = 1 − r − t
Valid when: the body's own thermal emission in the measured band is negligible.
Outside that: the radiation measured leaving each face includes the body's emission; subtract it first, t = (transmitted − emitted)/incident and r = (reflected − emitted)/incident; then a = 1 − r − t and, by Kirchhoff's law, ε = a (Petty ch. 6; Wallace & Hobbs §4.3).

### Radiative forcing linear in concentration, ΔF = (dF/dC)₀ ΔC
Valid when: the concentration change is small compared with the reference concentration.
Outside that: CO₂ ΔF = K ln(C/C₀), K ≈ 5.35 W m⁻² (equivalently K = C₀ (dF/dC)₀); CH₄ ΔF = 0.036(√M − √M₀) − [f(M, N₀) − f(M₀, N₀)] W m⁻², f(M, N) = 0.47 ln[1 + 2.01 × 10⁻⁵ (MN)^0.75 + 5.31 × 10⁻¹⁵ M (MN)^1.52], M and N (N₂O) in ppb (Myhre et al. 1998; IPCC TAR Table 6.2).

### Equilibrium or steady-state value: ΔT = F/λ, C = S τ
Valid when: the elapsed time is long compared with the adjustment time τ (t ≳ 4τ).
Outside that: first-order relaxation, from zero at t = 0: X(t) = X_eq(1 − e^(−t/τ)); slab ocean τ = ρ c h/λ; chemical box τ = 1/Σk; for chained species integrate dC/dt = P − kC for each. In a budget, total source = loss + rate of increase of the burden (Hartmann §10.6; Jacob ch. 3).

## Chemistry

### Pseudo-first-order kinetics with the excess reactant held constant
Valid when: [B]₀ ≫ [A]₀.
Outside that: second-order integrated law ln([B][A]₀/([A][B]₀)) = ([B]₀ − [A]₀) k t; half-life of A t½ = ln(2 − [A]₀/[B]₀)/(k([B]₀ − [A]₀)) (Atkins, Physical Chemistry, integrated rate laws; Seinfeld & Pandis ch. 3).

### Lifetime against a single sink, τ = 1/(k[X])
Valid when: other loss processes are negligible.
Outside that: parallel first-order losses add: 1/τ = Σ k_i[X_i] + J + k_dep + λ (reaction, photolysis, deposition, radioactive decay). In a ventilated box at steady state: inflow + emission = (outflow + V Σ k) C (Seinfeld & Pandis ch. 2; Jacob ch. 3).

### Mixing ratio ↔ concentration at standard conditions (24.45 L/mol; 2.46 × 10¹⁹ molecules cm⁻³)
Valid when: p ≈ 1 atm and T ≈ 25 °C.
Outside that: number density of air n = p/(k_B T), molar concentration p/(RT); mass concentration = mixing ratio × M × p/(RT) at the actual p and T (Seinfeld & Pandis ch. 1).

### Henry's law [X(aq)] = H p with the gas-phase partial pressure unchanged
Valid when: the gas is replenished (open system), or the dissolved fraction is small, H R T L ≪ 1 (L = liquid-water volume per volume of air).
Outside that: closed parcel: [X(aq)] = H p₀/(1 + H R T L), p₀ the partial pressure before dissolution; dissolved fraction H R T L/(1 + H R T L), R = 0.08206 L atm mol⁻¹ K⁻¹ (Seinfeld & Pandis ch. 7).

### Physical Henry's law for a gas that dissociates in water
Valid when: pH ≪ pK_a (dissociation negligible).
Outside that: effective constant H* = H(1 + K_a1/[H⁺] + K_a1 K_a2/[H⁺]²); total dissolved = H* p (Seinfeld & Pandis ch. 7).

### Weak-acid approximation [H⁺] = √(K_a C); strong acids counted as fully dissociated
Valid when: C ≫ K_a and K_a C ≫ K_w; for the second proton of H₂SO₄, [H⁺] ≪ K_a2 ≈ 1.2 × 10⁻² M.
Outside that: solve K_a = x²/(C − x) as a quadratic. At very low acidity include water: [H⁺]² ≈ K_w + Σ K_a,i [HA_i] (charge balance, [HA_i] the undissociated concentrations). At high [H⁺] solve the HSO₄⁻ ⇌ H⁺ + SO₄²⁻ equilibrium instead of counting two protons per H₂SO₄ (Seinfeld & Pandis ch. 7; Jacob ch. 13).

### Equilibrium constants applied to concentrations
Valid when: dilute solutions, ionic strength I ≲ 10⁻³–10⁻² M.
Outside that: equilibrium constants hold for activities a_i = γ_i [i]; I = ½ Σ c_i z_i²; Davies equation log₁₀ γ = −0.51 z² (√I/(1 + √I) − 0.3 I) at 25 °C (up to I ≈ 0.5 M); solve self-consistently (Stumm & Morgan ch. 3; Seinfeld & Pandis ch. 7).

### Standard reaction Gibbs energy ΔG° as the spontaneity criterion
Valid when: all species are at their standard states (reaction quotient Q ≈ 1).
Outside that: ΔG = ΔG° + RT ln Q, Q = Π a_i^ν_i, with a = p/p° for ideal gases (Atkins, Physical Chemistry; Seinfeld & Pandis ch. 7).

### Product formed set by one reactant or by rate × time ("the other reactant is in excess")
Valid when: the co-reactant, or the reservoir feeding the process, is larger than what is consumed.
Outside that: the product is capped by the limiting reactant (amount_i/stoichiometric coefficient_i, smallest wins), and a constant rate × time cannot exceed the finite reservoir it draws on (stoichiometry; Seinfeld & Pandis ch. 7, oxidant-limited sulfate).

### Leighton photostationary state [O₃] = j[NO₂]/(k₁[NO])
Valid when: peroxy radicals (HO₂, RO₂) are negligible (clean or night-like air).
Outside that: j[NO₂] = (k₁[O₃] + k_RO₂[RO₂])[NO], so [O₃] = j[NO₂]/(k₁[NO]) − k_RO₂[RO₂]/k₁ (Seinfeld & Pandis ch. 6).

## Aerosols

### Stokes drag without slip correction
Valid when: particle radius ≫ mean free path of air λ (≈ 0.066 µm at sea level), Kn = λ/r ≪ 1.
Outside that: multiply the Stokes settling speed (and the mobility and diffusivity) by the Cunningham factor C = 1 + Kn[1.257 + 0.4 exp(−1.1/Kn)] (Seinfeld & Pandis ch. 9; Hinds ch. 3).

### Mass of a population taken as N particles of the median diameter
Valid when: the distribution is nearly monodisperse (geometric standard deviation σ_g ≲ 1.1).
Outside that: lognormal moments (Hatch–Choate): mean D³ = D_g³ exp(4.5 ln²σ_g), so mass = N ρ (π/6) D_g³ exp(4.5 ln²σ_g); mass median diameter = D_g exp(3 ln²σ_g) (Seinfeld & Pandis ch. 8; Hinds ch. 4).

### Deposition velocity equal to the gravitational settling speed
Valid when: settling dominates Brownian diffusion across the near-surface layer (Péclet number v_s δ/D ≫ 1, typically diameters above a few µm).
Outside that: settling plus diffusion across a stagnant layer of thickness δ: v_d = v_s/(1 − exp(−v_s δ/D)), D = k_B T C/(3π μ d) (slip-corrected Stokes–Einstein) (Lai & Nazaroff 2000; Seinfeld & Pandis ch. 9).

## Boundary layer

### Neutral logarithmic wind profile u = (u*/κ) ln(z/z₀)
Valid when: |z/L| ≪ 1 (L = Obukhov length).
Outside that: Monin–Obukhov similarity. Stable: u = (u*/κ)[ln(z/z₀) + 5z/L] (Businger–Dyer, z/L ≲ 1); unstable: u = (u*/κ)[ln(z/z₀) − ψ_m(z/L)] with φ_m = (1 − 16 z/L)^(−1/4) (Stull §9.7; Garratt ch. 3).

### Logarithmic profile referenced to the ground over a tall canopy
Valid when: the height z is many canopy heights h above the ground.
Outside that: u = (u*/κ) ln((z − d)/z₀), zero-plane displacement d ≈ (2/3) h, roughness length z₀ ≈ 0.1 h (Oke ch. 2; Stull §9.7).

### Surface energy balance R_n = H + LE
Valid when: the ground (storage) heat flux is negligible, e.g. daily means over dense vegetation.
Outside that: R_n = H + LE + G; at midday over bare or dry soil G is commonly tens of W m⁻² or more (Oke ch. 1; Arya ch. 4).

### Near-field lateral plume spread σ_y = σ_v t
Valid when: travel time t ≪ the Lagrangian integral time scale T_L.
Outside that: Taylor (1921) with an exponential autocorrelation: σ_y² = 2σ_v² T_L² (t/T_L − 1 + e^(−t/T_L)); far field σ_y ≈ √(2 σ_v² T_L t) (Seinfeld & Pandis ch. 18).

## Air quality

### Gaussian plume without boundaries
Valid when: σ_z is small compared with the source height H and with the mixing height h.
Outside that: add image sources. Ground reflection: C = Q/(2π u σ_y σ_z) e^(−y²/2σ_y²) [e^(−(z−H)²/2σ_z²) + e^(−(z+H)²/2σ_z²)] (at ground level Q/(π u σ_y σ_z) e^(−H²/2σ_z²)). A capping inversion at h adds images at ±H + 2nh; once σ_z ≳ 1.6 h the plume is uniformly mixed: C = Q/(√(2π) u σ_y h) (Turner 1970; Seinfeld & Pandis ch. 18).

### Infinite line source C = √(2/π) q/(u σ_z)
Valid when: the source length is much greater than σ_y.
Outside that: finite line of length L, receptor opposite its midpoint: multiply by erf(L/(2√2 σ_y)) (Turner 1970; Seinfeld & Pandis ch. 18).

### Exponential removal laws: plug-flow reactor C/C₀ = e^(−kτ); Deutsch–Anderson collector η = 1 − exp(−wA/Q)
Valid when: the reactor has no back-mixing (plug flow); the collector has turbulent mixing that keeps the concentration uniform across each gas passage.
Outside that: a well-mixed (continuously stirred) reactor at steady state: C/C₀ = 1/(1 + kτ), fraction removed kτ/(1 + kτ); a laminar, unmixed collector: η = wA/Q (at most 1) (Levenspiel ch. 5; Cooper & Alley ch. 7).

### Haldane relation with oxyhaemoglobin taken as the total haemoglobin
Valid when: COHb is a small fraction of the binding sites (low CO exposure).
Outside that: conserve binding sites: [COHb]/[O₂Hb] = M p_CO/p_O₂ with [O₂Hb] + [COHb] = total capacity, so the COHb fraction = MR/(1 + MR), R = p_CO/p_O₂ (Douglas, Haldane & Haldane 1912; Coburn, Forster & Kane 1965 for uptake kinetics; de Nevers, Air Pollution Control Engineering).

## Observation

### Flat-Earth radar beam height h = r sin θ
Valid when: slant range r is short (tens of km or less).
Outside that: with standard refraction (effective Earth radius kR, k = 4/3): h = √(r² + (kR)² + 2 r kR sin θ) − kR (Doviak & Zrnić §2.2; Rinehart).

### Doppler radar radial velocity equal to the true radial velocity
Valid when: |v| < the Nyquist velocity v_N = λ·PRF/4.
Outside that: the velocity is aliased: folded into [−v_N, v_N) by adding or subtracting multiples of 2v_N (Doviak & Zrnić §3.5).

## Hydrology

### Cooper–Jacob straight-line drawdown s = (Q/4πT)(−0.5772 − ln u), u = r² S/(4 T t)
Valid when: u ≲ 0.01–0.05 (late time or close to the well).
Outside that: full Theis solution s = (Q/4πT) W(u), W(u) = E₁(u) = −0.5772 − ln u + u − u²/(2·2!) + u³/(3·3!) − … (Theis 1935; Kruseman & de Ridder §3.2).
