# Regime-template expansion: design spec (2026-09-26)

The five pilot templates (`PILOT` in `__init__.py`) showed that a template trap is good only if
(a) the parent's method is what a solver reaches for by default, and (b) the text gives everything
needed for the replacement without naming it. This file lists the 26 mechanisms added on
2026-09-26 and the rules every template follows. Condition (a) can only be checked empirically, on a
calibration model kept out of the main analysis: the original 30 were screened on deepseek-v4-flash
(`results/prereg/`, sealed), later batches on deepseek and Qwen3.6-27B (`results/e0/`). There are 72
templates in 32 clusters (`coverage_clusters.py`); the counts below are those of the 2026-09-26 batch.

## Rules (each is checked or argued in the module)

1. **One trigger.** `trigger` returns one key; inside and outside differ in that given only, and the
   `given` text is the outside text plus one inserted sentence (`check_templates.check_texts`).
2. **Gaps.** Shortcut within 2 % of the key inside, at least 15 % away outside (`base.pair`). Choose
   ranges so that `pair` rarely rejects a draw; do not rely on rejection to carve the regime.
3. **Fair from the givens.** Every quantity the correct answer needs is stated in *all three* arms
   (constants included). The inside arm carries the same givens even where they do not matter.
4. **Do not name the replacement** in inside/outside. Keep only qualifiers that move the answer by
   more than the 5 % tolerance (the pilot trim rule): every extra qualifier is a hint.
5. **Conventions.** Any published alternative a competent solver could use for the correct physics
   must land within 5 % of the key across the whole outside grid; `check()` measures the spread.
6. **Gradable answer.** Not an absolute temperature near 290 K (the grader converts °C/K, so ±14 K
   would pass); keyed value comfortably non-zero; unit in `check_templates.CONVERTED` where possible.
7. **Rounded givens.** Sampled values carry 2–3 significant figures (`rnd`) so text = solver input.
8. **`check()`** in every new module: an independent recomputation of `correct` (different algebra
   or numerics), the inside/outside gap bounds over a grid covering the sampling ranges, the
   convention spread, and a literature anchor value where one exists. Returns a one-line summary.
9. **Docstring** in the pilot format: parent's method, where it fails, fairness from the text,
   inside justification, conventions, reference, and any related mechanism.

10. **A trap, not just a fair problem** (2026-10-01, from the extended-set reviews). The outside text must
    carry only ordinary data whose implication the solver has to recognise unprompted (a radius, a humidity, a
    time span, an albedo). If the omitted process is *named with its parameter* ("entrains at 0.5 per km",
    "a surface resistance r_c = 184 s/m", "hygroscopicity κ = 0.49, a measure of water uptake"), nobody drops it:
    the problem is fair but tests nothing — reject. The reverse failure is worse: moving a number the key needs
    into the `given` arm only makes the outside arm unfair (rule 3). A mechanism that cannot satisfy both is not a
    template. Never define the asked quantity by the shortcut formula (the key would contradict the question).
11. **Independent review before screening.** Every new template is reviewed by a reviewer who did not write it:
    an independent recomputation of one outside key, the physics against the standard references, fairness
    (every number the key needs, and the spread of any recalled one), cues, plausibility of the outside value,
    and grading hazards — including a *second wrong method* that lands within 5 % of the key on much of the grid.
    Of the first 22 extended drafts, 7 were rejected and 10 fixed; two keys were physically wrong.

## The 26 mechanisms

Category in brackets. "Shortcut" = the parent's method; "correct" = the physics that holds on both
sides. Ranges are starting points; a module may move them and must say why in its docstring.

### Dynamics
| name | shortcut → correct | trigger: inside / outside |
|---|---|---|
| `gradient_wind_curvature` | geostrophic V_g → cyclonic gradient wind V = −fR/2 + √(f²R²/4 + fRV_g) | radius of curvature R: ≥ 6000 km (with weak gradients) / 100–300 km |
| `coriolis_neglect` | air from rest under a constant pressure-gradient force: v = a t → with rotation \|v\| = (2a/f)\|sin(ft/2)\| | elapsed time t: 5–15 min / ft ≈ 2–4.5 |
| `rossby_deformation` | non-divergent Rossby wave c = −β/k² → free-surface shallow water c = −β/(k² + L_d⁻²), L_d = √(gH)/f₀; ask westward speed (positive) | zonal wavelength: 1500–3000 km / 12 000–25 000 km |
| `anelastic_continuity` | kinematic w from constant divergence with constant density w = −D z → mass continuity in an isothermal atmosphere w = −D H(e^{z/H} − 1); ask w in cm/s (convergence, upward) | height z: 50–250 m / 3–8 km |
| `inertia_gravity_wave` | hydrostatic non-rotating gravity-wave frequency ω = N k/m → with rotation ω² = f² + N²k²/m²; ask the period (h) | horizontal wavelength: 10–50 km / 1000–3000 km |

### Radiation, cloud physics, observation
| name | shortcut → correct | trigger: inside / outside |
|---|---|---|
| `mie_geometric` | Rayleigh λ⁻⁴: extinction ratio 450 nm / 700 nm = (700/450)⁴ → full Mie ratio (→ ≈ 1 for large drops); non-absorbing sphere, m = 1.33 | radius: ≤ 10 nm / 5–20 μm |
| `line_broadening` | pressure-broadened Lorentz HWHM α_L = α₀ p/p₀ → Voigt HWHM (Olivero–Longbothum), Doppler-dominated aloft; CO₂ 667 cm⁻¹, T fixed within a problem | pressure: 400–1000 hPa / 0.1–0.4 hPa (so a Doppler-only answer is within 5 %) |
| `optically_thin_emission` | microwave brightness temperature of an isothermal layer against cold space T_b = τT → T(1 − e^{−τ}) | optical depth τ: 0.005–0.04 / 0.4–2 |
| `kelvin_curvature` | equilibrium vapour pressure over a pure-water drop = flat-surface e_s → e_s exp(2σ/(ρ_w R_v T r)); e_s formula given | radius: 0.5–5 μm / 3–6 nm |
| `solid_angle` | Earth's solid angle from a spacecraft πR²/d² → 2π(1 − √(1 − R²/d²)) | altitude: ~1.5 × 10⁶ km (L1) or lunar distance / 300–2000 km |
| `radar_beam_height` | flat-Earth beam height r sin θ → 4/3-effective-radius formula √(r² + (kR)² + 2rkR sin θ) − kR | slant range: 3–15 km / 120–250 km |
| `doppler_aliasing` | displayed radial velocity = true → folded into [−v_N, v_N), v_N = λ·PRF/4; away = positive | true speed: ≤ 0.9 v_N / 1.2–1.8 v_N (so the folded value is not near 0) |

### Chemistry
| name | shortcut → correct | trigger: inside / outside |
|---|---|---|
| `pseudo_first_order` | night-time NO + O₃ titration, NO half-life ln2/(k[O₃]₀) → mixed second-order ln(2 − A₀/B₀)/(k(B₀ − A₀)) | initial NO: 0.05–0.3 ppb / O₃/NO = 1.2–1.5 |
| `henry_depletion` | aqueous concentration H·p (gas reservoir unchanged) → closed parcel H p/(1 + H R T L), mixing ratio given *before* the cloud forms | Henry constant: 1–100 / 5 × 10⁴–10⁵ M atm⁻¹ |
| `effective_henry` | total dissolved formic acid H·p → H p(1 + K_a/[H⁺]); open system (gas partial pressure held) | pH: 1.0–2.0 / 3.8–4.6 |
| `weak_acid_dissociation` | [H⁺] = √(K_a C) → quadratic root; acetic acid | formal concentration: 0.02–0.1 M / 1–4 × 10⁻⁵ M |
| `arrhenius_temperature` | lifetime against OH with k(298 K) → with k(T) = k₂₉₈ exp[−(E/R)(1/T − 1/298)] | temperature: 297–299 K / 210–230 K |

### Boundary layer, air quality
| name | shortcut → correct | trigger: inside / outside |
|---|---|---|
| `log_wind_stability` | neutral log law → stable Monin–Obukhov, ψ = −5z/L (Businger–Dyer; Beljaars–Holtslag within 5 %) | Obukhov length: 2000–10 000 m / 20–40 m at z = 10 m |
| `taylor_dispersion` | near-field σ_y = σ_v t → Taylor (exponential autocorrelation); outside kept at t/T_L ≥ 12 so the far-field √(2σ_v²T_L t) is within 5 % whatever the autocorrelation shape | downwind distance: t/T_L ≤ 0.1 / 15–40 |
| `plume_mixing_lid` | Gaussian plume with ground reflection only → image sum with a mixing lid (well-mixed limit within 5 % outside) | downwind distance: σ_z ≤ 0.3h / σ_z ≥ 2h |
| `finite_line_source` | infinite line source √(2/π) q/(u σ_z) → × erf(L/(2√2 σ_y)), receptor opposite the midpoint | road length: 2–10 km / 40–120 m |
| `ppm_conversion_pressure` | ppb → μg m⁻³ with 24.45 L mol⁻¹ → ideal gas at the stated p (T = 25 °C in all arms) | pressure: 1000–1030 hPa / 500–800 hPa |

### Thermodynamics, climate
| name | shortcut → correct | trigger: inside / outside |
|---|---|---|
| `hydrostatic_constant_density` | Δp = ρ₀ g z → isothermal p₀(1 − e^{−gz/(R_d T)}) | height: 30–150 m / 3–6 km |
| `cc_scaling` | e_s increase factor 1 + L ΔT/(R_v T²) → exp[(L/R_v)(1/T − 1/(T+ΔT))] (constant L; Bolton ratio within 5 %) | warming ΔT: 0.3–1.5 K / 12–25 K |
| `planck_linearization` | ΔT = ΔF/(4σT³) → (T⁴ + ΔF/σ)^{1/4} − T | forcing: 1–5 / 150–400 W m⁻² |
| `mixed_layer_response` | equilibrium warming F/λ → slab ocean F/λ (1 − e^{−t/τ}), τ = ρ c h/λ | elapsed time: ≥ 4τ (60–150 yr) / 1–5 yr |

Dropped after design review: `kohler_solute` (the dilute and exact Raoult forms and NaCl
non-ideality differ by more than 5 % wherever the solute effect is large enough to trap),
`cyclostrophic_coriolis` (one trigger cannot move a vortex from dust-devil to cyclone scale without
an absurd pressure gradient on one side; the gradient-wind physics is already covered),
`raindrop_terminal_velocity` (published large-drop laws differ by 7–11 %), `polar_cap_area` and
`doppler_elevation` (no plausible lure: solvers use the exact formula by default).
Related pairs, to be clustered in any per-mechanism statistic: `hydrostatic_constant_density` /
`anelastic_continuity` (density decrease with height); `gradient_wind_curvature` /
`coriolis_neglect` (rotation vs curvature in horizontal balance); `optically_thin_emission` /
`spherical_slant_path` (both Beer-law path/opacity).

## Status after implementation (2026-09-26)

All 26 are implemented and pass `check_templates` (31 templates in total, 9300 graded rows). Final
ranges are in each module's docstring; the table records where they departed from the spec above
and the expected strength of the lure. The lure column is a **prior from reading the text, not a
measurement** — screening on a calibration model decides it.

| template | departure from spec (reason in the module) | expected lure |
|---|---|---|
| gradient_wind_curvature | outside R 60–150 km; latitude/gradient narrowed; text reworded (no "centre of low") | medium: stated radius is a cue |
| coriolis_neglect | inside 0.2–0.5 h (same time unit in both arms) | strong |
| rossby_deformation | outside λ ≤ 18 000 km (fits the latitude circle); "β-plane", "free surface" in all arms | medium–strong |
| anelastic_continuity | — | medium: isothermal T, R_d, g stated |
| inertia_gravity_wave | inside λ_x = 8–12 λ_z; N 0.015–0.020 s⁻¹; 30–50°N | weak (mirrors the failed `nonhydrostatic_wave`) |
| mie_geometric | outside r 25–60 μm (ripple below 20 μm breaks the ratio-1 answer); wavelengths sampled | strong |
| line_broadening | R and c not stated (universal constants; stating c would cue Doppler) | strong |
| optically_thin_emission | outside τ ≤ 1.5; RJ definition and "nothing else radiates" in all arms | medium |
| kelvin_curvature | outside r 4.0–6.5 nm | medium: σ stated |
| solid_angle | inside down to 30 000 km | weak |
| radar_beam_height | elevation 1.5–2.5°; inside 3–8 km, outside 150–250 km | medium: 4/3-radius sentence is a cue |
| doppler_aliasing | outside 1.25–1.75 v_N | strong |
| pseudo_first_order | — | medium |
| henry_depletion | answer in μmol/L (grader floor); "no gas enters or leaves" in all arms | strong |
| effective_henry | inside pH 1.5–2.0; "in all its forms" in all arms | medium |
| weak_acid_dissociation | — | medium–strong |
| arrhenius_temperature | — | weak (tests "forgot to adjust k", E/R stated) |
| log_wind_stability | outside L 30–40 m, z₀ 0.01–0.03 m (narrow window between gap and Högström spread) | medium |
| taylor_dispersion | outside t/T_L 15–30; autocorrelation shape not stated | medium |
| plume_mixing_lid | outside σ_z ≥ 1.5h | strong |
| finite_line_source | outside L = 1–2.5 σ_y (12–75 m); worded as a section of road | strong |
| ppm_conversion_pressure | — | weak (p and R stated) |
| hydrostatic_constant_density | — | strong |
| ~~cc_scaling~~ | **dropped 2026-09-26** after independent review (see below) | — |
| planck_linearization | outside ΔF ≥ 200 W m⁻² | strong |
| mixed_layer_response | — | weak–medium: "t years after" invites a transient |

Grading notes: temperature-change answers (K) grade correctly when given in °C (bare-number
fallback, checked 2026-09-26); mol/L, μmol/L and sr cannot be unit-converted by the grader, so those
templates skip the converted-unit check.

## Lead review (2026-09-26) and a prediction registered before screening

Checked by hand: one inside/outside/given triple of every new template read in full; the keyed
value of one outside item per template recomputed independently of the module (all 26 agree to the
printed digits); trap texts scanned for words naming the replacement (none do — the hits are an
instrument name "Doppler radar", the stated "Rayleigh–Jeans definition", "acid dissociation
constant", and the free surface / refraction / inversion givens that fairness requires in all arms).

**Two kinds of template.** The pilot's two failures (`vorticity_stretching`, `nonhydrostatic_wave`)
share a feature: the correct method is the *un-approximated form of the same relation*, which a
code-writing solver can write as cheaply as the approximation. The three good pilots need a
*physical ingredient absent from the parent relation* (slip, sphericity, latent heat). Sorting the
26 the same way:

- **A — a new ingredient must be recalled (12):** mie_geometric, line_broadening,
  kelvin_curvature, coriolis_neglect, doppler_aliasing, henry_depletion, effective_henry,
  log_wind_stability, plume_mixing_lid, finite_line_source, radar_beam_height, taylor_dispersion.
- **B — exact form of the same relation (10):** planck_linearization, hydrostatic_constant_density,
  weak_acid_dissociation, solid_angle, optically_thin_emission, inertia_gravity_wave,
  anelastic_continuity, mixed_layer_response, ppm_conversion_pressure, pseudo_first_order.
- **Borderline (3):** gradient_wind_curvature, rossby_deformation (a term of the full relation, but
  one many solvers never learned), arrhenius_temperature (the ingredient is stated as a given).

**Prediction:** screening passes a larger share of A than of B. If it holds, the regime deficit is
about *recalling a missing physical ingredient*, not about approximations in general — a sharper
claim than REPORT.md makes, and B then serves as a contrast set rather than being discarded. If A
and B pass alike, that distinction is not what drives it. Screening decisions must not be tuned to
make this come out.

**Known risks to measure in screening, not fix now:**
- (moot, template dropped) `cc_scaling` stated the local rate L/(R_v T²) in every arm. It is the textbook "7 % per K"
  framing and the text says "near a temperature T", but a solver who multiplies it by ΔT is
  arguably following the text. Read its failures before counting them.
- `henry_depletion`, `effective_henry` (μmol/L), `weak_acid_dissociation` (mol/L) and
  `solid_angle` (sr) use units the grader cannot convert: a right answer in another unit fails in
  every arm. Symmetric across arms, but noise — count unit-mismatch failures per template.
- Some trigger values read oddly in isolation (an "air mass at 298 K" inside `arrhenius`, isobars
  curved over 6000–9000 km inside `gradient_wind_curvature`); the magnitude of the trigger is the
  regime cue by design, so these are left as they are.

## Independent review (2026-09-26, four fresh reviewers, findings adjudicated by the lead)

Each template was re-read by a reviewer who had not written it, against seven questions: alternative
readings that move the answer > 5 %, physics over the corners of the sampled range (including other
approximations failing inside), the given sentence applied literally, fairness, cues, realism and
grading, and whether `check()` is independent of `correct()`. No key was wrong. Changes made:

- **Dropped `cc_scaling`** (25 new templates remain, 30 in total). Its key is a factor: inside it is
  1.06–1.10, so the 5 % tolerance is ±50–85 % of the *increase* — the control arm was graded far
  more leniently than the trap arm, which inflates any Trap Gap. Keying the percentage increase
  instead makes legitimate e_s formulas (compounding at fixed T) fail by ~8 %. With the stated-rate
  fairness concern on top, no version was worth keeping. General lesson, now rule 10 below.
- **Ambiguity removed:** `ppm_conversion_pressure` (monitoring stations report μg m⁻³ at reference
  conditions, i.e. exactly the shortcut — now "ambient air at the measurement site", outside p
  650–850 hPa); `doppler_aliasing` (operational radars dealias — now single PRF, estimate reported
  without further processing, motion along the beam, true speed ≤ 45 m/s); `weak_acid_dissociation`
  (CO₂-equilibrated water and activity readings — ranges moved so both stay < 5 %, at 4.8 % and 4.7 %);
  `rossby_deformation` (full shallow-water cubic reached 4.6 % — outside λ ≤ 15 000 km, now 3.3 %);
  `anelastic_continuity` (convergence "at every height" was physically inconsistent — now the layer
  up to the level considered).
- **Cues removed:** `henry_depletion` (R stated but used only by the replacement), `line_broadening`
  (molar mass used only by the Doppler width). Both constants now appear only in the given sentence.
- **Realism:** `log_wind_stability` u* 0.15–0.30 m/s (the old values implied −200 to −360 W m⁻²
  stable heat fluxes); `coriolis_neglect` 40–60°N, gradients ≤ 2 hPa/100 km (keys < 40 m/s);
  `arrhenius_temperature` pre-exponential factor ≤ 2.6 × 10⁻¹¹ (was above the gas-kinetic limit);
  `plume_mixing_lid` inside keys ≤ 2.3 mg m⁻³; `taylor_dispersion` inside t/T_L ≥ 0.05;
  `finite_line_source` unused, inconsistent receptor distance removed; `effective_henry` fog, pH 1.7–2.0.
- **Checks made independent:** `effective_henry` (speciation solve), `ppm_conversion_pressure`
  (tabulated factor), `mie_geometric` (+ anomalous diffraction), `radar_beam_height` (+ ray trace),
  `arrhenius_temperature` (+ JPL ethane); docstring errors fixed in `planck_linearization`,
  `mixed_layer_response`, `gradient_wind_curvature`, `kelvin_curvature`.
- **Screening rule added:** `line_broadening` — many texts define the Doppler width as the 1/e
  half-width, 1.17–1.19 × the HWHM key. Outside failures in that band are "recognised, wrong width
  convention", not regime misses; tag them before counting.
- Not changed, recorded: `solid_angle`, `kelvin_curvature`, `inertia_gravity_wave` `check()`s
  partly restate the key's algebra (anchors supply the independence); `kelvin_curvature` states σ
  and ρ_w, which serve only the Kelvin term, so its lure is expected weak.

**Rule 10 (from `cc_scaling`).** The keyed quantity must carry the physics being tested at the same
relative scale in both arms: never key a quantity of the form 1 + (small term) where the small
term is what the solver computes.

## Blind-solve check (2026-09-26)

One inside/outside/given triple of each of the 30 templates (90 problems) was shuffled, stripped of
keys and arm labels, and split across three Sonnet solvers told to read only the problem file,
solve with stdlib Python and flag any ambiguous or inconsistent text. **89/90 answers match the key
within 5 %**; the one miss is `line_broadening` outside, answered with the pressure-scaled Lorentz
width — the shortcut, not a text defect. The solvers' flags were all either the trap working as
designed ("the text does not say to include X") or realism notes already recorded (pH 1.7–2.0 fog,
Rayleigh inside at x ≈ 0.34, which the Mie key already absorbs: 4.888 vs 4.91). No new defect.

What this does and does not show: every text is solvable to the key by a careful expert, with no
alternative reading that any solver took. It says **nothing about lure strength** — the solvers
were told to check whether approximations hold (a stronger nudge than E2's generic hint), and arms of
the same template often landed in the same file, so a given sentence could inform its outside twin
(two solvers noticed the pairing). Lure strength is for the screening run.

## Domain and measurement-design review (2026-09-27)

Two further independent reviews looked at what the per-template reviews could not: whether each
problem is a real textbook exercise in a realistic situation (a senior atmospheric scientist's
reading), and the set as a measurement instrument. No key was wrong. Decisions and changes:

- **Removed (27 templates remain):** `vorticity_stretching` and `nonhydrostatic_wave`, the two
  pilot failures — textbooks derive the full relation first, so the "shortcut" is not the method
  a solver reaches for; and `arrhenius_temperature` — E_a/R is stated only for the replacement and
  cannot be removed fairly, so it tests "use the given", not recognition.
- **Rule 3, reworded.** Every quantity the correct answer needs is available in every arm — but a
  universal or handbook constant that only the *replacement* uses (Ω, g, R, R_d, σ of water,
  K_a of formic acid, sea-water ρ and c) is stated only in the given sentence, provided the
  spread of its common values moves the key by < 5 % (rule 5, checked in each `check()`). A
  constant stated only to serve the replacement announces that the parent method is incomplete.
- **Rule 11 (new): rounding-safe magnitudes.** The calibration model ends 17 % of its solvers with
  `round(x, 2)` or `round(x, 4)` (stored core runs), which turns a key of order 1e-4 into 0 in one
  arm only. Keys and rendered givens use units that put them at order 1–1000, and no given
  switches between plain and × 10^n notation across arms (`line_broadening` MHz,
  `weak_acid_dissociation` μmol/L and mmol/L, `solid_angle` deg², `henry_depletion` nmol/L,
  `cunningham_slip` μm/s, `kelvin_curvature` radii in nm).
- **Realism:** real weather-radar wavelengths; 25 °C only at 760–850 hPa; a queue of idling
  vehicles instead of an isolated 12–75 m road; convergence to at most 6 km; a corrected given
  sentence in `effective_henry` (the outside pH is *above* the pK_a).

**The A/B contrast is exploratory, not confirmatory.** The definition cannot be applied
mechanically: pairs that are the same situation carry different labels (`coriolis_neglect` A vs
`inertia_gravity_wave` B, both "add rotation"; `taylor_dispersion` A vs `mixed_layer_response` B,
both "a relaxation time scale"), and A templates carry a quantity only the replacement uses far
more often than B (11/12 vs 4/10), so A vs B is also "hint vs no hint". The labels stay as
registered and the contrast is computed and reported, but no confirmatory claim rests on it
(`docs/PREREG_TEMPLATE_SCREENING.md`).

**Mechanism clusters.** Templates that share an underlying mechanism are not independent units.
Any per-mechanism statistic is also reported with each cluster collapsed to one unit, and any
later trained / held-out split assigns **whole clusters**, never single templates:

| cluster | templates |
|---|---|
| rotation vs its time scale 1/f | coriolis_neglect, inertia_gravity_wave |
| rotating balance or wave with an extra term | gradient_wind_curvature, rossby_deformation |
| density decrease with height | hydrostatic_constant_density, anelastic_continuity, ppm_conversion_pressure |
| spherical-Earth geometry | radar_beam_height, solid_angle, spherical_slant_path |
| aqueous equilibria | henry_depletion, effective_henry, weak_acid_dissociation |
| plume and particle dispersion | plume_mixing_lid, finite_line_source, taylor_dispersion |
| linearised radiative emission | planck_linearization, optically_thin_emission |
| particle size vs a length scale | mie_geometric, kelvin_curvature, cunningham_slip; added below: ice_saturation |
| singletons | doppler_aliasing, line_broadening, log_wind_stability, pseudo_first_order, mixed_layer_response, saturated_lapse; added below: evaporative_cooling_limit, lognormal_mass |

That was 14 independent units; with the three templates added below, 16.

**Coverage gap, recorded.** Of the 27, thermodynamics, cloud physics and aerosols — three of the
core set's largest categories — have 2, 1 and 1 templates; observation has 3. A claim of
generality across atmospheric science needs more there; category is also confounded with A/B
(boundary layer all A, climate all B).

**Lure priors lowered by the domain review** (priors, not measurements): planck_linearization
strong → weak; hydrostatic_constant_density and doppler_aliasing strong → medium;
radar_beam_height, gradient_wind_curvature, taylor_dispersion → weak–medium; kelvin_curvature
weak until its σ cue was removed. `line_broadening`'s text endorses the parent relation ("this
width is proportional to pressure") in every arm — kept, since the pressure scaling is a given
the solver needs, and recorded.

**Blind re-check (2026-09-27).** The 17 templates whose texts changed in this review (constants
moved to the given sentence, new units, reworded scenarios) were re-solved blind: one
inside/outside/given triple each, 51 problems, shuffled, three Sonnet solvers told to supply and
name any constant the text does not state. **49 of 51 match the key within 5 %**; the two misses
are outside arms answered with the shortcut (`cunningham_slip` plain Stokes, `line_broadening`
pressure-scaled Lorentz width), i.e. the trap working, not a text defect. Every constant removed
from the texts was supplied from memory within tolerance (Ω, g, σ of water, K_a of formic acid,
R). One wording change followed: `ppm_conversion_pressure` no longer draws 760 hPa, which a solver
read as "760 mmHg = 1 atm". As before, this checks that the texts are well posed, not lure
strength (the solvers were told to check approximations).

## Three templates added for coverage (2026-09-27)

To narrow the coverage gap recorded above, three templates were added, one per thin category.
`evaporative_cooling_limit` and `lognormal_mass` are new mechanisms (singletons); `ice_saturation`
joins `kelvin_curvature`'s cluster (see the review below), so 16 independent units:

| name | category | shortcut → correct | trigger: inside / outside | lure (prior) |
|---|---|---|---|---|
| `evaporative_cooling_limit` | thermodynamics | all the rain evaporates, ΔT = L_v M/c_p → evaporation stops at saturation, isobaric wet-bulb depression | rain made available M: 0.5–1.8 / 8–12 g kg⁻¹ (T 22–26 °C, p 1000–1020 hPa, RH 25–50 %) | medium, two-sided |
| `ice_saturation` | cloud physics | saturation vapour pressure over liquid water (formula stated, labelled "over a plane liquid-water surface") → over ice | temperature: −0.2 to −1.5 °C / −20.0 to −30.0 °C (cloud level 600–850 hPa, not in the key) | weak |
| `lognormal_mass` | aerosols | N particles of the count median diameter → Hatch–Choate third moment, × exp(4.5 ln²σ_g) | geometric standard deviation: 1.03–1.06 / 1.50–1.90 (N 1000–3000 cm⁻³) | weak |

They follow rules 1–11 and the reworded rule 3 (the ice formula and the Hatch–Choate relation
appear only in the given sentence; no saturation formula is stated in `evaporative_cooling_limit`,
whose standard forms move the key by 0.1 %). They were added **after** the A/B classification was
registered, so they carry the label `post-reg` in `screening.KIND` and stay out of the exploratory
A/B contrast; they are screened like every other template.

**Blind solve (2026-09-27).** Two triples per template (seed 777), 18 problems, three Sonnet
solvers, one arm of each pair per solver, told only to solve carefully (no hint to check
approximations): **18 of 18 match the key within 5 %**, the six outside arms included — so the
texts are well posed and this solver saw through all three lures; lure strength is for screening.
Two ranges were narrowed afterwards because a solver's legitimate route came close to the
tolerance: `ice_saturation` derived e_i from the stated e_w with a constant latent heat of fusion
(4.6 % off at −35 °C, so the outside range now stops at −30 °C, ≤ 3.2 %), and Stull's (2011)
wet-bulb fit reached 4.7–4.9 % at the warm, low-pressure corner of `evaporative_cooling_limit`
(grid narrowed to 22–26 °C and 1000–1020 hPa, ≤ 3.3 %). Both routes are now checked in `check()`.

**Review of the three (2026-09-27).** A domain reviewer and a measurement-design reviewer
(independent, read-only) recomputed every key — no key was wrong, no reading lands > 5 % away —
and found:

- **Repeated items (must-fix).** `ice_saturation` depended on T alone, so 20 pairs held 10–11
  distinct texts; `rnd`'s two significant figures did the same, more mildly, to single-parameter
  templates (`weak_acid_dissociation` 12 distinct inside texts, `solid_angle` 16, and 1–3 repeats
  scattered elsewhere). Fixed at three levels: `build_templates` now redraws any pair whose inside
  or outside text already occurred in that template; `check_templates` builds 100 pairs per template,
  so every template must have at least 100 distinct pairs; and the thin grids were widened (ice: one
  decimal in both arms plus the cloud's pressure level; weak acid and solid angle: three
  significant figures; `ppm_conversion_pressure`: integer ppb; `pseudo_first_order`: integer ozone).
  Items built for a given seed therefore differ from those built before this change.
- **`ice_saturation` near-duplicated `kelvin_curvature`'s frame** (text similarity 0.76; both state
  the flat-water e_s and ask what vapour pressure keeps a particle in equilibrium). Reworded (crystals
  falling through a cloud layer; deposition vs sublimation), and put in the same cluster, so no
  trained / held-out split can separate them. The integer-vs-decimal temperature format cue went with
  the grid change.
- **`evaporative_cooling_limit`:** "closed parcel" holding 8–12 g/kg of rain contradicted "rain falls
  into the layer" and implied 5–10× a heavy-rain water content — reworded to water *made available*
  over the shower; "warm, dry" dropped. The lure is two-sided (wet-bulb cooling is itself textbook
  lore; criterion (a) catches a habitual wet-bulb solver), and a vapour-budget check made at the
  initial temperature returns the shortcut — both recorded in the module and the pre-registration.
- **`lognormal_mass`:** the gloss "(ln σ_g is the standard deviation of ln D)" drew attention to the
  width in every arm — dropped; N capped at 3000 cm⁻³ (5000 at D_g = 0.25 μm was severe-haze level).
- **Coverage** is narrowed, not closed: thermodynamics 3, cloud physics 2, aerosols 2 of 30, against
  dynamics 5.

**Blind re-check after the review (2026-09-27).** The three reworded templates, two triples each
(seed 901), 18 problems, three Sonnet solvers told only to solve carefully and flag anything odd:
**17 of 18 match the key**; the miss is an `ice_saturation` outside arm answered with the stated
water formula, whose solver noted that ice would be lower but "no ice formula was given" — the lure
working (the other outside solver recalled the ice/water ratio, 2.6 % from the key). Two solvers
noted that the cloud's pressure level does not enter the answer — intended, and it is in every arm.
