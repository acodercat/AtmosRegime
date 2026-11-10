# AtmosCoder-Bench — Dataset

Atmospheric-science computational problems for evaluating LLMs via code generation.
A model sees **only the problem text** and must write a `solve()` function whose
numerical output is checked against the known answer within a relative tolerance
(default 5%).

## Files

| File | Count | What |
|------|------:|------|
| `core.json` | 436 | Confirmed core problems |
| `variants_numeric.json` | 1730 | 5 **parameter-perturbed** variants per *perturbable* base (contamination-resistant) |
| `variants_paraphrase.json` | 2180 | 5 **reworded** variants per base — same numbers & answer, different phrasing (expression-robustness) |
| `traps.json` | 335 | **Probe set, not in `SETS`** (run with `--input`): single-trigger perturbations of 335 distinct core problems whose untouched parent is the control; six mechanism families (geometry-detail 130, regime-boundary 103, sign/direction 43, formula-selection 26, definition-confusion 21, averaging-space 12). 67 built 2026-08, 268 added 2026-09-07/13 in six pre-registered passes (`expansion` field distinguishes them; the 20 `2026-09-phase7b` geometry records come from a different parent-selection rule than the earlier geometry records — see the protocol) under `pipeline/reports/trap_expansion_protocol.md` (two-witness fairness certification stored per record; the 35 `2026-09-phase10` records answer to a stricter criterion 10 — all three witnesses must reproduce the key including its sign, flagged as `certification.sign_unanimity`); the 67 originals were metadata-normalised to the expansion schema on 2026-09-07 (`category`/`difficulty` from the parent, `topic = trap_type`, `separation_max` added — protocol amendment; no text, solver or answer changed). The 5 `2026-09-phase11` records (2026-09-12/13) are a targeted top-up requested by the downstream study — regime traps in cloud physics (2) and air quality (2) and a geometry trap in atmospheric chemistry (1) — hand-built rather than generated (`provenance: hand_built+human_adjudicated`); two of them change more than one given because the trigger forces it, and each says so in `review.c6`. |
| `traps_extended.json` | 18 | **Out-of-core probe, not in `SETS`**: certified `regime_boundary` traps on parents outside `core.json` (cross-domain: soil 14, hydrology 2, environmental chemistry 2). Never merged into `traps.json` — those parents have no `core_code` baseline. Carries `parent_source`, `parent_domain`, `parent_web_available`. Concentrated: 14/18 soil mechanics, 6/18 on one trigger. Parents are publicly available, so a Trap Gap here can be biased upward by control-arm memorisation. Phase 8A, 2026-09-10 (`eval.verify` 18/18). |
| `unseen_book.json` | 12 | **Held-out probe, not in `SETS`**: problems from a 14th textbook (Lamb & Verlinde, *Physics and Chemistry of Clouds*) that no `core.json` problem comes from, so no model trained on this corpus has seen a sibling from the same source. Every record carries `probe_role: "unseen_book"`. **Deliberately outside the frozen 436** — it is in no split, has no numeric or paraphrase variants, and no stored `core_code` run covers it; merging it into `core.json` would destroy the held-out property that is its only purpose. Consensus ground truth with `recertified_on_final_text` (`eval.verify` 12/12). Renamed from `core_v2.json` on 2026-09-11: the old name recorded which onboarding batch it came from, not what it is for. |
| `traps_regime_twins.json` | 31 | **Per-family probe, not in `SETS`**: certified `regime_boundary` traps for parents whose `traps.json` record is the `geometry_detail` twin. Same schema plus `twin_of`, ids `trap_<parent>_regime`. Use only in analyses that treat one family at a time — pooling with `traps.json` counts a parent twice. 30 were exported 2026-09-09 from Phase 2, where both proposals passed all ten criteria and assembly kept geometry; 1 (`trap_ry_7.7_regime`, `expansion: 2026-09-phase11`) was hand-built on 2026-09-14 for the targeted top-up, and is written by `sources/extracted/trap_expansion_phase11.py --apply`, not by the Phase-2 export, which now refuses to overwrite the file. `eval.verify` 31/31. |

Of the 436 core problems, **346 are `numeric_variantable`** (their text-visible
inputs can be safely perturbed) and carry 5 numeric variants each (346 × 5 =
1730); the remaining **90 are core-only** (perturbation would break them — e.g.
convention-sensitive or coupled inputs) and carry no numeric variants. **Every**
core problem carries 5 paraphrases (436 × 5 = 2180). The three files are kept
separate by role so a run can compare base vs numeric-variant vs paraphrase
accuracy; `parent_id` joins each variant back to its base.

> The 436-set was curated for QUALITY from a larger pool: low-value problems
> (trivial one-step plug-ins, pure unit conversions, non-atmospheric items) were
> removed by a **two-model** value audit (gpt-5.5 flags + Claude Opus confirms),
> a handful of out-of-scope/ill-posed/GT-defective problems were dropped, and
> borderline cases were salvaged (sign/unit twins, missing-assumption fills).
> Provenance: `pipeline/reports/quality*.json`, `recategorize_final.json`.

Sources (13 textbooks):

| Count | Textbook |
|------:|----------|
| 156 | Practical Meteorology (Stull) |
| 57 | Atmospheric Science: An Introductory Survey (Wallace & Hobbs) |
| 31 | An Introduction to Dynamic Meteorology (Holton) |
| 31 | Air Pollution Control Engineering (de Nevers) |
| 28 | Introduction to Atmospheric Chemistry (Jacob) |
| 28 | An Introduction to Atmospheric Physics (Andrews) |
| 26 | Fundamentals of Atmospheric Modeling (Jacobson) |
| 25 | Atmospheric Chemistry and Physics (Seinfeld & Pandis) |
| 20 | A Short Course in Cloud Physics (Rogers & Yau) |
| 13 | Workbook of Atmospheric Dispersion Estimates (Turner / EPA) |
| 12 | Chemistry of the Upper and Lower Atmosphere (Finlayson-Pitts & Pitts) |
| 7 | Air Pollution Control: A Design Approach (Cooper & Alley) |
| 2 | Air Pollutant Concentration Models |

## Two orthogonal labels

**`category`** — 10-class atmospheric-environment taxonomy (two-vendor consensus
re-identification, `pipeline/recategorize.py`):

| category | n | | category | n |
|----------|--:|--|----------|--:|
| atmospheric_dynamics | 116 | | atmospheric_radiation | 25 |
| atmospheric_thermodynamics | 89 | | atmospheric_aerosols | 24 |
| atmospheric_chemistry | 51 | | climate_dynamics | 19 |
| air_quality | 37 | | observation_and_modeling | 14 |
| boundary_layer | 32 | | | |
| cloud_physics | 29 | | | |

**`difficulty`** — intrinsic low/medium/high (gpt-5.5 6-dimension rubric,
`pipeline/classify_difficulty.py`): low 44 / medium 258 / high 134.

## Record schema

Base (`core.json`):
```json
{"id, book, problem, code, sub_answers:[{sub,value,unit}], category, topic, knowledge_points, verified, numeric_variantable, difficulty"}
```
Numeric variant (`variants_numeric.json`) adds: `parent_id`, `variant`,
`parameters` (perturbed inputs), `skipped_params` (frozen), `gates` (generation
diagnostics), `k2` (independent-verification verdict). Paraphrase variant
(`variants_paraphrase.json`) shares the parent's `code` and `sub_answers`; only
`problem` is reworded. Carries its own `k2`.

## How answers are grounded

Base answers come from one of two routes, both stronger than a single source:

- **Official textbook answer** (gold standard) — for books that ship a solutions
  manual (e.g. *Introduction to Atmospheric Chemistry*, *An Introduction to
  Atmospheric Physics*): a blind `solve()` must reproduce the textbook's stated
  answer within tolerance, and a second independent model must reproduce it too.
- **Multi-model consensus** (for books with no published solutions — e.g.
  *Atmospheric Chemistry and Physics*, *An Introduction to Dynamic Meteorology*,
  *Fundamentals of Atmospheric Modeling*): independent strong models from
  different vendors each blindly solve the text; admitted only when an author
  solver's answer is reproduced by the others within tolerance.

Every admitted base additionally passes an anti-hardcode / dead-input audit, a
self-verification (its stored solver reproduces its stored answer at 5%), and an
independent critical review. A final GT audit over the zero-confirmation set
(problems no evaluation model solved) by an independent blind solve found no GT
defects.

- **Numeric-variant ground truth** is computed by the core problem's *faithful*
  `solve()` at the perturbed inputs — never produced by an LLM.
- **Paraphrase ground truth** is the parent's answer unchanged (only wording moves).

## Variant certification

Each numeric variant passes deterministic safety gates at generation (constants
frozen out of the signature, only text-visible inputs perturbed, signature=params
AST-verified, answer moves vs base, physical-domain bounds, no zero/ill-conditioned
GT, text↔param consistency), then a **k≥2 independent check**: two independent
vendor models (GPT-5.5 reasoning, Claude Opus 4.8) each blindly solve the variant
text and must reproduce the stored GT (`pipeline/k2_certify.py`). Paraphrases are
certified by **semantic equivalence**: two independent vendor judges must each
agree the reworded text is the same problem as the parent, recorded as `EQUIV_K2`
(`pipeline/certify_paraphrase.py`).

`k2` verdict distribution:

| verdict | `variants_numeric.json` | meaning |
|---------|------------------------:|---------|
| `PASS_K2` | 1651 | both models independently reproduced GT |
| `PASS_K1` |   56 | one model reproduced GT, the other abstained |
| `PASS_K2_DUALSIGN` | 8 | both reproduced GT up to an ambiguous sign convention |
| `PASS_K2_UNITSIGN` | 6 | both reproduced GT up to a unit-scale (×10ᵏ) or sign convention |
| `REVIEW`  | 9 | neither model converged (hard problem); GT still solver-constructed |

≈96% of numeric variants carry direct two-model backing and ≈99.5% at least one;
all 2180 paraphrases are `EQUIV_K2`. A sync auditor (`pipeline/audit_variants.py`)
keeps the three files coherent (0 stale / 0 missing); every set verifies 100% at 5%
and the base×5 invariant holds.

## Caveat (scientific honesty)

"No known problems" is not "provably zero errors": a flaw that *both* independent
models reproduce would not be caught. A human expert audit of a random sample,
reporting a residual-error-rate 95% CI, is the remaining step for a publishable
guarantee.
