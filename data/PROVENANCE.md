# Data provenance

Where every file in `data/` came from, and every time one was replaced. **What each file
*is* and how this project uses it is in `data/README.md`** — this file is only the paper
trail.

Data is vendored rather than referenced so this repository is self-contained: clone it alone
and every split, reward and baseline reported here reproduces. The files are not arranged by
origin — origin is recorded here, in metadata, which is where it belongs.

## The corpus

| `role` | `frozen_split_member` | records | what it is |
|---|---|--:|---|
| `core` | `true` | 436 | the parent problems the frozen partition in `splits/` is defined over |
| `core` | `false` | 12 | a 14th textbook onboarded upstream after the release; in no split, not trainable, an unseen-book probe |
| `trap` | `false` | 335 | single-trigger perturbations; the control of each is its untouched `parent_id`, which is a `core` record in the same file |

`load_problems()` assembles these in memory from `core.json`, `unseen_book.json` and
`traps.json`; there is no built corpus file. Two properties are load-bearing and are asserted
on every load: `load_core()` returns exactly the 436 frozen ids in their source order, and
every trap's `parent_id` resolves to one of them. Merging without those would silently
repartition training or destroy the trap probe's control. `uv run python -m
atmoscoder_rl.check_corpus` reports the composition and exercises them.

| field | value |
|---|---|
| source repo | `AtmosCoder-Bench` (local working copy: `../AtmosCoder-Bench-Lab`) |
| source commit | `6891123b3e901318d98ac85bbbbaa2ef3f41eaf3` |
| commit date | 2026-09-06 |
| plus upstream trap-set commits | `e0c9601` (expansion 67 -> 265), `258bb50` (67 originals metadata-normalised), `e0a1865` (protocol review note), `a33eace` (regime twins probe), `ab98286` (Phase 6, 265 -> 269) |
| first copied on | 2026-09-02 (upstream `43c3592`, 2026-08-18) |
| re-vendored on | 2026-09-07 (trap expansion; then trap metadata + one trap), 2026-09-09 (regime twins; Phase 6) |

## Files

Hashes are not kept in this table, because they change on every re-vendor: `uv run python -m atmoscoder_rl.check_corpus` prints the current sha256 of the three corpus sources, and each dated entry below records the value it installed.

| path | records | role here |
|---|--:|---|
| `core.json` | 436 | source, unmodified |
| `unseen_book.json` | 12 | source, unmodified; was `core_v2.json` until 2026-09-11 |
| `traps.json` | 335 | source, unmodified |
| `variants_numeric.json` | 1730 | training signal (5 per perturbable parent; inputs re-samplable) |
| `variants_paraphrase.json` | 2180 | linguistic-robustness probe |
| `scaffolding_ablation/{original,stripped}.json` | 169 each | scaffolding-dependence probe (held-out subset used) |
| `cross_domain/*.json` | 131 | out-of-domain transfer probe (never trained on) |
| `traps_extended.json` | 18 | **out-of-core probe**: certified `regime_boundary` traps whose parents are NOT in `core.json` (soil 14, hydrology 2, environmental chemistry 2). Never part of the corpus — those parents have no record here, so a trap on them has no control in this repo. Its domain mix is an upstream artefact of a generator transport failure, not a finding; see the upstream protocol before quoting any rate over it |
| `traps_regime_twins.json` | 31 | **per-family probe**: certified `regime_boundary` traps whose parent's corpus record is the `geometry_detail` twin. Never part of the corpus — its parents already have a trap there, and the assembly asserts one trap per parent. Within regime_boundary each parent still has one trap and is still the control. 30 come from upstream's Phase-2 assembly and 1 (`trap_ry_7.7_regime`) from its Phase-11 top-up; see the dated entries below. How many have a parent outside `train` is a property of the split, not of the file — compute it, do not quote it from here |
| `DATASET_CARD.md` | — | the dataset's own card from the Bench release; it describes the benchmark, `README.md` describes this project's use of it |

`atmoscoder_rl/engine.py` is likewise a verbatim copy of the upstream
`eval/engine.py` (pure standard library, no project dependencies). It is **not
modified**: the grading rule used for reward here is bit-for-bit the rule the
benchmark paper reports accuracy under. Upstream changes must be re-vendored
deliberately, never merged silently.

## Re-vendored 2026-09-07

Three upstream changes were brought over; `core.json`, both variant files and the
scaffolding-ablation set were byte-identical and were not touched, so **the frozen
partition in `splits/` is unaffected** (it is defined over `core.json`).

| file | before | after | why |
|---|---|---|---|
| `benchmark/traps.json` | 67 | 264 | upstream expanded the two under-populated trap families under a pre-registered protocol: geometry_detail 2 -> 109, regime_boundary 5 -> 95. Every record re-verified upstream (`eval.verify` 264/264); the 197 new ones carry `expansion`, `mechanism`, `certification` (two independent blind witnesses) and `review` fields |
| `benchmark/cross_domain/*.json` | units empty | 209 units filled | upstream bug fix: an empty expected unit disables unit-aware grading, so a correct answer given in a commensurate unit was marked wrong |
| `benchmark/README.md` | — | — | upstream dataset card, refreshed |
| `benchmark/core_v2.json` | absent | 12 | new upstream set (Lamb & Verlinde, *Physics and Chemistry of Clouds*), consensus ground truth, `eval.verify` 12/12. Vendored as a separate probe — see the note below |

### Consequence for the trap probe — read before using the new traps

The frozen split removes **the 67 original trap parents** from training (policy item 3).
It was built on 2026-09-02, when those were the only traps. The 197 new traps have 197
*different* parents, and those were partitioned like any other core problem:

| where the new trap's parent sits | new traps | trap probe is |
|---|--:|---|
| `train` (incl. `train_50` / `train_100`) | 104 (+1 on the second pass: `trap_p_51_gen`, parent `p_51` is in `train`) | **contaminated** — the model trains on the very problem whose canonical method the trap defeats |
| `test_id`, `test_core_only`, `test_book` | 93 | clean, same status as the original 67 |

Neither the split nor the training pool was changed to fix this, deliberately: re-freezing
under the current policy would drop the trainable pool from 317 to 144 parents and `train`
from 190 to about 86, invalidating every baseline already measured. The partition above is
therefore a property to *report on*, and it happens to be a within-experiment contrast on
this project's own hypothesis (does execution-verified reward reinforce canonical-method
bias more when the parent was trained on?). Use the 93 clean traps for any headline trap
number, and the 104 contaminated ones only as that contrast.

### Second pass, 2026-09-07 evening: traps 264 -> 265, originals normalised

`_upstream/traps.json` re-copied from upstream `e0a1865` and `problems.json` rebuilt with
`build_corpus --apply` (712 -> 713). `core.json`, `core_v2.json`, variants, scaffolding and
cross-domain files unchanged; **the frozen partition is unaffected** (`build_splits --check`
passes; `load_core()` returns the same 436 ids in the same order).

| what | records | detail |
|---|--:|---|
| added | 1 | `trap_p_51_gen` (geometry_detail; both blind witnesses reproduced the key). Its parent `p_51` is in `train`, so it joins the contaminated bucket: 105 contaminated / 93 clean |
| changed | 67 | the original traps, metadata only: `difficulty` (25) and `category` (1) now inherited from the parent as the 198 expansion records already were, `topic = trap_type` (5), `provenance` filled (5), `separation_max` + `separation_definition` added (67). No problem text, answer value, solver method or `shortcut_values` changed (upstream proved its paper aggregations byte-identical) |

Nothing in this repo reads `difficulty`/`category`/`topic` of a *trap* record (the split
stratification reads core records), so no number measured here moves. Any result computed
on "the 197 new traps / 104 contaminated / 93 clean" stays valid as stated; the same
computation on the current file is 198 / 105 / 93.

### Third pass, 2026-09-09: Phase 6, traps 265 -> 269

Upstream re-tried the 60 held-out parents its first generator had declined for
`regime_boundary`, with a different generator and criteria 1-10 unchanged (pre-registered
before generation). Four were accepted, all `witness_confirmed`, and `problems.json` was
rebuilt with `build_corpus --apply`: **+4 records, 0 changed**, 713 -> 717.

| id | trigger | separation |
|---|---|--:|
| `trap_3.8_gen` | horizontal single-pressure path -> vertical layer | 0.280 |
| `trap_5.9_gen` | post-sunset Rayleigh friction just above zero, not exactly zero | 0.250 |
| `trap_air_301_gen` | condensate loading reduces updraft buoyancy | 0.191 |
| `trap_jacob_10.6_gen` | upper-stratospheric ClO + O loss | 10.017 |

All four parents are held out, so **clean regime traps go 48 -> 52** in `problems.json`, or
**64** counting the 12 held-out twins in `traps_regime_twins.json`. The frozen partition is
untouched (`build_splits --check` passes, `load_core()` unchanged). Contaminated trap parents
stay at 105; held-out trap parents 160 -> 164.

Upstream is **not** completing its 9-configuration trap sweep (the paper's `experiments/trap/`):
only deepseek-v4-flash, deepseek-v4-flash-reasoning and gpt55-reasoning hold the 264
pre-expansion ids there (gpt55-reasoning run 3: 263). See `splits/MANIFEST.md` on pinning
`trap_gap` before quoting it.

### `core_v2.json` — an unseen-book probe, not part of any split

The 12 problems come from a textbook that is **not one of the 13 in `core.json`**, so no
model trained here can have seen a sibling problem from the same source, and no split
change is needed to hold them out — they were never in. They carry no
`numeric_variantable` flag and therefore cannot supply rollouts; they are evaluation-only.
This makes them a stricter transfer probe than `test_book` (Andrews and Jacob are held out
of training, but the benchmark was built from them, so their style and conventions are
represented in the corpus). Sample size is small: 12 problems, so it resolves large effects
only.

**Do not run `build_splits --force`.** `atmoscoder_rl.splits` derives trap parents from
`traps.json` at runtime, so a rebuild today would exclude all 264 parents and silently
produce a different, much smaller training set. `--check` is unaffected and still passes.

## Not copied

`experiments/` (4.9 GB of stored runs) stays upstream. Baseline numbers are computed
from it by `atmoscoder_rl.baselines`, which takes the Bench working copy as an argument
and writes only the small aggregate result into this repo.

### Fourth pass, 2026-09-10: Phase 10, traps 295 -> 330 — 35 records, 9 of them useful here

`_upstream/traps.json` re-copied from upstream (sha256 `f09ef2812051aeda`) and
`problems.json` rebuilt: **+35 added, 0 removed, 0 changed**, 743 -> 778 records. All 778
re-verified (`atmoscoder_rl.verify`: 778/778).

The 35 are upstream's `sign_direction` pass: 34 `sign_direction` plus one
`definition_confusion` that its independent reviewer relabelled. The family goes 9 -> 43 and
`definition_confusion` 20 -> 21; nothing else moved.

**Read this before using them.** Upstream drew this phase's parent pool from *every* free
core parent, not from the held-out ones, because sign traps were never meant for this study's
regime pool. So of the 35 new records only **9 have a parent outside `train`**:

| | traps | parent in `train` | clean |
|---|--:|--:|--:|
| before this pass | 295 | 104 | 191 |
| Phase 10 adds | 35 | 26 | 9 |
| now | 330 | 130 | 200 |

Clean counts by family are now geometry_detail 72, regime_boundary 52, formula_selection 26,
definition_confusion 21, sign_direction 17, averaging_space 12. **The clean regime count is
unchanged at 52, or 64 counting the 12 held-out twins** — this pass produced no regime traps
and does not move the regime power calculation at all.

These records also answer to a stricter fairness bar than the rest of the file: for
`sign_direction`, upstream's criterion 10 requires all three blind witnesses to reproduce the
key *including its sign*, recorded per record as `certification.sign_unanimity`. One
candidate (`holton_44`) was rejected on that rule alone after two of three witnesses agreed.

Re-vendored later the same day after upstream's own review of the batch: **5 records changed,
0 added, 0 removed** (`trap_air_122_gen`, `trap_air_284_gen`, `trap_p_73_gen`, `trap_ry_1.3_gen`,
`trap_snp_64_gen`). The change is confined to each record's `review` provenance text, which had
carried a boilerplate claim contradicted by its own numbers. **No problem text, answer, solver or
`shortcut_values` moved**, so nothing computed from this file changes; `atmoscoder_rl.verify`
stays 778/778. Worth knowing if you read `review.c7`/`review.c9` on a trap: on the 2026-09-10
batch those fields were reviewer prose, and only these five were checked against the data before
this pass.

### 2026-09-11: `core_v2.json` renamed to `unseen_book.json`

Upstream renamed the file and added `probe_role: "unseen_book"` to each of its 12 records. Nothing
else changed: **12 records changed, 0 added, 0 removed**, and the change is the new field plus the
source filename. `atmoscoder_rl.verify` stays 778/778.

The rename answers a question that was raised and rejected: whether to merge those 12 into
`core.json` under a `version` field. They are not merged, because being outside the frozen 436 is
what makes them useful here — they come from a 14th textbook no `core.json` problem comes from, so
no model trained on this corpus has seen a sibling from the same source. They remain in no split
and `numeric_variantable` is still forced False.

`build_corpus.py`'s assertion that no upstream field is overwritten caught a real collision on the
first attempt: upstream's field was called `role`, which is the name this script forces onto every
record to mean core-or-trap. Upstream renamed it to `probe_role`. The assertion is why this was a
failed build rather than a silently overwritten field.

### 2026-09-11: two new per-record fields on every trap — read this before recomputing any trap arm

`_upstream/traps.json` re-vendored: **330 records changed, 0 added, 0 removed.** No problem text,
answer, solver or `shortcut_values` moved; the change is two adjudication fields upstream added
after auditing the whole corpus. `atmoscoder_rl.verify` stays 778/778.

| field | on | meaning |
|---|---|---|
| `review.measures_regime_judgement` | the 99 `regime_boundary` records | `true` when the solver must recognise the parent's relation has failed and produce the replacement; `false` when the trap STATES the replacement, so the solver reads it off the page |
| `review.magnitude_defect` | all 330 | `true` when the trap asserts a quantity the object it names cannot have |

**What this does to the arms of the regime-versus-geometry result:**

| arm | records | clean (parent outside `train`) |
|---|--:|--:|
| `regime_boundary`, all | 99 | 52 |
| `regime_boundary` that actually measures regime judgement | 69 | **36** |
| `geometry_detail` | 129 | 72 |

A trap that hands the solver its replacement relation is still a valid trap, but it cannot show
whether a model knows when a relation stops applying. If the claim is about regime JUDGEMENT, the
arm is 36 clean records, not 52.

**Seven records carry `magnitude_defect: true` and six of them are clean**, so they are inside any
arm computed today: `trap_air_64_gen`, `trap_p_35_gen`, `trap_p_37_gen`, `trap_jacob_13.1_gen`,
`trap_4.3_gen`, `trap_13.5_gen` (plus `trap_p_38_gen`, whose parent is in `train`). Each asserts
something its named object cannot be — Mars keeping 80 kPa pressure levels, 1368 W m-2 at 60° of
incidence, a 0.475 um frequency peak for the Sun, US precipitation at 0.7 mm/yr, an ITCZ at 65°S,
CO2 a million times below any terrestrial value. Exclude them from any arm.

### 2026-09-13: `data/` is one flat directory, and the corpus is assembled rather than stored

Entries above this one were written against the earlier layout — upstream copies under
`data/benchmark/_upstream/`, a built `data/benchmark/problems.json`, and `build_corpus --apply`
to rebuild it. All of that is gone; the intermediate steps are in git history.

| before | now |
|---|---|
| `data/benchmark/_upstream/{core,traps,unseen_book}.json` and every other vendored file under `data/benchmark/` | `data/*` |
| `data/benchmark/problems.json` (built) | none — `datasets.load_problems()` assembles the 783 records in memory |
| `atmoscoder_rl/build_corpus.py` | `atmoscoder_rl/check_corpus.py` — reports composition; the invariants run on every load and raise `CorpusError` |
| `data/benchmark/README.md` | `data/DATASET_CARD.md` |

No record changed: the assembled corpus equals the deleted `problems.json` record for record.
One consequence to keep in mind: `check_reward`'s isolation test used to try to read the answer key
from `problems.json`; it now targets `data/core.json`, which always exists, so a "blocked" result
still means the key was unreachable rather than that the path was wrong.

### 2026-09-12 to 09-14: Phase 11, the targeted top-up — six records

Upstream's response to the work order `docs/archive/TRAP_SPEC_2026-09-12.md` (retired 2026-09-28; `git show 75af38e:docs/archive/TRAP_SPEC_2026-09-12.md`). Current files: `traps.json` 335 records
(sha256 `77833afee04bdc98`), `traps_regime_twins.json` 31 (sha256 `0d350035be3d4b5b`), upstream commit `523cde8`.
`check_corpus`, `verify --set all` and `splits --check` pass.

| id | file | family | category | separation_max | trigger |
|---|---|---|---|--:|---|
| `trap_ry_5.1_p11` | traps | regime | cloud_physics | 0.218 | the cloud is fully glaciated: vapour equilibrates over ice, not water |
| `trap_ry_14.2_p11` | traps | regime | cloud_physics | 0.113 | fusion heat carries the parcel past ice saturation: deposition reverses to sublimation |
| `trap_ry_7.7_regime` | twins | regime | cloud_physics | 0.678 | 1 s after an updraft starts, supersaturation is still relaxing to its quasi-steady value |
| `trap_turner_2_p11` | traps | regime | air_quality | 0.645 | an inversion lid: Gaussian vertical spread becomes uniform mixing |
| `trap_jacob_13.4_p11` | traps | regime | air_quality | 2.198 | the valley starts clean, so the steady state has not been reached |
| `trap_fp_11.1_p11` | traps | geometry | atmospheric_chemistry | 0.563 | collision diameter of two unlike spheres |

All six have a parent outside `train`, measure what their family claims (the five regime records
carry `measures_regime_judgement: true`), have no magnitude defect, were reproduced by two blind
witnesses from vendors other than the author's, and were accepted by an independent reviewer.

**Where the spec stands** (arms from `analysis.arm_ids`, twins included):

| sub-domain | asked for | regime | geometry | both arms >= 6 |
|---|---|--:|--:|:--:|
| cloud_physics | 4 regime | 5 (was 2) | 6 | no — one regime short |
| air_quality | 2 regime | 6 (was 4) | 12 | yes |
| atmospheric_chemistry | 2 geometry | 7 | 5 (was 4) | no — one geometry short |

Sub-domains that support a within-sub-domain comparison: 4 of 6 (dynamics, thermodynamics,
radiation, air quality). The two open decisions are recorded in the spec's header.

**No run covers these six yet.** Following `analysis._trap_index`, the five `traps.json` records
belong under experiment id `trap_expansion` and the twin under `trap_regime_twins`. If
`trap_ry_14.2_p11` was run before 2026-09-14, those results are stale: its text changed that day
(see note 4).

**Read before analysing them:**

1. **`trap_ry_7.7_regime` adds little power.** Its parent is solved in 12 of 54 upstream runs and
   1 of 12 on 9B/27B models; a pair is informative only when the control is sometimes solved.
   `trap_ry_14.2_p11`'s parent is at 4 of 12 on those models, the other four at 75-92 %.
2. **Two controls are graded off-centre.** The parents of `ry_14.2` (keyed 1.3, solver 1.2604) and
   `fp_11.1` subs 2-3 (keyed 1.4, solver 1.3563) are keyed with textbook rounding about 3 % high, so
   an answer 2-5 % low passes the trap and fails the control. This can only lower a Trap Gap.
3. **`trap_fp_11.1_p11` accepts two values on subs 2 and 3**, as its parent does: "10 % of the
   excited molecules emit" has two defensible readings. `shortcut_values` stays single-valued.
4. **`trap_ry_14.2_p11` states its latent heats and c_p** (added 2026-09-14). Without them a
   Kirchhoff-corrected latent heat of fusion — used in 3 of the parent's 54 upstream runs — made the
   correct method miss by 10.3 % and the shortcut pass at 2.8 %. Stating them removes that inversion;
   the parent states no constants, so for those solvers the pair differs only in the direction that
   lowers the Trap Gap. Its separation (11 %) is the thinnest in the arm.
5. **`trap_ry_5.1_p11` and `trap_turner_2_p11` change more than one given**, because the trigger
   forces it (ice cannot exist at +10 C; at 500 m the plume never reaches the lid). The shortcut is
   handed the same changed givens, so the separation is the trigger alone, but the parent is not an
   equally hard control by construction. Each record's `review.c6` lists the edits.
6. **`trap_jacob_13.4_p11` sub 2 (pH) discriminates weakly** under a relative 5 % tolerance (about
   a factor 1.5 in [H+]); the regime test rests on sub 1.
7. **The independent review is not cross-vendor.** The records were authored and reviewed by
   Anthropic models; witnesses are cross-vendor, the reviewer is not.

What was built and rejected, with per-candidate reasons, is in upstream's trap-expansion protocol;
nothing here depends on it. Also removed in this phase: `data/traps_expansion.json`, a 263-record
duplicate of an older upstream snapshot that nothing read.
