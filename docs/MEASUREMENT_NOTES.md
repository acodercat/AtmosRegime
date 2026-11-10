# Measurement notes — defects found and what they cost

Kept because each of these changed a number we had already reported, and because the fix
is worth more than the finding. Written in the order they were found.

## 1. A network failure that read as a domain result (upstream, Phase 8A)

`generate_traps.py` had no retry on one call path (the evaluation runner always had one), so a
transient disconnect was recorded as `MODEL_ERROR` and read downstream exactly like
"the generator looked and declined". 83 of 143 parents were never evaluated: 82 server
disconnects, 1 SSL EOF.

**The loss was not random — it was selected by prompt length.** Disconnected parents were
systematically longer (problem text median 523 vs 340 characters, Mann-Whitney z = 5.58,
p < 0.001; solver code 750 vs 552), and the corpus sources differ sharply in length:

| source | median problem length | Phase 8A loss |
|---|--:|--:|
| hydrology | 525 | 33/37 = 89% |
| environmental chemistry | 499 | 14/21 = 67% |
| ecology | 478 | 11/19 = 58% |
| soil | 306 | 13/54 = 24% |

Soil is the shortest source in the corpus, lost the least, and therefore dominated the 18
survivors — from which we read "regime traps transfer to soil mechanics, barely to
hydrology". Two independent measurements say the opposite: the physical-level census finds
regime problems concentrated in **hydrology** (6/34 = 17.6%) against soil (2/50 = 4.0%),
and the backfilled generation run passes hydrology 20, soil 10, env chem 8, ecology 1.

The conclusion is withdrawn. What makes this worth reporting is that the mechanism is
named and checkable: a length-correlated failure channel in a generation pipeline will
select for whichever domain writes shortest, and produce a clean, plausible, entirely
artefactual domain finding.

**Also withdrawn:** the claim that the regime pool was exhausted. It rested on two
generation passes reported as yielding "4 and 20 accepts, none regime", which is the same
missing-data artefact.

**A limit on the fix.** The backfill cannot be read as "what Phase 8A should have found"
*between batches*: Phase 8A evaluated the short parents and the backfill the long ones, so
the gate rates (24/60 = 40% vs 51/62 = 82%) are confounded with length. What the backfill
buys is that the **union** covers all 143 parents, so the rate over all 143 is unbiased.
Per-batch rates are not.

**The fix is now structural**, not a convention: `gate()` returns `UNEVALUATED` for
`MODEL_ERROR`, the run summary prints the corrected denominator, and it carries the line
*do not read a parent that was never asked as a zero*.

## 2. Our own evaluation, checked for the same defect

18,462 stored records across every experiment: **0 error records**, and every run file
holds its full id set. The evaluation runner retries API failures and only writes an `error` if
they never clear, so the channel that produced the Phase 8A artefact is closed here. This
was verified, not assumed, with the same Mann-Whitney test.

One real gap did surface: the random-reward control arm had 196 of 228 expansion traps and
nothing on the extended and twin sets, so a control column over the full trap corpus would
have compared different id sets. Completed.

## 3. Confidence intervals on a census

The physical-level classification is a **census of all 567 parents**, not a sample. Its
uncertainty is 49 undecided boundary cases, and the reported ranges are what those cases
produce if they all fall one way. Quoting a 95% CI there describes an uncertainty the
measurement does not have while hiding the one it does.

The structure of the disagreement decides which numbers are quotable: 37 of the 49 are
`closed_form` vs `parameterization` — the fuzziest boundary — and only 9 involve
`regime_judgement`. **So the regime count is firm and the closed-form count is soft**, the
opposite of how a "94.8% of problems are closed-form" sentence would use them. That
sentence is dropped; a point estimate of 94.8% is in any case above the census upper bound
of 92.8%, and folding parameterization problems (4.2–10.9%) into "not closed form" would
inflate the scarcity of regime problems by counting a different thing.

**What survives, and is stronger:** problems that require judging whether a method still
applies are **16–25 of 567, 2.8–4.4%** — about one in thirty — and they are not spread
evenly: close to one in five in hydrology, none at all in ecology or environmental
chemistry.

## 4. What the training split actually contained

Cutting the census by the frozen split:

| split | n | regime problems | |
|---|--:|---|--:|
| **train** | 190 | **3–6** | **1.6–3.2%** |
| held-out core | 183 | 5–10 | 2.7–5.5% |
| train_50 | 50 | **0** | — |
| train_100 | 100 | 2 | 2.0% |

Each training parent contributes itself plus 5 numeric variants, so regime problems are
**18–36 of 1140 training rows**. That turns "RL does not learn regime judgement" from an
observation into a quantified mechanism, and it agrees with the independent pass@64
attribution (the base model produces a correct solution to a regime trap 0.2 times in 64).

**A warning for any data-scaling ablation:** the 50-problem arm contains **zero** regime
problems and the 100-problem arm two. A flat regime curve across that ladder would measure
the absence of signal, not a dose-response.

## 5. The witness gate selects traps that reasoning models solve (2026-09-23, exploratory)

Criterion 10 of the trap protocol admitted a trap only if blind witnesses — reasoning models
(gpt55-reasoning, deepseek-v4-pro-reasoning; opus48 in Phase 6, gpt54 in Phase 11) — solved it.
Of the 46 traps in the held-out regime arm, 42 passed through that gate; only 4 are original
traps built before it existed. Two consequences, measured on the archived runs:

**Reasoning models score high on the gated traps partly by construction.** gpt55-reasoning
(a witness) and deepseek-v4-flash-reasoning (a sibling of one) solve 84–100% of the regime arm
with no gap to the parents. That cannot be read as "reasoning removes the deficit": the arm was
selected for traps a reasoning model solves. On the 4 un-gated traps the same models show no
clear gap either, but 4 traps settle nothing.

**The gate also selected easier traps for everyone.** deepseek-v4-flash (non-reasoning, never a
witness): 25% on the un-gated regime traps, 57% on the gated ones (three repetitions pooled).

**What the gate does not hide:** the same non-reasoning model shows the pattern the study
reports for Qwen after RL, on a model we never trained — regime parents 68–74% vs traps 48–55%,
discordant pairs 10:26 pooled over three repetitions; geometry 18:19, no gap. The repetitions
are the same model on the same items, so this shows a stable direction, not a p-value.

**Cost:** the current trap set cannot answer whether test-time reasoning removes the regime
deficit — the question a reviewer asks first. A trap set for the final study has to record
witness outcomes without admitting on them, with fairness evidenced some other way
(`DESIGN.md` §2). Also: the Qwen3.5 and Qwen3.6 trap files in the upstream archive are
entirely error records from a dead endpoint, not zero scores.

## 6. Two grader properties that bear on what a problem may ask (2026-09-24)

Found while writing regime templates (`atmoscoder_rl/templates/`), confirmed against
`atmoscoder_rl/engine.py`. Neither is fixed: `engine.py` is the Bench grader, byte-identical by
rule. Both are constraints on problem design.

**Absolute temperatures are graded to about ±14 K.** Relative tolerance is 5 %, and a
temperature compared in kelvin is ~290 K, so the window is ~14 K wide. An answer expected in K is
always compared that way; one expected in °C is compared in °C when the model answers in °C, but
in kelvin when it answers in K — so an answer that is wrong by 6 °C passes if reported in K.
Scope: 49 of the 436 core problems (45 subs in K, 23 in °C) and 38 trap records have such a sub;
accuracy on those is lenient. Traps are not at risk of passing a shortcut this way: their
admission required at least 10 % separation measured in the answer's own unit, i.e. more than
28 K for an answer in K. **Consequence for new problems:** never ask for an absolute temperature;
ask for a difference or a rate. `saturated_lapse` asks for K km^-1 for this reason; °C km^-1
and K m^-1 are reconciled correctly.

**A bare numerical match passes regardless of the unit declared.** Expected 10 km, answered 10
kg: passes. Unit reconciliation only helps an answer in a different but commensurate unit; it
never rejects one. The effect needs a numerical coincidence, so it can inflate accuracy only
slightly, but it means the grader does not check dimensions.

## 7. Shortcut capture inherited the grader's Celsius window (2026-09-26)

Shortcut capture ("a wrong answer equals the parent method's answer") converted units under the
benchmark rule, which compares °C in kelvin at a *relative* 2 %: any two Celsius values within
~6 °C matched, and a bare `C` (coulomb) was read as Celsius. 70 of 6,516 stored trap records —
all of them *correct* answers — were counted as reproducing a shortcut a few degrees away.
`capture.py` now uses a strict unit rule. Re-derived on the server on 2026-09-27: **no failed
record changes its verdict**, so every stage-1 capture rate stands (`explore/REPORT.md`, notes).

## 8. Capture measures clean execution, not only a failure to check (2026-09-28)

The costliest finding of this round. "RL doubles shortcut capture on regime traps" (12 % →
21–29 % of wrong answers) was read as RL making the parent method more of a default. It is not:
a capture needs the parent method computed *exactly*, so the count rises whenever execution gets
cleaner. On the 13 textbook regime traps whose parent the base solves, among parent-right pairs,
the blind judge finds "not recognised" 15 vs 16 and "recognised, replacement wrong" 12 vs 11
before and after RL; what changed is how often a kept parent method lands exactly on the
shortcut, 3/15 → 10/16 (the judge covers one RL run). Recognition (trap right given parent right)
is 17 % before and 16–20 % after on those 13 traps (18 % and 14–20 % on all 46). **Rule: measure "not checking" by recognition (and the judge's A/B), never by
capture alone.** Details: `results/prereg/README.md`, the PREREG_TEMPLATE_RL section.

## 9. A magnitude-right answer with the wrong sign is graded wrong (2026-09-27)

The grader compares signed values. When a problem asks for a speed, a cooling, or any magnitude,
a model that returns the signed quantity fails with the right number. Seen in screening
(`rossby_deformation` given arm 0/20, every answer exactly −key, because its given sentence drops
the minus sign of the Rossby phase speed; smaller counts in three other templates) and in the
family contrast (9 of 42 `sign_direction` and 10 of 36 `definition_confusion` parent failures are
the key negated). Not fixed in the grader. **Consequence for new problems:** ask for a signed
quantity with its convention stated, or for a magnitude in words that cannot be read as signed;
report sign flips separately when analysing.

## 10. Template items repeated within a template (2026-09-27)

`ice_saturation` depended on its trigger alone, so 20 pairs held 10–11 distinct texts; `rnd()`
rounds to two significant figures, which made single-parameter templates repeat items too. A
repeated prompt at T = 0.3 is a correlated re-measurement, not a new item. Fixed before any
screening: `build_templates` redraws a pair whose inside or outside text already occurred, and
`check_templates` requires every template to yield 100 distinct pairs.

## 11. Paired trap-vs-parent leans follow the parent's ceiling (2026-09-27)

A parent-only discordance needs a solvable parent, so the parent-better lean of any trap group
grows with its parents' accuracy: in the family contrast, items whose parents are solved ≥ 2/3 of
the time lean 11:1 (applicability) and 8:3 (control), those below 1/3 lean 3:5 and 2:7. A
contrast between trap groups is therefore partly a contrast between their parents' difficulty.
Stratify by parent accuracy, or match it, before comparing groups.

## 12. Two confounds in the stage-1 model comparisons (2026-09-24)

Thinking-off runs use T = 0.3 (as the stored base runs), thinking-on runs use Qwen's recommended
T = 0.6, so every on/off contrast is also a temperature contrast. Qwen3-4B and Qwen3-30B-A3B exist
on the server only as the 2507 Instruct releases (non-thinking, newer post-training), so they are
not the same post-training as the other Qwen3 sizes; say so in any scale comparison.

## 13. A "replacement given" sentence can make an item easier than its parent (2026-09-28)

The given versions of the 46 regime traps were built by stating the relation the trap's own
solver computes. On average they were solved **more** often than their parents (RL 63 % vs 47 %):
on items the models cannot solve even as parents, the sentence also spelled out what makes the
parent hard. So "given version vs parent" does not isolate the recognition step; the comparison
that does is conditional on the parent being right (trap 20 % vs given version 77 %).
**Rule:** a control that hands over a relation must be checked for handing over more than the
parent's own difficulty — compare conditional on a solved parent, or write sentences that name
the replacement without spelling out a computation the parent also needs.

## Software environment: one drift, restored; vLLM 0.29 rejected (2026-09-15)

**Drift.** transformers was upgraded 5.9.0 → 5.10.0 in the live venv on 2026-09-14 21:38 for a
Gemma-4 serving probe. Every RL run, seed, the random-reward control and all base evaluations
predate it; the gold-SFT and RAFT arms (training and evaluation) ran after it. Restored to the
verl lock with `uv sync --frozen` on 2026-09-15 (venv then identical to the 256-package
pre-probe snapshot). Token sequences were fingerprinted under both versions — both SFT corpora
(full conversations and prompts) and all 436 evaluation prompts with thinking on and off — and
all are identical. Model code paths in transformers (FSDP forward) were not compared; the SFT
and RAFT conclusions rest on held-out accuracy staying at base level, which a small numerical
difference would not produce.

**vLLM 0.29.** Tried in a clone for Gemma-4 support. Three layered incompatibilities, each
fixable: standalone flash-attn compiled against torch 2.11; `flashinfer-cubin` 0.6.18 not on
PyPI (flashinfer-ai/flashinfer#4781; it is on `https://flashinfer.ai/whl`); FlashInfer JIT
needing ninja. After all three it served Qwen3-8B. Not adopted: the clone was patched in place
rather than built from a lock, and verl's own lock (main, 2026-09-15) still pins vLLM 0.24 —
verl's rollout worker calls vLLM internals, so an unpinned engine is an untested training stack.
The clone and the caches it wrote to `$HOME` were removed.


## 14. Answers below 1e-15 in SI are graded almost without tolerance (2026-10-05)

`engine._units_reconcile` converts both sides to SI and compares `|actual − expected| / max(|expected|, 1e-15)`.
When the expected value is below 1e-15 in SI the denominator is the floor, so the window becomes an absolute 5e-17 —
any answer of the right sign and a small magnitude passes, a shortcut included. Found while writing a template
whose key is a radar reflectivity factor in mm⁶ m⁻³ (≈ 1e-18 m³ in SI).

Scan of every key (core, traps, cross-domain, all templates; `si_magnitude` in `atmoscoder_rl/grading.py`):
**one** problem is affected — core `ry_6.1` (4.5e-8 cm³ min⁻¹ = 7.5e-16 m³ s⁻¹). Stored verdicts on that one problem
are lenient; no trap and no template is affected. `engine.py` is frozen and is not changed. Templates now fail
`check_templates` if a key falls below `SI_FLOOR` = 1e-14; `drop_size_truncation` uses a unit label the parser
does not read, so it is compared as a plain number.
