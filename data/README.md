# Data

One flat directory, one copy of every record. What each file is, and what it is for here.

## The corpus — what training and reward run on

`load_problems()` returns **783 records**, assembled in memory from the three files below
every time it is loaded. There is no built corpus file to keep in step, and no record is
stored twice.

| file | records | role in this project |
|---|--:|---|
| `core.json` | 436 | the parent problems. **The frozen partition in `splits/` is defined over these ids, in this order** — every split, baseline and result here rests on that, so the file is never reordered or edited |
| `traps.json` | 335 | single-trigger perturbations of core problems. Each one's control is its untouched `parent_id`, which is a record in `core.json`; the difference in accuracy between them is the Trap Gap |
| `unseen_book.json` | 12 | a 14th textbook that no `core.json` problem comes from. In no split and not trainable (`numeric_variantable` is forced `False`), so it is held out by construction rather than by policy — a stricter transfer probe than `test_book` |

Assembly adds exactly two fields, `role` and `frozen_split_member`, and refuses to run if a
source record already carries either. Three assertions run on every load: unique ids across
the sources, every trap's parent is one of the 436, and at most one trap per parent. Run
`uv run python -m atmoscoder_rl.check_corpus` to see the composition and exercise them.

## Training signal

| file | records | role |
|---|--:|---|
| `variants_numeric.json` | 1730 | 5 per perturbable parent, inputs re-samplable — the rollout pool. A variant leaks its parent's method, so parent and variants always land on the same side of a split boundary |
| `variants_paraphrase.json` | 2180 | same values, reworded text — linguistic robustness |

## Probes — evaluation only, never trained on

| file | records | what it measures |
|---|--:|---|
| `traps_regime_twins.json` | 31 | `regime_boundary` traps for parents whose corpus record is the `geometry_detail` twin. **Never merge into the corpus** — those parents already have a trap there and `check_corpus` asserts one per parent. Use within a single family only |
| `traps_extended.json` | 18 | `regime_boundary` traps whose parents are *not* in `core.json`, so they have no control here. Its domain mix (14 of 18 soil) is an upstream artefact of a generator transport failure, not a finding |
| `cross_domain/*.json` | 131 | hydrology, soil, environmental chemistry, ecology — out-of-domain transfer |
| `scaffolding_ablation/{original,stripped}.json` | 169 each | accuracy with vs. without handed-over knowledge. `verdicts.json` (197) is the adjudication behind the pairing |

## The other two files

`PROVENANCE.md` records where every file came from and every time one was replaced —
that is metadata about the data, not a property of the layout, which is why nothing in
this directory is arranged by origin.

`DATASET_CARD.md` is the dataset's own card as published with the AtmosCoder-Bench release:
category and difficulty counts, construction and certification. It describes the benchmark,
not this project's use of it — this file does that.

## Authored here (`authored/`)

Everything else in this directory is vendored from AtmosCoder-Bench. `authored/` holds problems
this project writes itself (new problems are authored here; the vendored files are not edited):

- `authored/regime_given_versions.json` (2026-09-28) — for each of the 46 regime-arm traps, the same
  text with one inserted sentence stating the replacement relation the trap's own solver computes;
  same key, same solver. Built to control for the recognition step on the same items
  (`docs/PREREG_GIVEN_VERSIONS.md`). Each record carries the author's recomputation, any problem
  the author flagged with the trap itself, and a blind-solve check (45 / 46 match; the other is an
  ambiguity of the parent text, noted in the record).

## Changing a file

Replace it, run `check_corpus` (composition and assertions) and `check_solvers` (every solver still
reproduces its answer), confirm `build_splits --check` still passes, then record the new sha256 and
what moved in `PROVENANCE.md`. Answers and solvers are never hand-edited here.
