# Stored runs imported from AtmosCoder-Bench (2026-09-26)

`experiments/` is gitignored: run results are artifacts, not source. To make this repository
self-contained, the API-model runs that `baselines`, `check_reward --replay` and
`explore/family_report.py` read were copied from the AtmosCoder-Bench working copy on
2026-09-26. They are the stored outputs of the benchmark's own sweeps; nothing here re-ran them.

| experiment id | run files | fingerprint (sha256 over sorted file hashes, first 16) |
|---|--:|---|
| `core_code` | 54 | `e0847002eea1675f` |
| `trap` | 27 | `6b13f676d85a7a15` |
| `scaffolding_ablation` | 24 | `2d40664db74e8a0b` |
| `cross_domain` | 20 | `b86aaab3a27f5222` |

Recompute a fingerprint with
`find experiments/<id> -type f -name '*.json' -print0 | sort -z | xargs -0 sha256sum | sha256sum`.

**Checked at import.** `atmoscoder_rl.baselines` over the imported copy is identical to the same
module over the Bench working copy. It is *not* identical to the committed
`splits/baselines.json`, which was computed on 2026-09-02: since then the Bench trap runs grew
(the expansion traps were back-filled) and cross-domain runs were re-graded, which moves
`trap_gap` and `cross_domain`. `baselines.json` is left as its dated snapshot; regenerate it
deliberately if a current number is needed.

**Token counts are not comparable across the two runners.** These imported files store `usage`
as an o200k recount of the stored text (prompt, completion and reasoning disjoint). This
repository's runner stores what the provider reported (completion includes reasoning; see
`evaluation/store.py`). Do not pool token statistics across the two, and do not resume an
imported file with the new runner.

**The training box's runs (done 2026-09-27, on host-D).**
Everything the Bench-era runner wrote there for this project, under
`/path/to/AtmosCoder-Bench/experiments/` (16 experiment directories: the stage-1
`explore_*` runs and the RL checkpoints' and controls' runs inside `core_code/`, `trap*/`,
`scaffolding_ablation/`, `cross_domain/`, ...), was **copied** — not moved; the Bench checkout is
untouched — to `$ATMOS_ROOT/experiments/` (405 files, every sha256 verified against the source).
On the server this repository's `experiments` is a symlink to that directory (the home disk is
small), so `analysis`, the explore reports and `explore/e1_run.sh` read and write it in place.
