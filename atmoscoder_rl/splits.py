"""The frozen train/test partition.

Built once, committed, and never regenerated with a different policy — every number
this project reports is conditioned on this partition, so a silent reshuffle would
invalidate the comparisons rather than improve them. `atmoscoder_rl.build_splits`
refuses to overwrite an existing `splits/splits.json` without `--force`.

Policy, in the order the exclusions are applied:

1. **Cross-domain** (131 problems, four non-atmospheric domains) is a separate corpus.
   It never enters training under any configuration.
2. **Held-out books** — *An Introduction to Atmospheric Physics* (Andrews, 28) and
   *Introduction to Atmospheric Chemistry* (Jacob, 28) are removed whole, giving a
   leave-book-out test on 56 problems spanning nine of the ten categories.

   Wallace & Hobbs (57) was the obvious first choice on size and coverage and is the wrong
   one: it is the easiest substantial book in the corpus (5% high-difficulty against 50-71%
   for these two), and every evaluated model already scores 92-98% on it, so it cannot
   register an improvement or a regression. Andrews and Jacob together are the same size,
   cover the same nine categories, sit at 41-84% for the models this project trains, and
   are the two books whose answers come from a published solutions manual rather than
   multi-model consensus — the cleanest ground truth available. They also cost 14 fewer
   variantable parents, so the training pool is larger than under the W&H holdout.
3. **Trap parents** (67) are removed. The trap probe asks whether training induces a
   general canonical-method bias; if the model had trained on a trap's parent, a trap
   result would confound that with memorisation of one problem. Removing the parents
   also keeps each trap's control baseline clean.
4. **Scaffolding probe** — every one of the 169 paired problems that does not end up in
   `train`. That is 77 pairs, and all of them are free: 41 were already removed by (2) or
   (3), 13 more are core-only and therefore never trainable, and 27 land in `test_id`.
   No pair is reserved at the training pool's expense, and the probe is still large enough
   to resolve the effect the benchmark paper measures on 169.

What remains (317 problems) is partitioned by `numeric_variantable`:

- **variantable** parents split ~75/25 into `train` and `test_id`, stratified by
  (difficulty, category) so the two are comparable. Only `train` supplies RL rollouts,
  because only a variantable parent can have its inputs re-sampled.
- **core-only** parents (perturbation would break them) cannot be trained on at all, so
  all of them become `test_core_only` — free held-out data.

`train` additionally carries nested subsets (`train_50`, `train_100`) for the
data-efficiency curve, with `train` itself as the top rung, so the 50-parent run is a
strict subset of the 100-parent run and scale is the only thing that changes.
"""

from __future__ import annotations

import random
from typing import Any

from .datasets import Problem, index_by_id, load_core, load_cross_domain, load_scaffolding, load_traps

SPLIT_SEED = 20260902
HELD_OUT_BOOKS = (
    "An Introduction to Atmospheric Physics",   # Andrews — official-answer GT
    "Introduction to Atmospheric Chemistry",    # Jacob   — official-answer GT
)
TEST_ID_FRACTION = 0.25
SCALING_LADDER = (50, 100)


def _stratum(problem: Problem) -> tuple[str, str]:
    return (problem.get("difficulty", "unknown"), problem.get("category", "unknown"))


def _stratified_sample(problems: list[Problem], fraction: float, rng: random.Random) -> set[str]:
    """Sample a `fraction` of ids, proportionally within each (difficulty, category) cell.

    Largest-remainder allocation: each stratum contributes floor(n*fraction) ids, then
    the leftover quota goes to the strata with the largest fractional parts. This keeps
    the sample size exact and stops small strata from being systematically dropped.
    """
    strata: dict[tuple[str, str], list[Problem]] = {}
    for problem in problems:
        strata.setdefault(_stratum(problem), []).append(problem)

    target = round(len(problems) * fraction)
    quotas: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    for key, members in strata.items():
        exact = len(members) * fraction
        quotas[key] = int(exact)
        remainders.append((exact - int(exact), key))

    shortfall = target - sum(quotas.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, key in remainders[:shortfall]:
        quotas[key] += 1

    sampled: set[str] = set()
    for key, members in sorted(strata.items()):
        chosen = rng.sample(sorted(members, key=lambda p: p["id"]), quotas[key])
        sampled.update(problem["id"] for problem in chosen)
    return sampled


def make_splits() -> dict[str, Any]:
    """Construct the partition. Deterministic given `SPLIT_SEED` and the vendored data."""
    rng = random.Random(SPLIT_SEED)

    core = load_core()
    by_id = index_by_id(core)
    trap_parents = {trap["parent_id"] for trap in load_traps()}
    scaffolding_ids = {problem["id"] for problem, _ in zip(*load_scaffolding())}
    book_ids = {problem["id"] for problem in core if problem["book"] in HELD_OUT_BOOKS}
    for book in HELD_OUT_BOOKS:
        if not any(problem["book"] == book for problem in core):
            raise ValueError(f"held-out book {book!r} matches no problem in core.json")

    missing = (trap_parents | scaffolding_ids) - set(by_id)
    if missing:
        raise ValueError(f"probe ids absent from core.json: {sorted(missing)}")

    excluded = trap_parents | book_ids
    remaining = [problem for problem in core if problem["id"] not in excluded]
    variantable = [problem for problem in remaining if problem["numeric_variantable"]]
    core_only = [problem for problem in remaining if not problem["numeric_variantable"]]

    test_id = _stratified_sample(variantable, TEST_ID_FRACTION, rng)
    train = [problem for problem in variantable if problem["id"] not in test_id]

    # Free by construction: a scaffolding pair is a probe unless training claims it.
    train_ids = {problem["id"] for problem in train}
    probe_scaffolding = scaffolding_ids - train_ids

    # Nested scaling subsets: shuffle once, then take prefixes.
    order = sorted(train, key=lambda p: p["id"])
    rng.shuffle(order)
    if max(SCALING_LADDER) >= len(order):
        raise ValueError(f"scaling ladder {SCALING_LADDER} must sit strictly below train ({len(order)})")
    ladder = {f"train_{size}": [problem["id"] for problem in order[:size]] for size in SCALING_LADDER}

    def entry(name: str, ids: list[str] | set[str], role: str, note: str) -> dict[str, Any]:
        return {"role": role, "note": note, "n": len(ids), "ids": sorted(ids)}

    splits = {
        "train": entry(
            "train", [p["id"] for p in train], "train",
            "variantable parents; each contributes itself plus its five stored numeric variants (1,140 training rows)",
        ),
        "test_id": entry(
            "test_id", test_id, "test",
            "held-out variantable parents, same strata as train — in-distribution test",
        ),
        "test_core_only": entry(
            "test_core_only", [p["id"] for p in core_only], "test",
            "parents that cannot be perturbed, hence never trainable — free held-out data",
        ),
        "test_book": entry(
            "test_book", book_ids, "test",
            "every problem from " + " and ".join(HELD_OUT_BOOKS) + " — leave-book-out",
        ),
        "probe_trap": entry(
            "probe_trap", trap_parents, "probe",
            "parents of the 67 traps; the traps themselves live in traps.json and are "
            "graded against these parents as controls",
        ),
        "probe_scaffolding": entry(
            "probe_scaffolding", probe_scaffolding, "probe",
            "scaffolding-ablation pairs held out of training — measured in both the "
            "original and stripped condition",
        ),
        "probe_cross_domain": entry(
            "probe_cross_domain", [p["id"] for p in load_cross_domain()], "probe",
            "131 non-atmospheric problems; out-of-domain transfer, never trained on",
        ),
    }
    for name, ids in ladder.items():
        splits[name] = entry(
            name, ids, "train_subset",
            f"nested prefix of train ({len(ids)} parents) for the data-efficiency curve",
        )

    return {
        "seed": SPLIT_SEED,
        "policy": {
            "held_out_books": list(HELD_OUT_BOOKS),
            "test_id_fraction": TEST_ID_FRACTION,
            "stratified_by": ["difficulty", "category"],
            "split_unit": "parent problem id",
            "scaling_ladder": list(SCALING_LADDER),
        },
        "splits": splits,
    }


def check_splits(splits: dict[str, Any]) -> list[str]:
    """Invariants that must hold before any training runs. Returns failure messages."""
    failures: list[str] = []
    entries = splits["splits"]
    core_ids = {problem["id"] for problem in load_core()}

    trainable = set(entries["train"]["ids"])
    for name in ("test_id", "test_core_only", "test_book", "probe_trap", "probe_scaffolding"):
        overlap = trainable & set(entries[name]["ids"])
        if overlap:
            failures.append(f"train overlaps {name}: {sorted(overlap)[:5]}")

    atmospheric = [n for n in entries if n != "probe_cross_domain"]
    covered: set[str] = set()
    for name in ("train", "test_id", "test_core_only", "test_book", "probe_trap"):
        covered |= set(entries[name]["ids"])
    if covered != core_ids:
        failures.append(
            f"train+test+trap-parents cover {len(covered)} of {len(core_ids)} core problems "
            f"(missing {len(core_ids - covered)}, extra {len(covered - core_ids)})"
        )

    for name in atmospheric:
        unknown = set(entries[name]["ids"]) - core_ids
        if unknown:
            failures.append(f"{name} contains non-core ids: {sorted(unknown)[:5]}")

    scaffolding_ids = {problem["id"] for problem, _ in zip(*load_scaffolding())}
    if set(entries["probe_scaffolding"]["ids"]) != scaffolding_ids - trainable:
        failures.append("probe_scaffolding is not exactly the scaffolding pairs outside train")

    for size in SCALING_LADDER:
        key = f"train_{size}"
        if key not in entries:
            failures.append(f"{key} missing")
            continue
        subset = set(entries[key]["ids"])
        if not subset <= trainable:
            failures.append(f"{key} is not a subset of train")
        if len(subset) != size:
            failures.append(f"{key} has {len(subset)} ids, expected {size}")
    for smaller, larger in zip(SCALING_LADDER, SCALING_LADDER[1:]):
        a, b = f"train_{smaller}", f"train_{larger}"
        if a in entries and b in entries and not set(entries[a]["ids"]) <= set(entries[b]["ids"]):
            failures.append(f"{a} is not nested inside {b}")

    return failures
