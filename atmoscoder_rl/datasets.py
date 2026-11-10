"""Loading this repository's dataset and the frozen splits.

`data/` is this repository's dataset: one flat directory, one copy of every record, and
`data/README.md` says what each file is. Where the files came from is recorded in
`data/PROVENANCE.md` — that is metadata about the data, not a property of the layout.

`load_problems()` returns the corpus the splits, the reward and every experiment here are
defined over. It is **assembled in memory** from `core.json`, `unseen_book.json` and
`traps.json` rather than read from a built file, so no record is ever stored twice and there
is no artefact that can silently go stale. The invariants that make the corpus usable are
re-checked on every load and raise `CorpusError`; `python -m atmoscoder_rl.check_corpus`
prints what the corpus is made of. Nothing here mutates a problem: loaders return fresh
lists so a caller can filter and reorder freely.

The unit that splits are defined over is the **parent problem id**, never the
variant id — a numeric variant leaks its parent's method, so parent and variants
must always land on the same side of a split boundary.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

from .paths import DATA_DIR, SPLITS_PATH

Problem = dict[str, Any]

CROSS_DOMAINS = ("hydrology", "soil", "environmental_chemistry", "ecology")


class CorpusError(Exception):
    """A corpus invariant was violated. Never caught here: the data must be fixed."""


class CorpusSource(NamedTuple):
    """One file the corpus is assembled from, and how this project may use its records."""

    file: str
    role: str                     # "core" or "trap"
    frozen_split_member: bool     # inside the partition `splits/` is defined over?
    forced: dict[str, Any]        # fields set on every record of this source


# Order is load-bearing: `load_core()` must return the 436 frozen ids in this order, which is
# what the frozen partition in `splits/` is defined over.
CORPUS_SOURCES = (
    CorpusSource("core.json", "core", True, {}),
    CorpusSource("unseen_book.json", "core", False, {"numeric_variantable": False}),
    CorpusSource("traps.json", "trap", False, {}),
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=None)
def _cached(path: Path) -> str:
    """Cache raw file text, not parsed objects — callers get an independent copy."""
    return path.read_text(encoding="utf-8")


def _load(name: str) -> list[Problem]:
    """One data file, by its name under `data/`."""
    return json.loads(_cached(DATA_DIR / name))


def _assemble_corpus() -> list[Problem]:
    """The three sources merged, with the two fields that say how this project may use a record.

    The invariants below are the contract, and they are the reason this is assembled rather
    than hand-maintained: unique ids, every trap's parent a frozen core problem, one trap per
    parent. Merging without them would silently repartition training or destroy a trap's
    control, so a violation raises `CorpusError` rather than asserting — `assert` is stripped
    under `python -O` and these must never be skipped.
    """
    records: list[Problem] = []
    for source in CORPUS_SOURCES:
        for rec in _load(source.file):
            for key in ("role", "frozen_split_member", *source.forced):
                if key in rec:
                    raise CorpusError(
                        f"{source.file}:{rec.get('id')} already carries {key!r}; refusing to overwrite")
            records.append({**rec, "role": source.role,
                            "frozen_split_member": source.frozen_split_member, **source.forced})

    dupes = [i for i, n in Counter(r["id"] for r in records).items() if n > 1]
    if dupes:
        raise CorpusError(f"duplicate ids across sources: {dupes[:5]}")
    core_ids = {r["id"] for r in records if r["role"] == "core" and r["frozen_split_member"]}
    orphans = [r["id"] for r in records if r["role"] == "trap" and r["parent_id"] not in core_ids]
    if orphans:
        raise CorpusError(f"traps whose parent is not a frozen core problem: {orphans[:5]}")
    parents = Counter(r["parent_id"] for r in records if r["role"] == "trap")
    multi = [p for p, n in parents.items() if n > 1]
    if multi:
        raise CorpusError(f"more than one trap per parent: {multi[:5]}")
    return records


def load_problems() -> list[Problem]:
    """Every problem in one list: the `role="core"` records and the `role="trap"` ones.

    Assembled from `CORPUS_SOURCES` on every call — the file reads behind it are cached, the
    assembly is not, so the invariants are re-checked rather than trusted from an earlier
    call. The subsets below are views on it, kept because two of them are load-bearing:
    `load_core()` must return exactly the 436 ids the frozen partition is defined over, and a
    trap must stay distinguishable from the parent it perturbs, or the probe measures nothing.
    """
    return _assemble_corpus()


def load_core(include_unsplit: bool = False) -> list[Problem]:
    """The core problems.

    By default the 436 the frozen partition covers. With `include_unsplit=True` also the 12
    from the textbook onboarded after the release (`frozen_split_member=False`): never in any
    split, never trainable, an unseen-book probe. They are excluded by default so that
    anything indexing core against `splits.json` cannot silently pick up ids it has no split
    for.
    """
    return [p for p in load_problems()
            if p["role"] == "core" and (include_unsplit or p["frozen_split_member"])]


def load_numeric_variants() -> list[Problem]:
    """The 1730 parameter-perturbed variants (5 per perturbable parent)."""
    return _load("variants_numeric.json")


def load_paraphrase_variants() -> list[Problem]:
    """The 2180 reworded variants (5 per parent; values and answers unchanged)."""
    return _load("variants_paraphrase.json")


def load_traps() -> list[Problem]:
    """Every trap: one baited shortcut each, its control the untouched `parent_id` in core.

    **Not all of them are usable for a headline number here.** A trap whose parent is inside
    the frozen `train` split measures a problem the model was trained on, so the two arms are
    not comparable. Filter on that before aggregating — `check_corpus` prints the current
    split, and `data/PROVENANCE.md` records why each pass landed where it did. The counts are
    deliberately not repeated here: they move every time upstream is re-vendored, and a number
    frozen into a docstring is a number that will be wrong."""
    return [p for p in load_problems() if p["role"] == "trap"]


def load_regime_twins() -> list[Problem]:
    """The 31 certified `regime_boundary` traps whose parent already carries a
    `geometry_detail` trap in `load_traps()`. Same schema; `twin_of` names the geometry record. **Per-family use only**: within regime_boundary each parent still has
    exactly one trap and the parent is still the control, but pooling these with
    `load_traps()` counts a parent twice. As with `load_traps()`, a twin whose parent is
    inside `train` is not usable for a headline number; how many that leaves is a property of
    the frozen split, not of this file, so it is not repeated here."""
    return _load("traps_regime_twins.json")


def load_extended_traps() -> list[Problem]:
    """Certified `regime_boundary` traps whose parents are *outside* core (cross-domain), each
    measured against its own parent's cross-domain run. Never pooled with `load_traps()`."""
    return _load("traps_extended.json") if (DATA_DIR / "traps_extended.json").exists() else []


def load_scaffolding() -> tuple[list[Problem], list[Problem]]:
    """The 169 paired problems, as (original, stripped) aligned by id."""
    original = _load("scaffolding_ablation/original.json")
    stripped = _load("scaffolding_ablation/stripped.json")
    by_id = {problem["id"]: problem for problem in stripped}
    return original, [by_id[problem["id"]] for problem in original]


def load_cross_domain(domain: str | None = None) -> list[Problem]:
    """The 131 non-atmospheric problems, all four domains or just one."""
    domains = (domain,) if domain else CROSS_DOMAINS
    problems: list[Problem] = []
    for name in domains:
        if name not in CROSS_DOMAINS:
            raise ValueError(f"unknown domain {name!r}; expected one of {CROSS_DOMAINS}")
        problems.extend(_load(f"cross_domain/{name}.json"))
    return problems


def index_by_id(problems: Iterable[Problem]) -> dict[str, Problem]:
    return {problem["id"]: problem for problem in problems}


def group_by_parent(variants: list[Problem]) -> dict[str, list[Problem]]:
    """Group variants under their `parent_id`, preserving file order within a parent."""
    grouped: dict[str, list[Problem]] = {}
    for variant in variants:
        grouped.setdefault(variant["parent_id"], []).append(variant)
    return grouped


def load_splits() -> dict[str, Any]:
    """The frozen splits. Build them with `atmoscoder_rl.build_splits` if missing."""
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"{SPLITS_PATH} not found — run `uv run python -m atmoscoder_rl.build_splits` first"
        )
    return _read_json(SPLITS_PATH)


def split_ids(name: str) -> list[str]:
    """Parent ids belonging to one split (see `splits/MANIFEST.md` for the roster)."""
    splits = load_splits()["splits"]
    if name not in splits:
        raise KeyError(f"unknown split {name!r}; have {sorted(splits)}")
    return list(splits[name]["ids"])


def train_families() -> list[tuple[Problem, list[Problem]]]:
    """The training material: each `train` parent with its numeric variants, in split order.

    The one definition of what RL, SFT and RAFT train on (`export_verl`, `export_sft`,
    `sample_raft`). A `train` id missing from core is a broken split and raises.
    """
    core = index_by_id(load_core())
    variants = group_by_parent(load_numeric_variants())
    return [(core[pid], variants.get(pid, [])) for pid in split_ids("train")]


def _unseen_book() -> list[Problem]:
    return [p for p in load_core(include_unsplit=True) if not p["frozen_split_member"]]


#: Every problem file under a name, for the CLIs that take `--set` (`check_solvers`, the
#: evaluation runner) and for `check_reward`'s gold-solver pass. `core` is the 436 the frozen
#: partition covers; the unseen book is its own set. The scaffolding halves share ids by
#: design (169 aligned pairs), and the regime twins' parents already carry a trap in `traps`,
#: so each is its own set rather than pooled.
PROBLEM_SETS: dict[str, Callable[[], list[Problem]]] = {
    "problems": load_problems,
    "core": load_core,
    "unseen_book": _unseen_book,
    "traps": load_traps,
    "traps_regime_twins": load_regime_twins,
    "traps_extended": load_extended_traps,
    "variants_numeric": load_numeric_variants,
    "variants_paraphrase": load_paraphrase_variants,
    "cross_domain": load_cross_domain,
    "scaffolding_original": lambda: load_scaffolding()[0],
    "scaffolding_stripped": lambda: load_scaffolding()[1],
}
