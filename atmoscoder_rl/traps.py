"""Which traps exist, which experiment holds each one's runs, and the analysis arms.

Everything that selects traps goes through here, so an arm means the same ids in every
report, in the training export and in the explore scripts. Depends only on `datasets`.
"""

from __future__ import annotations

from .datasets import Problem, load_extended_traps, load_regime_twins, load_traps, split_ids

#: Every experiment a trap record is measured under, one per corpus file: the 67
#: pre-expansion traps under `trap`, everything with an `expansion` tag under `trap_expansion`,
#: the regime twins and the cross-domain traps under their own ids. (The imported API-model
#: sweeps wrote every trap into `trap`.)
TRAP_EXPERIMENTS = ("trap", "trap_expansion", "trap_regime_twins", "trap_extended")
#: The atmospheric ones: every trap experiment but the cross-domain file.
ATMOSPHERIC_TRAP_EXPERIMENTS = TRAP_EXPERIMENTS[:3]

#: The three atmospheric arms every paired comparison reports: the claim, its internal
#: control (replacement handed over) and the family control.
ARMS = ("regime", "regime_given", "geometry")

TrapIndex = dict[str, tuple[Problem, str]]


def trap_index() -> TrapIndex:
    """Every trap record, paired with the experiment id that holds its runs.

    A function that reads only the first two files silently reports on a subset — which is
    how the trap tables were wrong before this existed.
    """
    out: TrapIndex = {}
    for t in load_traps():
        out[t["id"]] = (t, "trap_expansion" if t.get("expansion") else "trap")
    for t in load_regime_twins():
        out[t["id"]] = (t, "trap_regime_twins")
    for t in load_extended_traps():
        out[t["id"]] = (t, "trap_extended")
    return out


def adjudication(trap: Problem) -> dict:
    """The upstream reviewers' verdicts on one trap (`measures_regime_judgement`, `magnitude_defect`)."""
    return trap.get("review") or {}


def measures_recognition(trap: Problem) -> bool:
    """The trap leaves the replacement relation unstated, so it tests whether the solver notices."""
    return adjudication(trap).get("measures_regime_judgement") is True


def states_replacement(trap: Problem) -> bool:
    """The trap writes the replacement relation into its own text (the internal control)."""
    return adjudication(trap).get("measures_regime_judgement") is False


def sound(trap: Problem) -> bool:
    """The trap does not assert a quantity its referent cannot have."""
    return not adjudication(trap).get("magnitude_defect")


def family(trap: Problem) -> str:
    """The trap's family, with regime traps split the way the arms split them."""
    if trap["trap_type"] != "regime_boundary":
        return trap["trap_type"]
    if measures_recognition(trap):
        return "regime:recognise"
    return "regime:given" if states_replacement(trap) else "regime:unjudged"


def arm_ids(kind: str, index: TrapIndex, train: set[str]) -> list[str]:
    """The ids of one analysis arm, under the adjudication the corpus now carries.

    `regime` is the claim's arm and needs all three filters at once:
      - the parent is outside `train`, or the parent's own accuracy is a training artefact;
      - `measures_recognition`, or the trap writes the replacement relation into its own text
        and cannot test whether the model knows the old one failed;
      - `sound`, i.e. the trap does not assert a quantity its referent cannot have.
    `regime_given` is the same family with the second filter inverted — the internal control
    that holds physics fixed and varies only the recognition step.
    """
    def held_out(t: Problem) -> bool:
        return t["parent_id"] not in train

    def atmospheric(experiment: str) -> bool:
        return experiment in ATMOSPHERIC_TRAP_EXPERIMENTS

    regime = lambda t: t["trap_type"] == "regime_boundary"                  # noqa: E731
    tests = {
        "regime":       lambda t, e: regime(t) and atmospheric(e) and held_out(t) and sound(t) and measures_recognition(t),
        "regime_given": lambda t, e: regime(t) and atmospheric(e) and held_out(t) and sound(t) and states_replacement(t),
        "regime_xdom":  lambda t, e: e == "trap_extended" and held_out(t) and sound(t) and measures_recognition(t),
        "geometry":     lambda t, e: t["trap_type"] == "geometry_detail" and held_out(t) and sound(t),
    }
    if kind not in tests:
        raise ValueError(f"unknown arm {kind!r}; have {sorted(tests)}")
    return sorted(i for i, (t, e) in index.items() if tests[kind](t, e))


def targeted_regime_ids() -> list[str]:
    """The regime traps eligible as *training* material: the mirror image of the `regime` arm.

    Same two adjudication filters the arm uses, and the parent filter inverted — parent inside
    `train`, so adding these to the training table leaves the frozen split untouched. Disjoint
    from `regime`, `regime_given` and `geometry` by construction, since each of those requires
    the parent to be outside `train`. Pre-registered in `docs/PREREG_TARGETED_REGIME.md`; used
    by `export_verl --targeted-regime`.
    """
    train = set(split_ids("train"))
    return sorted(i for i, (t, e) in trap_index().items()
                  if t["trap_type"] == "regime_boundary" and e in ATMOSPHERIC_TRAP_EXPERIMENTS
                  and t["parent_id"] in train and measures_recognition(t) and sound(t))
