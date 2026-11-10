"""Stored runs: where run files live, how they are read back, and which records count.

A run file is `experiments/<exp-id>/<config>[.run<N>].json` with shape
`{"metrics": {...}, "results": [record, ...]}`, written by `atmoscoder_rl.evaluation.runner`
(and, for the imported runs, by AtmosCoder-Bench's runner, which writes the same shape).
`experiments/` is gitignored: results are artifacts, not source.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .grading import TOLERANCE, verify


Record = dict[str, Any]
Rep = dict[str, Record]          # one run file: problem id -> record

_RUN_FILE = re.compile(r"^(?P<config>.+?)(?:\.run(?P<run>\d+))?\.json$")


def _parse(path: Path) -> re.Match[str]:
    match = _RUN_FILE.match(Path(path).name)
    if not match:
        raise ValueError(f"not a run file: {path}")
    return match


def config_name(path: Path) -> str:
    """The configuration a run file belongs to: `qwen3-8b.think-on.run2.json` -> `qwen3-8b.think-on`."""
    return _parse(path).group("config")


def rep_index(path: Path) -> int | None:
    """The repetition number of a run file (`.run2.json` -> 2), or None for an un-numbered one."""
    run = _parse(path).group("run")
    return int(run) if run is not None else None


def _read_rep(path: Path) -> Rep:
    return {record["id"]: record for record in json.loads(Path(path).read_text())["results"]}


def load_reps(experiments: Path, exp_id: str, pattern: str = "*.json") -> list[Rep]:
    """Every run file of `exp_id` matching `pattern`, in file-name order."""
    return [_read_rep(path) for path in sorted((Path(experiments) / exp_id).glob(pattern))]


def load_runs(experiments: Path, exp_id: str) -> dict[str, list[Rep]]:
    """Every configuration under `exp_id`, each with its repetitions in file-name order."""
    runs: dict[str, list[Rep]] = defaultdict(list)
    for path in sorted((Path(experiments) / exp_id).glob("*.json")):
        runs[config_name(path)].append(_read_rep(path))
    return dict(runs)


def load_numbered_reps(experiments: Path, exp_id: str) -> dict[str, dict[int, Rep]]:
    """Every configuration under `exp_id`, its numbered repetitions keyed by repetition number —
    for pairing repetition k of one experiment with repetition k of another. Un-numbered run
    files are left out."""
    runs: dict[str, dict[int, Rep]] = defaultdict(dict)
    for path in (Path(experiments) / exp_id).glob("*.run*.json"):
        index = rep_index(path)
        if index is not None:
            runs[config_name(path)][index] = _read_rep(path)
    return dict(runs)


def is_gradable(record: Record | None) -> bool:
    """False for an infrastructure error, which is not a wrong answer.

    The runner stores an unreachable-endpoint or never-returned record with
    `num_attempts: 0`, no `attempts` and an `error` string, and excludes it from its own
    accuracy denominator. Re-grading such a record yields `passed` = False, so counting it as
    a failure invents a wrong answer the model never gave: the trap back-fill was still
    running when several configurations were swept, and treating its gaps as failures put one
    model at 0/197 and another at 26 %, against 57 % and 91 % on the same models' finished
    records. Every caller that uses a verdict must filter on this first.
    """
    if record is None or "passed" not in record:
        return False
    if record.get("error") and not (record.get("attempts") or record.get("details")):
        return False
    return not (record.get("num_attempts") == 0 and not record.get("details"))


def regrade(record: Record, subs: list[dict] | None) -> bool:
    """Re-derive one verdict from the record's stored `details` under the benchmark unit rule,
    rather than trusting its stored `passed` — a grading correction (units restored to a
    problem file, a tolerance) then needs no re-run. Falls back to the stored verdict only
    when there is nothing to re-grade (no details: an ungradable response is a real fail).

    The two verdicts a record can yield: this one (`analysis`) and the stored `passed`
    (`baselines`, which reports the grading in force when each run was made, and the explore
    reports, whose runs are graded by this repository's current rule anyway)."""
    details = record.get("details") or []
    if not details or subs is None:
        return bool(record.get("passed"))
    if any("sub" not in d for d in details):
        # An error marker rather than a per-sub comparison, e.g. {"error": "empty results"}
        # when the solver returned no keys. There is nothing to re-grade and the stored
        # verdict is already a fail. One such record exists across 20,422 details.
        return bool(record.get("passed"))
    actual = {
        str(d["sub"]): {"value": d.get("actual"), "unit": d.get("actual_unit") or ""}
        for d in details
    }
    if any(v["value"] is None for v in actual.values()):
        return False
    try:
        ok, _ = verify(actual, subs, TOLERANCE, units="benchmark")
    except (TypeError, ValueError, KeyError):    # a malformed stored detail; never seen in 30,004
        return bool(record.get("passed"))
    return bool(ok)
