"""The run file: one JSON per (experiment, config, repetition), written as results arrive.

Shape: `{"metrics": {...}, "results": [record, ...]}` (records as `episode.evaluate` returns
them). The store owns persistence so the runner does not have to:

- **Written per result.** Every `add` rewrites the file (temp file + `os.replace`, under a
  lock), so a killed run loses nothing and never leaves truncated JSON.
- **Resume is an unbiased continuation.** Reopening keeps every *completed* measurement —
  passes and gradable fails alike — and drops only `error` records, which are re-run together
  with any id missing from the file. Re-running only the fails would bias accuracy upward.
- **Selective replacement.** `replace_ids` drops exactly those ids so the caller re-runs them;
  the aggregate metrics are recomputed from the records on every write, so the file equals a
  clean run of the final id set.
- **Accuracy excludes errors**: `passed / (passed + failed)`.

`usage` (per record and summed in `metrics`) is what the provider reported: `completion_tokens`
*includes* reasoning and `total_tokens` = prompt + completion. The imported AtmosCoder-Bench runs
store a different quantity — an o200k recount of the stored text, with prompt, completion and
reasoning disjoint — so token counts from the two sources are not comparable, and a run file
resumed across the two definitions mixes them (docs/IMPORTED_RUNS.md).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .client import USAGE_KEYS



class RunStore:
    def __init__(self, path: Path, meta: dict[str, Any], resume: bool = True,
                 replace_ids: set[str] | None = None) -> None:
        self.path = Path(path)
        self.meta = dict(meta)
        self._lock = threading.Lock()
        self.results: list[dict[str, Any]] = []
        if not self.path.exists():
            return
        try:
            previous = json.loads(self.path.read_text()).get("results", [])
        except (json.JSONDecodeError, OSError):
            return
        if replace_ids is not None:
            self.results = [record for record in previous if record.get("id") not in replace_ids]
        elif resume:
            self.results = [record for record in previous if not record.get("error")]

    @property
    def done_ids(self) -> set[str]:
        return {record["id"] for record in self.results}

    def counts(self) -> dict[str, int]:
        # An `error` record is an API failure that never cleared; `episode.evaluate` writes it
        # with no attempts and no details, which is exactly what `runs.is_gradable` rejects.
        passed = sum(1 for record in self.results if record.get("passed"))
        errors = sum(1 for record in self.results if record.get("error"))
        return {"total": len(self.results), "passed": passed,
                "failed": len(self.results) - passed - errors, "errors": errors}

    def add(self, record: dict[str, Any]) -> dict[str, int]:
        """Append one record, rewrite the file, and return the live counts."""
        with self._lock:
            self.results.append(record)
            self._write()
            return self.counts()

    def _write(self) -> None:
        counts = self.counts()
        graded = counts["passed"] + counts["failed"]
        usage = {key: sum((record.get("usage") or {}).get(key, 0) for record in self.results)
                 for key in USAGE_KEYS}
        metrics = {**self.meta, "timestamp": datetime.now().isoformat(), **counts,
                   "accuracy": counts["passed"] / graded if graded else 0, "usage": usage}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"metrics": metrics, "results": self.results},
                                        indent=2, ensure_ascii=False))
        os.replace(temporary, self.path)
