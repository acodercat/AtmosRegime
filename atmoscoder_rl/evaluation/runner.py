"""Run one model over a problem set and store the graded results.

    uv run --extra eval python -m atmoscoder_rl.evaluation.runner --model local-e1-off --input <problems.json> --exp-id <id>
    uv run --extra eval python -m atmoscoder_rl.evaluation.runner --model local-e1-on --set core --exp-id core_code --run 2

(On the training box, run it in the eval venv: README, "Model calls on the server".)

`--model` names a `models.toml` section. The problems are a named set (`--set`) or any JSON
list of `{id, problem, sub_answers}` (`--input`). Results stream to
`experiments/<exp-id>/<name>[.run<N>].json`, where `<name>` is `--out-name` or the model
section; re-running the same command resumes it (see `store.py`). `--run N` writes a separate
repetition and offsets the sampling seed so repetitions are independent.

Grading uses the benchmark's unit rule (`grading.verify(units="benchmark")`), recorded as `units` in the
run's metrics; the reward uses a stricter one (`grading.py`).
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from ..datasets import PROBLEM_SETS, index_by_id
from ..grading import TOLERANCE
from ..prompts import SYSTEM
from ..paths import EXPERIMENTS_DIR
from .client import build_model
from .episode import MAX_ATTEMPTS, evaluate
from .store import RunStore

def _problems(args: argparse.Namespace) -> list[dict[str, Any]]:
    problems = PROBLEM_SETS[args.set]() if args.set else json.loads(Path(args.input).read_text())
    if args.ids:
        by_id = index_by_id(problems)
        missing = [i for i in args.ids if i not in by_id]
        if missing:
            print(f"WARNING: {len(missing)} id(s) not in the problem set: {missing}")
        return [by_id[i] for i in dict.fromkeys(args.ids) if i in by_id]
    if args.sample:
        rng = random.Random(args.seed + (args.run or 0))
        return rng.sample(problems, min(args.sample, len(problems)))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="a models.toml section")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--set", choices=sorted(PROBLEM_SETS), help="a named problem set (datasets.PROBLEM_SETS)")
    source.add_argument("--input", help="a JSON list of {id, problem, sub_answers}")
    parser.add_argument("--exp-id", help="experiment id (default: <model>_<date>)")
    parser.add_argument("--out-name", help="run-file name instead of --model, so two endpoints "
                                           "of one logical model share a file and its resume")
    parser.add_argument("--run", type=int, help="repetition index: writes <name>.run<N>.json")
    parser.add_argument("--sample", type=int, help="evaluate a random sample of this many problems")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed for --sample")
    parser.add_argument("--ids", nargs="+", help="re-run exactly these ids and merge them into "
                                                 "the run file, replacing their old records")
    parser.add_argument("--no-skip", action="store_true", help="ignore the resume cache")
    parser.add_argument("--fresh", action="store_true",
                        help="refuse to start if the run file already exists (a pre-registered run "
                             "must not resume or overwrite anything)")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                        help="content tries per problem, including self-repairs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR,
                        help="the experiments directory (default: this repository's)")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    model = build_model(args.model)
    problems = _problems(args)
    if not problems:
        print("No problems to run.")
        return 1

    exp_id = args.exp_id or f"{args.model}_{datetime.now():%Y%m%d}"
    name = (args.out_name or args.model).replace("/", "--")
    out_dir = args.experiments / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"{name}.run{args.run}.json" if args.run is not None else f"{name}.json")
    if args.fresh and out_path.exists():
        print(f"refusing: {out_path} exists and --fresh was given")
        return 1
    meta: dict[str, Any] = {"model": args.out_name or args.model, "model_id": model.model_id,
                            "exp_id": exp_id, "mode": "code", "tolerance": args.tolerance,
                            "units": "benchmark", "system": SYSTEM}
    if args.run is not None:
        meta["run"] = args.run
    store = RunStore(out_path, meta, resume=not args.no_skip,
                     replace_ids={problem["id"] for problem in problems} if args.ids else None)

    remaining = [problem for problem in problems if problem["id"] not in store.done_ids]
    if not remaining:
        print(f"All {len(problems)} problems complete; --no-skip re-runs them.")
        return 0
    print(f"Model:    {args.model} ({model.model_id})")
    print(f"Exp ID:   {exp_id}{f' (run {args.run})' if args.run is not None else ''}")
    print(f"Problems: {len(remaining)} remaining / {len(problems)} total, {args.workers} workers")
    print(f"Output:   {out_path}\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, problem, model, args.tolerance, args.max_attempts)
                   for problem in remaining]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            store.add(record)
            status = ("PASS" if record["passed"] else
                      f"ERROR: {record['error'][:50]}" if record.get("error") else "FAIL")
            tries = f" (x{record['num_attempts']})" if record.get("num_attempts", 1) > 1 else ""
            print(f"[{done}/{len(remaining)}] {record['id']}... {status}{tries}")

    counts = store.counts()
    graded = counts["passed"] + counts["failed"]
    print(f"\n{'=' * 50}")
    print(f"Accuracy: {counts['passed']}/{graded} ({counts['passed'] / graded if graded else 0:.1%}) "
          f"[errors excluded: {counts['errors']}]")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
