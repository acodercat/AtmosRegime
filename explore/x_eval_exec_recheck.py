"""Endpoint-evaluation counterpart of the X-24 re-check: did the evaluation's own execution report false failures?

`explore/x_eval.sh` grades through `engine.run_solver` (a plain subprocess), not the RL sandbox, so X-24's
"a failure counts only if it fails again" does not cover it, and X-24's fault was machine-level. This re-executes,
twice, the code of every attempt the evaluation recorded as an execution failure ("ungradable: ..." other than
"no code in response") and counts how many now run. Real failures are deterministic (X-24: 1,659 / 1,659), so an
attempt that runs now is a candidate false failure. Execution only: nothing is compared with an answer, so no
accuracy is read.

    $ATMOS_ROOT/eval-venv/bin/python -m explore.x_eval_exec_recheck --root $ATMOS_ROOT/experiments/_x_eval \
        --out $ATMOS_ROOT/results_scratch/x_eval_exec_recheck.json [--workers 48]
"""
import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atmoscoder_rl.engine import extract_code, run_solver

FILES = ("final", "judge", "test_id", "textbook")


def kind(error: str) -> str:
    """The last non-empty line of an execution error, digits masked: the failure type, not its instance."""
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    last = lines[-1] if lines else error.strip()
    return "".join("#" if c.isdigit() else c for c in last)[:120]


def failures(root: Path):
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("dryrun")):
        for name in FILES:
            path = label_dir / f"{name}.json"
            for record in json.loads(path.read_text())["results"]:
                calls = record.get("attempts") or []
                for i, call in enumerate(calls):
                    outcome = call.get("outcome", "")
                    if outcome.startswith("ungradable:") and "no code in response" not in outcome:
                        yield dict(label=label_dir.name, file=name, id=record["id"], attempt=i,
                                   last=i == len(calls) - 1, error=outcome, code=extract_code(call.get("response", "")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    items = list(failures(args.root))
    with ThreadPoolExecutor(args.workers) as pool:
        reruns = list(pool.map(lambda item: [run_solver(item["code"])[0] for _ in range(2)], items))
    hosts = {p.name: json.loads((p / "serving.json").read_text())["host"] for p in args.root.iterdir()
             if (p / "serving.json").exists()}
    labels = collections.defaultdict(lambda: collections.Counter())
    kinds = collections.defaultdict(collections.Counter)
    ran = []
    for item, (first, second) in zip(items, reruns):
        c = labels[item["label"]]
        c["exec_failures"] += 1
        c["ran_both"] += first and second
        c["ran_one"] += first != second
        c["ran_both_on_last_attempt"] += first and second and item["last"]
        kinds[item["label"]][kind(item["error"])] += 1
        if first or second:
            ran.append({k: item[k] for k in ("label", "file", "id", "attempt", "last")} | dict(reruns=[first, second],
                       error=item["error"][:300]))
    report = dict(attempts_rechecked=len(items),
                  labels={label: dict(host=hosts.get(label, "?"), **c, top_kinds=kinds[label].most_common(5))
                          for label, c in sorted(labels.items())},
                  ran_now=ran)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for label, c in report["labels"].items():
        print(f"{label:28s} {c['host'][-3:]:4s} exec_failures {c['exec_failures']:4d} ran_both {c['ran_both']:3d} "
              f"ran_one {c['ran_one']:3d} on_last {c['ran_both_on_last_attempt']:3d}")
    print(f"total {len(items)} rechecked, {sum(r['reruns'] == [True, True] for r in ran)} run on both re-executions")


if __name__ == "__main__":
    main()
