"""D-5 textbook replication report (docs/PREREG_MAIN.md revisions D-5 §4, D-5a): D5-1 and the secondary numbers.

    python -m explore.d5_report --input experiments/_inputs/d5/textbook.json \
        --run-dir $ROOT/experiments/families_d5/families_context_v2 \
        [--d-input experiments/_inputs/families_v4/textbook.json --d-run-dir DIR ...] --out results/d5/report.json

A configuration counts when its three runs (<config>.textbook.run{1,2,3}.json) hold a gradable record for every input id;
the others are listed as incomplete and left out of every number (D-5 §4). Per trap: parent right - trap right, averaged
over the three runs (`d2_report.textbook_gaps`, as S4); pooled over configurations per trap; regime:recognise minus
geometry_detail, each family resampled independently by trap, 10,000 draws, `random.Random("d5-pool:20261013")`, a fresh
generator for every interval (`d2_report.pooled`). With --d-run-dir, D's own textbook gaps of the same configurations are
pooled with the new ones (old + new, described as decided after the old result was known).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.runs import is_gradable
from explore.d2_report import find, load, pooled, textbook_gaps

SEED = "d5-pool:20261013"
CONFIGS = ("qwen3-8b.think-off", "qwen3-8b.think-on", "qwen3.5-9b.think-off", "qwen3.5-9b.think-on",
           "qwen3.5-27b.think-off", "qwen3.5-27b.think-on", "gemma4-e4b.think-off", "gemma4-e4b.think-on",
           "gemma4-31b.think-off", "gemma4-31b.think-on", "granite4.2-8b.think-off", "granite4.2-8b.think-on",
           "granite4.2-30b.think-off", "granite4.2-30b.think-on", "ministral3-8b.think-off", "ministral3-14b.think-off")
REPLICATES = "在独立陷阱上重复出现"
NUMBERS = "只报数字"


def runs(run_dirs: list[Path], config: str, ids: set[str]) -> list[dict] | None:
    """The three repetitions of a configuration, or None when one is missing or lacks a gradable record for an id."""
    reps = []
    for r in (1, 2, 3):
        try:
            rep = load(find(run_dirs, f"{config}.textbook.run{r}.json"))
        except SystemExit:
            return None
        if not all(is_gradable(rep.get(i)) for i in ids):
            return None
        reps.append(rep)
    return reps


def report(items: list[dict], run_dirs: list[Path], d_items: list[dict] | None, d_run_dirs: list[Path] | None) -> dict:
    ids = {r["id"] for r in items}
    gaps, incomplete = {}, []
    for c in CONFIGS:
        reps = runs(run_dirs, c, ids)
        if reps is None:
            incomplete.append(c)
        else:
            gaps[c] = textbook_gaps(items, reps)
    done = [c for c in CONFIGS if c in gaps]
    main = pooled([gaps[c] for c in done], SEED)
    verdict = (REPLICATES if main and main["ci95"][0] > 0 and main["mean"] > 0 else NUMBERS)
    out = {"D5-1": main, "verdict": verdict, "configurations": done, "incomplete": incomplete,
           "note": "主结果只在完成的配置上计算" if incomplete else None,
           "per_configuration": {c: pooled([gaps[c]], SEED) for c in done},
           "by_mode": {m: pooled([gaps[c] for c in done if c.endswith(m)], SEED) for m in (".think-off", ".think-on")},
           # descriptive: each family's own parent - trap gap behind D5-1 (trap mean of the configuration mean)
           "family_gaps": {f: fmean(fmean(gaps[c][f][i] for c in done) for i in gaps[done[0]][f])
                           for f in ("regime:recognise", "geometry_detail")} if done else None}
    if d_items is not None and d_run_dirs:
        d_ids = {r["id"] for r in d_items}
        merged, skipped = [], []
        for c in done:
            reps = runs(d_run_dirs, c, d_ids)
            if reps is None:   # missing, duplicated across --d-run-dir, or not full: listed, never dropped silently
                skipped.append(c)
                continue
            old = textbook_gaps(d_items, reps)
            merged.append({f: {**old[f], **gaps[c][f]} for f in ("regime:recognise", "geometry_detail")})
        out["old_plus_new (事后决定：加题时已知旧结果)"] = pooled(merged, SEED)
        out["old_plus_new_skipped"] = skipped
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, action="append", required=True)
    ap.add_argument("--d-input", type=Path, help="D's textbook input (families_v4/textbook.json), for old + new")
    ap.add_argument("--d-run-dir", type=Path, action="append", help="directories holding D's textbook runs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    items = json.loads(args.input.read_text())
    d_items = json.loads(args.d_input.read_text()) if args.d_input else None
    r = report(items, args.run_dir, d_items, args.d_run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    m = r["D5-1"]
    print(f"D5-1 over {len(r['configurations'])} configurations: "
          + ("-" if m is None else f"{100 * m['mean']:.1f} [{100 * m['ci95'][0]:.1f}, {100 * m['ci95'][1]:.1f}] (traps {m['units']})")
          + f" -> {r['verdict']}; incomplete: {r['incomplete'] or 'none'}"
          + (f"; old + new skipped: {r['old_plus_new_skipped'] or 'none'}" if "old_plus_new_skipped" in r else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
