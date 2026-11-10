"""PREREG_OPD §4.3 / §3: the main test set Z ∩ "privileged teacher knows" (OPD-6; should have been recorded in OPD-1).

    python -m explore.opd_main_set --kc experiments/_inputs/kc --samples experiments/_x/qwen3-8b/warm_samples_F1.jsonl \
        experiments/_x/qwen3-8b/warm_samples_F2.jsonl --report results/x/report_qwen3-8b.json --out results/opd/main_set.json

"Knows" (§3): the base model's accuracy on the template's `given` rows, read from the archived warm-start samples of W_f
(explore.x_data warm-sample: every given row of the fold, 8 completions, T 1.0, no hint in the prompt), graded at 2 % as in X
(atmoscoder_rl.reward.grade, strict mode, the row's own sub_answers). Template accuracy = mean over its given rows of the
share of the 8 completions that pass; main set = Z (P1 of the X report, base pass@64 = 0 outside) with given accuracy >= 0.5.
Also reported, not used for the set: pass@8 per template (any of 8 right, averaged over rows). The samples are the
archive's bytes (sha256 recorded); nothing is sampled anew.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.reward import grade

TOL, KNOWS = 0.02, 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kc", type=Path, required=True)
    ap.add_argument("--samples", type=Path, nargs="+", required=True)
    ap.add_argument("--report", type=Path, required=True, help="results/x/report_<family>.json (P1.Z)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows = {}
    for fold in ("F1", "F2"):
        for r in json.loads((args.kc / f"train_{fold}.json").read_text()):
            rows[r["id"]] = r
    recs, shas = [], {}
    for path in args.samples:
        shas[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        recs += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert all(r["arm"] == "G" and r["id"] in rows and rows[r["id"]]["variant"] == "given" for r in recs)
    errors = [r["id"] for r in recs if r.get("error") or not r["completions"]]

    def one(rec):
        subs = rows[rec["id"]]["sub_answers"]
        return rec["id"], [grade(c, subs, tolerance=TOL).passed for c in rec["completions"]]
    with ThreadPoolExecutor(args.workers) as pool:
        verdicts = dict(pool.map(one, [r for r in recs if r["completions"]]))
    per_row = {i: {"template": rows[i]["template"], "n": len(v), "c": sum(v)} for i, v in verdicts.items()}
    by_tpl = defaultdict(list)
    for i, v in per_row.items():
        by_tpl[v["template"]].append(v)
    per_tpl = {t: {"rows": len(v), "given_acc": fmean(x["c"] / x["n"] for x in v), "pass@8": fmean(float(x["c"] > 0) for x in v),
                   "fold": rows[next(i for i in per_row if per_row[i]["template"] == t)]["cluster_fold"]}
               for t, v in sorted(by_tpl.items())}
    z = json.loads(args.report.read_text())["P1"]["Z"]
    missing = [t for t in z if t not in per_tpl]
    main_set = sorted(t for t in z if t in per_tpl and per_tpl[t]["given_acc"] >= KNOWS)
    out = {"criterion": f"Z (X report P1, base pass@64 = 0) and base given-row accuracy >= {KNOWS} at {TOL:.0%} over the W_f warm samples",
           "samples_sha256": shas, "rows_graded": len(per_row), "rows_without_samples": errors, "Z": z, "Z_missing_from_samples": missing,
           "Z_given": {t: per_tpl[t] for t in z if t in per_tpl}, "main_set": main_set, "main_set_size": len(main_set),
           "judgement_allowed": len(main_set) >= 5, "per_template": per_tpl}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(per_row)} given rows graded ({len(errors)} without samples); Z = {len(z)}, missing {missing}")
    for t in z:
        if t in per_tpl:
            p = per_tpl[t]
            print(f"  {t:36s} {p['fold']} rows {p['rows']:2d} given_acc {p['given_acc']:.3f} pass@8 {p['pass@8']:.3f} {'KNOWS' if p['given_acc'] >= KNOWS else ''}")
    print(f"main set ({len(main_set)}; judgement {'allowed' if len(main_set) >= 5 else 'numbers only'}): {main_set}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
