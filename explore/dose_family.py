"""Dose-gradient design for a family without an X chain (PREREG_DOSE_MINISTRAL): units from the family's reachability probe
(explore/family_reach.py) instead of the X report, the same random dose assignment as explore.dose, the same design.json
schema (explore.dose build and explore/dose_chain.sh read it unchanged).

    python -m explore.dose_family design --reach results/x/reach_ministral3-8b.json --out results/dose_ministral/design.json

Units (per fold, sorted): training-fold mechanisms of the fold, not plume_mixing_lid, base pass@64 <= 0.10 (explore/x_passk.sh,
n = 64, T 1.0), at least 4 warm-start pairs (explore.x_data.warm_rows over the family's own warm samples, as family_reach
counts them). PREREG_DOSE's second criterion (A arm outside accuracy < 0.5) cannot be applied: the family has no A arm; the
dose-0 observations of the design are its RL-only control on every unit. Doses: explore.dose.assign (levels 0/1/2/4, balanced,
random.Random("dose:<fold>:<seed>")), seeds 42 and 43 independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore.dose import EXCLUDED, FOLDS, LEVELS, MAX_PASS64, MIN_PAIRS, SEEDS, assign, sha256


def design(args) -> int:
    reach = json.loads(args.reach.read_text())
    per = reach["per_mechanism"]
    u = {f: sorted(t for t, r in per.items() if r["fold"] == f and t not in EXCLUDED and r.get("pass@64") is not None
                   and r["pass@64"] <= MAX_PASS64 and r["pairs"] >= MIN_PAIRS) for f in FOLDS}
    doses = {f: {str(s): assign(u[f], f, s) for s in SEEDS} for f in FOLDS}
    out = {"levels": list(LEVELS), "rule": {"max_base_pass64": MAX_PASS64, "max_A_mean": None, "min_pairs": MIN_PAIRS,
                                            "excluded": list(EXCLUDED), "note": "no A arm for this family; dose 0 is the RL-only control"},
           "inputs": {"reach": str(args.reach), "reach_sha256": sha256(args.reach), "family": reach["family"],
                      "reach_inputs_sha256": reach["inputs_sha256"]},
           "units": u,
           "unit_facts": {t: {"fold": per[t]["fold"], "base_pass64": per[t]["pass@64"], "base_pass1": per[t]["pass@1"],
                              "given_acc": per[t].get("given_acc"), "pairs": per[t]["pairs"]} for f in FOLDS for t in u[f]},
           "doses": doses}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for f in FOLDS:
        print(f, len(u[f]), "units:", " ".join(u[f]))
        for s in SEEDS:
            print(f"  s{s}", " ".join(f"{t}={d}" for t, d in doses[f][str(s)].items()))
    counts = {lv: sum(d == lv for f in FOLDS for s in SEEDS for d in doses[f][str(s)].values()) for lv in LEVELS}
    print("observations per level", counts, "total", sum(counts.values()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design"); d.add_argument("--reach", type=Path, required=True); d.add_argument("--out", type=Path, required=True)
    d.set_defaults(fn=design)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
