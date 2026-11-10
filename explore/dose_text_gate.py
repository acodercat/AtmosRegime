"""PREREG_DOSE_TEXT §4: the format gate before any RL — run after the four text warm starts are evaluated, before RL starts.

    python -m explore.dose_text_gate --design results/dose_text/design.json --name qwen3-8b \
        --eval-dir $ATMOS_ROOT/experiments/_x_eval --final $ATMOS_ROOT/experiments/_inputs/kc/final.json \
        --inputs $ATMOS_ROOT/experiments/_inputs/x_eval --out gate.json

Per text warm start (sft-dtext-<name>-<fold>-s<seed>): (a) the share of final.json rows still ungradable after the content
attempts (no verdict details, not passed); (b) its inside accuracy over the fold's non-unit training mechanisms against the
DOSE warm start of the same fold and seed (sft-dose-...), x_report's regrade. The gate passes iff every model has
(a) <= 5 % and (b) not more than 5 points below its DOSE counterpart. Nothing about a unit's outside rows or pass@64 is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from explore.x_report import Family, P1_EXCLUDED

MAX_UNGRADABLE = 0.05
MAX_INSIDE_DROP = 0.05


def ungradable_share(eval_dir: Path, label: str) -> float | None:
    p = eval_dir / label / "final.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())["results"]
    return sum(1 for r in rows if not r.get("passed") and not r.get("details")) / len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.seeds = [42, 43]
    design = json.loads(args.design.read_text())
    fam = Family(args)
    out, ok = {}, True
    for fold, units in design["units"].items():
        nonunit = [t for t in fam.templates(fold, "train") if t not in units and t not in P1_EXCLUDED]
        for s in (42, 43):
            text, dose = f"sft-dtext-{args.name}-{fold}-s{s}", f"sft-dose-{args.name}-{fold}-s{s}"
            missing = fam.missing(text)
            ung = ungradable_share(args.eval_dir, text)
            it, idose = fam.accuracy(text, "inside"), fam.accuracy(dose, "inside")
            a_text = fmean(it[t] for t in nonunit if t in it) if any(t in it for t in nonunit) else None
            a_dose = fmean(idose[t] for t in nonunit if t in idose) if any(t in idose for t in nonunit) else None
            drop = None if a_text is None or a_dose is None else a_dose - a_text
            passed = not missing and ung is not None and ung <= MAX_UNGRADABLE and drop is not None and drop <= MAX_INSIDE_DROP
            ok &= passed
            out[text] = {"missing": missing, "ungradable_share": ung, "non_unit_inside_text": a_text, "non_unit_inside_dose": a_dose,
                         "drop": drop, "non_unit_mechanisms": len(nonunit), "passed": passed}
            print(f"{text}: ungradable {ung if ung is None else f'{100 * ung:.2f} %'}; non-unit inside {a_text if a_text is None else f'{100 * a_text:.1f}'} "
                  f"vs DOSE warm {a_dose if a_dose is None else f'{100 * a_dose:.1f}'} (drop {drop if drop is None else f'{100 * drop:+.1f}'}); "
                  f"missing {missing or 'none'} -> {'pass' if passed else 'FAIL'}")
    out["gate"] = "pass" if ok else "FAIL"
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("GATE", out["gate"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
