"""D without the three reworded templates (docs/PREREG_PAPER_ANALYSES.md §19; post hoc, numbers only).

    python -m explore.d_reword_sensitivity [--out results/paper/d_reword.json]
    python -m explore.d_reword_sensitivity --selftest

The training study rewords line_broadening, drop_ventilation and clausius_clapeyron_extrapolation in every version; D ran
the original wording. For each of the 16 D configurations this recomputes the template contrasts with the aggregation of
`atmoscoder_rl.diagnosis.template_summary` (all templates averaged within cluster, clusters weighted equally, `mean_ci`
with the D bootstrap seed) on the stored per-template accuracies, first with all 72 templates (must equal the stored
contrasts exactly) and then without the three, and re-applies `diagnosis.verdict` with the textbook side unchanged.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.diagnosis import mean_ci, verdict

ROOT = Path(__file__).resolve().parents[1]
REWORDED = ("line_broadening", "drop_ventilation", "clausius_clapeyron_extrapolation")
SOURCES = [*sorted((ROOT / "results/d2").glob("families_report_*.json")),
           *(ROOT / "experiments/d_execution_review" / f for f in ("qwen3_8b.json", "qwen35_9b.json", "gemma_e4b.json"))]
ARMS = ("inside", "given", "flag")


def aggregate(templates: dict, drop: tuple[str, ...] = ()) -> dict:
    """`template_summary.aggregate` on stored rows: returns {'all': ..., 'usable_only': ..., 'usable': bool}."""
    rows = {n: r for n, r in templates.items() if n not in drop}

    def agg(eligible_only: bool) -> dict:
        clusters = defaultdict(list)
        for r in rows.values():
            if not eligible_only or r["usable"]:
                clusters[r["cluster"]].append(r["accuracy"])
        means = {c: {v: fmean(a[v] for a in rs) for v in ("inside", "outside", "given", "flag")} for c, rs in sorted(clusters.items())}
        return {"cluster_count": len(clusters),
                "contrasts": {a + "_minus_outside": mean_ci([m[a] - m["outside"] for m in means.values()]) for a in ARMS}}

    usable = agg(True)
    return {"all": agg(False), "usable_only": usable, "usable": usable["cluster_count"] >= 15, "templates": len(rows)}


def configurations() -> list[dict]:
    out, seen = [], set()
    for f in SOURCES:
        for c in json.loads(f.read_text())["configurations"]:
            if "templates" in c and c["key"] not in seen:
                seen.add(c["key"])
                out.append({**c, "source": str(f.relative_to(ROOT))})
    return out


def analyse(c: dict) -> dict:
    stored = c["templates"]
    full, cut = aggregate(stored["templates"]), aggregate(stored["templates"], REWORDED)
    for side in ("all", "usable_only"):
        if full[side]["contrasts"] != stored[side]["contrasts"] or full[side]["cluster_count"] != stored[side]["cluster_count"]:
            raise SystemExit(f"{c['key']}: recomputed {side} contrasts differ from the stored report")
    if verdict(c["textbook"], {**stored, "usable": full["usable"]}) != c["D1"]:
        raise SystemExit(f"{c['key']}: recomputed D-1 differs from the stored verdict")
    d1_cut = verdict(c["textbook"], {"all": cut["all"], "usable": cut["usable"]})
    io = lambda r: r["all"]["contrasts"]["inside_minus_outside"]["mean"]
    return {"configuration": c["key"], "source": c["source"],
            "templates_72": {"clusters": full["all"]["cluster_count"], "usable_clusters": full["usable_only"]["cluster_count"], "contrasts": full["all"]["contrasts"]},
            "templates_69": {"clusters": cut["all"]["cluster_count"], "usable_clusters": cut["usable_only"]["cluster_count"], "contrasts": cut["all"]["contrasts"]},
            "inside_minus_outside_change": io(cut) - io(full),
            "D1_72": c["D1"], "D1_69": d1_cut, "D1_changed": d1_cut != c["D1"]}


def selftest() -> int:
    acc = lambda i, o: {"inside": i, "outside": o, "given": 1.0, "flag": 0.0}
    t = {f"t{k}": {"cluster": f"c{k}", "accuracy": acc(1.0, 0.5), "usable": True} for k in range(16)}
    t["line_broadening"] = {"cluster": "line_broadening", "accuracy": acc(1.0, 0.0), "usable": True}
    a, b = aggregate(t), aggregate(t, REWORDED)
    assert a["all"]["cluster_count"] == 17 and b["all"]["cluster_count"] == 16
    assert abs(a["all"]["contrasts"]["inside_minus_outside"]["mean"] - (16 * .5 + 1) / 17) < 1e-12
    assert b["all"]["contrasts"]["inside_minus_outside"]["mean"] == 0.5 and b["usable"] and b["templates"] == 16
    t2 = {k: v for k, v in t.items() if k in ("t0", "line_broadening")}
    assert aggregate(t2)["usable"] is False
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper/d_reword.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    rows = [analyse(c) for c in configurations()]
    if len(rows) != 16:
        raise SystemExit(f"expected 16 configurations, found {len(rows)}")
    changes = [abs(r["inside_minus_outside_change"]) for r in rows]
    holds = not any(r["D1_changed"] for r in rows) and max(changes) <= .03 + 1e-12
    out = {"analysis": "PREREG_PAPER_ANALYSES §19", "dropped": list(REWORDED), "configurations": rows,
           "max_abs_inside_minus_outside_change": max(changes), "D1_changed": [r["configuration"] for r in rows if r["D1_changed"]],
           "reading_holds": holds}
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for r in rows:
        a, b = r["templates_72"]["contrasts"]["inside_minus_outside"], r["templates_69"]["contrasts"]["inside_minus_outside"]
        print(f"{r['configuration']:24s} {100*a['mean']:5.1f} [{100*a['ci95'][0]:5.1f}, {100*a['ci95'][1]:5.1f}] -> {100*b['mean']:5.1f} "
              f"[{100*b['ci95'][0]:5.1f}, {100*b['ci95'][1]:5.1f}]  usable {r['templates_72']['usable_clusters']}->{r['templates_69']['usable_clusters']}  "
              f"D-1 {r['D1_72']} -> {r['D1_69']}")
    print(f"max |change| {100*max(changes):.1f} pt; D-1 changed: {out['D1_changed'] or 'none'}; reading holds: {holds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
