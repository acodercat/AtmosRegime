"""PREREG_PAPER_ANALYSES §11: the textbook traps of AtmosCoder-Bench by category and family, for the stored frontier runs.

    python -m explore.paper_traps --traps <Bench-Lab>/benchmark/traps.json --trap-dir <Bench-Lab>/experiments/trap \
        --core-dir <Bench-Lab>/experiments/core_code [--types results/paper/approx_types.json] --out results/paper/traps_frontier.json

Per configuration (one file per run, 3 runs): a trap's accuracy = mean of its runs' `passed`; its parent's (control_id, same run
index in core_code) likewise; paired gap = parent - trap; shortcut capture = a failed run whose every sub-answer is within 2 %
of `shortcut_values` (TRAP_RESULTS Table 4's rule), shared over failed runs. Records without `details` (API errors) and traps
not in traps.json are dropped and counted; traps flagged review.magnitude_defect are excluded and counted. Groups: the six
trap families, the regime traps that measure the regime judgement (review.measures_regime_judgement), categories (five kept,
the rest merged into "other"), and the coded approximation types when given. Unit = trap, resampled by parent_id, 10,000
draws, random.Random("paper:traps-<config>-<group>:20261029"); a group needs MIN_TRAPS traps for an interval. Numbers only.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

DRAWS = 10_000
MIN_TRAPS = 6
TOL = 0.02
KEEP_CATEGORIES = ("atmospheric_thermodynamics", "atmospheric_dynamics", "atmospheric_chemistry", "air_quality", "climate_dynamics")


def close(a, b, tol: float = TOL) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1e-12)
    except (TypeError, ValueError):
        return False


def captured(rec: dict, shortcut: dict) -> bool:
    if rec.get("passed") or not shortcut:
        return False
    actual = {d.get("sub"): d.get("actual") for d in rec.get("details") or [] if isinstance(d, dict)}
    return all(sub in actual and close(actual[sub], v) for sub, v in shortcut.items())


def pct(vals: list[float]) -> list[float] | None:
    if not vals:
        return None
    v = sorted(vals)
    return [v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]]


def runs_of(d: Path, model: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(d / f"{model}.run*.json"))):
        out.append({r["id"]: r for r in json.loads(Path(p).read_text())["results"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traps", type=Path, required=True)
    ap.add_argument("--trap-dir", type=Path, required=True)
    ap.add_argument("--core-dir", type=Path, required=True)
    ap.add_argument("--types", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    traps = {r["id"]: r for r in json.loads(args.traps.read_text())}
    types = json.loads(args.types.read_text())["traps"] if args.types else {}
    models = sorted({Path(p).name.split(".run")[0] for p in glob.glob(str(args.trap_dir / "*.run*.json"))})
    out = {"configs": {}, "rules": {"tolerance": TOL, "min_traps": MIN_TRAPS, "draws": DRAWS}}
    for m in models:
        trap_runs, core_runs = runs_of(args.trap_dir, m), runs_of(args.core_dir, m)
        per, dropped = {}, {"no_details": 0, "not_in_traps": 0, "magnitude_defect": 0, "parent_missing_runs": 0}
        for tid in sorted({i for run in trap_runs for i in run}):
            if tid not in traps:
                dropped["not_in_traps"] += 1
                continue
            t = traps[tid]
            if (t.get("review") or {}).get("magnitude_defect"):
                dropped["magnitude_defect"] += 1
                continue
            acc, par, caps = [], [], []
            for run, core in zip(trap_runs, core_runs + [{}] * (len(trap_runs) - len(core_runs))):
                r = run.get(tid)
                if r is None or "details" not in r:
                    dropped["no_details"] += r is not None
                    continue
                acc.append(float(bool(r["passed"])))
                if not r["passed"]:
                    caps.append(captured(r, t.get("shortcut_values") or {}))
                p = core.get(t["control_id"])
                if p is None or "details" not in p:
                    dropped["parent_missing_runs"] += 1
                else:
                    par.append(float(bool(p["passed"])))
            if not acc:
                continue
            per[tid] = {"family": t["trap_type"], "category": t["category"], "parent": t["control_id"],
                        "must_recognise": bool((t.get("review") or {}).get("measures_regime_judgement")), "type": types.get(tid),
                        "runs": len(acc), "trap": fmean(acc), "parent_acc": fmean(par) if par else None,
                        "gap": (fmean(par) - fmean(acc)) if par else None, "failed_runs": len(caps), "captured_runs": sum(caps)}
        groups = {"all": list(per)}
        for fam in sorted({v["family"] for v in per.values()}):
            groups[f"family:{fam}"] = [k for k, v in per.items() if v["family"] == fam]
        groups["regime:must_recognise"] = [k for k, v in per.items() if v["family"] == "regime_boundary" and v["must_recognise"]]
        groups["regime:other"] = [k for k, v in per.items() if v["family"] == "regime_boundary" and not v["must_recognise"]]
        for cat in KEEP_CATEGORIES:
            groups[f"regime_mr:{cat}"] = [k for k in groups["regime:must_recognise"] if per[k]["category"] == cat]
        groups["regime_mr:other_categories"] = [k for k in groups["regime:must_recognise"] if per[k]["category"] not in KEEP_CATEGORIES]
        for cat in KEEP_CATEGORIES:
            groups[f"all:{cat}"] = [k for k, v in per.items() if v["category"] == cat]
        if types:
            for ty in sorted({v["type"] for v in per.values() if v["type"]}):
                groups[f"regime_type:{ty}"] = [k for k, v in per.items() if v["family"] == "regime_boundary" and v["type"] == ty]
        summary = {}
        for g, ids in groups.items():
            if not ids:
                continue
            rows = [per[k] for k in ids]
            by_parent = defaultdict(list)
            for k in ids:
                by_parent[per[k]["parent"]].append(per[k])
            parents = sorted(by_parent)
            rng = random.Random(f"paper:traps-{m}-{g}:20261029")
            stat = {"traps": len(ids), "parents": len(parents), "trap_acc": fmean(r["trap"] for r in rows),
                    "parent_acc": fmean(r["parent_acc"] for r in rows if r["parent_acc"] is not None) if any(r["parent_acc"] is not None for r in rows) else None,
                    "gap": fmean(r["gap"] for r in rows if r["gap"] is not None) if any(r["gap"] is not None for r in rows) else None,
                    "failed_runs": sum(r["failed_runs"] for r in rows), "captured_runs": sum(r["captured_runs"] for r in rows)}
            stat["capture_share"] = stat["captured_runs"] / stat["failed_runs"] if stat["failed_runs"] else None
            if len(ids) >= MIN_TRAPS:
                boots = []
                for _ in range(DRAWS):
                    draw = [r for p in (rng.choice(parents) for _ in parents) for r in by_parent[p] if r["gap"] is not None]
                    if draw:
                        boots.append(fmean(r["gap"] for r in draw))
                stat["gap_ci95"] = pct(boots)
                boots = [fmean(r["trap"] for p in (rng.choice(parents) for _ in parents) for r in by_parent[p]) for _ in range(DRAWS)]
                stat["trap_acc_ci95"] = pct(boots)
            summary[g] = stat
        failing = sorted((k for k, v in per.items() if v["trap"] < 0.5), key=lambda k: (per[k]["family"], per[k]["category"]))
        out["configs"][m] = {"runs": len(trap_runs), "traps": len(per), "dropped": dropped, "groups": summary,
                             "failing_traps": [{"id": k, **{x: per[k][x] for x in ("family", "category", "trap", "parent_acc", "captured_runs", "failed_runs")}} for k in failing],
                             "per_trap": per}
        print(f"{m:30s} traps {len(per):3d} dropped {dropped} | all: trap {100 * summary['all']['trap_acc']:.1f} parent "
              f"{100 * (summary['all']['parent_acc'] or 0):.1f} gap {100 * (summary['all']['gap'] or 0):+.1f} | regime must-recognise: "
              f"{summary.get('regime:must_recognise', {}).get('traps', 0)} traps, gap {100 * (summary.get('regime:must_recognise', {}).get('gap') or 0):+.1f} "
              f"{summary.get('regime:must_recognise', {}).get('gap_ci95')}")
        for g in sorted(k for k in summary if k.startswith("regime_mr:")):
            s = summary[g]
            print(f"    {g:40s} n {s['traps']:2d} trap {100 * s['trap_acc']:5.1f} gap {100 * (s['gap'] or 0):+5.1f} {s.get('gap_ci95')} capture {s['capture_share']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
