"""PREREG_PAPER_ANALYSES §12: the physical cost of a silent error — how far the standard formula's value lies from the correct one.

    python -m explore.paper_cost --traps <Bench-Lab>/benchmark/traps.json --final experiments/_inputs/kc/final.json \
        --trap-runs <Bench-Lab>/experiments/trap --out results/paper/cost.json

Per sub-answer: shortcut s, expected e (first acceptable value); R = |s - e| / |e|, F = max(|s|/|e|, |e|/|s|); s*e < 0 is a sign flip,
e = 0 or s = 0 undefined (counted, not binned). Per trap the max over subs (R_max is separation_max's definition); per template the
median over its 20 outside rows. Bins: <= 5 % (inside the grading tolerance), 5-20, 20-50, 50-100 %, 2-10x, > 10x, sign flip.
Thresholds and bin edges carry a 1e-9 slack (EPS): 50 of the 280 defined traps have F_max = 2 up to rounding, 22 have R_max = 0.5.
Realised errors: every failed record of the frontier trap runs (three repetitions), its actual answers against the expected ones,
same bins; the share with R_max > 50 % and the share equal to the shortcut (every sub within 2 %). Intervals: 10,000 resamples of
traps / templates / records, random.Random("paper:cost:20261029"). Numbers only.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

DRAWS = 10_000
MERGED = {"soil": "non_atmospheric_environmental", "environmental_chemistry": "non_atmospheric_environmental",
          "hydrology": "non_atmospheric_environmental", "oceanography": "non_atmospheric_environmental"}
PRIMARY_CONFIGS = ("deepseek-v4-flash", "deepseek-v4-flash-reasoning", "gpt55-reasoning")
BINS = ("<=5%", "5-20%", "20-50%", "50-100%", "2-10x", ">10x")
EPS = 1e-9   # bin edges and thresholds are compared with this slack: many keys differ from the shortcut by exactly 2 (1.9999999999999996)


def first(v):
    if isinstance(v, list):
        v = v[0] if v else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pair_stats(s, e):
    """-> ('ok', R, F) | ('sign_flip',) | ('undefined',)."""
    if s is None or e is None or e == 0 or s == 0:
        return ("undefined",)
    if s * e < 0:
        return ("sign_flip",)
    return ("ok", abs(s - e) / abs(e), max(abs(s) / abs(e), abs(e) / abs(s)))


def bin_of(r, f):
    if r <= 0.05 + EPS:
        return "<=5%"
    if r <= 0.20 + EPS:
        return "5-20%"
    if r <= 0.50 + EPS:
        return "20-50%"
    if f <= 2 + EPS:
        return "50-100%"
    if f <= 10 + EPS:
        return "2-10x"
    return ">10x"


def item_max(subs):
    """subs: list of pair_stats -> dict(kind, R_max, F_max) for the item (max over its defined subs)."""
    ok = [x for x in subs if x[0] == "ok"]
    if any(x[0] == "sign_flip" for x in subs):
        return {"kind": "sign_flip"}
    if not ok:
        return {"kind": "undefined"}
    return {"kind": "ok", "R_max": max(x[1] for x in ok), "F_max": max(x[2] for x in ok)}


def pct(vals):
    if not vals:
        return None
    v = sorted(vals)
    return [v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]]


def summarise(items, rng):
    """items: list of item_max dicts. Quartiles of R_max and F_max, bins, counts, bootstrap of the median and of P(R_max > 0.5)."""
    ok = [x for x in items if x["kind"] == "ok"]
    out = {"n": len(items), "defined": len(ok), "sign_flip": sum(x["kind"] == "sign_flip" for x in items),
           "undefined": sum(x["kind"] == "undefined" for x in items), "bins": {b: 0 for b in BINS}}
    for x in ok:
        out["bins"][bin_of(x["R_max"], x["F_max"])] += 1
    if ok:
        r = sorted(x["R_max"] for x in ok); f = sorted(x["F_max"] for x in ok)
        q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
        out["R_quartiles"] = [q(r, .25), median(r), q(r, .75)]
        out["F_quartiles"] = [q(f, .25), median(f), q(f, .75)]
        out["share_R_gt_50"] = sum(x > 0.5 + EPS for x in r) / len(r)
        out["share_F_ge_2"] = sum(x >= 2 - EPS for x in f) / len(f)
        out["count_F_ge_2"] = sum(x >= 2 - EPS for x in f)
        out["exactly_factor_2"] = sum(abs(x - 2) <= EPS for x in f)      # construction signature: keys that are exactly twice / half the shortcut
        out["exactly_R_half"] = sum(abs(x - 0.5) <= EPS for x in r)
        if len(ok) >= 6:
            meds, shares = [], []
            for _ in range(DRAWS):
                draw = [rng.choice(ok) for _ in ok]
                meds.append(median(x["R_max"] for x in draw)); shares.append(sum(x["R_max"] > 0.5 + EPS for x in draw) / len(draw))
            out["R_median_ci95"] = pct(meds); out["share_R_gt_50_ci95"] = pct(shares)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traps", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--trap-runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random("paper:cost:20261029")
    out = {"definition": __doc__.strip().splitlines()[0], "traps": {}, "templates": {}, "realised": {}}

    # --- traps -------------------------------------------------------------------------------------------------------------
    traps = json.loads(args.traps.read_text())
    per_trap = {}
    for t in traps:
        sv = t.get("shortcut_values") or {}
        subs = [pair_stats(first(sv.get(sa["sub"])), first(sa["value"])) for sa in t["sub_answers"] if sa["sub"] in sv]
        x = item_max(subs) if subs else {"kind": "undefined"}
        x.update(id=t["id"], category=MERGED.get(t["category"], t["category"]), trap_type=t["trap_type"],
                 magnitude_defect=bool((t.get("review") or {}).get("magnitude_defect")),
                 must_recognise=bool((t.get("review") or {}).get("measures_regime_judgement")))
        per_trap[t["id"]] = x
    items = list(per_trap.values())
    clean = [x for x in items if not x["magnitude_defect"]]
    T = out["traps"]
    T["all"] = summarise(items, rng); T["all_without_magnitude_defect"] = summarise(clean, rng)
    T["by_category"] = {c: summarise([x for x in clean if x["category"] == c], rng) for c in sorted({x["category"] for x in clean})}
    T["by_trap_type"] = {c: summarise([x for x in clean if x["trap_type"] == c], rng) for c in sorted({x["trap_type"] for x in clean})}
    reg = [x for x in clean if x["trap_type"] == "regime_boundary"]
    T["regime_by_must_recognise"] = {"must_recognise": summarise([x for x in reg if x["must_recognise"]], rng),
                                     "other": summarise([x for x in reg if not x["must_recognise"]], rng)}
    T["regime_must_recognise_by_category"] = {c: summarise([x for x in reg if x["must_recognise"] and x["category"] == c], rng)
                                              for c in sorted({x["category"] for x in reg if x["must_recognise"]})}
    T["within_tolerance_ids"] = sorted(x["id"] for x in items if x["kind"] == "ok" and x["R_max"] <= 0.05)
    T["sign_flip_ids"] = sorted(x["id"] for x in items if x["kind"] == "sign_flip")
    T["undefined_ids"] = sorted(x["id"] for x in items if x["kind"] == "undefined")
    T["largest"] = sorted(({"id": x["id"], "F_max": x["F_max"], "category": x["category"], "trap_type": x["trap_type"]}
                           for x in clean if x["kind"] == "ok"), key=lambda z: -z["F_max"])[:15]
    T["per_trap"] = per_trap

    # --- templates ---------------------------------------------------------------------------------------------------------
    rows = [r for r in json.loads(args.final.read_text()) if r["variant"] == "outside"]
    by_t = defaultdict(list); cat_t = {}
    for r in rows:
        sv = r.get("shortcut_values") or {}
        subs = [pair_stats(first(sv.get(sa["sub"])), first(sa["value"])) for sa in r["sub_answers"] if sa["sub"] in sv]
        by_t[r["template"]].append(item_max(subs) if subs else {"kind": "undefined"}); cat_t[r["template"]] = MERGED.get(r["category"], r["category"])
    per_template = {}
    for tpl, xs in by_t.items():
        ok = [x for x in xs if x["kind"] == "ok"]
        rec = {"rows": len(xs), "defined": len(ok), "sign_flip_rows": sum(x["kind"] == "sign_flip" for x in xs), "category": cat_t[tpl]}
        if len(ok) >= len(xs) / 2:
            rec.update(kind="ok", R_max=median(x["R_max"] for x in ok), F_max=median(x["F_max"] for x in ok))
        else:
            rec["kind"] = "sign_flip" if rec["sign_flip_rows"] > len(xs) / 2 else "undefined"
        per_template[tpl] = rec
    titems = list(per_template.values())
    P = out["templates"]
    P["all"] = summarise(titems, rng)
    P["by_category"] = {c: summarise([x for x in titems if x["category"] == c], rng) for c in sorted({x["category"] for x in titems})}
    P["largest"] = sorted(({"template": k, "F_max": v["F_max"], "category": v["category"]} for k, v in per_template.items() if v["kind"] == "ok"),
                          key=lambda z: -z["F_max"])[:15]
    P["within_tolerance"] = sorted(k for k, v in per_template.items() if v["kind"] == "ok" and v["R_max"] <= 0.05)
    P["per_template"] = per_template

    # --- realised errors in the frontier trap runs ---------------------------------------------------------------------------
    expected = {t["id"]: t for t in traps}
    runs = defaultdict(list)
    for p in sorted(args.trap_runs.glob("*.run*.json")):
        runs[p.name.split(".run")[0]].append(p)
    R = out["realised"]
    for cfg, paths in runs.items():
        recs, by_cat = [], defaultdict(list)
        n_fail = n_nodetail = n_error = 0
        for p in paths:
            for r in json.loads(p.read_text())["results"]:
                t = expected.get(r["id"])
                if t is None or r.get("passed") or t["id"] not in per_trap or per_trap[t["id"]]["magnitude_defect"]:
                    continue
                if r.get("error"):          # an API / infrastructure error, not a model failure (Bench excludes it from accuracy)
                    n_error += 1
                    continue
                n_fail += 1
                det = [d for d in (r.get("details") or []) if isinstance(d, dict)]
                if not det:
                    n_nodetail += 1
                    continue
                subs = [pair_stats(first(d.get("actual")), first(d.get("expected"))) for d in det]
                x = item_max(subs)
                sv = t.get("shortcut_values") or {}
                act = {d.get("sub"): first(d.get("actual")) for d in det}
                x["captured"] = bool(sv) and all(act.get(s) is not None and abs(act[s] - float(v)) <= 0.02 * max(abs(float(v)), 1e-12) for s, v in sv.items())
                recs.append(x); by_cat[per_trap[t["id"]]["category"]].append(x)
        s = summarise(recs, rng)
        # descriptive split of the realised errors (not a §12 bin): shortcut-captured; a factor within 2 % of a unit-conversion
        # ratio (power of ten, 60, 3600, 1440, 86400, 24) or beyond 1e6 — a unit or code error rather than a physics choice; the rest.
        unit_like = lambda f: f > 1e6 or any(abs(f - u) <= 0.02 * u for u in (10, 100, 1000, 1e4, 1e5, 1e6, 60, 3600, 1440, 86400, 24, 365))
        defined = [x for x in recs if x["kind"] == "ok"]
        s.update(config=cfg, primary=cfg in PRIMARY_CONFIGS, repetitions=len(paths), failed_records=n_fail, failed_without_details=n_nodetail, error_records=n_error,
                 captured_share=(sum(x.get("captured", False) for x in recs) / len(recs)) if recs else None,
                 split={"captured_shortcut": sum(x.get("captured", False) for x in recs),
                        "unit_like_or_runaway": sum((not x.get("captured")) and unit_like(x["F_max"]) for x in defined),
                        "other_within_10x": sum((not x.get("captured")) and (not unit_like(x["F_max"])) and x["F_max"] <= 10 for x in defined),
                        "other_beyond_10x": sum((not x.get("captured")) and (not unit_like(x["F_max"])) and x["F_max"] > 10 for x in defined),
                        "sign_flip_or_undefined": sum(x["kind"] != "ok" and not x.get("captured") for x in recs)},
                 by_category={c: summarise(v, rng) for c, v in sorted(by_cat.items())})
        R[cfg] = s

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    def line(name, s):
        if not s or not s.get("defined"):
            return f"{name:42s} n {s.get('n', 0)} defined 0"
        b = s["bins"]
        return (f"{name:42s} n {s['n']:3d} def {s['defined']:3d} sign {s['sign_flip']:2d} undef {s['undefined']:2d} | R q {[round(x, 2) for x in s['R_quartiles']]} "
                f"F q {[round(x, 2) for x in s['F_quartiles']]} | >50% {100 * s['share_R_gt_50']:.0f}% {([round(100 * x) for x in s['share_R_gt_50_ci95']] if 'share_R_gt_50_ci95' in s else '')} "
                f"| bins {' '.join(f'{k}:{v}' for k, v in b.items())}")
    print("== traps"); print(line("all", T["all"])); print(line("without magnitude_defect", T["all_without_magnitude_defect"]))
    for c, s in T["by_category"].items(): print(line("  cat " + c, s))
    for c, s in T["by_trap_type"].items(): print(line("  type " + c, s))
    for c, s in T["regime_by_must_recognise"].items(): print(line("  regime " + c, s))
    for c, s in T["regime_must_recognise_by_category"].items(): print(line("  regime-mr " + c, s))
    print("  within tolerance:", T["within_tolerance_ids"], "| sign flips:", len(T["sign_flip_ids"]), "| undefined:", T["undefined_ids"])
    print("  largest:", [(z["id"], round(z["F_max"], 1)) for z in T["largest"][:8]])
    print("== templates"); print(line("all", P["all"]))
    for c, s in P["by_category"].items(): print(line("  cat " + c, s))
    print("  within tolerance:", P["within_tolerance"]); print("  largest:", [(z["template"], round(z["F_max"], 1)) for z in P["largest"][:8]])
    print("== realised (failed records, 3 reps, magnitude_defect excluded)")
    for cfg, s in R.items():
        print(line(("* " if s["primary"] else "  ") + cfg, s) + f" | captured {100 * (s['captured_share'] or 0):.0f}% nodetail {s['failed_without_details']} errors {s['error_records']} | split {s['split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
