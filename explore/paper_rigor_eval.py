"""PREREG_PAPER_ANALYSES §3, §4, §5, §7 and the §2 partial correlation — the items that need the stored evaluations (server).

    python -m explore.paper_rigor_eval --name qwen3-8b --seeds 42 43 --eval-dir $ATMOS_ROOT/experiments/_x_eval \
        --final $ATMOS_ROOT/experiments/_inputs/kc/final.json --inputs $ATMOS_ROOT/experiments/_inputs/x_eval \
        --passk $ATMOS_ROOT/experiments/_x/qwen3-8b/passk_final_outside.x18.jsonl --out results/paper/rigor_eval_qwen3-8b.json

One execution pass per label: every gradable final.json response is run once (as x_report.regrade_response) and its returned
values are cached next to the run file (<label>/final_values.json under the sha256 of final.json); grading at tolerances
{2, 5, 10} % (atmoscoder_rl.grading.verify, units="benchmark") and shortcut matching at {1, 2, 5} % (every sub-answer within
the threshold of shortcut_values) are then read off those values. The 5 % verdicts are checked against x_report's cached
regrade. Clusters and sides as x_report.Family (seeds averaged within a cluster before resampling, 10,000 draws,
random.Random("paper:<item>:20261029")).
§3 held-out asymmetry: per arm, (outside endpoint - start) - (inside endpoint - start) over held-out clusters.
§4 interaction: (WD - WA) - (D - A) over training-fold clusters.
§5 shortcut share: share of outside rows answered with the shortcut value, endpoint - start, held-out and training fold,
per arm; R arm - base where the R arm exists.
§7 P1 Spearman(pass@64, training-fold outside accuracy) at each tolerance; S1-like share of wrong base answers at the
shortcut for each matching threshold.
§2 partial Spearman of pass@64 and outside accuracy controlling the base's given-version accuracy per mechanism.
Numbers only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path
from statistics import fmean

from atmoscoder_rl import pilot
from atmoscoder_rl.engine import extract_code
from atmoscoder_rl.grading import verify
from atmoscoder_rl.runs import is_gradable
from atmoscoder_rl.sandbox import run_solver_sandboxed
from explore.x_report import FOLDS, Family, avg, passk_table, spearman

DRAWS = 10_000
TOLS = (0.02, 0.05, 0.10)
SHORTCUT_TOLS = (0.01, 0.02, 0.05)
ARMS = ("A", "D", "F", "WA", "WD")


def seed(item: str) -> random.Random:
    return random.Random(f"paper:{item}:20261029")


def pct(vals: list[float]) -> list[float] | None:
    if not vals:
        return None
    v = sorted(vals)
    return [v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]]


def ci_clusters(values: dict[str, float], item: str) -> dict:
    keys = sorted(values)
    rng = seed(item)
    boots = [fmean(values[k] for k in (rng.choice(keys) for _ in keys)) for _ in range(DRAWS)] if keys else []
    return {"mean": fmean(values.values()) if keys else None, "ci95": pct(boots), "units": len(keys)}


def close(a, b, tol: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1e-12)
    except (TypeError, ValueError):
        return False


class Values:
    """Per label: id -> the solver's returned values (None when the response produced none)."""
    def __init__(self, fam: Family) -> None:
        self.fam = fam

    def of(self, label: str) -> dict | None:
        path = self.fam.dir / label / "final.json"
        if not path.exists():
            return None
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cache = self.fam.dir / label / "final_values.json"
        got = json.loads(cache.read_text()) if cache.exists() else {}
        if got.get("source_sha256") != sha:
            todo = [(i, r) for i, r in self.fam.records(label, "final").items() if i in self.fam.rows and is_gradable(r)]
            def one(pair):
                i, r = pair
                code = extract_code((r["attempts"][-1]["response"] if r.get("attempts") else "") or "")
                if not code.strip():
                    return i, None
                try:
                    ok, results, _ = run_solver_sandboxed(code)
                except Exception:  # noqa: BLE001
                    return i, None
                if not ok or not isinstance(results, dict):
                    return i, None
                results.pop(pilot.JUDGE_KEY, None)
                return i, {k: (v.get("value") if isinstance(v, dict) else v) for k, v in results.items()}
            with ThreadPoolExecutor(16) as pool:
                res = dict(pool.map(one, todo))
            got = {"source_sha256": sha, "values": res}
            cache.write_text(json.dumps(got) + "\n")
        return got["values"]


def accuracy(fam: Family, values: dict, variant: str, tol: float) -> dict[str, float]:
    acc = defaultdict(list)
    for i, r in fam.rows.items():
        if r["variant"] != variant or i not in values:
            continue
        v = values[i]
        ok = bool(v) and verify({k: {"value": x} for k, x in v.items()}, list(r["sub_answers"]), tol, units="benchmark")[0]
        acc[r["template"]].append(float(ok))
    return {t: fmean(v) for t, v in acc.items()}


def shortcut_share(fam: Family, values: dict, tol: float, wrong_only: bool = False) -> dict[str, float]:
    """template -> share of outside rows whose every sub-answer is within tol of shortcut_values (over all outside rows, or over wrong ones)."""
    acc = defaultdict(list)
    for i, r in fam.rows.items():
        if r["variant"] != "outside" or i not in values:
            continue
        v, sv = values[i] or {}, r.get("shortcut_values") or {}
        hit = bool(sv) and all(k in v and close(v[k], s, tol) for k, s in sv.items())
        if wrong_only:
            right = bool(v) and verify({k: {"value": x} for k, x in v.items()}, list(r["sub_answers"]), 0.05, units="benchmark")[0]
            if right:
                continue
        acc[r["template"]].append(float(hit))
    return {t: fmean(v) for t, v in acc.items()}


def empty_share(fam: Family, values: dict) -> dict[str, float]:
    """template -> share of outside rows with no gradable output (the stored values are empty: no code, crash, or no return)."""
    acc = defaultdict(list)
    for i, r in fam.rows.items():
        if r["variant"] != "outside" or i not in values:
            continue
        acc[r["template"]].append(float(not values[i]))
    return {t: fmean(v) for t, v in acc.items()}


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> float | None:
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    d = (1 - rxz ** 2) * (1 - ryz ** 2)
    return (rxy - rxz * ryz) / sqrt(d) if d > 0 else None


def main() -> int:
    from atmoscoder_rl.paths import EXPERIMENTS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--passk", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fam = Family(args)
    vals = Values(fam)
    labels = [fam.base()] + [fam.warm(f) for f in FOLDS] + [fam.arm(a, f, s) for a in ARMS for f in FOLDS for s in args.seeds]
    labels += [fam.arm("R", f, 42) for f in FOLDS if (fam.dir / fam.arm("R", f, 42) / "final.json").exists()]
    present = [l for l in labels if (fam.dir / l / "final.json").exists()]
    missing = [l for l in labels if l not in present]
    V = {l: vals.of(l) for l in present}
    # grading at every tolerance, shortcut shares, and the 5 % check against x_report's regrade
    acc = {l: {v: {t: accuracy(fam, V[l], v, t) for t in TOLS} for v in ("inside", "outside", "given")} for l in present}
    sc = {l: {t: shortcut_share(fam, V[l], t) for t in SHORTCUT_TOLS} for l in present}
    check = {}
    for l in present:
        ref = fam.accuracy(l, "outside")
        mine = acc[l]["outside"][0.05]
        check[l] = {"templates": len(ref), "max_abs_diff_vs_x_report_regrade": max((abs(ref[t] - mine.get(t, 0)) for t in ref), default=None)}

    def cluster_mean(per_t: dict, tpls: list[str]) -> dict:
        return fam.clusters(per_t, tpls)

    def side(label_of, side_name: str, measure) -> dict:
        """cluster -> measure of the label(s) on `side_name` templates, folds pooled; label_of(fold) -> list of labels (seeds)."""
        out = {}
        for f in FOLDS:
            tpls = fam.templates(f, side_name)
            out.update(avg([cluster_mean(measure(l), tpls) for l in label_of(f)]))
        return out

    def arm_labels(a):
        return lambda f: [fam.arm(a, f, s) for s in args.seeds if fam.arm(a, f, s) in present]
    def start_labels(a):
        return lambda f: [fam.start(a, f)]

    out = {"family": args.name, "seeds": args.seeds, "labels": present, "missing": missing, "regrade_check_5pct": check}
    # §3 asymmetry on the held-out side (tolerance 5 %)
    asym = {}
    for a in ARMS:
        if not all(arm_labels(a)(f) for f in FOLDS):
            continue
        e_out, s_out = side(arm_labels(a), "held", lambda l: acc[l]["outside"][0.05]), side(start_labels(a), "held", lambda l: acc[l]["outside"][0.05])
        e_in, s_in = side(arm_labels(a), "held", lambda l: acc[l]["inside"][0.05]), side(start_labels(a), "held", lambda l: acc[l]["inside"][0.05])
        d = {c: (e_out[c] - s_out[c]) - (e_in[c] - s_in[c]) for c in e_out if c in s_out and c in e_in and c in s_in}
        asym[a] = {"outside_gain": ci_clusters({c: e_out[c] - s_out[c] for c in d}, f"asym-{a}-out"), "inside_gain": ci_clusters({c: e_in[c] - s_in[c] for c in d}, f"asym-{a}-in"),
                   "asymmetry": ci_clusters(d, f"asym-{a}")}
    out["asymmetry_held_out"] = asym
    # §4 interaction on the training side
    tr = {a: side(arm_labels(a), "train", lambda l: acc[l]["outside"][0.05]) for a in ARMS if all(arm_labels(a)(f) for f in FOLDS)}
    if all(a in tr for a in ("A", "D", "WA", "WD")):
        inter = {c: (tr["WD"][c] - tr["WA"][c]) - (tr["D"][c] - tr["A"][c]) for c in tr["A"] if all(c in tr[a] for a in ("D", "WA", "WD"))}
        out["interaction_train"] = {"(WD-WA)-(D-A)": ci_clusters(inter, "interaction"), "WA-A": ci_clusters({c: tr["WA"][c] - tr["A"][c] for c in inter}, "wa-a"),
                                    "D-A": ci_clusters({c: tr["D"][c] - tr["A"][c] for c in inter}, "d-a")}
    # §5 shortcut share
    short = {}
    for t in SHORTCUT_TOLS:
        st = {}
        for a in ARMS + ("R",):
            lab = arm_labels(a) if a != "R" else (lambda f: [fam.arm("R", f, 42)] if fam.arm("R", f, 42) in present else [])
            if not all(lab(f) for f in FOLDS):
                continue
            start = start_labels(a) if a != "R" else start_labels("A")
            for s_name in ("held", "train"):
                e, s0 = side(lab, s_name, lambda l, t=t: sc[l][t]), side(start, s_name, lambda l, t=t: sc[l][t])
                st[f"{a}:{s_name}"] = ci_clusters({c: e[c] - s0[c] for c in e if c in s0}, f"shortcut-{a}-{s_name}-{t}")
                st[f"{a}:{s_name}:endpoint_level"] = ci_clusters(e, f"shortcut-level-{a}-{s_name}-{t}")
        st["base:held+train:level"] = ci_clusters(side(start_labels("A"), "held", lambda l, t=t: sc[l][t]) | side(start_labels("A"), "train", lambda l, t=t: sc[l][t]), f"shortcut-base-{t}")
        short[str(t)] = st
    out["shortcut_share"] = short
    # Review 2026-10-29 (results/paper/README.md §5 caveat): the share over all outside rows mixes "the wrong answer is the standard
    # formula" with "an output exists at all" — a run that turns empty outputs into executing ones raises it mechanically. Same
    # deltas at 2 % over wrong rows only, and the share of rows with no gradable output, so the two can be separated.
    extra_src = {"wrong_only_0.02": {l: shortcut_share(fam, V[l], 0.02, wrong_only=True) for l in present},
                 "empty_output": {l: empty_share(fam, V[l]) for l in present}}
    for name, src in extra_src.items():
        st = {}
        for a in ARMS + ("R",):
            lab = arm_labels(a) if a != "R" else (lambda f: [fam.arm("R", f, 42)] if fam.arm("R", f, 42) in present else [])
            if not all(lab(f) for f in FOLDS):
                continue
            start = start_labels(a) if a != "R" else start_labels("A")
            for s_name in ("held", "train"):
                e, s0 = side(lab, s_name, lambda l: src[l]), side(start, s_name, lambda l: src[l])
                st[f"{a}:{s_name}"] = ci_clusters({c: e[c] - s0[c] for c in e if c in s0}, f"{name}-{a}-{s_name}")
                st[f"{a}:{s_name}:endpoint_level"] = ci_clusters(e, f"{name}-level-{a}-{s_name}")
        st["base:held+train:level"] = ci_clusters(side(start_labels("A"), "held", lambda l: src[l]) | side(start_labels("A"), "train", lambda l: src[l]), f"{name}-base")
        out["shortcut_share_" + name if name.startswith("wrong") else name + "_share"] = st
    # §7 tolerance sweep on P1 and S1-like shares
    pk = passk_table(args.passk, ks=(64,))
    p1 = {}
    for t in TOLS:
        per_mech = defaultdict(list)
        for a in ("A", "D", "F"):
            for f in FOLDS:
                for l in arm_labels(a)(f):
                    for m, v in acc[l]["outside"][t].items():
                        if fam.fold_of.get(m) == f:
                            per_mech[m].append(v)
        pairs = [(pk[m][64], fmean(v)) for m, v in per_mech.items() if m in pk]
        p1[str(t)] = {"spearman_pass64_vs_outside": spearman([p[0] for p in pairs], [p[1] for p in pairs]), "mechanisms": len(pairs),
                      "mean_training_fold_outside_A": fmean(v for l in [fam.arm("A", f, s) for f in FOLDS for s in args.seeds if fam.arm("A", f, s) in present]
                                                            for m, v in acc[l]["outside"][t].items() if fam.fold_of.get(m) == l.split("-")[-3])}
    out["tolerance_sweep"] = p1
    out["s1_base_wrong_at_shortcut"] = {str(t): fmean(shortcut_share(fam, V[fam.base()], t, wrong_only=True).values()) for t in SHORTCUT_TOLS}
    # §2 partial Spearman controlling the base's given accuracy
    given = acc[fam.base()]["given"][0.05]
    per_mech = defaultdict(list)
    for a in ("A", "D", "F"):
        for f in FOLDS:
            for l in arm_labels(a)(f):
                for m, v in acc[l]["outside"][0.05].items():
                    if fam.fold_of.get(m) == f:
                        per_mech[m].append(v)
    mechs = sorted(m for m in per_mech if m in pk and m in given)
    out["partial_spearman_given"] = {"rho_partial": partial_spearman([pk[m][64] for m in mechs], [fmean(per_mech[m]) for m in mechs], [given[m] for m in mechs]),
                                     "rho_given_vs_outside": spearman([given[m] for m in mechs], [fmean(per_mech[m]) for m in mechs]),
                                     "rho_pass64_vs_given": spearman([pk[m][64] for m in mechs], [given[m] for m in mechs]), "mechanisms": len(mechs)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"== {args.name}: labels {len(present)}, missing {missing}; 5 % regrade check max |diff| {max(v['max_abs_diff_vs_x_report_regrade'] or 0 for v in check.values()):.4f}")
    for a, v in asym.items():
        print(f"§3 {a}: outside gain {100 * v['outside_gain']['mean']:+.1f}, inside gain {100 * v['inside_gain']['mean']:+.1f}, asymmetry {100 * v['asymmetry']['mean']:+.1f} {[round(100 * x, 1) for x in v['asymmetry']['ci95']]}")
    if "interaction_train" in out:
        i = out["interaction_train"]
        print(f"§4 (WD-WA)-(D-A) {100 * i['(WD-WA)-(D-A)']['mean']:+.1f} {[round(100 * x, 1) for x in i['(WD-WA)-(D-A)']['ci95']]}")
    for t, st in short.items():
        print(f"§5 shortcut tol {t}: " + "; ".join(f"{k} {100 * v['mean']:+.1f} {[round(100 * x, 1) for x in v['ci95']]}" for k, v in st.items() if "level" not in k))
    for key in ("shortcut_share_wrong_only_0.02", "empty_output_share"):
        print(f"§5 {key}: " + "; ".join(f"{k} {100 * v['mean']:+.1f} [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]" for k, v in out[key].items() if "level" not in k)
              + " | levels: " + "; ".join(f"{k.split(':endpoint')[0]} {100 * v['mean']:.1f}" for k, v in out[key].items() if "level" in k))
    print("§7 tolerance:", {t: round(v["spearman_pass64_vs_outside"], 3) for t, v in p1.items()}, "| S1 base wrong at shortcut:", {t: round(v, 3) for t, v in out["s1_base_wrong_at_shortcut"].items()})
    print("§2 partial Spearman | given:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["partial_spearman_given"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
