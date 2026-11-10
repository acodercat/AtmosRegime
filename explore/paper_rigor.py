"""PREREG_PAPER_ANALYSES §1, §2 (the parts that need no evaluation files), §6 and §8 — from committed results and the raw pass@k samples.

    python -m explore.paper_rigor --dose-report results/dose/report_qwen3-8b.json --dose-posthoc results/dose/posthoc_qwen3-8b.json \
        --x-report qwen3-8b=results/x/report_qwen3-8b.json qwen35-9b=results/x/report_qwen35-9b.json \
        --passk qwen3-8b=experiments/x_passk/qwen3-8b/passk_final_outside.x18.jsonl qwen35-9b=experiments/x_passk/qwen35-9b/passk_final_outside.x18.jsonl \
        --passk-sens qwen3-8b=experiments/x_passk/qwen3-8b/passk_final_outside.sens.jsonl qwen35-9b=experiments/x_passk/qwen35-9b/passk_final_outside.sens.jsonl \
        --conditional results/conditional/d.json --x5 results/x5/report.json --out results/paper/rigor.json

§1 DOSE randomisation inference: doses re-assigned within each fold x seed by explore.dose.assign's rule, every observation's RL
accuracy kept, H1 recomputed 10,000 times; p = share >= the observed H1 (ties included). Leave-one-mechanism / leave-one-run
minima of H1's mechanism-bootstrap lower bound (dose_report.boot).
§2 P1 null and sensitivity: learned = any of A/D/F, any seed, >= 0.5 on the training-fold outside rows; Fisher exact (two-sided)
of Z (base pass@64 = 0) vs not Z against learned; AUC of base pass@64 and pass@1 for learned, mechanism bootstrap; Spearman
(pass@k, mean outside accuracy over A/D/F and seeds) for k in {1, 8, 16, 32, 64}; the pass@32 / T 0.7 sample's Z and how many
of them stay <= 0.10; leave-one-mechanism minimum Spearman at k = 64; cluster-resampled interval; count of (mechanism, arm,
seed) accuracies in (0.1, 0.9).
§6 p-curve: AUC of the warm start's per-sample p for learned over DOSE's dosed observations, mechanism bootstrap, plus the
isotonic (pool-adjacent-violators) curve; out of sample on X's mechanisms: base pass@1 >= 0.02 vs < 0.02 against learned, Fisher.
§8 frontier batch split: templates first committed 2026-09-26/27 (the 30 screened by E0) vs later; deepseek-v4-flash thinking-off
P(outside | inside) by batch, cluster bootstrap; X5 beside it.
Seeds random.Random("paper:<item>:20261029"), 10,000 draws. Numbers only.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.coverage_clusters import unit_of
from explore import dose_report
from explore.dose import LEVELS
from explore.x_report import passk_table, spearman

DRAWS = 10_000
LEARNED = 0.5
P_THRESHOLD = 0.02
ROOT = Path(__file__).resolve().parents[1]


def seed(item: str) -> random.Random:
    return random.Random(f"paper:{item}:20261029")


def pct(vals: list[float]) -> list[float] | None:
    if not vals:
        return None
    v = sorted(vals)
    return [v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]]


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """2x2 [[a, b], [c, d]], hypergeometric two-sided p (sum of tables at most as probable)."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    def prob(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p_obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if prob(x) <= p_obs * (1 + 1e-9)))


def auc(pairs: list[tuple[float, bool]]) -> float | None:
    """Rank AUC of score for the positive class; ties count half."""
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    return sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg) / (len(pos) * len(neg))


def auc_boot(pairs_by_mech: dict[str, tuple[float, bool]], item: str) -> dict:
    mechs = sorted(pairs_by_mech)
    point = auc([pairs_by_mech[m] for m in mechs])
    rng = seed(item)
    vals = [v for v in (auc([pairs_by_mech[rng.choice(mechs)] for _ in mechs]) for _ in range(DRAWS)) if v is not None]
    return {"auc": point, "ci95": pct(vals), "draws_used": len(vals), "n": len(mechs), "positives": sum(y for _, y in pairs_by_mech.values())}


def pav(points: list[tuple[float, float]]) -> list[dict]:
    """Isotonic (non-decreasing) fit by pool-adjacent-violators: blocks of (x range, fitted y, n)."""
    pts = sorted(points)
    blocks = [[p[0], p[0], p[1], 1] for p in pts]   # x_lo, x_hi, mean y, n
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2]:
            a, b = blocks[i], blocks[i + 1]
            blocks[i:i + 2] = [[a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / (a[3] + b[3]), a[3] + b[3]]]
            i = max(i - 1, 0)
        else:
            i += 1
    return [{"p_from": b[0], "p_to": b[1], "learned_share": b[2], "n": b[3]} for b in blocks]


# ---- §1 DOSE ---------------------------------------------------------------------------------------------------------

def random_assign(mechs: list[str], rng: random.Random) -> dict[str, int]:
    n = len(mechs)
    levels = list(LEVELS) * (n // len(LEVELS)) + rng.sample(LEVELS, n % len(LEVELS))
    return dict(zip(rng.sample(sorted(mechs), n), levels))


def dose_section(report: dict) -> dict:
    obs = [o for o in report["observations"] if o["rl"] is not None]
    observed = dose_report.h1_stat(obs)
    rng = seed("dose-rand")
    runs = sorted({(o["fold"], o["seed"]) for o in obs})
    by_run = {r: [o for o in obs if (o["fold"], o["seed"]) == r] for r in runs}
    ge, vals = 0, []
    for _ in range(DRAWS):
        fake = []
        for r, os_ in by_run.items():
            assign = random_assign([o["mechanism"] for o in os_], rng)
            fake += [{**o, "dose": assign[o["mechanism"]]} for o in os_]
        v = dose_report.h1_stat(fake)
        vals.append(v)
        ge += v is not None and v >= observed - 1e-12
    loo_mech = {m: dose_report.boot([o for o in obs if o["mechanism"] != m], dose_report.h1_stat)["ci95"] for m in sorted({o["mechanism"] for o in obs})}
    loo_run = {f"{f}-s{s}": dose_report.boot([o for o in obs if (o["fold"], o["seed"]) != (f, s)], dose_report.h1_stat)["ci95"] for f, s in runs}
    return {"observed_H1": observed, "randomisation_p": (ge + 1) / (DRAWS + 1), "draws": DRAWS, "runs": [f"{f}-s{s}" for f, s in runs],
            "null_quantiles": {"50": sorted(vals)[len(vals) // 2], "95": sorted(vals)[int(.95 * len(vals))], "99": sorted(vals)[int(.99 * len(vals))]},
            "loo_mechanism_min_lower": min(v[0] for v in loo_mech.values()), "loo_mechanism": loo_mech,
            "loo_run_min_lower": min(v[0] for v in loo_run.values()), "loo_run": loo_run,
            "per_run_sign": {f"{f}-s{s}": dose_report.h1_stat(by_run[(f, s)]) for f, s in runs}}


# ---- §2 P1 ---------------------------------------------------------------------------------------------------------

def p1_section(per: dict, passk_path: Path, sens_path: Path, item: str) -> dict:
    arms = ("A", "D", "F")
    mechs = sorted(m for m, v in per.items() if v.get("pass@64") is not None)
    def acc_values(m):
        return [per[m][k] for a in arms for k in (f"{a}_s42", f"{a}_s43") if per[m].get(k) is not None]
    learned = {m: any(v >= LEARNED for v in acc_values(m)) for m in mechs}
    mean_acc = {m: fmean(acc_values(m)) for m in mechs}
    z = [m for m in mechs if per[m]["pass@64"] == 0]
    nz = [m for m in mechs if m not in z]
    table = [[sum(learned[m] for m in z), sum(not learned[m] for m in z)], [sum(learned[m] for m in nz), sum(not learned[m] for m in nz)]]
    fisher = fisher_two_sided(table[0][0], table[0][1], table[1][0], table[1][1])
    pk = passk_table(passk_path, ks=(1, 8, 16, 32, 64))
    spear = {}
    for k in (1, 8, 16, 32, 64):
        pairs = [(pk[m][k], mean_acc[m]) for m in mechs if m in pk]
        rng = seed(f"{item}-spearman-k{k}")
        boots = [spearman([p[0] for p in d], [p[1] for p in d]) for d in ([pairs[rng.randrange(len(pairs))] for _ in pairs] for _ in range(DRAWS))]
        spear[k] = {"rho": spearman([p[0] for p in pairs], [p[1] for p in pairs]), "ci95_mechanism": pct(boots), "n": len(pairs)}
    # cluster-resampled interval at k = 64
    by_u = defaultdict(list)
    for m in mechs:
        if m in pk:
            by_u[unit_of(m)].append((pk[m][64], mean_acc[m]))
    units = sorted(by_u)
    rng = seed(f"{item}-spearman-cluster")
    boots = []
    for _ in range(DRAWS):
        d = [p for u in (rng.choice(units) for _ in units) for p in by_u[u]]
        boots.append(spearman([p[0] for p in d], [p[1] for p in d]))
    spear[64]["ci95_cluster"] = pct(boots)
    spear[64]["loo_mechanism_min"] = min(spearman([pk[m][64] for m in mechs if m != x and m in pk], [mean_acc[m] for m in mechs if m != x and m in pk]) for x in mechs)
    sens = passk_table(sens_path, ks=(1, 32))
    z_sens = sorted(m for m in mechs if m in sens and sens[m][32] == 0)
    at_or_below_10 = sum(all(v <= 0.10 for v in acc_values(m)) for m in z_sens)
    in_between = sum(0.1 < v < 0.9 for m in mechs for v in acc_values(m))
    return {"mechanisms": len(mechs), "Z": len(z), "learned_definition": "any of A/D/F, any seed, >= 0.5 on training-fold outside rows",
            "fisher_table_[Z_learned, Z_not],[nonZ_learned, nonZ_not]": table, "fisher_p_two_sided": fisher,
            "auc_pass64": auc_boot({m: (per[m]["pass@64"], learned[m]) for m in mechs}, f"{item}-auc64"),
            "auc_pass1": auc_boot({m: (pk[m][1], learned[m]) for m in mechs if m in pk}, f"{item}-auc1"),
            "spearman_by_k": spear, "sensitivity_pass32_T0.7": {"Z": len(z_sens), "stay_at_or_below_0.10_every_arm_seed": at_or_below_10, "mechanisms": z_sens},
            "accuracies_in_(0.1,0.9)": {"count": in_between, "of": sum(len(acc_values(m)) for m in mechs)},
            "learned": sorted(m for m in mechs if learned[m])}


# ---- §6 p-curve -----------------------------------------------------------------------------------------------------

def pcurve_section(posthoc: dict, per_by_family: dict, passk_paths: dict) -> dict:
    dosed = [o for o in posthoc["observations"] if o["dose"] >= 1 and o["warm_p"] is not None and o["rl"] is not None]
    # each mechanism has at most two dosed observations; resample mechanisms carrying both
    by_m = defaultdict(list)
    for o in dosed:
        by_m[o["mechanism"]].append((o["warm_p"], o["rl"] >= LEARNED))
    mechs = sorted(by_m)
    point = auc([p for m in mechs for p in by_m[m]])
    rng = seed("dose-pcurve-auc")
    vals = [v for v in (auc([p for m in (rng.choice(mechs) for _ in mechs) for p in by_m[m]]) for _ in range(DRAWS)) if v is not None]
    out = {"dose": {"auc_warm_p": point, "ci95": pct(vals), "observations": len(dosed), "mechanisms": len(mechs), "isotonic": pav([(o["warm_p"], float(o["rl"] >= LEARNED)) for o in dosed])}}
    for fam, per in per_by_family.items():
        pk = passk_table(passk_paths[fam], ks=(1,))
        mechs = sorted(m for m in per if per[m].get("pass@64") is not None and m in pk)
        learned = {m: any(per[m].get(k, 0) >= LEARNED for a in ("A", "D", "F") for k in (f"{a}_s42", f"{a}_s43") if per[m].get(k) is not None) for m in mechs}
        hi = [m for m in mechs if pk[m][1] >= P_THRESHOLD]
        lo = [m for m in mechs if m not in hi]
        table = [[sum(learned[m] for m in hi), sum(not learned[m] for m in hi)], [sum(learned[m] for m in lo), sum(not learned[m] for m in lo)]]
        out[f"x_{fam}"] = {"threshold": P_THRESHOLD, "table_[hi_learned, hi_not],[lo_learned, lo_not]": table,
                           "fisher_p_two_sided": fisher_two_sided(*table[0], *table[1]), "share_learned_hi": table[0][0] / len(hi) if hi else None,
                           "share_learned_lo": table[1][0] / len(lo) if lo else None}
    return out


# ---- §8 frontier batch split ----------------------------------------------------------------------------------------

def batch_section(conditional: dict, x5: dict) -> dict:
    dates = {}
    for p in sorted((ROOT / "atmoscoder_rl" / "templates").glob("*.py")):
        if p.name.startswith(("_", "base")):
            continue
        d = subprocess.run(["git", "log", "--diff-filter=A", "--format=%ad", "--date=short", "--", str(p)], cwd=ROOT, capture_output=True, text=True).stdout.split()
        dates[p.stem] = d[-1] if d else None
    pre = sorted(t for t, d in dates.items() if d and d <= "2026-09-27")
    post = sorted(t for t, d in dates.items() if d and d > "2026-09-27")
    cfg = conditional["deepseek-v4-flash.think-off"]["per_template"]
    def group(tpls, item):
        vals = {t: cfg[t]["outside"] for t in tpls if t in cfg}
        by_u = defaultdict(list)
        for t, v in vals.items():
            by_u[unit_of(t)].append(v)
        units = sorted(by_u)
        rng = seed(item)
        boots = [fmean(x for u in (rng.choice(units) for _ in units) for x in by_u[u]) for _ in range(DRAWS)]
        return {"templates": len(vals), "kept_of": len(tpls), "clusters": len(units), "mean_rec_outside": fmean(vals.values()) if vals else None, "ci95_cluster": pct(boots),
                "failing_at_or_below_0.5": sorted(t for t, v in vals.items() if v <= 0.5)}
    x5d = x5.get("deepseek-v4-flash.think-off", {}).get("accuracy", {})
    return {"batches": {"pre_screening_2026-09-26/27": group(pre, "batch-pre"), "post_screening": group(post, "batch-post")},
            "first_commit_dates": dates,
            "x5_deepseek_off": {"templates": len(x5d), "mean_outside": fmean(v["outside"] for v in x5d.values()) if x5d else None,
                                "mean_inside": fmean(v["inside"] for v in x5d.values()) if x5d else None,
                                "failing_outside_at_or_below_0.5": sorted(t for t, v in x5d.items() if v["outside"] <= 0.5)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dose-report", type=Path, required=True)
    ap.add_argument("--dose-posthoc", type=Path, required=True)
    ap.add_argument("--x-report", nargs="+", required=True, metavar="FAMILY=PATH")
    ap.add_argument("--passk", nargs="+", required=True, metavar="FAMILY=PATH")
    ap.add_argument("--passk-sens", nargs="+", required=True, metavar="FAMILY=PATH")
    ap.add_argument("--conditional", type=Path, required=True)
    ap.add_argument("--x5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    kv = lambda items: {k: Path(v) for k, v in (a.split("=", 1) for a in items)}
    xrep, passk, sens = kv(args.x_report), kv(args.passk), kv(args.passk_sens)
    per = {f: json.loads(p.read_text())["P1"]["per_mechanism"] for f, p in xrep.items()}
    out = {"dose_randomisation": dose_section(json.loads(args.dose_report.read_text())),
           "p1": {f: p1_section(per[f], passk[f], sens[f], f"p1-{f}") for f in per},
           "p_curve": pcurve_section(json.loads(args.dose_posthoc.read_text()), per, passk),
           "frontier_batch": batch_section(json.loads(args.conditional.read_text()), json.loads(args.x5.read_text()))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    d = out["dose_randomisation"]
    print(f"§1 DOSE H1 {100 * d['observed_H1']:.1f}: randomisation p {d['randomisation_p']:.4f} (null 95th {100 * d['null_quantiles']['95']:.1f}, 99th {100 * d['null_quantiles']['99']:.1f}); "
          f"LOO lower bound min: mechanism {100 * d['loo_mechanism_min_lower']:.1f}, run {100 * d['loo_run_min_lower']:.1f}")
    for f, p in out["p1"].items():
        print(f"§2 {f}: Z {p['Z']}/{p['mechanisms']}, Fisher table {p['fisher_table_[Z_learned, Z_not],[nonZ_learned, nonZ_not]']} p {p['fisher_p_two_sided']:.2e}; "
              f"AUC pass@64 {p['auc_pass64']['auc']:.3f} {p['auc_pass64']['ci95']}, pass@1 {p['auc_pass1']['auc']:.3f} {p['auc_pass1']['ci95']}")
        print("   Spearman by k: " + ", ".join(f"k{k} {v['rho']:.3f} {[round(x, 2) for x in v['ci95_mechanism']]}" for k, v in p["spearman_by_k"].items())
              + f"; k64 cluster CI {[round(x, 2) for x in p['spearman_by_k'][64]['ci95_cluster']]}, LOO min {p['spearman_by_k'][64]['loo_mechanism_min']:.3f}")
        s = p["sensitivity_pass32_T0.7"]
        print(f"   pass@32/T0.7: Z {s['Z']}, stay <= 0.10 in every arm and seed {s['stay_at_or_below_0.10_every_arm_seed']}; accuracies in (0.1,0.9): {p['accuracies_in_(0.1,0.9)']}")
    c = out["p_curve"]
    print(f"§6 DOSE AUC(warm p -> learned) {c['dose']['auc_warm_p']:.3f} {c['dose']['ci95']} n {c['dose']['observations']}; isotonic {[(round(b['p_from'], 4), round(b['p_to'], 4), round(b['learned_share'], 2), b['n']) for b in c['dose']['isotonic']]}")
    for k, v in c.items():
        if k.startswith("x_"):
            print(f"   {k}: p>=0.02 learned {v['share_learned_hi']:.2f}, p<0.02 learned {v['share_learned_lo']:.2f}, table {v['table_[hi_learned, hi_not],[lo_learned, lo_not]']}, Fisher p {v['fisher_p_two_sided']:.2e}")
    b = out["frontier_batch"]
    for k, v in b["batches"].items():
        print(f"§8 {k}: {v['templates']}/{v['kept_of']} templates, {v['clusters']} clusters, deepseek-off P(outside|inside) {100 * v['mean_rec_outside']:.1f} {[round(100 * x, 1) for x in v['ci95_cluster']]}, failing {len(v['failing_at_or_below_0.5'])}")
    print(f"   X5 deepseek-off: outside {100 * b['x5_deepseek_off']['mean_outside']:.1f}, inside {100 * b['x5_deepseek_off']['mean_inside']:.1f}, failing {b['x5_deepseek_off']['failing_outside_at_or_below_0.5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
