"""PREREG_PAPER_ANALYSES §9-§11: the failure atlas — every per-mechanism result in one table, and how the failures organise.

    python -m explore.paper_atlas --conditional results/conditional/d.json \
        --families results/d2/families_report_*.json experiments/d_execution_review/qwen3_8b.json experiments/d_execution_review/qwen35_9b.json \
        --x6 results/x6/report_kany.json --x7-strict results/x7/k2_table_strict.json --x7-lenient results/x7/k2_table_lenient.json \
        --x-report qwen3-8b=results/x/report_qwen3-8b.json qwen35-9b=results/x/report_qwen35-9b.json \
        --warm qwen3-8b=results/x/warm_knowledge_qwen3-8b.json qwen35-9b=results/x/warm_knowledge_qwen35-9b.json \
        --dose results/dose/posthoc_qwen3-8b.json --x5 results/x5/report.json [--types results/paper/approx_types.json] --out-dir results/paper

Numbers only, no verdict words. Outcome = P(outside right | inside right) per template and configuration (results/conditional/d.json,
17 configurations); a template "fails" a configuration at <= 0.5. Universality = failing configurations among those keeping the
template. Agreement between configurations = mean pairwise Spearman of the template ranking (templates kept in >= MIN_KEPT
configurations), mechanism-bootstrap interval. Variance decomposition = share of the cell variance carried by template means,
permutation null (template labels shuffled within each configuration). Organisation = ICC(1) of the template mean under category /
cluster / type labels with a permutation null, and ICC(cluster | category) with cluster labels shuffled within category. Rescue =
thinking-on minus thinking-off within a family and large minus small within a family, per template; a template is thinking-
(scale-)resistant when it fails thinking-off (small) in every pair it appears in and gains < RESCUE in every pair.
Seeds: random.Random("paper:atlas-<item>:20261029"), 10,000 draws.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pvariance

from atmoscoder_rl.coverage_clusters import unit_of
from explore.x_report import spearman

DRAWS = 10_000
FAIL = 0.5
MIN_KEPT = 10
RESCUE = 0.30
LOCAL = ["qwen3-8b.think-off", "qwen3-8b.think-on", "qwen3.5-9b.think-off", "qwen3.5-9b.think-on", "qwen3.5-27b.think-off",
         "qwen3.5-27b.think-on", "gemma4-e4b.think-off", "gemma4-e4b.think-on", "gemma4-31b.think-off", "gemma4-31b.think-on",
         "granite4.2-8b.think-off", "granite4.2-8b.think-on", "granite4.2-30b.think-off", "granite4.2-30b.think-on",
         "ministral3-8b.think-off", "ministral3-14b.think-off"]
FRONTIER = ["deepseek-v4-flash.think-off"]
THINK_PAIRS = [(m + ".think-off", m + ".think-on") for m in ("qwen3-8b", "qwen3.5-9b", "qwen3.5-27b", "gemma4-e4b", "gemma4-31b", "granite4.2-8b", "granite4.2-30b")]
SIZE_PAIRS = [("qwen3.5-9b.think-off", "qwen3.5-27b.think-off"), ("gemma4-e4b.think-off", "gemma4-31b.think-off"),
              ("granite4.2-8b.think-off", "granite4.2-30b.think-off"), ("ministral3-8b.think-off", "ministral3-14b.think-off")]
MERGED = {"soil": "non_atmospheric_environmental", "environmental_chemistry": "non_atmospheric_environmental",
          "hydrology": "non_atmospheric_environmental", "oceanography": "non_atmospheric_environmental"}


def seed(item: str) -> random.Random:
    return random.Random(f"paper:atlas-{item}:20261029")


def pct(vals: list[float]) -> list[float] | None:
    if not vals:
        return None
    v = sorted(vals)
    return [v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]]


def categories() -> dict[str, str]:
    import importlib
    out = {}
    for p in sorted((Path(__file__).resolve().parents[1] / "atmoscoder_rl" / "templates").glob("*.py")):
        if p.name.startswith(("_", "base")):
            continue
        t = getattr(importlib.import_module(f"atmoscoder_rl.templates.{p.stem}"), "TEMPLATE", None)
        if t is not None:
            out[t.name] = t.category
    return out


def load_families(paths: list[str]) -> dict[str, dict[str, dict]]:
    """config key -> template -> {inside, outside, given, flag, usable}."""
    out = {}
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            for c in json.loads(Path(p).read_text())["configurations"]:
                if c.get("status") != "complete":
                    continue
                out[c["key"]] = {t: {**v["accuracy"], "usable": v.get("usable")} for t, v in c["templates"]["templates"].items()}
    return out


def icc1(groups: dict[str, list[float]]) -> float | None:
    """One-way ICC(1) of the group means; None with fewer than 2 groups of size >= 2 in total."""
    g = {k: v for k, v in groups.items() if len(v) >= 1}
    n_total = sum(len(v) for v in g.values())
    if len(g) < 2 or n_total <= len(g):
        return None
    grand = fmean(x for v in g.values() for x in v)
    ssb = sum(len(v) * (fmean(v) - grand) ** 2 for v in g.values())
    ssw = sum((x - fmean(v)) ** 2 for v in g.values() for x in v)
    msb, msw = ssb / (len(g) - 1), ssw / (n_total - len(g))
    k0 = (n_total - sum(len(v) ** 2 for v in g.values()) / n_total) / (len(g) - 1)
    denom = msb + (k0 - 1) * msw
    return (msb - msw) / denom if denom else None


def icc_with_null(values: dict[str, float], labels: dict[str, str], item: str, within: dict[str, str] | None = None) -> dict:
    """ICC(1) of per-template values under `labels`, permutation p (labels shuffled, within `within` strata if given)."""
    tpls = sorted(t for t in values if t in labels and (within is None or t in within))
    def icc_of(lab):
        groups = defaultdict(list)
        for t in tpls:
            groups[lab[t]].append(values[t])
        return icc1(groups)
    obs = icc_of(labels)
    rng = seed(item)
    strata = defaultdict(list)
    for t in tpls:
        strata[within[t] if within else "all"].append(t)
    ge, n = 0, 0
    for _ in range(DRAWS):
        lab = {}
        for s, ts in strata.items():
            perm = rng.sample([labels[t] for t in ts], len(ts))
            lab.update(zip(ts, perm))
        v = icc_of(lab)
        if v is not None:
            n += 1
            ge += v >= obs
    return {"icc": obs, "templates": len(tpls), "groups": len({labels[t] for t in tpls}), "perm_p": (ge + 1) / (n + 1) if n else None, "draws": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditional", type=Path, required=True)
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--x6", type=Path, required=True)
    ap.add_argument("--x7-strict", type=Path, required=True)
    ap.add_argument("--x7-lenient", type=Path, required=True)
    ap.add_argument("--x-report", nargs="+", required=True, metavar="FAMILY=PATH")
    ap.add_argument("--warm", nargs="+", required=True, metavar="FAMILY=PATH")
    ap.add_argument("--dose", type=Path, required=True)
    ap.add_argument("--x5", type=Path, required=True)
    ap.add_argument("--types", type=Path, help="results/paper/approx_types.json (blind coding); type analyses need it")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cat = categories()
    cluster = {t: unit_of(t) for t in cat}
    cond = json.loads(args.conditional.read_text())
    fam = load_families(args.families)
    x6 = json.loads(args.x6.read_text())
    k2s, k2l = json.loads(args.x7_strict.read_text()), json.loads(args.x7_lenient.read_text())
    xrep = {k: json.loads(Path(v).read_text())["P1"]["per_mechanism"] for k, v in (a.split("=", 1) for a in args.x_report)}
    warm = {k: json.loads(Path(v).read_text())["per_mechanism"] for k, v in (a.split("=", 1) for a in args.warm)}
    dose = json.loads(args.dose.read_text())["observations"]
    x5 = json.loads(args.x5.read_text())
    types = json.loads(args.types.read_text())["templates"] if args.types else {}
    configs = [c for c in LOCAL + FRONTIER if c in cond]

    # ---- the long table -------------------------------------------------------------------------------------------------
    rows = []
    for t in sorted(cat):
        for c in configs:
            pt = cond[c]["per_template"].get(t) or {}
            fa = fam.get(c, {}).get(t) or {}
            rows.append({"template": t, "category": cat[t], "cluster": cluster[t], "type": types.get(t, ""), "config": c,
                         "kept": t in cond[c]["per_template"], "rec_outside": pt.get("outside"), "rec_given": pt.get("given"),
                         "rec_flag": pt.get("flag"), "inside": fa.get("inside"), "outside": fa.get("outside"), "given": fa.get("given"),
                         "flag": fa.get("flag"), "usable": fa.get("usable")})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "atlas_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- per-template summary -------------------------------------------------------------------------------------------
    rec = {t: {c: cond[c]["per_template"][t]["outside"] for c in configs if t in cond[c]["per_template"]} for t in cat}
    local = {t: {c: v for c, v in d.items() if c in LOCAL} for t, d in rec.items()}
    dose_tag = {}
    for o in dose:
        tag = dose_tag.setdefault(o["mechanism"], {"dose": {}, "rl": {}})
        tag["dose"][str(o["seed"])], tag["rl"][str(o["seed"])] = o["dose"], o["rl"]
    per_t = {}
    for t in sorted(cat):
        kept = local[t]
        s = {"category": cat[t], "category_merged": MERGED.get(cat[t], cat[t]), "cluster": cluster[t], "type": types.get(t),
             "configs_kept_local": len(kept), "configs_kept_all": len(rec[t]),
             "universality_local": sum(v <= FAIL for v in kept.values()), "universality_all": sum(v <= FAIL for v in rec[t].values()),
             "mean_rec_outside_local": fmean(kept.values()) if kept else None, "deepseek_off": rec[t].get("deepseek-v4-flash.think-off"),
             "K_any": {m: x6[m]["per_template"].get(t, {}).get("K") for m in ("qwen3-8b", "qwen35-9b", "deepseek")},
             "K2_strict": {m: k2s[m]["per_template"].get(t, {}).get("K2") for m in ("qwen3-8b", "qwen35-9b", "deepseek")},
             "R2": {m: k2s[m]["per_template"].get(t, {}).get("R2") for m in ("qwen3-8b", "qwen35-9b", "deepseek")},
             "K2_lenient": {m: k2l[m]["per_template"].get(t, {}).get("K2") for m in ("qwen3-8b", "qwen35-9b", "deepseek")},
             "rl": {f: {k: v for k, v in xrep[f].get(t, {}).items()} for f in xrep},
             "warm": {f: {k: warm[f].get(t, {}).get(k) for k in ("W_f", "WA", "WD", "wf_pass64", "pairs")} for f in warm},
             "dose": dose_tag.get(t)}
        per_t[t] = s

    # ---- A1: universality and agreement between configurations ---------------------------------------------------------
    pool = [t for t in cat if len(rec[t]) >= MIN_KEPT]
    def mean_pairwise(tpls):
        vals = []
        for i, a in enumerate(configs):
            for b in configs[i + 1:]:
                pairs = [(rec[t][a], rec[t][b]) for t in tpls if a in rec[t] and b in rec[t]]
                if len(pairs) >= 8:
                    vals.append(spearman([p[0] for p in pairs], [p[1] for p in pairs]))
        return fmean(vals) if vals else None
    rng = seed("agreement")
    boots = [v for v in (mean_pairwise([rng.choice(pool) for _ in pool]) for _ in range(DRAWS)) if v is not None]
    agreement = {"templates": len(pool), "configs": len(configs), "mean_pairwise_spearman": mean_pairwise(pool), "ci95": pct(boots), "draws_used": len(boots)}
    pair_matrix = {a: {b: (spearman([rec[t][a] for t in pool if a in rec[t] and b in rec[t]], [rec[t][b] for t in pool if a in rec[t] and b in rec[t]])
                        if sum(1 for t in pool if a in rec[t] and b in rec[t]) >= 8 else None) for b in configs} for a in configs}
    # variance decomposition: share of cell variance explained by template means, permutation null within configuration
    cells = [(t, c, rec[t][c]) for t in pool for c in configs if c in rec[t]]
    def share(cells):
        tot = pvariance([v for _, _, v in cells])
        by_t = defaultdict(list)
        for t, _, v in cells:
            by_t[t].append(v)
        grand = fmean(v for _, _, v in cells)
        between = sum(len(v) * (fmean(v) - grand) ** 2 for v in by_t.values()) / len(cells)
        return between / tot if tot else None
    obs_share = share(cells)
    rng = seed("variance")
    by_c = defaultdict(list)
    for t, c, v in cells:
        by_c[c].append((t, v))
    ge = 0
    for _ in range(DRAWS):
        perm = []
        for c, tv in by_c.items():
            vals = rng.sample([v for _, v in tv], len(tv))
            perm += [(t, c, v) for (t, _), v in zip(tv, vals)]
        ge += (share(perm) or 0) >= (obs_share or 0)
    variance = {"template_share": obs_share, "perm_p": (ge + 1) / (DRAWS + 1), "cells": len(cells)}

    # ---- A2: organisation by category / cluster / type ----------------------------------------------------------------
    mean_local = {t: per_t[t]["mean_rec_outside_local"] for t in cat if per_t[t]["mean_rec_outside_local"] is not None}
    cat_m = {t: MERGED.get(cat[t], cat[t]) for t in cat}
    org = {"category": icc_with_null(mean_local, cat_m, "icc-category"),
           "cluster": icc_with_null(mean_local, cluster, "icc-cluster"),
           "cluster_within_category": icc_with_null(mean_local, cluster, "icc-cluster-in-category", within=cat_m)}
    if types:
        org["type"] = icc_with_null(mean_local, {t: ty for t, ty in types.items() if ty}, "icc-type")
    def group_means(lab: dict[str, str], item: str) -> dict:
        out = {}
        for g in sorted(set(lab.values())):
            ts = [t for t in mean_local if lab.get(t) == g]
            units = sorted({cluster[t] for t in ts})
            if not ts:
                continue
            rng = seed(f"{item}-{g}")
            by_u = defaultdict(list)
            for t in ts:
                by_u[cluster[t]].append(mean_local[t])
            boots = [fmean(x for u in (rng.choice(units) for _ in units) for x in by_u[u]) for _ in range(DRAWS)]
            out[g] = {"templates": len(ts), "clusters": len(units), "mean_rec_outside_local": fmean(mean_local[t] for t in ts), "ci95_cluster": pct(boots),
                      "deepseek_off": fmean(v for t in ts for v in [rec[t].get("deepseek-v4-flash.think-off")] if v is not None) if any(rec[t].get("deepseek-v4-flash.think-off") is not None for t in ts) else None,
                      "universality_local_mean": fmean(per_t[t]["universality_local"] for t in ts)}
        return out
    groups = {"category": group_means(cat_m, "cat"), "cluster": group_means(cluster, "clu")}
    if types:
        groups["type"] = group_means({t: ty for t, ty in types.items() if ty}, "type")

    # ---- A3 / A10: rescue by thinking and by scale ---------------------------------------------------------------------
    def rescue(pairs, item):
        per = {}
        for t in cat:
            d = {f"{a}->{b}": rec[t][b] - rec[t][a] for a, b in pairs if a in rec[t] and b in rec[t]}
            if d:
                per[t] = {"pairs": d, "mean": fmean(d.values()), "n": len(d),
                          "fails_before_every_pair": all(rec[t][a] <= FAIL for a, b in pairs if a in rec[t] and b in rec[t]),
                          "rescued_in": sum(v >= RESCUE for v in d.values())}
        resistant = sorted(t for t, v in per.items() if v["n"] >= 4 and v["fails_before_every_pair"] and v["rescued_in"] == 0)
        rescued = sorted(t for t, v in per.items() if v["n"] >= 4 and v["fails_before_every_pair"] and v["rescued_in"] >= v["n"] / 2)
        return {"per_template": per, "resistant": resistant, "rescued_in_at_least_half": rescued}
    think, size = rescue(THINK_PAIRS, "think"), rescue(SIZE_PAIRS, "size")

    # ---- X5 ---------------------------------------------------------------------------------------------------------
    x5_rows = {m: {t: v for t, v in x5[m]["accuracy"].items()} for m in x5 if isinstance(x5[m], dict) and "accuracy" in x5[m]}

    out = {"configs": configs, "templates": len(cat), "per_template": per_t, "A1_agreement": agreement, "A1_pairwise_spearman": pair_matrix,
           "A1_variance": variance, "A2_organisation": org, "group_means": groups, "A10_thinking_rescue": think, "A10_scale_rescue": size,
           "x5_accuracy": x5_rows, "types_file": str(args.types) if args.types else None}
    (args.out_dir / "atlas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"atlas: {len(cat)} templates x {len(configs)} configurations; pool for agreement {len(pool)}")
    print(f"A1 mean pairwise Spearman {agreement['mean_pairwise_spearman']:.3f} {agreement['ci95']}; template share of variance "
          f"{variance['template_share']:.3f} (perm p {variance['perm_p']:.4f})")
    for k, v in org.items():
        print(f"A2 ICC {k}: {v['icc'] if v['icc'] is None else f'{v['icc']:.3f}'} (groups {v['groups']}, templates {v['templates']}, perm p {v['perm_p']})")
    for k, g in groups.items():
        print(f"-- by {k}")
        for name, v in sorted(g.items(), key=lambda kv: kv[1]["mean_rec_outside_local"]):
            print(f"   {name:34s} n {v['templates']:2d} units {v['clusters']:2d} rec_outside {100 * v['mean_rec_outside_local']:5.1f} {[round(100 * x, 1) for x in v['ci95_cluster']]} "
                  f"universality {v['universality_local_mean']:.1f}/{len([c for c in configs if c in LOCAL])}")
    print("universal failures (fail in every local config kept):", [t for t in sorted(cat) if per_t[t]["configs_kept_local"] >= MIN_KEPT and per_t[t]["universality_local"] == per_t[t]["configs_kept_local"]])
    print("thinking-resistant:", think["resistant"]); print("thinking-rescued (>= half of pairs):", think["rescued_in_at_least_half"])
    print("scale-resistant:", size["resistant"]); print("scale-rescued:", size["rescued_in_at_least_half"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
