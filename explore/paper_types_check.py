"""Review checks on the by-type analysis (PREREG_PAPER_ANALYSES §9-§10, 2026-10-30 amendment). Reads the committed atlas and coding.

  python -m explore.paper_types_check --atlas results/paper/atlas.json --types results/paper/approx_types.json \
      --out results/paper/types_check.json

1. thinking / scale rescue by type (§10: reported only for types with >= 3 clusters that have both arms), cluster bootstrap;
2. Holm over the organisation family (category, cluster, type permutation p; §9);
3. sensitivity: group means and ICC with the pre-merge codes (8 types; the two small types reported but below the §10 floor);
4. how far the type labels follow the cluster labels (cluster = the design unit the templates were built in).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from explore.paper_atlas import DRAWS, icc_with_null, pct, seed


def boot_by_type(vals: dict[str, float], lab: dict[str, str], cluster: dict[str, str], item: str) -> dict:
    out = {}
    for g in sorted(set(lab[t] for t in vals if t in lab)):
        ts = [t for t in vals if lab.get(t) == g]
        by_u = defaultdict(list)
        for t in ts:
            by_u[cluster[t]].append(vals[t])
        units = sorted(by_u)
        rng = seed(f"types-check-{item}-{g}")
        boots = [fmean(x for u in (rng.choice(units) for _ in units) for x in by_u[u]) for _ in range(DRAWS)]
        out[g] = {"templates": len(ts), "clusters": len(units), "mean": fmean(vals[t] for t in ts),
                  "ci95_cluster": pct(boots), "reportable": len(units) >= 3}
    return out


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        adj[k] = run
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--types", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    atlas, coding = json.loads(a.atlas.read_text()), json.loads(a.types.read_text())
    per = atlas["per_template"]
    cluster = {t: v["cluster"] for t, v in per.items()}
    mean_local = {t: v["mean_rec_outside_local"] for t, v in per.items() if v.get("mean_rec_outside_local") is not None}
    merged = {t: ty for t, ty in coding["templates"].items() if t in mean_local}
    pre = {t: ty for t, ty in coding["templates_premerge"].items() if t in mean_local}

    rescue = {}
    for name, key in (("thinking", "A10_thinking_rescue"), ("scale", "A10_scale_rescue")):
        vals = {t: v["mean"] for t, v in atlas[key]["per_template"].items() if t in merged}
        rescue[name] = boot_by_type(vals, merged, cluster, name)

    org = atlas["A2_organisation"]
    ps = {k: org[k]["perm_p"] for k in ("category", "cluster", "type")}
    sens = {"icc": icc_with_null(mean_local, pre, "icc-type-premerge"),
            "group_means": boot_by_type(mean_local, pre, cluster, "premerge")}

    by_c = defaultdict(set)
    for t, ty in merged.items():
        by_c[cluster[t]].add(ty)
    pure = [c for c, s in by_c.items() if len(s) == 1]
    overlap = {"clusters": len(by_c), "single_type_clusters": len(pure),
               "templates_in_single_type_clusters": sum(cluster[t] in pure for t in merged), "templates": len(merged),
               "icc_type_within_cluster": icc_with_null(mean_local, merged, "icc-type-in-cluster", within=cluster)}

    out = {"rescue_by_type": rescue, "organisation_holm": {"raw": ps, "holm": holm(ps)}, "premerge_sensitivity": sens,
           "type_vs_cluster": overlap}
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for name, g in rescue.items():
        print(f"-- {name} rescue by type (mean change in P(outside right | original right), points)")
        for ty, v in sorted(g.items(), key=lambda x: -x[1]["mean"]):
            ci = v["ci95_cluster"]
            print(f"   {ty[:44]:44s} n{v['templates']:3d} cl{v['clusters']:3d} {100*v['mean']:+6.1f} [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]{'' if v['reportable'] else '  (below floor)'}")
    print("-- organisation p, raw -> Holm:", {k: f"{ps[k]:.3f} -> {out['organisation_holm']['holm'][k]:.3f}" for k in ps})
    print("-- pre-merge ICC:", round(sens["icc"]["icc"], 3), "p", round(sens["icc"]["perm_p"], 3))
    for ty, v in sorted(sens["group_means"].items(), key=lambda x: x[1]["mean"]):
        ci = v["ci95_cluster"]
        print(f"   {ty[:44]:44s} n{v['templates']:3d} cl{v['clusters']:3d} {100*v['mean']:5.1f} [{100*ci[0]:.1f}, {100*ci[1]:.1f}]")
    print("-- type vs cluster:", {k: (v if not isinstance(v, dict) else {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in overlap.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
