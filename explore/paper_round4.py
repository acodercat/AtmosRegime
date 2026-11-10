"""PREREG_PAPER_ANALYSES §15-17 (2026-11-03): the fourth round of post-hoc readings. Numbers only, plus the readings fixed there.

    # §15 what was learned: inside-row cost and incidental judgement on the learned mechanisms (server: every label is in _x_eval)
    python -m explore.paper_round4 --conditional --eval-dirs $E/_x_eval --final $E/_inputs/kc/final.json --inputs $E/_inputs/x_eval \
        --out results/paper/round4_conditional.json
    # §16 three-family reachability concordance (local, committed reports only)
    python -m explore.paper_round4 --families --x-report qwen3-8b=results/x/report_qwen3-8b.json qwen35-9b=results/x/report_qwen35-9b.json \
        --reach ministral3-8b=results/x/reach_ministral3-8b.json --atlas results/paper/atlas.json \
        --dose-posthoc qwen3-8b=results/dose/posthoc_qwen3-8b.json ministral3-8b=results/dose_ministral/posthoc_ministral3-8b.json \
        --out results/paper/round4_families.json
    # §17 injected knowledge on the 46 textbook regime traps (local: textbook.json of every label is in experiments/)
    python -m explore.paper_round4 --traps --eval-dirs experiments/dose/eval experiments/dose_ministral/eval experiments/opd/eval \
        experiments/x_base_eval --inputs experiments/x_textbook/inputs --trap-meta <Bench-Lab>/benchmark/traps.json \
        --traps-rl qwen3-8b=results/traps_rl/report_qwen3-8b.json --out results/paper/round4_traps.json

§15 units are (mechanism, intervention) on the training fold. Interventions and starts: X arm A (base), DOSE RL endpoints (the same
seed's warm start), OPD arms (base). Accuracy comes from x_report.Family (regraded final.json, inside / outside rows); H and F from
Family.flag0 (judge.json scored as P2). "Learned" = endpoint outside >= 0.5 and start outside < 0.5. Intervals resample mechanisms
(or traps in §17), 10,000 draws, random.Random("paper:round4:<item>:20261103").
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

from atmoscoder_rl.runs import is_gradable
from explore.dose_report import rl_label, warm_label
from explore.opd_report import OPD_ARMS
from explore.paper_rigor import auc, pct
from explore.traps_rl_report import KEEP, close
from explore.x_report import FOLDS, P1_EXCLUDED, Family, avg, spearman

DRAWS = 10_000
TAG = "20261103"
LEARNED = 0.5
COND, SWAP = -0.10, -0.25          # §15: conditional if the inside drop is at most 0.10, swap if more than 0.25
RHO_LOW, KAPPA_LOW = 0.3, 0.4      # §16 readings
BINS = ((0.0, 0.005), (0.005, 0.02), (0.02, 2.0))
OPD = ("P", "S", "Pk3", "Pk2", "Pshuf")
SEEDS = {"qwen3-8b": [42, 43], "qwen35-9b": [42], "ministral3-8b": [42, 43]}
EVAL_FILES = ("final", "judge", "test_id", "textbook")


def rng(item: str) -> random.Random:
    return random.Random(f"paper:round4:{item}:{TAG}")


def boot(values: dict[str, float], item: str) -> dict:
    """mean of a per-unit dict with the unit-resampling interval."""
    keys = sorted(values)
    if not keys:
        return {"mean": None, "ci95": None, "units": 0}
    r = rng(item)
    draws = [fmean(values[k] for k in (r.choice(keys) for _ in keys)) for _ in range(DRAWS)]
    return {"mean": fmean(values.values()), "ci95": pct(draws), "units": len(keys)}


# ---- §15 ---------------------------------------------------------------------------------------------------------------

class Dirs:
    """Stands in for Family.dir: `dirs / label` is the first evaluation directory that holds the label."""
    def __init__(self, dirs: list[Path]) -> None:
        self.dirs = dirs

    def __truediv__(self, label: str) -> Path:
        return next((d / label for d in self.dirs if (d / label / "final.json").exists()), self.dirs[0] / label)


def family(name: str, eval_dirs: list[Path], final: Path, inputs: Path) -> Family:
    fam = Family(SimpleNamespace(name=name, seeds=SEEDS[name], eval_dir=Dirs(eval_dirs), final=final, inputs=inputs))
    return fam


def interventions(name: str, fam: Family) -> dict[str, list[tuple[str, list[str], list[str]]]]:
    """intervention -> [(fold, endpoint labels, start labels)]; a start list of one label is shared by every endpoint, a list as
    long as the endpoints pairs each endpoint with its own start (DOSE: the same seed's warm model)."""
    out: dict[str, list] = {}
    if name != "ministral3-8b":
        out["A"] = [(f, [fam.arm("A", f, s) for s in SEEDS[name]], [fam.base()]) for f in FOLDS]
    if name in ("qwen3-8b", "ministral3-8b"):
        out["DOSE"] = [(f, [rl_label(name, f, s) for s in SEEDS[name]], [warm_label(name, f, s) for s in SEEDS[name]]) for f in FOLDS]
    if name == "qwen3-8b":
        for a in OPD:
            seeds = [42, 43] if a == "P" else [42]
            out[a] = [(f, [f"opd-{name}-{f}-{a}-s{s}" for s in seeds], [fam.base()]) for f in FOLDS]
    return out


def classify(d_inside: float | None) -> str | None:
    if d_inside is None:
        return None
    return "conditional" if d_inside >= COND else "swap" if d_inside < SWAP else "between"


def measures(fam: Family, label: str, templates: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(dict)
    for variant, key in (("outside", "outside"), ("inside", "inside")):
        for t, v in fam.accuracy(label, variant).items():
            if t in templates:
                out[t][key] = v
    for variant, key in (("outside", "H"), ("inside", "F")):
        for t, v in fam.flag0(label, variant).items():
            if t in templates:
                out[t][key] = v
    return out


def pool(cells: list[dict[str, dict]]) -> dict[str, dict]:
    """Seed average of per-template measure dicts (a key missing in one seed is averaged over the seeds that have it)."""
    out: dict[str, dict] = defaultdict(dict)
    for key in ("outside", "inside", "H", "F"):
        for t, v in avg([{t: c[t][key] for t in c if key in c[t]} for c in cells]).items():
            out[t][key] = v
    return out


def summarise_intervention(per: dict[str, dict], item: str) -> dict:
    learned = {t: v for t, v in per.items() if v["learned"]}
    rest = {t: v for t, v in per.items() if v["learned"] is False}
    classes = {c: sorted(t for t, v in learned.items() if v["class"] == c) for c in ("conditional", "between", "swap")}
    d_in = boot({t: v["d_inside"] for t, v in learned.items() if v["d_inside"] is not None}, f"{item}:learned:inside")
    d_hf = boot({t: v["d_hf"] for t, v in learned.items() if v["d_hf"] is not None}, f"{item}:learned:hf")
    out = {"learned": {"n": len(learned), "mechanisms": sorted(learned), "d_inside": d_in, "classes": classes,
                       "class_counts": {c: len(v) for c, v in classes.items()}, "d_hf": d_hf,
                       "inside_start_mean": fmean(v["inside_start"] for v in learned.values()) if learned else None,
                       "inside_end_mean": fmean(v["inside_end"] for v in learned.values()) if learned else None},
           "not_learned": {"n": len(rest),
                           "d_inside": boot({t: v["d_inside"] for t, v in rest.items() if v["d_inside"] is not None}, f"{item}:rest:inside"),
                           "d_hf": boot({t: v["d_hf"] for t, v in rest.items() if v["d_hf"] is not None}, f"{item}:rest:hf")},
           "per_mechanism": per}
    n_c, n_s = len(classes["conditional"]), len(classes["swap"])
    lo = d_in["ci95"][0] if d_in["ci95"] else None
    out["reading_a"] = (None if not learned else
                        "该干预学到的主要是无条件的公式替换" if n_s >= n_c else
                        "学会越界答案没有以原题为代价" if n_c > len(classes["between"]) + n_s and lo is not None and lo > COND else "只报数字")
    lo_hf = d_hf["ci95"][0] if d_hf["ci95"] else None
    out["reading_b"] = None if not learned else ("学会替代关系顺带提高了失效判断" if lo_hf is not None and lo_hf > 0 else "只报数字")
    return out


def conditional(args) -> dict:
    out = {"families": {}}
    for name in args.conditional_families:
        fam = family(name, args.eval_dirs, args.final, args.inputs)
        fam_out = {"interventions": {}, "labels": {}}
        for inter, folds in interventions(name, fam).items():
            per: dict[str, dict] = {}
            for fold, ends, starts in folds:
                if any(fam.records(l, "final") is None for l in ends + starts):
                    fam_out["interventions"][inter] = {"missing": [l for l in ends + starts if fam.records(l, "final") is None]}
                    break
                templates = set(fam.templates(fold, "train")) - set(P1_EXCLUDED)
                end = pool([measures(fam, l, templates) for l in ends])
                start = pool([measures(fam, l, templates) for l in starts])
                for t in sorted(templates):
                    e, s = end.get(t, {}), start.get(t, {})
                    if "outside" not in e or "outside" not in s:
                        continue
                    d_inside = e["inside"] - s["inside"] if "inside" in e and "inside" in s else None
                    hf_e = e["H"] - e["F"] if "H" in e and "F" in e else None
                    hf_s = s["H"] - s["F"] if "H" in s and "F" in s else None
                    per[t] = {"fold": fold, "outside_start": s["outside"], "outside_end": e["outside"],
                              "inside_start": s.get("inside"), "inside_end": e.get("inside"), "d_inside": d_inside,
                              "H_start": s.get("H"), "F_start": s.get("F"), "H_end": e.get("H"), "F_end": e.get("F"),
                              "hf_start": hf_s, "hf_end": hf_e, "d_hf": (hf_e - hf_s) if hf_e is not None and hf_s is not None else None,
                              "learned": e["outside"] >= LEARNED and s["outside"] < LEARNED, "class": None}
                    per[t]["class"] = classify(d_inside) if per[t]["learned"] else None
            else:
                fam_out["interventions"][inter] = {"endpoints": [l for _, ends, _ in folds for l in ends],
                                                   "starts": sorted({l for _, _, starts in folds for l in starts}),
                                                   **summarise_intervention(per, f"{name}:{inter}")}
        # appendix: test_id and the four overall accuracies per label (runner verdicts, errors excluded)
        labels = sorted({l for v in fam_out["interventions"].values() for k in ("endpoints", "starts") for l in v.get(k, [])})
        for l in labels:
            cell = {}
            for kind in EVAL_FILES:
                recs = fam.records(l, kind)
                g = [float(bool(r["passed"])) for r in (recs or {}).values() if is_gradable(r)]
                cell[kind] = {"accuracy": fmean(g) if g else None, "graded": len(g)}
            fam_out["labels"][l] = cell
        out["families"][name] = fam_out
    return out


# ---- §16 ---------------------------------------------------------------------------------------------------------------

def spearman_boot(pairs: dict[str, tuple[float, float]], item: str) -> dict:
    keys = sorted(pairs)
    if len(keys) < 3:
        return {"rho": None, "ci95": None, "units": len(keys)}
    r = rng(item)
    draws = []
    for _ in range(DRAWS):
        s = [pairs[k] for k in (r.choice(keys) for _ in keys)]
        v = spearman([a for a, _ in s], [b for _, b in s])
        if v == v:
            draws.append(v)
    rho = spearman([pairs[k][0] for k in keys], [pairs[k][1] for k in keys])
    return {"rho": rho if rho == rho else None, "ci95": pct(draws), "units": len(keys), "draws_used": len(draws)}


def kappa(a: dict[str, bool], b: dict[str, bool], keys: list[str]) -> float | None:
    n = len(keys)
    if not n:
        return None
    po = sum(a[k] == b[k] for k in keys) / n
    pa, pb = sum(a[k] for k in keys) / n, sum(b[k] for k in keys) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def kappa_boot(a: dict[str, bool], b: dict[str, bool], item: str) -> dict:
    keys = sorted(set(a) & set(b))
    r = rng(item)
    draws = [v for v in (kappa(a, b, [r.choice(keys) for _ in keys]) for _ in range(DRAWS)) if v is not None]
    table = {"both": sorted(k for k in keys if a[k] and b[k]), "a_only": sorted(k for k in keys if a[k] and not b[k]),
             "b_only": sorted(k for k in keys if b[k] and not a[k]), "neither": sorted(k for k in keys if not a[k] and not b[k])}
    return {"kappa": kappa(a, b, keys), "ci95": pct(draws), "units": len(keys), "draws_used": len(draws),
            "table": {k: len(v) for k, v in table.items()}, "mechanisms": table}


def auc_boot_units(units: dict[str, list[tuple[float, bool]]], item: str) -> dict:
    """AUC over every (score, label) pair, resampling the units (mechanisms) with all their pairs."""
    keys = sorted(units)
    point = auc([p for k in keys for p in units[k]])
    r = rng(item)
    draws = [v for v in (auc([p for k in (r.choice(keys) for _ in keys) for p in units[k]]) for _ in range(DRAWS)) if v is not None]
    return {"auc": point, "ci95": pct(draws), "units": len(keys), "pairs": sum(len(v) for v in units.values()),
            "positives": sum(y for v in units.values() for _, y in v), "draws_used": len(draws)}


def families(args) -> dict:
    pass64: dict[str, dict[str, float]] = {}
    learned_a: dict[str, dict[str, bool]] = {}
    for spec in args.x_report:
        name, path = spec.split("=", 1)
        per = json.loads(Path(path).read_text())["P1"]["per_mechanism"]
        pass64[name] = {t: v["pass@64"] for t, v in per.items() if t not in P1_EXCLUDED and v.get("pass@64") is not None}
        learned_a[name] = {t: fmean(v[f"A_s{s}"] for s in SEEDS[name] if f"A_s{s}" in v) >= LEARNED for t, v in per.items()
                           if t in pass64[name] and any(f"A_s{s}" in v for s in SEEDS[name])}
    for spec in args.reach:
        name, path = spec.split("=", 1)
        per = json.loads(Path(path).read_text())["per_mechanism"]
        pass64[name] = {t: v["pass@64"] for t, v in per.items() if t not in P1_EXCLUDED and v.get("pass@64") is not None}
    names = sorted(pass64)
    common = sorted(set.intersection(*(set(v) for v in pass64.values())))
    out = {"families": names, "common_mechanisms": len(common), "pass64": {n: {t: pass64[n][t] for t in common} for n in names}}
    # S16-a
    out["S16a"] = {"pairs": {}, "zero_in_all": sorted(t for t in common if all(pass64[n][t] == 0 for n in names)),
                   "zero_in_exactly_one": {n: sorted(t for t in common if pass64[n][t] == 0 and all(pass64[m][t] > 0 for m in names if m != n)) for n in names},
                   "zero_count": {n: sum(pass64[n][t] == 0 for t in common) for n in names}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["S16a"]["pairs"][f"{a}~{b}"] = spearman_boot({t: (pass64[a][t], pass64[b][t]) for t in common}, f"families:{a}~{b}")
    if args.atlas:
        atlas = json.loads(args.atlas.read_text())["per_template"]
        out["S16a"]["vs_atlas_mean_rec_outside_local"] = {
            n: spearman_boot({t: (pass64[n][t], atlas[t]["mean_rec_outside_local"]) for t in common if t in atlas and atlas[t].get("mean_rec_outside_local") is not None},
                             f"families:atlas:{n}") for n in names}
    lows = [v["ci95"][0] for v in out["S16a"]["pairs"].values() if v["ci95"]]
    out["S16a"]["reading"] = "可达性首先是机制的属性，其次才是模型的" if lows and len(lows) == len(out["S16a"]["pairs"]) and all(x > RHO_LOW for x in lows) else "只报数字"
    # S16-b
    if len(learned_a) == 2:
        a, b = sorted(learned_a)
        kb = kappa_boot(learned_a[a], learned_a[b], f"families:kappa:{a}~{b}")
        kb["discordant_pass64"] = {t: {n: pass64[n].get(t) for n in names} for t in kb["mechanisms"]["a_only"] + kb["mechanisms"]["b_only"]}
        kb["a"], kb["b"] = a, b
        kb["reading"] = "两家族学会的是同一批机制" if kb["ci95"] and kb["ci95"][0] > KAPPA_LOW else "只报数字"
        out["S16b"] = kb
    # S16-c
    if args.dose_posthoc:
        obs = {}
        for spec in args.dose_posthoc:
            name, path = spec.split("=", 1)
            obs[name] = json.loads(Path(path).read_text())["observations"]
        c: dict = {"per_family": {}, "pooled": {}}
        units_obs: dict[str, list] = {}
        units_mech: dict[str, tuple] = {}
        for name, rows in obs.items():
            dosed = [o for o in rows if o["dose"] >= 1]
            u_obs = defaultdict(list)
            for o in dosed:
                u_obs[f"{name}:{o['mechanism']}"].append((o["warm_p"], o["rl"] >= LEARNED))
            units_obs.update(u_obs)
            u_mech = {k: (fmean(p for p, _ in v), fmean(float(y) for _, y in v) >= LEARNED) for k, v in u_obs.items()}
            units_mech.update(u_mech)
            c["per_family"][name] = {"dosed_observations": len(dosed),
                                     "auc_observations": auc_boot_units(u_obs, f"families:auc-obs:{name}"),
                                     "auc_mechanisms": auc_boot_units({k: [v] for k, v in u_mech.items()}, f"families:auc-mech:{name}")}
        c["pooled"]["auc_observations"] = auc_boot_units(units_obs, "families:auc-obs:pooled")
        c["pooled"]["auc_mechanisms"] = auc_boot_units({k: [v] for k, v in units_mech.items()}, "families:auc-mech:pooled")
        c["pooled"]["dosed_by_warm_p"] = {f"[{lo}, {hi})": {"n": sum(1 for v in units_obs.values() for p, _ in v if lo <= p < hi),
                                                           "learned": sum(1 for v in units_obs.values() for p, y in v if lo <= p < hi and y)} for lo, hi in BINS}
        # family-stratified dose >= 1 minus dose 0, weighted by observations, mechanisms resampled within family
        def strat(sample: dict[str, dict[str, list[dict]]]) -> float | None:
            num = den = 0.0
            for name, by_mech in sample.items():
                rows = [o for v in by_mech.values() for o in v]
                d1 = [o["rl"] for o in rows if o["dose"] >= 1]
                d0 = [o["rl"] for o in rows if o["dose"] == 0]
                if d1 and d0:
                    num += (fmean(d1) - fmean(d0)) * len(rows); den += len(rows)
            return num / den if den else None
        by_fam = {name: defaultdict(list) for name in obs}
        for name, rows in obs.items():
            for o in rows:
                by_fam[name][o["mechanism"]].append(o)
        point = strat(by_fam)
        r = rng("families:stratified-h1")
        draws = []
        for _ in range(DRAWS):
            s = {name: {} for name in by_fam}
            for name, by_mech in by_fam.items():
                keys = sorted(by_mech)
                for i, k in enumerate(r.choice(keys) for _ in keys):
                    s[name][f"{k}#{i}"] = by_mech[k]
            v = strat(s)
            if v is not None:
                draws.append(v)
        c["pooled"]["stratified_dose_contrast"] = {"value": point, "ci95": pct(draws), "draws_used": len(draws),
                                                   "per_family": {name: (fmean(o["rl"] for v in m.values() for o in v if o["dose"] >= 1) - fmean(o["rl"] for v in m.values() for o in v if o["dose"] == 0)) for name, m in by_fam.items()}}
        out["S16c"] = c
    return out


# ---- §17 ---------------------------------------------------------------------------------------------------------------

def textbook_records(eval_dirs: list[Path], label: str) -> dict | None:
    for d in eval_dirs:
        p = d / label / "textbook.json"
        if p.exists():
            return {r["id"]: r for r in json.loads(p.read_text())["results"] if is_gradable(r)}
    return None


def per_trap(recs_list: list[dict], ids: list[str], parent_of: dict, meta: dict) -> dict:
    """trap -> mean trap passed, mean parent passed, failed rows, captured rows over the records of several labels
    (the loop of traps_rl_report.per_trap)."""
    out = {}
    for i in ids:
        t, p, failed, cap = [], [], 0, 0
        for recs in recs_list:
            r, pr = recs.get(i), recs.get(parent_of[i])
            if r is None or pr is None:
                continue
            t.append(float(bool(r["passed"]))); p.append(float(bool(pr["passed"])))
            if not r["passed"]:
                failed += 1
                actual = {d.get("sub"): d.get("actual") for d in (r.get("details") or []) if isinstance(d, dict)}
                sv = (meta.get(i) or {}).get("shortcut_values") or {}
                cap += bool(sv) and all(s in actual and close(actual[s], v) for s, v in sv.items())
        if t:
            out[i] = {"trap": fmean(t), "parent": fmean(p), "failed": failed, "captured": cap, "n": len(t)}
    return out


def trap_summary(pt: dict, gains: dict | None, item: str) -> dict:
    keys = sorted(pt)
    r = rng(f"traps:{item}")
    s = {"traps": len(keys), "trap_acc": fmean(pt[k]["trap"] for k in keys) if keys else None,
         "trap_acc_ci95": pct([fmean(pt[k]["trap"] for k in (r.choice(keys) for _ in keys)) for _ in range(DRAWS)]) if keys else None,
         "parent_acc": fmean(pt[k]["parent"] for k in keys) if keys else None,
         "failed_rows": sum(pt[k]["failed"] for k in keys), "captured_rows": sum(pt[k]["captured"] for k in keys)}
    s["capture_share"] = s["captured_rows"] / s["failed_rows"] if s["failed_rows"] else None
    if gains is not None:
        g = {k: gains[k] for k in keys if k in gains}
        s["gain_over_start"] = boot(g, f"traps:{item}:gain")
        s["discordant_vs_start"] = {"trap_right_only_after": sum(v > 0 for v in g.values()), "trap_right_only_before": sum(v < 0 for v in g.values())}
    return s


def conditional_share(recs_list: list[dict], ids: list[str], parent_of: dict) -> dict:
    num = den = 0
    for recs in recs_list:
        for i in ids:
            r, pr = recs.get(i), recs.get(parent_of[i])
            if r is not None and pr is not None and pr["passed"]:
                den += 1; num += bool(r["passed"])
    return {"trap_right": num, "parent_right": den, "share": num / den if den else None}


def traps(args) -> dict:
    groups = json.loads((args.inputs / "textbook_groups.json").read_text())
    parent_of, ids = groups["parent_of"], sorted(groups["regime_arm"])
    meta = {r["id"]: r for r in json.loads(args.trap_meta.read_text())} if args.trap_meta else {}
    meta.update({r["id"]: r for r in json.loads((args.inputs / "textbook.json").read_text())})
    cat = {i: (meta[i]["category"] if meta.get(i, {}).get("category") in KEEP else "other") for i in ids}
    ref = {}
    for spec in args.traps_rl:
        name, path = spec.split("=", 1)
        ref[name] = json.loads(Path(path).read_text())["arms"]
    out = {"traps": len(ids), "families": {}}
    for name in args.trap_families:
        base = f"x-{name}-base"
        groups_of: dict[str, list[tuple[str, str]]] = {   # group -> [(endpoint label, start label)]
            "base": [(base, None)],
            "dose_warm": [(warm_label(name, f, s), base) for f in FOLDS for s in SEEDS[name]],
            "dose_rl": [(rl_label(name, f, s), warm_label(name, f, s)) for f in FOLDS for s in SEEDS[name]],
            "dose_rl_vs_base": [(rl_label(name, f, s), base) for f in FOLDS for s in SEEDS[name]]}
        if name == "qwen3-8b":
            for a in OPD:
                groups_of[f"opd_{a}"] = [(f"opd-{name}-{f}-{a}-s{s}", base) for f in FOLDS for s in ([42, 43] if a == "P" else [42])]
        fam_out = {"groups": {}, "per_trap": {}}
        recs_cache: dict[str, dict | None] = {}
        def recs(label):
            if label not in recs_cache:
                recs_cache[label] = textbook_records(args.eval_dirs, label)
            return recs_cache[label]
        for g, pairs in groups_of.items():
            present = [(e, s) for e, s in pairs if recs(e) is not None and (s is None or recs(s) is not None)]
            if not present:
                fam_out["groups"][g] = {"missing": [e for e, _ in pairs]}
                continue
            pt = per_trap([recs(e) for e, _ in present], ids, parent_of, meta)
            gains = None
            if present[0][1] is not None:
                per_pair = [per_trap([recs(e)], ids, parent_of, meta) for e, _ in present]
                per_start = [per_trap([recs(s)], ids, parent_of, meta) for _, s in present]
                gains = {i: fmean(pe[i]["trap"] - ps[i]["trap"] for pe, ps in zip(per_pair, per_start) if i in pe and i in ps)
                         for i in ids if any(i in pe and i in ps for pe, ps in zip(per_pair, per_start))}
            cell = {"endpoints": [e for e, _ in present], "starts": sorted({s for _, s in present if s}),
                    "all": trap_summary(pt, gains, f"{name}:{g}"), "conditional": conditional_share([recs(e) for e, _ in present], ids, parent_of),
                    "by_category": {c: trap_summary({k: v for k, v in pt.items() if cat[k] == c}, {k: v for k, v in (gains or {}).items() if cat[k] == c} if gains else None, f"{name}:{g}:{c}")
                                    for c in sorted(set(cat.values()))}}
            fam_out["groups"][g] = cell
            fam_out["per_trap"][g] = pt
        # DOSE RL against the X A endpoints of the same family (traps_rl per_trap, same traps)
        if name in ref and "A" in ref[name] and "dose_rl" in fam_out["per_trap"]:
            a_pt = ref[name]["A"]["per_trap"]
            d_pt = fam_out["per_trap"]["dose_rl"]
            diff = {i: d_pt[i]["trap"] - a_pt[i]["trap"] for i in ids if i in d_pt and i in a_pt}
            cell = {"dose_rl_minus_A": boot(diff, f"{name}:dose_rl-A"), "A_trap_acc": ref[name]["A"]["all"]["trap_acc"],
                    "dose_rl_trap_acc": fam_out["groups"]["dose_rl"]["all"]["trap_acc"],
                    "base_trap_acc": ref[name]["base"]["all"]["trap_acc"] if "base" in ref[name] else None}
            lo = cell["dose_rl_minus_A"]["ci95"][0] if cell["dose_rl_minus_A"]["ci95"] else None
            cell["reading"] = "注入的模板知识没有额外迁移到教科书陷阱" if lo is not None and cell["dose_rl_minus_A"]["ci95"][1] >= 0 >= lo else "只报数字（单家族、46 道）"
            if "WA" in ref[name]:   # the same warm-started recipe without the dose (numbers only, added at the review of 2026-11-03)
                w_pt = ref[name]["WA"]["per_trap"]
                cell["dose_rl_minus_WA"] = boot({i: d_pt[i]["trap"] - w_pt[i]["trap"] for i in ids if i in d_pt and i in w_pt}, f"{name}:dose_rl-WA")
                cell["WA_trap_acc"] = ref[name]["WA"]["all"]["trap_acc"]
            fam_out["vs_x_A"] = cell
        out["families"][name] = fam_out
    out["categories"] = {c: sorted(i for i in ids if cat[i] == c) for c in sorted(set(cat.values()))}
    return out


# ---- printing, selftest, main ----------------------------------------------------------------------------------------

def fmt(b: dict, scale: float = 100) -> str:
    if not b or b.get("mean") is None:
        return "-"
    return f"{scale * b['mean']:+.1f} [{scale * b['ci95'][0]:+.1f}, {scale * b['ci95'][1]:+.1f}] n {b['units']}"


def show_conditional(out: dict) -> None:
    for name, fam in out["families"].items():
        for inter, v in fam["interventions"].items():
            if "missing" in v:
                print(f"{name} {inter}: missing {v['missing']}"); continue
            L = v["learned"]
            print(f"{name} {inter:6s} learned {L['n']:2d}: inside {100 * (L['inside_start_mean'] or 0):.0f} -> {100 * (L['inside_end_mean'] or 0):.0f}, "
                  f"d_inside {fmt(L['d_inside'])}; classes {L['class_counts']}; d(H-F) {fmt(L['d_hf'])} | not learned {v['not_learned']['n']}: "
                  f"d_inside {fmt(v['not_learned']['d_inside'])}, d(H-F) {fmt(v['not_learned']['d_hf'])} | {v['reading_a']} / {v['reading_b']}")
            if L["classes"]["swap"]:
                print(f"      swap: {L['classes']['swap']}")


def show_families(out: dict) -> None:
    s = out["S16a"]
    print(f"common mechanisms {out['common_mechanisms']}; zero in all {len(s['zero_in_all'])} {s['zero_in_all']}; zero count {s['zero_count']}")
    for k, v in s["pairs"].items():
        print(f"  rho {k}: {v['rho']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    for k, v in s.get("vs_atlas_mean_rec_outside_local", {}).items():
        print(f"  rho {k} ~ atlas: {v['rho']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print(f"  zero in exactly one: " + "; ".join(f"{n} {len(v)}" for n, v in s["zero_in_exactly_one"].items()) + f" -> {s['reading']}")
    if "S16b" in out:
        b = out["S16b"]
        print(f"S16b A learned {b['a']} vs {b['b']}: table {b['table']}, kappa {b['kappa']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] -> {b['reading']}")
    if "S16c" in out:
        c = out["S16c"]
        def a(v):
            return f"{v['auc']:.3f} {[round(x, 3) for x in v['ci95']]}" if v["auc"] is not None else f"- (positives {v['positives']} of {v['pairs']})"
        for n, v in c["per_family"].items():
            print(f"S16c {n}: dosed {v['dosed_observations']}, AUC obs {a(v['auc_observations'])}, mech {a(v['auc_mechanisms'])}")
        p = c["pooled"]
        print(f"S16c pooled: AUC obs {a(p['auc_observations'])}, mech {a(p['auc_mechanisms'])}; "
              f"bins " + " | ".join(f"{k}: {v['learned']}/{v['n']}" for k, v in p["dosed_by_warm_p"].items())
              + f"; stratified dose contrast {100 * p['stratified_dose_contrast']['value']:+.1f} {[round(100 * x, 1) for x in p['stratified_dose_contrast']['ci95']]}")


def show_traps(out: dict) -> None:
    for name, fam in out["families"].items():
        for g, v in fam["groups"].items():
            if "missing" in v:
                print(f"{name} {g}: missing {v['missing']}"); continue
            s, c = v["all"], v["conditional"]
            print(f"{name} {g:16s} trap {100 * s['trap_acc']:5.1f} {[round(100 * x, 1) for x in s['trap_acc_ci95']]} parent {100 * s['parent_acc']:5.1f} "
                  f"cond {c['trap_right']}/{c['parent_right']} capture {s['capture_share'] if s['capture_share'] is None else round(s['capture_share'], 2)}"
                  + (f" | gain {fmt(s['gain_over_start'])} discordant {s['discordant_vs_start']}" if "gain_over_start" in s else ""))
        if "vs_x_A" in fam:
            v = fam["vs_x_A"]
            print(f"{name} DOSE RL - X A: {fmt(v['dose_rl_minus_A'])} (base {100 * (v['base_trap_acc'] or 0):.1f}, A {100 * v['A_trap_acc']:.1f}, DOSE RL {100 * v['dose_rl_trap_acc']:.1f}) -> {v['reading']}")


def selftest() -> int:
    # (i) §15 classification and readings
    assert classify(-0.05) == "conditional" and classify(-0.3) == "swap" and classify(-0.2) == "between" and classify(None) is None
    per = {f"m{i}": {"learned": True, "d_inside": -0.5, "d_hf": 0.0, "class": "swap", "inside_start": 1.0, "inside_end": 0.5} for i in range(4)}
    per.update({f"c{i}": {"learned": True, "d_inside": 0.0, "d_hf": 0.3, "class": "conditional", "inside_start": 1.0, "inside_end": 1.0} for i in range(2)})
    per["n0"] = {"learned": False, "d_inside": 0.0, "d_hf": 0.0, "class": None, "inside_start": 1.0, "inside_end": 1.0}
    s = summarise_intervention(per, "selftest:swap")
    assert s["reading_a"] == "该干预学到的主要是无条件的公式替换" and s["learned"]["class_counts"] == {"conditional": 2, "between": 0, "swap": 4}, s["reading_a"]
    per2 = {f"c{i}": {"learned": True, "d_inside": -0.02 * (i % 3), "d_hf": 0.2 + 0.01 * i, "class": "conditional", "inside_start": 1.0, "inside_end": 0.98} for i in range(8)}
    s2 = summarise_intervention(per2, "selftest:cond")
    assert s2["reading_a"] == "学会越界答案没有以原题为代价" and s2["reading_b"] == "学会替代关系顺带提高了失效判断", (s2["reading_a"], s2["reading_b"])
    # (ii) §16 concordance: identical order -> rho 1 and reading; shuffled -> no reading
    mechs = [f"t{i}" for i in range(40)]
    same = {t: (i / 40, i / 40) for i, t in enumerate(mechs)}
    assert spearman_boot(same, "selftest:same")["rho"] == 1.0
    r = random.Random(1); shuffled = {t: (i / 40, r.random()) for i, t in enumerate(mechs)}
    assert spearman_boot(shuffled, "selftest:shuf")["ci95"][0] < RHO_LOW
    a = {t: i % 2 == 0 for i, t in enumerate(mechs)}
    kb = kappa_boot(a, dict(a), "selftest:kappa"); assert kb["kappa"] == 1.0 and kb["table"] == {"both": 20, "a_only": 0, "b_only": 0, "neither": 20}
    kb2 = kappa_boot(a, {t: not v for t, v in a.items()}, "selftest:kappa2"); assert kb2["kappa"] == -1.0
    u = {"m1": [(0.5, True), (0.4, True)], "m2": [(0.01, False)], "m3": [(0.02, False), (0.03, True)]}
    assert auc_boot_units(u, "selftest:auc")["auc"] == auc([(0.5, True), (0.4, True), (0.01, False), (0.02, False), (0.03, True)])
    # (iii) §17 trap bookkeeping on six hand-made traps
    ids = [f"trap{i}" for i in range(6)]; parent_of = {i: f"p{i}" for i in ids}
    meta = {i: {"shortcut_values": {"1": 10.0}} for i in ids}
    def rec(passed, actual=None):
        return {"passed": passed, "details": [{"sub": "1", "actual": actual}], "attempts": [{"response": "x"}]}
    recs = {**{i: rec(k % 2 == 0, 10.0 if k in (1, 3) else 3.0) for k, i in enumerate(ids)}, **{f"p{i}": rec(k < 4) for k, i in enumerate(ids)}}
    pt = per_trap([recs], ids, parent_of, meta)
    assert len(pt) == 6 and sum(v["trap"] for v in pt.values()) == 3 and sum(v["failed"] for v in pt.values()) == 3 and sum(v["captured"] for v in pt.values()) == 2, pt
    assert conditional_share([recs], ids, parent_of) == {"trap_right": 2, "parent_right": 4, "share": 0.5}
    s3 = trap_summary(pt, {i: 0.0 for i in ids}, "selftest"); assert s3["trap_acc"] == 0.5 and s3["gain_over_start"]["mean"] == 0.0
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--conditional", action="store_true"); ap.add_argument("--families", action="store_true"); ap.add_argument("--traps", action="store_true")
    ap.add_argument("--eval-dirs", type=Path, nargs="*", default=[])
    ap.add_argument("--final", type=Path); ap.add_argument("--inputs", type=Path)
    ap.add_argument("--conditional-families", nargs="*", default=["qwen3-8b", "qwen35-9b", "ministral3-8b"])
    ap.add_argument("--x-report", nargs="*", default=[], help="name=results/x/report_<name>.json")
    ap.add_argument("--reach", nargs="*", default=[], help="name=results/x/reach_<name>.json")
    ap.add_argument("--atlas", type=Path)
    ap.add_argument("--dose-posthoc", nargs="*", default=[], help="name=posthoc file")
    ap.add_argument("--trap-meta", type=Path, help="Bench-Lab benchmark/traps.json (shortcut_values of the traps)")
    ap.add_argument("--traps-rl", nargs="*", default=[], help="name=results/traps_rl/report_<name>.json")
    ap.add_argument("--trap-families", nargs="*", default=["qwen3-8b", "ministral3-8b"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if sum([args.conditional, args.families, args.traps]) != 1 or args.out is None:
        ap.error("exactly one of --conditional / --families / --traps, and --out")
    if args.conditional:
        out = conditional(args); show = show_conditional
    elif args.families:
        out = families(args); show = show_families
    else:
        out = traps(args); show = show_traps
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    show(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
