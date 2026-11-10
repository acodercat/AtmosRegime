"""PREREG_OPD §3 report: the OPD arms (P privileged self-teacher, S same teacher without privilege, Pk3 sampled-token k3)
against the X arm A on the mechanisms RL could not reach (Z), with the pre-registered readings. Written before any OPD
endpoint was evaluated (OPD-6); every number comes from the archived evaluations under --eval-dir.

    cd $ATMOS_REPO && python -m explore.opd_report --name qwen3-8b --main-set results/opd/main_set.json \
        --opd-seeds 42 [43] --out results/opd/report_qwen3-8b.json

Labels: A endpoints are X's (x-<name>-<fold>-A-s<seed>, seeds 42 43, explore/x_eval.sh); OPD endpoints are the run names
opd-<name>-<fold>-<arm>-s<seed> evaluated with the same x_eval.sh into the same directory. Units are mechanisms
(= templates); every interval resamples mechanisms 10,000 times with random.Random("opd:<item>:20261029").

Main outcome (§3): a mechanism's outside accuracy on its training fold (the arm trained on that fold; final.json outside
rows, x_report's regrade at 5 %), seeds averaged, both folds pooled. Main test set = Z ∩ privileged-teacher-knows
(results/opd/main_set.json, explore/opd_main_set.py); Z and all templates are reported beside it.
  H-OPD1  P - A on the main set: mean >= 30 points and lower bound > 0 -> supports (b) "signal, not knowledge";
          interval contains 0 or mean < 10 points -> supports (a) "the barrier is not only sparse signal"; else numbers only.
  H-OPD2  P - S: mean >= 30 points and lower bound > 0 -> the gain is the privilege, not the KL regulariser; if H-OPD1 holds
          and H-OPD2 does not, H-OPD1 is read as numbers only.
  H-OPD3  Pk3 - A: interval contains 0 or mean < 10 -> sampled-token distillation is reachability-limited too;
          P - Pk3 >= 20 points -> crossing the barrier needs the signal on unsampled tokens.
  H-OPD4  untestable (OPD-4: no external teacher passed the gate).
Judgements need the main set to have >= 5 mechanisms and every endpoint of the compared arms evaluated on both folds;
otherwise numbers only. Secondary (numbers only): held-out outside / inside gains over the base (x_report.transfer's
cluster units, kept so the numbers are comparable with X); P - A on all templates; wrong-only shortcut share at 2 %
(paper_rigor_eval); inside accuracy on the training fold (a drop > 5 points is reported); test_id (x_report.safety rule);
textbook regime traps (x_report.textbook); truncation; P seed 42 vs 43 agreement on the main set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.paths import EXPERIMENTS_DIR, REPO_ROOT
from atmoscoder_rl.runs import is_gradable
from explore.paper_rigor_eval import Values, shortcut_share
from explore.x_report import DRAWS, FOLDS, Family, avg, ci, textbook, truncation

OPD_ARMS = ("P", "S", "Pk3")
TAG = "20261029"
NUMBERS = "只报数字"


def ci_opd(values: list[float], item: str) -> dict:
    return ci(values, seed=f"opd:{item}:{TAG}")


def diff_opd(a: dict, b: dict, item: str) -> dict:
    keys = sorted(set(a) & set(b))
    return {**ci_opd([a[k] - b[k] for k in keys], item), "per_unit": {k: a[k] - b[k] for k in keys}}


class OPDFamily(Family):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.opd_seeds = args.opd_seeds

    def arm(self, arm: str, fold: str, seed: int) -> str:
        return f"opd-{self.name}-{fold}-{arm}-s{seed}" if arm in OPD_ARMS else super().arm(arm, fold, seed)

    def seeds_of(self, arm: str) -> list[int]:
        return self.opd_seeds if arm in OPD_ARMS else self.seeds

    def labels(self, arm: str) -> list[str]:
        return [self.arm(arm, f, s) for f in FOLDS for s in self.seeds_of(arm)]

    def evaluated(self, label: str) -> bool:
        return self.records(label, "final") is not None and not self.missing(label)

    def side_acc(self, arm: str, side: str, variant: str) -> dict:
        """mechanism -> accuracy of `arm` on its `side` ("train" / "held") templates, seeds averaged, folds pooled;
        a seed whose endpoint is not evaluated is left out (reported in `missing`)."""
        out = {}
        for fold in FOLDS:
            tpls = set(self.templates(fold, side))
            per_seed = [self.accuracy(self.arm(arm, fold, s), variant) for s in self.seeds_of(arm)
                        if self.records(self.arm(arm, fold, s), "final") is not None]
            out.update({t: v for t, v in avg(per_seed).items() if t in tpls})
        return out

    def base_acc(self, variant: str) -> dict:
        return self.accuracy(self.base(), variant)


def subset(d: dict, keys) -> dict:
    return {k: d[k] for k in keys if k in d}


def level(d: dict, item: str) -> dict:
    return {**ci_opd(list(d.values()), item), "per_unit": d}


def main_outcome(fam: OPDFamily, sets: dict, complete: dict) -> dict:
    acc = {a: fam.side_acc(a, "train", "outside") for a in ("A", *OPD_ARMS)}
    out = {"per_arm": {}, "contrasts": {}, "hypotheses": {}}
    for name, keys in sets.items():
        out["per_arm"][name] = {a: level(subset(acc[a], keys), f"level-{a}-{name}") for a in acc}
        out["contrasts"][name] = {f"{x}-{y}": diff_opd(subset(acc[x], keys), subset(acc[y], keys), f"{x}-{y}-{name}")
                                  for x, y in (("P", "A"), ("P", "S"), ("Pk3", "A"), ("P", "Pk3"))}
    judge = sets.get("main_set") is not None and len(sets["main_set"]) >= 5
    c = out["contrasts"].get("main_set", {})

    def ok(name, lo_gt0: bool, min_mean: float) -> bool | None:
        d = c.get(name)
        if not d or d["mean"] is None or not complete.get(name, False) or not judge:
            return None
        return d["mean"] >= min_mean and (d["ci95"][0] > 0 if lo_gt0 else True)

    def null(name, max_mean: float) -> bool | None:
        d = c.get(name)
        if not d or d["mean"] is None or not complete.get(name, False) or not judge:
            return None
        return d["ci95"][0] <= 0 <= d["ci95"][1] or d["mean"] < max_mean

    h1 = ok("P-A", True, .30); h1_null = null("P-A", .10)
    h2 = ok("P-S", True, .30)
    h3_null = null("Pk3-A", .10); h3_gap = ok("P-Pk3", False, .20)
    H = out["hypotheses"]
    H["H-OPD1"] = {"P-A": c.get("P-A"), "verdict": NUMBERS if h1 is None else
                   "支持 b：稠密的特权信号让 RL 学不会的机制学会——缺的不是知识而是信号" if h1 else
                   "支持 a：即使教师逐 token 给出知道替代关系时的分布也学不会——屏障不只是信号稀疏" if h1_null else NUMBERS}
    H["H-OPD2"] = {"P-S": c.get("P-S"), "verdict": NUMBERS if h2 is None else
                   "P 的增益来自特权信息，不是 KL 正则" if h2 else NUMBERS}
    if h1 and h2 is False:
        H["H-OPD1"]["verdict"] = NUMBERS + "（H-OPD2 不成立）"
    H["H-OPD3"] = {"Pk3-A": c.get("Pk3-A"), "P-Pk3": c.get("P-Pk3"),
                   "verdict_Pk3-A": NUMBERS if h3_null is None else "只在学生采到的 token 上打分的蒸馏同样被可达性限制" if h3_null else NUMBERS,
                   "verdict_P-Pk3": NUMBERS if h3_gap is None else "越过屏障靠的是对未采样 token 的信号" if h3_gap else NUMBERS}
    H["H-OPD4"] = {"verdict": "不可检验（OPD-4：没有外部教师过门槛，T 臂未跑）"}
    H["judgement_allowed"] = {"main_set_size_ok": judge, "complete": complete}
    return out


def transfer(fam: OPDFamily) -> dict:
    """x_report.transfer's units (clusters) for the OPD arms: held-out outside / inside accuracy, endpoint - base."""
    out = {}
    for a in OPD_ARMS:
        cell = {}
        for variant in ("outside", "inside"):
            held = {}
            start = {}
            for fold in FOLDS:
                tpls = fam.templates(fold, "held")
                per_seed = [fam.clusters(fam.accuracy(fam.arm(a, fold, s), variant), tpls) for s in fam.seeds_of(a)
                            if fam.records(fam.arm(a, fold, s), "final") is not None]
                held.update(avg(per_seed))
                start.update(fam.clusters(fam.base_acc(variant), tpls))
            cell[variant] = {"held_out": ci_opd(list(held.values()), f"transfer-{a}-{variant}"),
                             "start": ci_opd(list(start.values()), f"transfer-start-{variant}"),
                             "endpoint-start": diff_opd(held, start, f"transfer-{a}-{variant}-diff")}
        out[a] = cell
    return out


def inside_cost(fam: OPDFamily) -> dict:
    base = fam.base_acc("inside")
    out = {}
    for a in ("A", *OPD_ARMS):
        end = fam.side_acc(a, "train", "inside")
        d = diff_opd(end, base, f"inside-{a}")
        out[a] = {"endpoint": ci_opd(list(end.values()), f"inside-level-{a}"), "endpoint-base": {k: v for k, v in d.items() if k != "per_unit"},
                  "note": None if d["mean"] is None or d["mean"] >= -.05 else "原题正确率下降 > 5 点"}
    return out


def shortcut(fam: OPDFamily) -> dict:
    """Wrong-only shortcut share at 2 % on the training fold's outside rows (paper_rigor_eval), endpoint - base, per arm."""
    vals = Values(fam)
    base_v = vals.of(fam.base())
    base = shortcut_share(fam, base_v, .02, wrong_only=True) if base_v is not None else {}
    out = {}
    for a in ("A", *OPD_ARMS):
        end = {}
        for fold in FOLDS:
            tpls = set(fam.templates(fold, "train"))
            per_seed = []
            for s in fam.seeds_of(a):
                v = vals.of(fam.arm(a, fold, s))
                if v is not None:
                    per_seed.append(shortcut_share(fam, v, .02, wrong_only=True))
            end.update({t: x for t, x in avg(per_seed).items() if t in tpls})
        out[a] = {"endpoint": ci_opd(list(end.values()), f"shortcut-level-{a}"),
                  "endpoint-base": {k: v for k, v in diff_opd(end, base, f"shortcut-{a}").items() if k != "per_unit"}}
    return out


def safety(fam: OPDFamily) -> dict:
    def acc(label: str) -> float | None:
        recs = fam.records(label, "test_id")
        g = [float(r["passed"]) for i, r in (recs or {}).items() if i in set(fam.inputs["test_id"]) and is_gradable(r)]
        return fmean(g) if g else None
    st = acc(fam.base())
    out = {}
    for a in OPD_ARMS:
        for lab in fam.labels(a):
            e = acc(lab)
            out[lab] = {"test_id": e, "start": st, "note": None if e is None or st is None or e >= st - .03 else "能力代价"}
    return out


def seed_agreement(fam: OPDFamily, main_set: list[str]) -> dict:
    if len(fam.opd_seeds) < 2:
        return {"note": "P 只有一个种子"}
    per = {}
    for s in fam.opd_seeds:
        d = {}
        for fold in FOLDS:
            lab = fam.arm("P", fold, s)
            if fam.records(lab, "final") is not None:
                d.update(subset(fam.accuracy(lab, "outside"), fam.templates(fold, "train")))
        per[s] = subset(d, main_set)
    s0, s1 = fam.opd_seeds[:2]
    both = sorted(set(per[s0]) & set(per[s1]))
    return {"per_seed": {str(s): level(per[s], f"seed-{s}") for s in per},
            "learned_at_50_both": [t for t in both if per[s0][t] >= .5 and per[s1][t] >= .5],
            "learned_at_50_either": [t for t in both if per[s0][t] >= .5 or per[s1][t] >= .5],
            "mean_abs_difference": fmean(abs(per[s0][t] - per[s1][t]) for t in both) if both else None}


def report(args) -> dict:
    fam = OPDFamily(args)
    ms = json.loads(args.main_set.read_text())
    assert ms["Z"] == json.loads(args.report.read_text())["P1"]["Z"], "main_set.json was built from another Z"
    sets = {"main_set": ms["main_set"], "Z": ms["Z"], "all": sorted(fam.fold_of)}
    labels = {a: fam.labels(a) for a in ("A", *OPD_ARMS)}
    missing = {lab: fam.missing(lab) for a in labels for lab in labels[a] if fam.missing(lab)}
    if fam.missing(fam.base()):
        missing[fam.base()] = fam.missing(fam.base())
    complete = {f"{x}-{y}": all(not fam.missing(lab) for lab in labels[x] + labels[y]) for x, y in (("P", "A"), ("P", "S"), ("Pk3", "A"), ("P", "Pk3"))}
    evaluated = [lab for a in labels for lab in labels[a] if fam.records(lab, "final") is not None]
    return {"family": args.name, "A_seeds": fam.seeds, "opd_seeds": fam.opd_seeds, "draws": DRAWS, "seed_tag": TAG,
            "main_set": {"mechanisms": ms["main_set"], "size": len(ms["main_set"]), "criterion": ms["criterion"]},
            "labels": labels, "missing": missing, "complete_pairs": complete,
            "infra_errors": {lab: e for lab in evaluated if (e := fam.infra_errors(lab))},
            "main_outcome": main_outcome(fam, sets, complete),
            "transfer": transfer(fam), "inside_training_fold": inside_cost(fam), "shortcut_wrong_only_0.02": shortcut(fam),
            "safety": safety(fam), "textbook": textbook(fam, [fam.base(), *evaluated]),
            "truncation": truncation(fam, [fam.base(), *evaluated])["per_label"],
            "seed_agreement_P": seed_agreement(fam, ms["main_set"])}


def pts(x) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def show(r: dict) -> None:
    c = lambda d: (f"{pts(d['mean'])} [{pts(d['ci95'][0])}, {pts(d['ci95'][1])}] n={d['units']}" if d and d.get("ci95") else pts((d or {}).get("mean")))
    print(f"== OPD report: {r['family']}, A seeds {r['A_seeds']}, OPD seeds {r['opd_seeds']}; main set {r['main_set']['size']}: {r['main_set']['mechanisms']}")
    print(f"missing: {r['missing'] or 'none'} | complete pairs: {r['complete_pairs']}")
    mo = r["main_outcome"]
    for name in ("main_set", "Z", "all"):
        print(f"-- {name}: " + "; ".join(f"{a} {c(v)}" for a, v in mo["per_arm"][name].items()))
        print("   " + "; ".join(f"{k} {c(v)}" for k, v in mo["contrasts"][name].items()))
    for h, v in mo["hypotheses"].items():
        print(f"{h}: " + "; ".join(f"{k}: {x}" for k, x in v.items() if isinstance(x, str)))
    for a, cell in r["transfer"].items():
        print(f"transfer {a}: " + "; ".join(f"{v} {c(cell[v]['endpoint-start'])}" for v in cell))
    print("inside (train fold) endpoint-base: " + "; ".join(f"{a} {c(v['endpoint-base'])}{' ' + v['note'] if v['note'] else ''}" for a, v in r["inside_training_fold"].items()))
    print("shortcut wrong-only 2 % endpoint-base: " + "; ".join(f"{a} {c(v['endpoint-base'])}" for a, v in r["shortcut_wrong_only_0.02"].items()))
    print("test_id: " + "; ".join(f"{lab} {pts(v['test_id'])} (base {pts(v['start'])}){' ' + v['note'] if v['note'] else ''}" for lab, v in r["safety"].items()))
    sa = r["seed_agreement_P"]
    print(f"P seeds: {sa.get('note') or 'both >= 50: ' + str(sa['learned_at_50_both']) + '; either: ' + str(sa['learned_at_50_either'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43], help="A seeds (X)")
    ap.add_argument("--opd-seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--main-set", type=Path, default=REPO_ROOT / "results" / "opd" / "main_set.json")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "x" / "report_qwen3-8b.json", help="X report (P1.Z)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    show(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
