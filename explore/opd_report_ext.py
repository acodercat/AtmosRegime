"""PREREG_OPD OPD-13 report: the two follow-up arms, against the same A and P endpoints and on the same mechanism sets as
explore/opd_report.py (which stays the report of record for H-OPD1-4 and is not changed). Written before either arm was trained.

    cd $ATMOS_REPO && python -m explore.opd_report_ext --name qwen3-8b --main-set results/opd/main_set.json --opd-seeds 42 \
        --eval-dir ... --final ... --inputs ... --out results/opd/report_ext_qwen3-8b.json

Arms (seed 42, both folds): Pk2 = privileged self-teacher, sampled-token k2 with verl's per-token loss clamp off;
Pshuf = privileged self-teacher shown another template's relation sentence (train/opd/build_opd_data.py --shuffle-hint).
Same units, outcome and resampling as opd_report (mechanisms, training-fold outside accuracy, 10,000 draws,
random.Random("opd:<item>:20261029")); judgements need the main set (>= 5 mechanisms) and both folds of both arms evaluated.
  H-OPD5  Pk2 - A interval contains 0 or mean < 10, and P - Pk2 >= 20 -> sampled-token distillation fails without the clamp
          too: H-OPD3's reading does not rest on the clamp.  Pk2 - A >= 30 with lower bound > 0 -> the clamp was what stopped
          Pk3: H-OPD3's "unsampled tokens" reading is withdrawn (numbers only).  Otherwise numbers only.
  H-OPD6  P - Pshuf >= 30 with lower bound > 0 -> the gain needs the right relation, not any extra sentence in the teacher's
          prompt.  Pshuf - A >= 30 with lower bound > 0 -> an extra sentence does most of it (H-OPD1's reading is narrowed
          to "a privileged prompt", numbers only).  Otherwise numbers only.
Secondary (numbers only), as opd_report: held-out transfer, training-fold inside accuracy, wrong-only shortcut share,
test_id, truncation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import explore.opd_report as R
from atmoscoder_rl.paths import EXPERIMENTS_DIR, REPO_ROOT
from explore.x_report import DRAWS, truncation

NEW = ("Pk2", "Pshuf")
PAIRS = (("Pk2", "A"), ("P", "Pk2"), ("Pshuf", "A"), ("P", "Pshuf"))


def outcome(fam: R.OPDFamily, sets: dict, complete: dict) -> dict:
    acc = {a: fam.side_acc(a, "train", "outside") for a in ("A", "P", *NEW)}
    out = {"per_arm": {}, "contrasts": {}, "hypotheses": {}}
    for name, keys in sets.items():
        out["per_arm"][name] = {a: R.level(R.subset(acc[a], keys), f"level-{a}-{name}") for a in acc}
        out["contrasts"][name] = {f"{x}-{y}": R.diff_opd(R.subset(acc[x], keys), R.subset(acc[y], keys), f"{x}-{y}-{name}")
                                  for x, y in PAIRS}
    judge = len(sets["main_set"]) >= 5
    c = out["contrasts"]["main_set"]

    def d(name):
        v = c.get(name)
        return v if v and v["mean"] is not None and complete.get(name) and judge else None

    def null(name, max_mean):
        v = d(name)
        return None if v is None else (v["ci95"][0] <= 0 <= v["ci95"][1] or v["mean"] < max_mean)

    def big(name, min_mean, lo_gt0):
        v = d(name)
        return None if v is None else (v["mean"] >= min_mean and (v["ci95"][0] > 0 if lo_gt0 else True))

    h5_null, h5_gap, h5_big = null("Pk2-A", .10), big("P-Pk2", .20, False), big("Pk2-A", .30, True)
    h6_gap, h6_big = big("P-Pshuf", .30, True), big("Pshuf-A", .30, True)
    H = out["hypotheses"]
    H["H-OPD5"] = {"Pk2-A": c.get("Pk2-A"), "P-Pk2": c.get("P-Pk2"), "verdict":
                   R.NUMBERS if h5_null is None or h5_gap is None else
                   "去掉截断后只在采样 token 上蒸馏仍学不会：H-OPD3 的读法不依赖截断" if h5_null and h5_gap else
                   "截断是 Pk3 学不会的原因：撤回 H-OPD3 的'未采样 token'读法，改为只报数字" if h5_big else R.NUMBERS}
    H["H-OPD6"] = {"P-Pshuf": c.get("P-Pshuf"), "Pshuf-A": c.get("Pshuf-A"), "verdict":
                   R.NUMBERS if h6_gap is None else
                   "增益需要正确的替代关系，不是教师题面里多了任何一句话" if h6_gap else
                   "多一句话本身就有大部分作用：H-OPD1 的读法收窄为'特权题面'，只报数字" if h6_big else R.NUMBERS}
    H["judgement_allowed"] = {"main_set_size_ok": judge, "complete": complete}
    return out


def report(args) -> dict:
    R.OPD_ARMS = ("P", *NEW)          # the arm labels and the secondary blocks of opd_report, for these arms only
    fam = R.OPDFamily(args)
    ms = json.loads(args.main_set.read_text())
    assert ms["Z"] == json.loads(args.report.read_text())["P1"]["Z"], "main_set.json was built from another Z"
    sets = {"main_set": ms["main_set"], "Z": ms["Z"], "all": sorted(fam.fold_of)}
    labels = {a: fam.labels(a) for a in ("A", "P", *NEW)}
    missing = {lab: fam.missing(lab) for a in labels for lab in labels[a] if fam.missing(lab)}
    complete = {f"{x}-{y}": all(not fam.missing(lab) for lab in labels[x] + labels[y]) for x, y in PAIRS}
    evaluated = [lab for a in NEW for lab in labels[a] if fam.records(lab, "final") is not None]
    return {"family": args.name, "A_seeds": fam.seeds, "opd_seeds": fam.opd_seeds, "draws": DRAWS, "seed_tag": R.TAG,
            "main_set": {"mechanisms": ms["main_set"], "size": len(ms["main_set"])},
            "labels": labels, "missing": missing, "complete_pairs": complete,
            "infra_errors": {lab: e for lab in evaluated if (e := fam.infra_errors(lab))},
            "main_outcome": outcome(fam, sets, complete),
            "transfer": R.transfer(fam), "inside_training_fold": R.inside_cost(fam),
            "shortcut_wrong_only_0.02": R.shortcut(fam), "safety": R.safety(fam),
            "truncation": truncation(fam, evaluated)["per_label"] if evaluated else {}}


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
    mo = r["main_outcome"]
    c = lambda d: (f"{R.pts(d['mean'])} [{R.pts(d['ci95'][0])}, {R.pts(d['ci95'][1])}] n={d['units']}" if d and d.get("ci95") else R.pts((d or {}).get("mean")))
    print(f"== OPD ext report: {r['family']}; main set {r['main_set']['size']}; missing: {r['missing'] or 'none'}")
    for name in ("main_set", "Z", "all"):
        print(f"-- {name}: " + "; ".join(f"{a} {c(v)}" for a, v in mo["per_arm"][name].items()))
        print("   " + "; ".join(f"{k} {c(v)}" for k, v in mo["contrasts"][name].items()))
    for h in ("H-OPD5", "H-OPD6"):
        print(f"{h}: {mo['hypotheses'][h]['verdict']}")
    print("inside (train fold) endpoint-base: " + "; ".join(f"{a} {c(v['endpoint-base'])}" for a, v in r["inside_training_fold"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
