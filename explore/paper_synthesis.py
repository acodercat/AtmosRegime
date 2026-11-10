"""PREREG_PAPER_ANALYSES §13: the cross-intervention picture (post hoc / exploratory; Qwen3-8B).

    cd $ATMOS_REPO && python -m explore.paper_synthesis --eval-dir $ATMOS_ROOT/experiments/_x_eval \
        --final $ATMOS_ROOT/experiments/_inputs/kc/final.json --inputs $ATMOS_ROOT/experiments/_inputs/x_eval \
        --out results/paper/synthesis.json
    python -m explore.paper_synthesis --selftest

S1 error anatomy: on each template's training-fold outside rows, the wrong answers (5 %, units="benchmark") split into
"equals the standard-formula value" (every sub-answer within 2 % of shortcut_values, as paper_rigor_eval.shortcut_share),
"no gradable output" and "another number"; per arm (A seeds 42/43 averaged; P, S, Pk3, Pk2, Pshuf seed 42) and the base.
Contrasts A - P, Pshuf - P, A - base, P - base on the shortcut share, templates with wrong answers on both sides.
S2 whose reachability bounds which method: templates with base outside accuracy < 0.5; predictors = the student's base
pass@64 (X report P1) and the teacher's given-row accuracy (results/opd/main_set.json); outcome = training-fold outside
accuracy >= 0.5; per outcome the AUC of each predictor and the template-paired AUC difference.
Every interval resamples templates 10,000 times with random.Random("paper:synthesis:<item>:20261101").
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import explore.opd_report as R
from atmoscoder_rl.grading import verify
from atmoscoder_rl.paths import EXPERIMENTS_DIR, REPO_ROOT
from explore.paper_rigor import auc, pct
from explore.paper_rigor_eval import Values, close
from explore.x_report import DRAWS, FOLDS, avg, ci

TAG = "20261101"
NUMBERS = "只报数字"
OPD = ("P", "S", "Pk3", "Pk2", "Pshuf")
ARMS = ("A", *OPD)
CATS = ("shortcut", "empty", "other")


def rng(item: str) -> random.Random:
    return random.Random(f"paper:synthesis:{item}:{TAG}")


def ci_syn(values: list[float], item: str) -> dict:
    return ci(values, seed=f"paper:synthesis:{item}:{TAG}")


def classify(v: dict | None, row: dict) -> str | None:
    """None for a right answer; else the wrong answer's kind."""
    if v and verify({k: {"value": x} for k, x in v.items()}, list(row["sub_answers"]), 0.05, units="benchmark")[0]:
        return None
    if not v:
        return "empty"
    sv = row.get("shortcut_values") or {}
    if sv and all(k in v and close(v[k], s, 0.02) for k, s in sv.items()):
        return "shortcut"
    return "other"


def anatomy(rows: dict, values: dict, templates: set[str]) -> dict[str, dict[str, float]]:
    """template -> {category: share of its wrong outside rows} (templates with at least one wrong row)."""
    per: dict[str, list[str]] = {}
    for i, r in rows.items():
        if r["variant"] != "outside" or r["template"] not in templates or i not in values:
            continue
        c = classify(values[i], r)
        if c is not None:
            per.setdefault(r["template"], []).append(c)
    return {t: {c: fmean(float(x == c) for x in cs) for c in CATS} for t, cs in per.items()}


def mean_anatomy(dicts: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for t in sorted({t for d in dicts for t in d}):
        have = [d[t] for d in dicts if t in d]
        out[t] = {c: fmean(h[c] for h in have) for c in CATS}
    return out


def s1(fam, vals) -> dict:
    train = {f: set(fam.templates(f, "train")) for f in FOLDS}
    every = set().union(*train.values())
    by_arm = {"base": anatomy(fam.rows, vals.of(fam.base()) or {}, every)}
    for a in ARMS:
        merged = {}
        for f in FOLDS:
            per_seed = [anatomy(fam.rows, v, train[f]) for s in fam.seeds_of(a) if (v := vals.of(fam.arm(a, f, s))) is not None]
            merged.update(mean_anatomy(per_seed))
        by_arm[a] = merged
    levels = {a: {c: ci_syn([d[c] for d in by_arm[a].values()], f"s1-{a}-{c}") for c in CATS} for a in by_arm}
    contrasts = {}
    for x, y in (("A", "P"), ("Pshuf", "P"), ("A", "base"), ("P", "base"), ("Pk2", "P"), ("S", "base"), ("Pk3", "base")):
        keys = sorted(set(by_arm[x]) & set(by_arm[y]))
        contrasts[f"{x}-{y}"] = ci_syn([by_arm[x][k]["shortcut"] - by_arm[y][k]["shortcut"] for k in keys], f"s1-{x}-{y}")
    d = contrasts["A-P"]
    verdict = (NUMBERS if d["ci95"] is None else
               "RL 让答错的越界题收拢到标准公式值，特权蒸馏不会" if d["ci95"][0] > 0 else NUMBERS)
    return {"levels": levels, "contrasts_shortcut": contrasts, "verdict_A-P": verdict,
            "per_template": {a: by_arm[a] for a in by_arm}}


def auc_pair(pairs: dict[str, tuple[float, float, bool]], item: str) -> dict:
    """pairs: template -> (student predictor, teacher predictor, learned). AUCs and the paired student - teacher difference."""
    ts = sorted(pairs)
    def both(keys):
        s = auc([(pairs[k][0], pairs[k][2]) for k in keys]); t = auc([(pairs[k][1], pairs[k][2]) for k in keys])
        return s, t
    s, t = both(ts)
    g = rng(item)
    diffs = []
    for _ in range(DRAWS):
        keys = [g.choice(ts) for _ in ts]
        bs, bt = both(keys)
        if bs is not None and bt is not None:
            diffs.append(bs - bt)
    return {"n": len(ts), "learned": sum(p[2] for p in pairs.values()), "auc_student": s, "auc_teacher": t,
            "student_minus_teacher": None if s is None or t is None else s - t, "ci95": pct(diffs), "draws_used": len(diffs)}


def s2(fam, per_mech: dict, given: dict) -> dict:
    base = fam.base_acc("outside")
    units = sorted(t for t, b in base.items() if b < 0.5 and t in per_mech and per_mech[t].get("pass@64") is not None and t in given)
    out = {"units": len(units), "per_outcome": {}}
    for a in ("A", "P", "Pk2", "Pshuf"):
        acc = fam.side_acc(a, "train", "outside")
        pairs = {t: (per_mech[t]["pass@64"], given[t], acc[t] >= 0.5) for t in units if t in acc}
        out["per_outcome"][a] = auc_pair(pairs, f"s2-{a}")
    A, P = out["per_outcome"]["A"], out["per_outcome"]["P"]
    ok = (A["ci95"] is not None and P["ci95"] is not None and A["ci95"][0] > 0 and P["ci95"][1] < 0)
    out["verdict"] = "每种方法受其信号源的可达性限制" if ok else NUMBERS
    return out


def matrix(fam, per_mech: dict, given: dict, types: dict, posthoc: dict) -> dict:
    base = fam.base_acc("outside")
    acc = {a: fam.side_acc(a, "train", "outside") for a in OPD}
    dose = {}
    for o in posthoc.get("observations", []):
        if o.get("dose", 0) > 0:
            dose.setdefault(o["mechanism"], []).append(o["rl"])
    rows = {}
    for t in sorted(fam.fold_of):
        pm = per_mech.get(t, {})
        rows[t] = {"type": types.get(t), "fold": fam.fold_of[t], "base_outside": base.get(t), "pass@64": pm.get("pass@64"),
                   "teacher_given": given.get(t),
                   **{a: (fmean(v) if (v := [pm[k] for k in (f"{a}_s42", f"{a}_s43") if pm.get(k) is not None]) else None) for a in ("A", "D", "F")},
                   "dose_rl": fmean(dose[t]) if t in dose else None, **{a: acc[a].get(t) for a in OPD}}
    return rows


def run(args) -> dict:
    R.OPD_ARMS = OPD
    fam = R.OPDFamily(SimpleNamespace(name="qwen3-8b", seeds=[42, 43], opd_seeds=[42], eval_dir=args.eval_dir,
                                      final=args.final, inputs=args.inputs))
    per_mech = json.loads(args.x_report.read_text())["P1"]["per_mechanism"]
    given = {t: v["given_acc"] for t, v in json.loads(args.main_set.read_text())["per_template"].items()}
    types = json.loads(args.types.read_text())["templates"]
    posthoc = json.loads(args.posthoc.read_text())
    vals = Values(fam)
    missing = {lab: fam.missing(lab) for a in ARMS for lab in fam.labels(a) if fam.missing(lab)}
    return {"family": "qwen3-8b", "draws": DRAWS, "seed_tag": f"paper:synthesis:<item>:{TAG}", "missing": missing,
            "S1_error_anatomy": s1(fam, vals), "S2_reachability_source": s2(fam, per_mech, given),
            "matrix": matrix(fam, per_mech, given, types, posthoc)}


def robustness(r: dict) -> dict:
    """§13 addendum (added after S1 was read): A - P on the shortcut share without the wrong-only selection (every outside
    row: wrong-only share x (1 - accuracy)) and on the templates neither arm learned (both < 0.5)."""
    m, pt = r["matrix"], r["S1_error_anatomy"]["per_template"]

    def uncond(arm: str, t: str) -> float | None:
        acc, s = m[t][arm], pt[arm].get(t, {}).get("shortcut")
        if acc is None:
            return None
        return 0.0 if acc >= 1.0 else None if s is None else s * (1 - acc)
    keys = [t for t in m if uncond("A", t) is not None and uncond("P", t) is not None]
    both = [t for t in m if m[t]["A"] is not None and m[t]["P"] is not None and m[t]["A"] < .5 and m[t]["P"] < .5
            and t in pt["A"] and t in pt["P"]]
    return {"unconditional": {"A": ci_syn([uncond("A", t) for t in keys], "rob-uncond-A"),
                              "P": ci_syn([uncond("P", t) for t in keys], "rob-uncond-P"),
                              "A-P": ci_syn([uncond("A", t) - uncond("P", t) for t in keys], "rob-uncond-A-P")},
            "both_unlearned_wrong_only_A-P": ci_syn([pt["A"][t]["shortcut"] - pt["P"][t]["shortcut"] for t in both], "rob-both-A-P")}


def rel_error(v: dict | None, row: dict) -> float | None:
    """§14 R2: max over sub-answers of |given - expected| / |expected| (expected 0 or non-numeric skipped)."""
    if not v:
        return None
    errs = []
    for sa in row["sub_answers"]:
        e = sa["value"][0] if isinstance(sa["value"], list) else sa["value"]
        x = v.get(sa["sub"])
        try:
            e, x = float(e), float(x)
        except (TypeError, ValueError):
            continue
        if e != 0:
            errs.append(abs(x - e) / abs(e))
    return max(errs) if errs else None


def quartiles(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "q25": q(.25), "median": q(.5), "q75": q(.75), "share_over_0.5": fmean(float(x > 0.5) for x in s)}


def round3(fam, vals, per_mech: dict, given: dict, types: dict, posthoc: dict) -> dict:
    m = matrix(fam, per_mech, given, types, posthoc)
    learned_A = [t for t, v in m.items() if v["base_outside"] is not None and v["base_outside"] < .5 and v["A"] is not None and v["A"] >= .5]
    r1 = {"units": len(learned_A), "per_arm": {}}
    for a in ("A", *OPD):
        xs = [m[t][a] for t in learned_A if m[t][a] is not None]
        r1["per_arm"][a] = {**ci_syn(xs, f"r1-{a}"), "learned": sum(x >= .5 for x in xs)}
    r1["A-S"] = ci_syn([m[t]["A"] - m[t]["S"] for t in learned_A if m[t]["S"] is not None], "r1-A-S")
    kept = {t: (m[t]["teacher_given"], m[t]["teacher_given"], m[t]["P"] >= .5) for t in learned_A if m[t]["P"] is not None}
    r1["P_kept_by_teacher"] = {"kept": sum(k[2] for k in kept.values()), "of": len(kept),
                               "teacher_median_kept": quartiles([k[0] for k in kept.values() if k[2]]).get("median"),
                               "teacher_median_lost": quartiles([k[0] for k in kept.values() if not k[2]]).get("median"),
                               "auc_teacher": auc([(k[0], k[2]) for k in kept.values()])}
    train = {f: set(fam.templates(f, "train")) for f in FOLDS}
    r2 = {}
    for a in ("base", "A", "P", "S", "Pk2", "Pshuf"):
        per_t: dict[str, list[float]] = {}
        if a == "base":
            labels = [(fam.base(), set().union(*train.values()))]
        else:
            labels = [(fam.arm(a, f, s), train[f]) for f in FOLDS for s in fam.seeds_of(a)]
        for lab, tpls in labels:
            v = vals.of(lab)
            if v is None:
                continue
            for i, row in fam.rows.items():
                if row["variant"] != "outside" or row["template"] not in tpls or i not in v:
                    continue
                if classify(v[i], row) in (None, "empty"):
                    continue
                e = rel_error(v[i], row)
                if e is not None:
                    per_t.setdefault(row["template"], []).append(e)
        meds = [sorted(x)[len(x) // 2] for x in per_t.values()]
        r2[a] = quartiles(meds)
    r3 = {}
    units = [t for t, v in m.items() if v["base_outside"] is not None and v["base_outside"] < .5]
    for ty in sorted({m[t]["type"] for t in units if m[t]["type"]}):
        ts = [t for t in units if m[t]["type"] == ty]
        cell = {"templates": len(ts)}
        for a in ("A", "P"):
            xs = [m[t][a] for t in ts if m[t][a] is not None]
            cell[a] = {"learned": sum(x >= .5 for x in xs), "of": len(xs)}
        ds = [m[t]["dose_rl"] for t in ts if m[t]["dose_rl"] is not None]
        cell["DOSE_dosed"] = {"learned": sum(x >= .5 for x in ds), "of": len(ds)}
        r3[ty] = cell
    return {"R1_dynamic_sampling_near_ablation": r1, "R2_error_magnitude": r2, "R3_type_by_method": r3}


def selftest() -> int:
    rows = {f"r{i}": {"variant": "outside", "template": "t", "sub_answers": [{"sub": "a", "value": 10.0, "unit": ""}],
                      "shortcut_values": {"a": 20.0}} for i in range(4)}
    vals = {"r0": {"a": 10.1}, "r1": {"a": 20.2}, "r2": None, "r3": {"a": 33.0}}
    got = [classify(vals[k], rows[k]) for k in ("r0", "r1", "r2", "r3")]
    assert got == [None, "shortcut", "empty", "other"], got
    assert abs(rel_error({"a": 15.0}, rows["r0"]) - 0.5) < 1e-12 and rel_error(None, rows["r0"]) is None
    assert rel_error({"a": 1.0}, {"sub_answers": [{"sub": "a", "value": [0.0, 2.0]}]}) is None   # expected 0 is skipped
    an = anatomy(rows, vals, {"t"})["t"]
    assert abs(an["shortcut"] - 1 / 3) < 1e-9 and abs(an["empty"] - 1 / 3) < 1e-9 and abs(an["other"] - 1 / 3) < 1e-9, an
    g = random.Random(0)
    world = {}
    for i in range(40):
        s, t = g.random(), g.random()
        world[f"m{i}"] = (s, t)
    a_pairs = {m: (s, t, s > 0.5) for m, (s, t) in world.items()}       # A learns iff the student can sample it
    p_pairs = {m: (s, t, t > 0.5) for m, (s, t) in world.items()}       # P learns iff the teacher knows it
    A, P = auc_pair(a_pairs, "selftest-A"), auc_pair(p_pairs, "selftest-P")
    assert A["ci95"][0] > 0 and P["ci95"][1] < 0, (A, P)
    A2 = auc_pair({m: (t, s, y) for m, (s, t, y) in a_pairs.items()}, "selftest-A2")   # swapped predictors flip the sign
    assert A2["ci95"][1] < 0, A2
    print("selftest passed: anatomy classes, AUC crossover detected, swapped labels flip")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--x-report", type=Path, default=REPO_ROOT / "results" / "x" / "report_qwen3-8b.json")
    ap.add_argument("--main-set", type=Path, default=REPO_ROOT / "results" / "opd" / "main_set.json")
    ap.add_argument("--types", type=Path, default=REPO_ROOT / "results" / "paper" / "approx_types.json")
    ap.add_argument("--posthoc", type=Path, default=REPO_ROOT / "results" / "dose" / "posthoc_qwen3-8b.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--robust", type=Path, help="synthesis.json -> its S1 robustness checks (§13 addendum), written to --out")
    ap.add_argument("--round3", action="store_true", help="§14 R1-R3 instead of §13, written to --out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.round3:
        if not a.out:
            ap.error("--out is required")
        R.OPD_ARMS = OPD
        fam = R.OPDFamily(SimpleNamespace(name="qwen3-8b", seeds=[42, 43], opd_seeds=[42], eval_dir=a.eval_dir,
                                          final=a.final, inputs=a.inputs))
        per_mech = json.loads(a.x_report.read_text())["P1"]["per_mechanism"]
        given = {t: v["given_acc"] for t, v in json.loads(a.main_set.read_text())["per_template"].items()}
        r3 = round3(fam, Values(fam), per_mech, given, json.loads(a.types.read_text())["templates"], json.loads(a.posthoc.read_text()))
        a.out.write_text(json.dumps(r3, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps(r3, ensure_ascii=False, indent=1))
        return 0
    if a.robust:
        if not a.out:
            ap.error("--out is required")
        rob = robustness(json.loads(a.robust.read_text()))
        a.out.write_text(json.dumps(rob, indent=1) + "\n")
        print(json.dumps(rob))
        return 0
    if not a.out:
        ap.error("--out is required")
    r = run(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    p = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"
    s1r = r["S1_error_anatomy"]
    print("S1 wrong-answer anatomy (shortcut / empty / other, % of wrong outside rows):")
    for arm, lv in s1r["levels"].items():
        print(f"  {arm:6s} " + "  ".join(f"{c} {p(lv[c]['mean'])}" for c in CATS) + f"  (n={lv['shortcut']['units']})")
    for k, d in s1r["contrasts_shortcut"].items():
        print(f"  {k:12s} {p(d['mean'])} [{p((d['ci95'] or [None, None])[0])}, {p((d['ci95'] or [None, None])[1])}] n={d['units']}")
    print("  verdict A-P:", s1r["verdict_A-P"])
    s2r = r["S2_reachability_source"]
    print(f"S2 units {s2r['units']}:")
    for arm, d in s2r["per_outcome"].items():
        f = lambda x: "-" if x is None else f"{x:.3f}"
        print(f"  {arm:6s} learned {d['learned']}/{d['n']}  AUC student {f(d['auc_student'])} teacher {f(d['auc_teacher'])}"
              f"  student-teacher {f(d['student_minus_teacher'])} {d['ci95']}")
    print("  verdict:", s2r["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
