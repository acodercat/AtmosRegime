"""PREREG_PAPER_ANALYSES §23: reports of the post-freeze runs.

    python -m explore.f23_report e1 [--eval-dir experiments/x_eval_d5] [--raft data/sft_raft_ids.json] --out results/paper/f23_e1.json
    python -m explore.f23_report e2 [--f23-dir experiments/families_f23_eval] [--d-dir experiments/d_textbook_eval] --out results/paper/f23_e2.json
    python -m explore.f23_report e4 --passk experiments/x_passk/qwen35-27b/passk_final_outside.x18.jsonl --out results/paper/f23_e4.json
    python -m explore.f23_report --selftest

E1 (unit: trap of the D-5 set): an arm's per-trap accuracy is the mean of its present endpoints (fold x seed, one run each);
its start is the base (A, D, F, R; OPD arms) or the warm-up of the same folds (WA, WD). S23-a: arm - start on the 54 regime
traps; S23-b: (regime gain) - (geometry gain), the two families resampled independently (as D-5); parent gain reported.
Warm-started arms and warm-ups also get the numbers without the traps whose parent (or a numeric variant of it) is in the
RAFT background of the warm-up data (--raft: a JSON list of those parent ids). E2: §22's S22-a on all 46 X regime traps for
D's four comparison configurations (D's three runs on the 38 shared traps + the three §23 runs on the 8 missing ones;
parents averaged over every record present), and §21's S21-b on the final-set rows for the two thinking-off ones.
E4: per-mechanism unbiased pass@64 of Qwen3.5-27B on the final-set outside rows (explore.x_report.passk_table), zero-hit
mechanisms, Spearman with Qwen3.5-9B and Qwen3-8B (results/x/report_*.json P1.per_mechanism, resampling mechanisms).
Intervals: percentile bootstrap, 10,000 draws, random.Random("paper:f23:<item>:20261108"). Readings (E1, the A arm of each
family): S23-a lower > 0 "the gain replicates on independent regime traps", contains 0 "no gain detected on independent traps", upper < 0 "decrease";
S23-b lower > 0 "the gain is concentrated on regime traps", contains 0 "no detectable difference from the geometry-trap gain (general improvement)", upper < 0 "larger gain on geometry traps".
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from explore.frontier21_report import clusters, gradable, per_id_accuracy, reading

ROOT = Path(__file__).resolve().parents[1]
DRAWS = 10_000
D5_INPUT = ROOT / "experiments/_inputs/d5/textbook.json"
ARMS = {"qwen3-8b": "x-qwen3-8b", "qwen35-9b": "x-qwen35-9b"}


def ci(values: list[float], item: str) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(f"paper:f23:{item}:20261108")
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": len(values)}


def diff_independent(a: list[float], b: list[float], item: str) -> dict:
    """mean(a) - mean(b), a and b resampled independently (D-5's convention for regime - geometry)."""
    if not a or not b:
        return {"mean": None, "ci95": None, "units": [len(a), len(b)]}
    rng = random.Random(f"paper:f23:{item}:20261108")
    boots = sorted(fmean(rng.choice(a) for _ in a) - fmean(rng.choice(b) for _ in b) for _ in range(DRAWS))
    return {"mean": fmean(a) - fmean(b), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": [len(a), len(b)]}


def reading_b(d: dict) -> str:
    if d["ci95"] is None:
        return "缺数据"
    lo, hi = d["ci95"]
    return "增益集中在 regime 陷阱" if lo > 0 else "几何陷阱增益更大" if hi < 0 else "与几何陷阱的增益无可检出差别（一般性提升）"


def reading_a(d: dict) -> str:
    if d["ci95"] is None:
        return "缺数据"
    lo, hi = d["ci95"]
    return "增益在独立的 regime 陷阱上重复" if lo > 0 else "下降" if hi < 0 else "独立陷阱上未检出增益"


# ---- E1 ------------------------------------------------------------------------------------------------------------------

def d5_groups(items: list[dict]) -> dict:
    traps = [r for r in items if "parent_id" in r]
    fam = lambda r: "regime" if r.get("trap_type") == "regime_boundary" else "geometry"  # noqa: E731
    return {"regime": sorted(r["id"] for r in traps if fam(r) == "regime"),
            "geometry": sorted(r["id"] for r in traps if fam(r) == "geometry"),
            "parent_of": {r["id"]: r["parent_id"] for r in traps}}


def label_acc(eval_dir: Path, label: str) -> dict | None:
    f = eval_dir / label / "textbook_d5.json"
    return per_id_accuracy([f])[0] if f.exists() else None


def fmean0(values) -> float | None:
    values = list(values)
    return fmean(values) if values else None


def mean_of(accs: list[dict]) -> dict:
    keys = set.intersection(*(set(a) for a in accs))
    return {k: fmean(a[k] for a in accs) for k in keys}


def e1_arms(eval_dir: Path) -> dict:
    """name -> {endpoints, start labels} for every arm with at least one evaluated endpoint and an evaluated start."""
    present = {p.name for p in eval_dir.iterdir() if (p / "textbook_d5.json").exists()} if eval_dir.exists() else set()
    out = {}
    for fam, pre in ARMS.items():
        base = f"{pre}-base"
        for arm in ("A", "D", "F", "R", "WA", "WD"):
            eps = sorted(l for l in present if re.fullmatch(rf"{re.escape(pre)}-F[12]-{arm}-s4[23]", l))
            if not eps:
                continue
            if arm in ("WA", "WD"):
                folds = sorted({l.split("-")[-3] for l in eps})
                start = [f"sft-{pre}-{f}" for f in folds]
            else:
                start = [base]
            if all(s in present for s in start):
                out[f"{fam}:{arm}"] = {"endpoints": eps, "start": start, "warm": arm in ("WA", "WD")}
        warms = sorted(l for l in present if re.fullmatch(rf"sft-{re.escape(pre)}-F[12]", l))
        if warms and base in present:
            out[f"{fam}:warm"] = {"endpoints": warms, "start": [base], "warm": True}
    for l in sorted(present):
        m = re.fullmatch(r"opd-qwen3-8b-F[12]-(\w+)-s(4[23])", l)
        if m and "x-qwen3-8b-base" in present:
            out.setdefault(f"qwen3-8b:OPD-{m.group(1)}", {"endpoints": [], "start": ["x-qwen3-8b-base"], "warm": False})["endpoints"].append(l)
    return out


def e1(eval_dir: Path, raft_parents: set[str]) -> dict:
    g = d5_groups(json.loads(D5_INPUT.read_text()))
    arms = e1_arms(eval_dir)
    out = {"analysis": "PREREG_PAPER_ANALYSES §23 E1", "traps": {k: len(g[k]) for k in ("regime", "geometry")},
           "labels_evaluated": sorted(p.name for p in eval_dir.iterdir() if (p / "textbook_d5.json").exists()),
           "raft_parent_traps": {k: sum(g["parent_of"][t] in raft_parents for t in g[k]) for k in ("regime", "geometry")},
           "arms": {}, "readings": {}}
    for name, a in arms.items():
        end = mean_of([label_acc(eval_dir, l) for l in a["endpoints"]])
        st = mean_of([label_acc(eval_dir, l) for l in a["start"]])
        def part(keep):  # noqa: E306
            gain = {k: [end[t] - st[t] for t in g[k] if t in end and t in st and keep(t)] for k in ("regime", "geometry")}
            pg = [end[p] - st[p] for p in sorted({g["parent_of"][t] for t in g["regime"] + g["geometry"] if keep(t)}) if p in end and p in st]
            return {"regime_level": {"arm": fmean0(end[t] for t in g["regime"] if t in end and keep(t)),
                                     "start": fmean0(st[t] for t in g["regime"] if t in st and keep(t))},
                    "geometry_level": {"arm": fmean0(end[t] for t in g["geometry"] if t in end and keep(t)),
                                       "start": fmean0(st[t] for t in g["geometry"] if t in st and keep(t))},
                    "S23a_regime_gain": ci(gain["regime"], f"{name}-regime"), "geometry_gain": ci(gain["geometry"], f"{name}-geometry"),
                    "S23b_regime_minus_geometry": diff_independent(gain["regime"], gain["geometry"], f"{name}-specific"),
                    "parent_gain": ci(pg, f"{name}-parent")}
        entry = {"endpoints": a["endpoints"], "start": a["start"], "all": part(lambda t: True)}
        if a["warm"] and raft_parents:
            entry["without_raft_parents"] = part(lambda t: g["parent_of"][t] not in raft_parents)
        out["arms"][name] = entry
    for fam in ARMS:
        a = out["arms"].get(f"{fam}:A")
        out["readings"][f"{fam}:A"] = ({"S23a": reading_a(a["all"]["S23a_regime_gain"]), "S23b": reading_b(a["all"]["S23b_regime_minus_geometry"])}
                                       if a else {"S23a": "缺数据", "S23b": "缺数据"})
    return out


# ---- E2 ------------------------------------------------------------------------------------------------------------------

E2_CONFIGS = ("qwen3.5-27b.think-off", "gemma4-31b.think-off", "qwen3.5-27b.think-on", "gemma4-31b.think-on")


def e2(f23_dir: Path, d_dir: Path, inputs: Path, heldout_dir: Path) -> dict:
    from explore.frontier21_report import heldout_9b
    from explore.frontier22_report import locate, trap_part
    from explore.frontier21_report import arm_per_trap
    groups = json.loads((inputs / "textbook_groups.json").read_text())
    traps = sorted(groups["regime_arm"])
    arms = {"qwen35-9b-A": arm_per_trap(ROOT / "results/traps_rl/report_qwen35-9b.json", "A")}
    rows = {r["id"]: r for r in json.loads((inputs / "final_think.json").read_text())}
    cluster_of = {r["template"]: r["cluster"] for r in rows.values()}
    held = heldout_9b(heldout_dir, inputs / "final_think.json", inputs)
    held_cl = {k: clusters(held[k], cluster_of) for k in ("outside", "inside")} if held else None
    out = {"analysis": "PREREG_PAPER_ANALYSES §23 E2", "traps": len(traps), "configurations": {}, "readings": {}}
    for key in E2_CONFIGS:
        d_files, _ = locate(d_dir, key)
        f_files = [f23_dir / f"{key}.missing.run{n}.json" for n in (1, 2, 3)]
        f_files = [f for f in f_files if f.exists()]
        if not d_files or len(f_files) < 3:
            out["configurations"][key] = {"present": False, "d_files": len(d_files), "f23_missing_runs": len(f_files)}
            continue
        acc, n = per_id_accuracy(d_files + f_files)
        entry = {"present": True, "textbook": trap_part(acc, n, traps, groups["parent_of"], arms, f"f23-{key}")}
        out["readings"][key] = {"traps46": reading(entry["textbook"]["vs"]["qwen35-9b-A"]["trap"])}
        fin = f23_dir / f"{key}.final.run1.json"
        if fin.exists() and held_cl is not None:
            facc, _ = per_id_accuracy([fin])
            per = {"inside": defaultdict(list), "outside": defaultdict(list)}
            for i, v in facc.items():
                per[rows[i]["variant"]][rows[i]["template"]].append(v)
            cl = {k: clusters({t: fmean(v) for t, v in d.items()}, cluster_of) for k, d in per.items()}
            from explore.frontier22_report import paired
            entry["templates"] = {"rows_graded": len(facc), "outside": ci(list(cl["outside"].values()), f"{key}-tpl-out"),
                                  "inside": ci(list(cl["inside"].values()), f"{key}-tpl-in"),
                                  "vs_9b_heldout": {"outside": paired(held_cl["outside"], cl["outside"], f"{key}-held-out"),
                                                    "inside": paired(held_cl["inside"], cl["inside"], f"{key}-held-in")}}
            out["readings"][key]["templates_outside"] = reading(entry["templates"]["vs_9b_heldout"]["outside"])
        out["configurations"][key] = entry
    return out


# ---- E4 ------------------------------------------------------------------------------------------------------------------

def e4(passk: Path) -> dict:
    from explore.x_report import passk_table, spearman_ci
    big = {t: v[64] for t, v in passk_table(passk).items() if 64 in v}
    out = {"analysis": "PREREG_PAPER_ANALYSES §23 E4", "mechanisms": len(big), "mean_pass@64": fmean(big.values()) if big else None,
           "zero_hit": sorted(t for t, v in big.items() if v == 0), "vs": {}}
    for name in ("qwen35-9b", "qwen3-8b"):
        other = {t: v["pass@64"] for t, v in json.loads((ROOT / f"results/x/report_{name}.json").read_text())["P1"]["per_mechanism"].items()
                 if v.get("pass@64") is not None}
        common = sorted(set(big) & set(other))
        out["vs"][name] = {"common": len(common), "spearman": spearman_ci([(big[t], other[t]) for t in common]),
                           "zero_hit_other": sorted(t for t in common if other[t] == 0),
                           "zero_hit_both": sorted(t for t in common if other[t] == 0 and big[t] == 0)}
    return out


def selftest() -> int:
    import tempfile
    assert reading_a({"ci95": [0.01, 0.2]}).startswith("增益") and reading_b({"ci95": [-0.1, 0.1]}).startswith("与几何")
    d = diff_independent([1.0, 1.0], [0.0, 0.0], "selftest")
    assert d["mean"] == 1.0 and d["ci95"] == [1.0, 1.0]
    with tempfile.TemporaryDirectory() as tmp:
        ev = Path(tmp)
        rec = lambda i, p: {"id": i, "passed": p, "num_attempts": 1, "attempts": [{}]}  # noqa: E731
        for label, passed in (("x-qwen3-8b-base", set()), ("x-qwen3-8b-F1-A-s42", {"r1", "g1"}),
                              ("sft-x-qwen3-8b-F1", {"g1"}), ("x-qwen3-8b-F1-WA-s42", {"r1", "g1"})):
            (ev / label).mkdir()
            (ev / label / "textbook_d5.json").write_text(json.dumps({"results": [rec(i, i in passed) for i in ("r1", "r2", "g1", "p1")]}))
        arms = e1_arms(ev)
        assert set(arms) == {"qwen3-8b:A", "qwen3-8b:WA", "qwen3-8b:warm"}, arms
        assert arms["qwen3-8b:WA"]["start"] == ["sft-x-qwen3-8b-F1"] and arms["qwen3-8b:A"]["start"] == ["x-qwen3-8b-base"]
        global D5_INPUT
        items = [{"id": "r1", "parent_id": "p1", "trap_type": "regime_boundary"}, {"id": "r2", "parent_id": "p1", "trap_type": "regime_boundary"},
                 {"id": "g1", "parent_id": "p1", "trap_type": "geometry_detail"}, {"id": "p1"}]
        (ev.parent / "d5.json").write_text(json.dumps(items))
        keep, D5_INPUT = D5_INPUT, ev.parent / "d5.json"
        try:
            r = e1(ev, {"p1"})
        finally:
            D5_INPUT = keep
        a = r["arms"]["qwen3-8b:A"]["all"]
        assert abs(a["S23a_regime_gain"]["mean"] - 0.5) < 1e-12 and abs(a["geometry_gain"]["mean"] - 1.0) < 1e-12   # (1, 0) / 2; 1
        assert abs(a["S23b_regime_minus_geometry"]["mean"] + 0.5) < 1e-12
        w = r["arms"]["qwen3-8b:WA"]
        assert abs(w["all"]["S23a_regime_gain"]["mean"] - 0.5) < 1e-12 and w["without_raft_parents"]["S23a_regime_gain"]["units"] == 0
        assert r["readings"]["qwen35-9b:A"]["S23a"] == "缺数据"
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", nargs="?", choices=("e1", "e2", "e4"))
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "experiments/x_eval_d5")
    ap.add_argument("--raft", type=Path, help="JSON list of D-5 parent ids present (or as numeric variants) in the RAFT background")
    ap.add_argument("--f23-dir", type=Path, default=ROOT / "experiments/families_f23_eval")
    ap.add_argument("--d-dir", type=Path, default=ROOT / "experiments/d_textbook_eval")
    ap.add_argument("--inputs", type=Path, default=ROOT / "experiments/_inputs/x_eval")
    ap.add_argument("--heldout-dir", type=Path, default=ROOT / "experiments/x_heldout_eval")
    ap.add_argument("--passk", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.part == "e1":
        out = e1(args.eval_dir, set(json.loads(args.raft.read_text())) if args.raft else set())
    elif args.part == "e2":
        out = e2(args.f23_dir, args.d_dir, args.inputs, args.heldout_dir)
    else:
        out = e4(args.passk)
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
