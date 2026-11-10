"""PREREG_PAPER_ANALYSES §21: the trained Qwen3.5-9B against frontier models evaluated in the same X harness.

    python -m explore.frontier21_report [--runs experiments/frontier21] [--out results/paper/frontier21.json]
    python -m explore.frontier21_report --selftest

Frontier configurations (`explore/frontier21.sh`): textbook.json (46 regime traps + 46 parents) three times each, and
final_think.json (the inside / outside rows of the final set, 70 templates x 20 pairs) once for the thinking-off ones.
S21-a: unit = trap (parents paired with their trap). Frontier per-trap accuracy = mean of the repetitions (gradable
records only); the 9B A arm's per-trap accuracy is read from results/traps_rl/report_qwen35-9b.json (mean of its four
endpoints), the base and the Qwen3-8B A arm likewise. Paired difference = mean over traps of (arm - frontier).
S21-b: unit = cluster (templates averaged within cluster, clusters equal, as X's transfer statistic). The 9B A arm's
held-out per-template accuracy is the endpoint trained on the other fold, averaged over seeds 42 and 43, through
explore.x_report.Family (same regrade cache as results/x); it needs the endpoints' final.json under --heldout-dir and is
skipped (reported as missing) when they are absent. S21-c: completion tokens per textbook row, numbers only.
Intervals: percentile bootstrap over units, 10,000 draws, random.Random("paper:frontier21:<item>:20261107").
Readings (fixed in the pre-registration, thinking-off configurations, S21-a trap difference and S21-b outside difference):
lower bound > 0 "exceeds", interval containing 0 "no detectable difference", upper bound < 0 "still below".
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DRAWS = 10_000
OFF = ("frontier21-deepseek-v4-flash-off", "frontier21-gpt-5.5-off", "frontier21-gpt-5.4-off", "frontier21-minimax-m3-off")
ON = ("frontier21-deepseek-v4-flash-on", "frontier21-gpt-5.5-on")
REPS = (1, 2, 3)
OTHER = {"F1": "F2", "F2": "F1"}
SEEDS = (42, 43)


def ci(values: list[float], item: str) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(f"paper:frontier21:{item}:20261107")
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": len(values)}


def paired(a: dict, b: dict, item: str) -> dict:
    keys = sorted(set(a) & set(b))
    return ci([a[k] - b[k] for k in keys], item)


def reading(d: dict) -> str:
    if d["ci95"] is None:
        return "缺数据"
    lo, hi = d["ci95"]
    return "超过" if lo > 0 else "仍低于" if hi < 0 else "无可检出差别"


def gradable(r: dict) -> bool:
    return not r.get("error") and bool(r.get("attempts")) and r.get("num_attempts", 0) > 0


# ---- frontier runs -----------------------------------------------------------------------------------------------------

def per_id_accuracy(files: list[Path]) -> tuple[dict, dict]:
    """id -> mean passed over the files that hold a gradable record for it; id -> number of records used."""
    acc, n = defaultdict(list), defaultdict(int)
    for f in files:
        for r in json.loads(f.read_text())["results"]:
            if gradable(r):
                acc[r["id"]].append(float(bool(r["passed"])))
                n[r["id"]] += 1
    return {i: fmean(v) for i, v in acc.items()}, dict(n)


def textbook_part(run_dir: Path, groups: dict, arms: dict[str, dict]) -> dict:
    traps = sorted(groups["regime_arm"])
    parent_of = groups["parent_of"]
    files = [run_dir / f"textbook.run{k}.json" for k in REPS]
    present = [f for f in files if f.exists()]
    acc, n = per_id_accuracy(present)
    trap_acc = {t: acc[t] for t in traps if t in acc}
    parent_acc = {t: acc[parent_of[t]] for t in traps if parent_of[t] in acc}
    out = {"repetitions": len(present),
           "records_per_trap": {"min": min(n.values()) if n else 0, "max": max(n.values()) if n else 0},
           "traps_covered": len(trap_acc), "parents_covered": len(parent_acc),
           "trap": ci(list(trap_acc.values()), "trap-level"), "parent": ci(list(parent_acc.values()), "parent-level"),
           "trap_minus_parent": paired(trap_acc, parent_acc, "trap-parent"),
           "usage": usage_summary(present), "vs": {}}
    for name, arm in arms.items():
        out["vs"][name] = {"arm_trap": fmean(arm["trap"][t] for t in trap_acc) if trap_acc else None,
                           "trap": paired(arm["trap"], trap_acc, f"vs-{name}-trap"),
                           "parent": paired(arm["parent"], parent_acc, f"vs-{name}-parent")}
    return out


def usage_summary(files: list[Path]) -> dict:
    toks = [r["usage"]["completion_tokens"] for f in files for r in json.loads(f.read_text())["results"]
            if gradable(r) and r.get("usage")]
    return {"rows": len(toks), "completion_tokens_median": median(toks) if toks else None,
            "completion_tokens_mean": fmean(toks) if toks else None}


def templates_part(run_dir: Path, rows: dict, held: dict | None) -> dict:
    """Per-template inside / outside accuracy of one frontier final run; cluster-level pairing with the 9B held-out arm."""
    f = run_dir / "final.json"
    if not f.exists():
        return {"present": False}
    acc, n = per_id_accuracy([f])
    per = {"inside": defaultdict(list), "outside": defaultdict(list)}
    for i, v in acc.items():
        r = rows[i]
        per[r["variant"]][r["template"]].append(v)
    tpl = {k: {t: fmean(v) for t, v in d.items()} for k, d in per.items()}
    cluster_of = {r["template"]: r["cluster"] for r in rows.values()}
    cl = {k: clusters(tpl[k], cluster_of) for k in tpl}
    out = {"present": True, "rows_graded": len(acc), "rows_total": len(rows), "templates": len(tpl["outside"]),
           "outside": ci(list(cl["outside"].values()), "tpl-outside"), "inside": ci(list(cl["inside"].values()), "tpl-inside"),
           "inside_minus_outside": paired(cl["inside"], cl["outside"], "tpl-in-out"),
           "per_template_outside": tpl["outside"]}
    if held is not None:
        hc = {k: clusters(held[k], cluster_of) for k in ("outside", "inside")}
        out["vs_9b_heldout"] = {"arm_outside": ci(list(hc["outside"].values()), "9b-held-outside"),
                                "arm_inside": ci(list(hc["inside"].values()), "9b-held-inside"),
                                "outside": paired(hc["outside"], cl["outside"], "vs-9b-held-outside"),
                                "inside": paired(hc["inside"], cl["inside"], "vs-9b-held-inside")}
    return out


def clusters(per_tpl: dict, cluster_of: dict) -> dict:
    acc = defaultdict(list)
    for t, v in per_tpl.items():
        acc[cluster_of[t]].append(v)
    return {c: fmean(v) for c, v in acc.items()}


# ---- the trained 9B ----------------------------------------------------------------------------------------------------

def arm_per_trap(report: Path, arm: str) -> dict:
    per = json.loads(report.read_text())["arms"][arm]["per_trap"]
    return {"trap": {t: v["trap"] for t, v in per.items()}, "parent": {t: v["parent"] for t, v in per.items()}}


def heldout_9b(heldout_dir: Path, final_rows: Path, inputs: Path) -> dict | None:
    """template -> held-out accuracy of the Qwen3.5-9B A arm (endpoint trained on the other fold, mean of seeds)."""
    from explore.x_report import Family
    labels = [f"x-qwen35-9b-{f}-A-s{s}" for f in OTHER for s in SEEDS]
    if not all((heldout_dir / l / "final.json").exists() for l in labels):
        return None
    fam = Family(SimpleNamespace(name="qwen35-9b", seeds=list(SEEDS), eval_dir=heldout_dir, final=final_rows, inputs=inputs))
    out = {}
    for variant in ("outside", "inside"):
        per = {}
        for fold in OTHER:  # templates of `fold` are held out for the arm trained on OTHER[fold]
            accs = [fam.accuracy(fam.arm("A", OTHER[fold], s), variant) for s in SEEDS]
            for t in fam.templates(OTHER[fold], "held"):
                per[t] = fmean(a[t] for a in accs)
        out[variant] = per
    return out


def arm_usage(textbook_eval: Path) -> dict:
    files = [textbook_eval / f"x-qwen35-9b-{f}-A-s{s}" / "textbook.json" for f in OTHER for s in SEEDS]
    return usage_summary([f for f in files if f.exists()])


# ---- main ----------------------------------------------------------------------------------------------------------------

def report(runs: Path, inputs: Path, traps_reports: dict[str, Path], heldout_dir: Path, textbook_eval: Path) -> dict:
    groups = json.loads((inputs / "textbook_groups.json").read_text())
    rows = {r["id"]: r for r in json.loads((inputs / "final_think.json").read_text())}
    arms = {"qwen35-9b-A": arm_per_trap(traps_reports["qwen35-9b"], "A"),
            "qwen35-9b-base": arm_per_trap(traps_reports["qwen35-9b"], "base"),
            "qwen3-8b-A": arm_per_trap(traps_reports["qwen3-8b"], "A")}
    held = heldout_9b(heldout_dir, inputs / "final_think.json", inputs)
    if held is not None:   # anchor: the cluster-weighted held-out outside accuracy must equal results/x transfer.A.held_out
        cluster_of = {r["template"]: r["cluster"] for r in rows.values()}
        got = fmean(clusters(held["outside"], cluster_of).values())
        want = json.loads((ROOT / "results/x/report_qwen35-9b_s42-43.json").read_text())["transfer"]["A"]["held_out"]["mean"]
        if abs(got - want) > 1e-9:   # the two-seed report is the one the paper quotes (\heldEndQwenBA)
            raise SystemExit(f"anchor: 9B A held-out outside {got:.6f} differs from results/x transfer {want:.6f}")
    out = {"analysis": "PREREG_PAPER_ANALYSES §21", "configurations": {}, "arm_usage_9b_A": arm_usage(textbook_eval),
           "heldout_9b_available": held is not None, "readings": {}}
    for cfg in OFF + ON:
        d = runs / cfg
        if not d.exists():
            out["configurations"][cfg] = {"present": False}
            continue
        tb = textbook_part(d, groups, arms)
        tp = templates_part(d, rows, held) if cfg in OFF else {"present": False}
        out["configurations"][cfg] = {"thinking": "off" if cfg in OFF else "on", "textbook": tb, "templates": tp}
        if cfg in OFF:
            out["readings"][cfg] = {"traps": reading(tb["vs"]["qwen35-9b-A"]["trap"]),
                                    "templates_outside": reading(tp["vs_9b_heldout"]["outside"]) if tp.get("vs_9b_heldout") else "缺数据"}
    return out


def selftest() -> int:
    import tempfile
    acc = lambda i, o: {"inside": i, "outside": o}  # noqa: E731
    # paired(): the difference over the common keys, and the three readings
    a = {"t1": 1.0, "t2": 1.0, "t3": 0.5}
    b = {"t1": 0.0, "t2": 0.5, "t3": 0.5, "t4": 1.0}
    d = paired(a, b, "selftest")
    assert d["units"] == 3 and abs(d["mean"] - 0.5) < 1e-12 and reading(d) in ("超过", "无可检出差别")   # (1, 0.5, 0) / 3
    assert reading({"ci95": [0.1, 0.3]}) == "超过" and reading({"ci95": [-0.3, -0.1]}) == "仍低于" and reading({"ci95": [-0.1, 0.1]}) == "无可检出差别"
    # textbook_part on a synthetic run: two reps, one API-error record excluded, parents paired with traps
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        groups = {"regime_arm": ["trap_a", "trap_b"], "parent_of": {"trap_a": "pa", "trap_b": "pb"}}
        rec = lambda i, p: {"id": i, "passed": p, "num_attempts": 1, "attempts": [{}], "usage": {"completion_tokens": 100}}  # noqa: E731
        (d / "textbook.run1.json").write_text(json.dumps({"results": [rec("trap_a", True), rec("trap_b", False), rec("pa", True), rec("pb", True)]}))
        (d / "textbook.run2.json").write_text(json.dumps({"results": [rec("trap_a", False), {"id": "trap_b", "passed": False, "num_attempts": 0, "error": "x"},
                                                                       rec("pa", True), rec("pb", False)]}))
        arms = {"arm": {"trap": {"trap_a": 1.0, "trap_b": 1.0}, "parent": {"pa": 1.0, "pb": 1.0}}}
        # the arm's per-trap dict is keyed by trap id for both sides, as results/traps_rl per_trap is
        arms["arm"]["parent"] = {"trap_a": 1.0, "trap_b": 1.0}
        tb = textbook_part(d, groups, arms)
        assert tb["repetitions"] == 2 and tb["records_per_trap"] == {"min": 1, "max": 2}
        assert abs(tb["trap"]["mean"] - 0.25) < 1e-12 and abs(tb["parent"]["mean"] - 0.75) < 1e-12
        assert abs(tb["vs"]["arm"]["trap"]["mean"] - 0.75) < 1e-12 and abs(tb["vs"]["arm"]["parent"]["mean"] - 0.25) < 1e-12
        assert tb["usage"] == {"rows": 7, "completion_tokens_median": 100, "completion_tokens_mean": 100}
        # templates_part: two templates in two clusters, held-out pairing
        rows = {}
        for t, c in (("t1", "c1"), ("t2", "c2")):
            for v in ("inside", "outside"):
                for k in range(2):
                    rows[f"{t}-{v}-{k}"] = {"id": f"{t}-{v}-{k}", "template": t, "cluster": c, "variant": v}
        res = [rec(i, (r["template"] == "t1")) for i, r in rows.items()]
        (d / "final.json").write_text(json.dumps({"results": res}))
        held = {"outside": {"t1": 0.5, "t2": 1.0}, "inside": {"t1": 1.0, "t2": 1.0}}
        tp = templates_part(d, rows, held)
        assert tp["templates"] == 2 and abs(tp["outside"]["mean"] - 0.5) < 1e-12
        assert abs(tp["vs_9b_heldout"]["outside"]["mean"] - 0.25) < 1e-12   # (0.5-1) + (1-0) over two clusters
        assert tp["vs_9b_heldout"]["inside"]["units"] == 2 and abs(tp["vs_9b_heldout"]["inside"]["mean"] - 0.5) < 1e-12
    print("selftest ok")
    return 0


def fmt(d: dict) -> str:
    if not d or d.get("mean") is None:
        return "—"
    return f"{100 * d['mean']:5.1f} [{100 * d['ci95'][0]:5.1f}, {100 * d['ci95'][1]:5.1f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "experiments/frontier21")
    ap.add_argument("--inputs", type=Path, default=ROOT / "experiments/_inputs/x_eval")
    ap.add_argument("--heldout-dir", type=Path, default=ROOT / "experiments/x_heldout_eval")
    ap.add_argument("--textbook-eval", type=Path, default=ROOT / "experiments/x_textbook/eval")
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper/frontier21.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    out = report(args.runs, args.inputs, {"qwen35-9b": ROOT / "results/traps_rl/report_qwen35-9b.json",
                                          "qwen3-8b": ROOT / "results/traps_rl/report_qwen3-8b.json"},
                 args.heldout_dir, args.textbook_eval)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{'configuration':36s} reps  trap                 parent               trap−parent          A−cfg trap            A−cfg parent          tokens(med)")
    for cfg, c in out["configurations"].items():
        if not c.get("textbook"):
            print(f"{cfg:36s} absent"); continue
        tb = c["textbook"]; v = tb["vs"]["qwen35-9b-A"]
        print(f"{cfg:36s} {tb['repetitions']}     {fmt(tb['trap'])}  {fmt(tb['parent'])}  {fmt(tb['trap_minus_parent'])}  {fmt(v['trap'])}  {fmt(v['parent'])}  {tb['usage']['completion_tokens_median']}")
    print(f"9B A arm textbook completion tokens: {out['arm_usage_9b_A']}")
    print(f"\n{'configuration':36s} outside              inside               in−out               A_held−cfg outside     A_held−cfg inside")
    for cfg, c in out["configurations"].items():
        tp = c.get("templates") or {}
        if not tp.get("present"):
            continue
        v = tp.get("vs_9b_heldout") or {}
        print(f"{cfg:36s} {fmt(tp['outside'])}  {fmt(tp['inside'])}  {fmt(tp['inside_minus_outside'])}  {fmt(v.get('outside'))}  {fmt(v.get('inside'))}")
    print("\nreadings:", json.dumps(out["readings"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
