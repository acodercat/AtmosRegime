"""PREREG_PAPER_ANALYSES §22: the trained Qwen3.5-9B against D's 16 local open configurations on the shared textbook traps.

    python -m explore.frontier22_report [--d-dir experiments/d_textbook_eval] [--out results/paper/frontier22.json]
    python -m explore.frontier22_report --selftest

D (`explore/d_launch.py`, same runner, temperature 0.3, 8,192 / 28,672 caps, 5 % benchmark units) ran every configuration
three times on `experiments/_inputs/families_v4/textbook.json` (398 rows), which holds 38 of the 46 regime traps of the X
textbook set (identical text and answers) and all 46 parents. S22-a: unit = shared trap (38; parents paired with their
trap). A D configuration's per-trap accuracy = mean of its three repetitions (gradable records only; API-error records are
excluded and counted). The 9B A arm, its base and the Qwen3-8B A arm are read from results/traps_rl (mean of endpoints)
and restricted to the 38; the §21 frontier configurations are restricted likewise (numbers only). Every stored verdict is
re-derived with explore.x_report.regrade_response (the grader §21 was checked with); disagreements are counted and, if any,
the regraded verdict is used. S22-b: unit = cluster (32); D cluster-level outside / inside accuracy comes from the committed
D reports (D's own parameter draw, 72 templates, one run), the 9B held-out values from frontier21_report.heldout_9b;
numbers only. File selection: a configuration's files come from the launch group with all three runs complete; when two
groups are complete the restart group named in PREFER is used and the other group's shared-trap accuracy is reported as a
check; the synthetic-probe group is never used. Intervals: percentile bootstrap over units, 10,000 draws,
random.Random("paper:frontier22:<item>:20261108"). Readings (S22-a trap difference, every configuration separately):
lower bound > 0 "exceeds", interval containing 0 "no detectable difference", upper bound < 0 "still below".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, median

from explore.frontier21_report import DRAWS, arm_per_trap, clusters, gradable, heldout_9b, per_id_accuracy, reading

ROOT = Path(__file__).resolve().parents[1]
SUBDIR = "outputs/families_context_v2"
D_CONFIGS = ("qwen3-8b.think-off", "qwen3-8b.think-on", "qwen3.5-9b.think-off", "qwen3.5-9b.think-on",
             "qwen3.5-27b.think-off", "qwen3.5-27b.think-on", "gemma4-e4b.think-off", "gemma4-e4b.think-on",
             "gemma4-31b.think-off", "gemma4-31b.think-on", "granite4.2-8b.think-off", "granite4.2-8b.think-on",
             "granite4.2-30b.think-off", "granite4.2-30b.think-on", "ministral3-8b.think-off", "ministral3-14b.think-off")
FRONTIER = ("frontier21-deepseek-v4-flash-off", "frontier21-gpt-5.5-off", "frontier21-gpt-5.4-off",
            "frontier21-minimax-m3-off", "frontier21-deepseek-v4-flash-on", "frontier21-gpt-5.5-on")
EXCLUDE_GROUPS = {"d_qwen35_27b_114_20261011"}          # synthetic probe (results/d_execution/current_status.json)
PREFER = {"qwen3-8b.think-on": "d_qwen3_8b_on_restart"}  # restart group when two groups are complete
ROWS = 398
REPORTS = (sorted((ROOT / "results/d2").glob("families_report_*.json"))
           + sorted((ROOT / "experiments/d_execution_review").glob("*.json")))


def ci(values: list[float], item: str) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(f"paper:frontier22:{item}:20261108")
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": len(values)}


def paired(a: dict, b: dict, item: str) -> dict:
    keys = sorted(set(a) & set(b))
    return ci([a[k] - b[k] for k in keys], item)


# ---- files ---------------------------------------------------------------------------------------------------------------

def locate(d_dir: Path, key: str) -> tuple[list[Path], dict[str, list[Path]]]:
    """(the three run files to use, other complete groups' files) under <d_dir>/<group>/[outputs/families_context_v2/],
    mirroring the server's $ATMOS_ROOT/experiments layout."""
    complete = {}
    for group in sorted(p for p in d_dir.iterdir() if p.is_dir() and p.name not in EXCLUDE_GROUPS):
        base = group / SUBDIR if (group / SUBDIR).is_dir() else group   # launch groups keep outputs/…; families_context_v2 is flat
        files = [base / f"{key}.textbook.run{n}.json" for n in (1, 2, 3)]
        if all(f.exists() for f in files) and all(len(json.loads(f.read_text())["results"]) == ROWS for f in files):
            complete[group.name] = files
    if not complete:
        return [], {}
    chosen = PREFER.get(key) if len(complete) > 1 else next(iter(complete))
    if chosen not in complete:
        raise SystemExit(f"{key}: complete in {sorted(complete)}, no selection rule")
    return complete[chosen], {g: f for g, f in complete.items() if g != chosen}


def regrade(files: list[Path], items: dict) -> tuple[dict, int]:
    """id -> regraded mean accuracy over the files; number of records whose regraded verdict differs from the stored one.
    Cached next to each file (<name>.regrade.json, keyed by the file's sha256)."""
    from explore.x_report import regrade_response
    acc, disagree = defaultdict(list), 0
    for f in files:
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        cache = f.with_name(f.name[:-5] + ".regrade.json")
        got = json.loads(cache.read_text()) if cache.exists() else {}
        if got.get("source_sha256") != sha:
            recs = [r for r in json.loads(f.read_text())["results"] if gradable(r)]
            def one(r):
                return r["id"], regrade_response(r["attempts"][-1].get("response", ""), items[r["id"]]["sub_answers"])[0]
            with ThreadPoolExecutor(16) as pool:
                got = {"source_sha256": sha, "passed": dict(pool.map(one, recs))}
            cache.write_text(json.dumps(got) + "\n")
        for r in json.loads(f.read_text())["results"]:
            if gradable(r):
                v = got["passed"][r["id"]]
                disagree += int(bool(v) != bool(r["passed"]))
                acc[r["id"]].append(float(bool(v)))
    return {i: fmean(v) for i, v in acc.items()}, disagree


# ---- S22-a ---------------------------------------------------------------------------------------------------------------

def trap_part(acc: dict, n: dict, shared: list[str], parent_of: dict, arms: dict, tag: str) -> dict:
    trap = {t: acc[t] for t in shared if t in acc}
    parent = {t: acc[parent_of[t]] for t in shared if parent_of[t] in acc}
    out = {"traps_covered": len(trap), "parents_covered": len(parent),
           "records_per_trap": {"min": min(n.get(t, 0) for t in shared), "max": max(n.get(t, 0) for t in shared)},
           "trap": ci(list(trap.values()), f"{tag}-trap"), "parent": ci(list(parent.values()), f"{tag}-parent"),
           "trap_minus_parent": paired(trap, parent, f"{tag}-trap-parent"), "vs": {}}
    for name, arm in arms.items():
        a_trap = {t: arm["trap"][t] for t in shared}
        a_parent = {t: arm["parent"][t] for t in shared}
        out["vs"][name] = {"arm_trap": fmean(a_trap.values()), "arm_parent": fmean(a_parent.values()),
                           "trap": paired(a_trap, trap, f"{tag}-vs-{name}-trap"),
                           "parent": paired(a_parent, parent, f"{tag}-vs-{name}-parent")}
    return out


def usage_median(files: list[Path], ids: set[str]) -> dict:
    toks = [r["usage"]["completion_tokens"] for f in files for r in json.loads(f.read_text())["results"]
            if gradable(r) and r["id"] in ids and r.get("usage") and r["usage"].get("completion_tokens") is not None]
    return {"rows": len(toks), "completion_tokens_median": median(toks) if toks else None}


def unique_parent_accuracy(files: list[Path], items: dict) -> float:
    """D's parent_accuracy_unique_parents: mean stored verdict over every trap's parent id, all three repetitions."""
    parents = sorted({it["parent_id"] for it in items.values() if "parent_id" in it})
    reps = [{r["id"]: r for r in json.loads(f.read_text())["results"]} for f in files]
    return fmean(int(bool(rep[p]["passed"])) for rep in reps for p in parents)


# ---- S22-b ---------------------------------------------------------------------------------------------------------------

def report_clusters(key: str) -> dict | None:
    """cluster -> {inside, outside} of one D configuration from the committed D reports (status complete)."""
    found = {}
    for f in REPORTS:
        for c in json.loads(f.read_text())["configurations"]:
            if c["key"] == key and c["status"] == "complete":
                found[str(f)] = c["templates"]["all"]["clusters"]
    if not found:
        return None
    vals = list(found.values())
    if any(v != vals[0] for v in vals[1:]):
        raise SystemExit(f"{key}: the D reports disagree on its clusters: {sorted(found)}")
    return vals[0]


# ---- main ----------------------------------------------------------------------------------------------------------------

def report(d_dir: Path, d_input: Path, inputs: Path, frontier_runs: Path, heldout_dir: Path, textbook_eval: Path) -> dict:
    items = {it["id"]: it for it in json.loads(d_input.read_text())}
    groups = json.loads((inputs / "textbook_groups.json").read_text())
    parent_of = groups["parent_of"]
    shared = sorted(t for t in groups["regime_arm"] if t in items)
    x_items = {it["id"]: it for it in json.loads((inputs / "textbook.json").read_text())}
    for t in shared:   # identical rows on both sides
        assert items[t]["problem"] == x_items[t]["problem"] and items[t]["sub_answers"] == x_items[t]["sub_answers"], t
        assert parent_of[t] in items and items[parent_of[t]]["problem"] == x_items[parent_of[t]]["problem"], t
    from atmoscoder_rl.diagnosis import family
    recognise = sorted(i for i, it in items.items() if "parent_id" in it and family(it) == "regime:recognise")
    if recognise != shared:
        raise SystemExit(f"D's regime:recognise set ({len(recognise)}) is not the shared set ({len(shared)})")
    arms = {"qwen35-9b-A": arm_per_trap(ROOT / "results/traps_rl/report_qwen35-9b.json", "A"),
            "qwen35-9b-base": arm_per_trap(ROOT / "results/traps_rl/report_qwen35-9b.json", "base"),
            "qwen3-8b-A": arm_per_trap(ROOT / "results/traps_rl/report_qwen3-8b.json", "A")}
    rows = {r["id"]: r for r in json.loads((inputs / "final_think.json").read_text())}
    cluster_of = {r["template"]: r["cluster"] for r in rows.values()}
    held = heldout_9b(heldout_dir, inputs / "final_think.json", inputs)
    held_cl = {k: clusters(held[k], cluster_of) for k in ("outside", "inside")} if held else None
    ids = set(shared) | {parent_of[t] for t in shared}
    out = {"analysis": "PREREG_PAPER_ANALYSES §22", "shared_traps": len(shared), "missing_traps": sorted(set(groups["regime_arm"]) - set(shared)),
           "arms_on_shared": {k: {"trap": fmean(v["trap"][t] for t in shared), "parent": fmean(v["parent"][t] for t in shared)} for k, v in arms.items()},
           "configurations": {}, "frontier21_on_shared": {}, "anchors": {}, "readings": {}}
    for key in D_CONFIGS:
        files, others = locate(d_dir, key)
        if not files:
            out["configurations"][key] = {"present": False}
            continue
        stored, n = per_id_accuracy(files)
        regraded, disagree = regrade(files, items)
        acc = regraded if disagree else stored
        errors = sum(1 for f in files for r in json.loads(f.read_text())["results"] if not gradable(r))
        entry = {"present": True, "group": files[0].relative_to(d_dir).parts[0], "thinking": key.rsplit("-", 1)[1],
                 "api_error_records": errors, "regrade_disagreements": disagree,
                 "textbook": trap_part(acc, n, shared, parent_of, arms, key), "usage_shared_rows": usage_median(files, ids)}
        if others:
            entry["other_complete_groups"] = {g: trap_part(per_id_accuracy(f)[0], per_id_accuracy(f)[1], shared, parent_of, {}, f"{key}-{g}")["trap"]["mean"]
                                              for g, f in others.items()}
        cl = report_clusters(key)
        if cl is not None and held_cl is not None:
            d_out = {c: v["outside"] for c, v in cl.items()}
            d_in = {c: v["inside"] for c, v in cl.items()}
            entry["templates"] = {"source": "committed D report, D's own draw, 72 templates, one run", "clusters": len(cl),
                                  "outside": ci(list(d_out.values()), f"{key}-tpl-outside"), "inside": ci(list(d_in.values()), f"{key}-tpl-inside"),
                                  "vs_9b_heldout": {"outside": paired(held_cl["outside"], d_out, f"{key}-vs-held-outside"),
                                                    "inside": paired(held_cl["inside"], d_in, f"{key}-vs-held-inside")}}
        out["configurations"][key] = entry
        out["readings"][key] = reading(entry["textbook"]["vs"]["qwen35-9b-A"]["trap"])
        if key == "qwen3.5-27b.think-off":
            want = None
            for f in REPORTS:
                for c in json.loads(f.read_text())["configurations"]:
                    if c["key"] == key and c["status"] == "complete":
                        want = c["textbook"]["parent_accuracy_unique_parents"]
            got = unique_parent_accuracy(files, items)
            out["anchors"]["qwen3.5-27b.think-off_parent_accuracy_unique_parents"] = {"recomputed": got, "report": want}
            if want is None or abs(got - want) > 1e-9:
                raise SystemExit(f"anchor: Qwen3.5-27B off unique-parent accuracy {got} vs report {want}")
            out["anchors"]["regime_pairs"] = 3 * len(shared)
    if held_cl is not None:
        out["heldout_9b"] = {"outside": ci(list(held_cl["outside"].values()), "9b-held-outside"),
                             "inside": ci(list(held_cl["inside"].values()), "9b-held-inside")}
    for cfg in FRONTIER:
        files = [frontier_runs / cfg / f"textbook.run{n}.json" for n in (1, 2, 3)]
        files = [f for f in files if f.exists()]
        if files:
            acc, n = per_id_accuracy(files)
            out["frontier21_on_shared"][cfg] = trap_part(acc, n, shared, parent_of, arms, cfg)
    base = textbook_eval / "x-qwen35-9b-base" / "textbook.json"
    if base.exists():
        acc, n = per_id_accuracy([base])
        out["anchors"]["x-qwen35-9b-base_on_shared"] = {"trap": fmean(acc[t] for t in shared), "parent": fmean(acc[parent_of[t]] for t in shared)}
    return out


def selftest() -> int:
    import tempfile
    assert reading({"ci95": [0.1, 0.3]}) == "超过" and reading({"ci95": [-0.3, -0.1]}) == "仍低于" and reading({"ci95": [-0.1, 0.1]}) == "无可检出差别"
    d = paired({"a": 1.0, "b": 0.5}, {"a": 0.0, "b": 0.5, "c": 1.0}, "selftest")
    assert d["units"] == 2 and abs(d["mean"] - 0.5) < 1e-12
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rec = lambda i, p: {"id": i, "passed": p, "num_attempts": 1, "attempts": [{"response": ""}], "usage": {"completion_tokens": 10}}  # noqa: E731
        # locate(): a complete group, an incomplete one (two runs) and an excluded one; the preferred group when two are complete
        global ROWS
        ROWS = 2
        for g, runs in (("d_a", (1, 2, 3)), ("d_b", (1, 2)), ("d_qwen35_27b_114_20261011", (1, 2, 3))):
            for n in runs:
                p = root / g / SUBDIR / f"k.think-off.textbook.run{n}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"results": [rec("t", n == 1), rec("p", True)]}))
        files, others = locate(root, "k.think-off")
        assert [f.relative_to(root).parts[0] for f in files] == ["d_a"] * 3 and others == {}
        # a flat group (no outputs/ subdirectory) is found too
        for n in (1, 2, 3):
            q = root / "families_context_v2" / f"flat.think-off.textbook.run{n}.json"
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text(json.dumps({"results": [rec("t", True), rec("p", True)]}))
        assert locate(root, "flat.think-off")[0][0].parent == root / "families_context_v2"
        for n in (1, 2, 3):
            p = root / "d_qwen3_8b_on_restart" / SUBDIR / f"qwen3-8b.think-on.textbook.run{n}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"results": [rec("t", True), rec("p", True)]}))
            q = root / "d_bridge" / SUBDIR / f"qwen3-8b.think-on.textbook.run{n}.json"
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text(json.dumps({"results": [rec("t", False), rec("p", True)]}))
        files, others = locate(root, "qwen3-8b.think-on")
        assert files[0].relative_to(root).parts[0] == "d_qwen3_8b_on_restart" and list(others) == ["d_bridge"]
        # trap_part(): mean over reps, parents paired, arm difference restricted to the shared traps, an error record excluded
        files = [root / "d_a" / SUBDIR / f"k.think-off.textbook.run{n}.json" for n in (1, 2, 3)]
        (root / "d_a" / SUBDIR / "k.think-off.textbook.run3.json").write_text(json.dumps({"results": [
            {"id": "t", "passed": False, "num_attempts": 0, "error": "x"}, rec("p", False)]}))
        acc, n = per_id_accuracy(files)
        arms = {"arm": {"trap": {"t": 1.0, "zz": 0.0}, "parent": {"t": 1.0, "zz": 0.0}}}
        tp = trap_part(acc, n, ["t"], {"t": "p"}, arms, "selftest")
        assert tp["records_per_trap"] == {"min": 2, "max": 2} and abs(tp["trap"]["mean"] - 0.5) < 1e-12   # (1, 0) / 2
        assert abs(tp["parent"]["mean"] - 2 / 3) < 1e-12 and abs(tp["vs"]["arm"]["trap"]["mean"] - 0.5) < 1e-12
        assert abs(tp["vs"]["arm"]["parent"]["mean"] - 1 / 3) < 1e-12 and tp["vs"]["arm"]["arm_trap"] == 1.0
        assert usage_median(files, {"t", "p"}) == {"rows": 5, "completion_tokens_median": 10}
        # unique_parent_accuracy(): parents of every trap family, all repetitions
        items = {"t": {"id": "t", "parent_id": "p"}, "p": {"id": "p"}}
        assert abs(unique_parent_accuracy(files, items) - 2 / 3) < 1e-12
    print("selftest ok")
    return 0


def fmt(d: dict | None) -> str:
    if not d or d.get("mean") is None:
        return "—"
    return f"{100 * d['mean']:5.1f} [{100 * d['ci95'][0]:5.1f}, {100 * d['ci95'][1]:5.1f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d-dir", type=Path, default=ROOT / "experiments/d_textbook_eval")
    ap.add_argument("--d-input", type=Path, default=ROOT / "experiments/_inputs/families_v4/textbook.json")
    ap.add_argument("--inputs", type=Path, default=ROOT / "experiments/_inputs/x_eval")
    ap.add_argument("--frontier-runs", type=Path, default=ROOT / "experiments/frontier21")
    ap.add_argument("--heldout-dir", type=Path, default=ROOT / "experiments/x_heldout_eval")
    ap.add_argument("--textbook-eval", type=Path, default=ROOT / "experiments/x_textbook/eval")
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper/frontier22.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    out = report(args.d_dir, args.d_input, args.inputs, args.frontier_runs, args.heldout_dir, args.textbook_eval)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    a = out["arms_on_shared"]["qwen35-9b-A"]
    print(f"shared traps {out['shared_traps']}; 9B A on them: trap {100 * a['trap']:.1f}, parent {100 * a['parent']:.1f}; "
          f"base trap {100 * out['arms_on_shared']['qwen35-9b-base']['trap']:.1f}")
    print(f"{'configuration':34s} err dis  trap                 parent               trap−parent          A−cfg trap            A−cfg parent          tokens  reading")
    for key, e in list(out["configurations"].items()) + list(out["frontier21_on_shared"].items()):
        if not e.get("present", True):
            print(f"{key:34s} missing")
            continue
        tb = e["textbook"] if "textbook" in e else e
        vs = tb["vs"]["qwen35-9b-A"]
        tok = e.get("usage_shared_rows", {}).get("completion_tokens_median")
        print(f"{key:34s} {e.get('api_error_records', 0):3d} {e.get('regrade_disagreements', 0):3d}  {fmt(tb['trap'])}  {fmt(tb['parent'])}  "
              f"{fmt(tb['trap_minus_parent'])}  {fmt(vs['trap'])}  {fmt(vs['parent'])}  {tok if tok is not None else '—':>6}  {out['readings'].get(key, '只报数字')}")
        if e.get("templates"):
            t = e["templates"]
            print(f"{'':34s}   templates (D draw): outside {fmt(t['outside'])} inside {fmt(t['inside'])}; "
                  f"A held-out − cfg: outside {fmt(t['vs_9b_heldout']['outside'])} inside {fmt(t['vs_9b_heldout']['inside'])}")
        if e.get("other_complete_groups"):
            print(f"{'':34s}   other complete groups (trap mean): {e['other_complete_groups']}")
    print("anchors:", json.dumps(out["anchors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
