"""PREREG_DOSE DOSE-4: post-hoc readings of the dose results, written after every dose number had been read. Numbers only.

    python -m explore.dose_posthoc --design results/dose/design.json --name qwen3-8b \
        --passk-dir experiments/dose/passk --eval-dir experiments/dose/eval --final <final.json> \
        --x-report results/x/report_qwen3-8b.json --x-report-2 results/x/report_qwen35-9b.json --out results/dose/posthoc_qwen3-8b.json

Per observation (unit mechanism, seed): the warm start's per-sample probability p of a correct outside answer (correct samples over all
samples of the mechanism's 20 final.json outside rows x 64), its pass@64, warm and RL outside accuracy, and how many of the 20 outside
rows the warm start and the RL endpoint answer with the standard-formula (shortcut) value within 2 %. Then: dosed observations binned
by p (< 0.005, 0.005-0.02, >= 0.02) with the number learned (RL >= 0.5); every RL failure's shortcut count; the same p bins over the
X report's base pass@1 against the A arm (no warm start) for all mechanisms; the units' source textbook traps (named in the template
docstrings) that the endpoint evaluation contains, with pass per label; and the second family's mechanisms that would satisfy the
unit rule's first two conditions (base pass@64 <= 0.10, A < 0.5). The p bin edges were chosen after seeing the dose results.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from explore.dose_report import rl_label, warm_label

SUCCESS = 0.5
BINS = ((0.0, 0.005), (0.005, 0.02), (0.02, 2.0))
TEMPLATES = Path(__file__).resolve().parents[1] / "atmoscoder_rl" / "templates"


def close(a, b, tol: float = 0.02) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1e-12)
    except (TypeError, ValueError):
        return False


def shortcut_count(results: dict, rows: list[dict]) -> int:
    n = 0
    for row in rows:
        r = results[row["id"]]
        if r["passed"]:
            continue
        actual = [d.get("actual") for d in (r.get("details") or [])]
        n += any(close(a, s) for a in actual for s in (row.get("shortcut_values") or {}).values())
    return n


def observations(design: dict, name: str, final: list[dict], eval_dir: Path, passk_dir: Path) -> list[dict]:
    outside = defaultdict(list)
    for row in final:
        if row["variant"] == "outside":
            outside[row["template"]].append(row)
    obs = []
    for fold, units in design["units"].items():
        for seed in (42, 43):
            w, rl = warm_label(name, fold, seed), rl_label(name, fold, seed)
            rw = {r["id"]: r for r in json.loads((eval_dir / w / "final.json").read_text())["results"]}
            rr = {r["id"]: r for r in json.loads((eval_dir / rl / "final.json").read_text())["results"]}
            samples = defaultdict(list)
            for line in (passk_dir / f"passk_final_outside.{w}.jsonl").read_text().splitlines():
                r = json.loads(line)
                samples[r["template"]].append((r["c"], r["n"]))
            for m in units:
                rows = outside[m]
                c = samples[m]
                obs.append({"fold": fold, "seed": seed, "mechanism": m, "dose": design["doses"][fold][str(seed)][m],
                            "warm_p": sum(ci for ci, _ in c) / sum(ni for _, ni in c),
                            "warm_pass64": sum(ci > 0 for ci, _ in c) / len(c),
                            "warm": sum(rw[r["id"]]["passed"] for r in rows) / len(rows),
                            "rl": sum(rr[r["id"]]["passed"] for r in rows) / len(rows),
                            "rows": len(rows), "shortcut_warm": shortcut_count(rw, rows), "shortcut_rl": shortcut_count(rr, rows)})
    return sorted(obs, key=lambda o: o["warm_p"])


def bins(items: list[tuple[float, bool]]) -> list[dict]:
    return [{"p_from": lo, "p_to": hi, "n": len(g := [ok for p, ok in items if lo <= p < hi]), "learned": sum(g)} for lo, hi in BINS]


def source_traps(units: list[str], eval_dir: Path, labels: list[str]) -> dict:
    out = {}
    textbook = {lab: {r["id"]: r["passed"] for r in json.loads((eval_dir / lab / "textbook.json").read_text())["results"]}
               for lab in labels}
    ids = set(textbook[labels[0]])
    for m in units:
        doc = (TEMPLATES / f"{m}.py").read_text().split('"""')[1]
        named = re.findall(r"source traps?:\s*([^)\n]+)", doc)
        names = [x.strip().rstrip(".") for s in named for x in s.split(",")]
        hits = sorted({i for n in names for i in ids if i in (n, "trap_" + n)})
        out[m] = {"named": names, "in_eval": {t: {lab: int(textbook[lab][t]) for lab in labels} for t in hits}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--passk-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--x-report", type=Path, help="results/x report of the same family (P1.per_mechanism); a family without an X chain omits it and the A-arm bins")
    ap.add_argument("--x-report-2", type=Path, help="results/x report of a second family, for the unit-rule count")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    design = json.loads(args.design.read_text())
    final = json.loads(args.final.read_text())
    obs = observations(design, args.name, final, args.eval_dir, args.passk_dir)
    dosed = [o for o in obs if o["dose"] >= 1]
    failures = [o for o in obs if o["rl"] < SUCCESS]
    x = json.loads(args.x_report.read_text())["P1"]["per_mechanism"] if args.x_report else {}
    labels = [lab for fold in design["units"] for seed in (42, 43) for lab in (warm_label(args.name, fold, seed), rl_label(args.name, fold, seed))]
    out = {
        "observations": obs,
        "dosed_by_warm_p": bins([(o["warm_p"], o["rl"] >= SUCCESS) for o in dosed]),
        "dose0_by_warm_p": bins([(o["warm_p"], o["rl"] >= SUCCESS) for o in obs if o["dose"] == 0]),
        "rl_failures": {"n": len(failures), "dosed": sum(o["dose"] >= 1 for o in failures),
                        "shortcut_rl_all_rows": sum(o["shortcut_rl"] == o["rows"] for o in failures),
                        "shortcut_warm": sorted(o["shortcut_warm"] for o in failures)},
        "x_a_arm_by_base_p": (bins([(v["pass@1"], (v["A_s42"] + v["A_s43"]) / 2 >= SUCCESS) for v in x.values()]) +
                              [{"p_from": 0.02, "p_to": 0.1, "n": len(g := [(v["A_s42"] + v["A_s43"]) / 2 >= SUCCESS for v in x.values() if 0.02 <= v["pass@1"] < 0.1]), "learned": sum(g)},
                               {"p_from": 0.1, "p_to": 2.0, "n": len(g := [(v["A_s42"] + v["A_s43"]) / 2 >= SUCCESS for v in x.values() if v["pass@1"] >= 0.1]), "learned": sum(g)}]) if x else None,
        "source_traps": source_traps(sorted(sum(design["units"].values(), [])), args.eval_dir, labels),
    }
    if args.x_report_2:
        x2 = json.loads(args.x_report_2.read_text())["P1"]["per_mechanism"]
        a = lambda v: (v["A_s42"] + v.get("A_s43", v["A_s42"])) / 2
        out["second_family_unit_candidates"] = sorted(m for m, v in x2.items() if v["pass@64"] <= 0.10 and a(v) < SUCCESS)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"{'fold':4s} {'seed':4s} {'mechanism':34s} dose {'warm p':>7s} pass@64  warm    RL  shortcut W->RL")
    for o in obs:
        print(f"{o['fold']:4s} {o['seed']:<4d} {o['mechanism']:34s} {o['dose']:<4d} {o['warm_p']:7.4f} {o['warm_pass64']:7.2f} "
              f"{o['warm']:5.2f} {o['rl']:5.2f}  {o['shortcut_warm']:2d}/{o['rows']} -> {o['shortcut_rl']:2d}/{o['rows']}")
    for key in ("dosed_by_warm_p", "dose0_by_warm_p", "x_a_arm_by_base_p"):
        if out[key] is not None:
            print(key, " | ".join(f"[{b['p_from']}, {b['p_to']}): {b['learned']}/{b['n']}" for b in out[key]))
    f = out["rl_failures"]
    print(f"RL failures {f['n']} ({f['dosed']} dosed): endpoint gives the shortcut on every outside row in {f['shortcut_rl_all_rows']}; "
          f"warm shortcut counts {f['shortcut_warm']}")
    for m, s in out["source_traps"].items():
        if s["in_eval"]:
            print(f"source traps of {m}: " + "; ".join(f"{t} passed in {sum(v.values())}/{len(v)} labels" for t, v in s["in_eval"].items()))
    if "second_family_unit_candidates" in out:
        print("second family, unit rule conditions 1-2:", len(out["second_family_unit_candidates"]), out["second_family_unit_candidates"])


if __name__ == "__main__":
    main()
