"""One row per mechanism across the whole study (supplementary table; numbers only).

    python -m explore.mechanism_table --out results/paper/mechanism_table.json --md results/paper/mechanism_table.md

Columns: category and cluster (results/paper/atlas.json), fold, base outside accuracy and pass@64 of the three families
(results/x/report_*.json P1.per_mechanism, results/x/reach_ministral3-8b.json), the X arms on the training fold (synthesis.json
matrix: A / D / F seeds averaged; Qwen3.5-9B A from its report), the OPD arms (matrix), and the DOSE observations (dose and RL
outside accuracy per seed, from the two posthoc files). "-" = the mechanism was not a unit of that experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]


def load(p: str):
    return json.loads((ROOT / p).read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper/mechanism_table.json")
    ap.add_argument("--md", type=Path, default=ROOT / "results/paper/mechanism_table.md")
    args = ap.parse_args()
    atlas = load("results/paper/atlas.json")["per_template"]
    matrix = load("results/paper/synthesis.json")["matrix"]
    q8 = load("results/x/report_qwen3-8b.json")["P1"]["per_mechanism"]
    q35 = load("results/x/report_qwen35-9b.json")["P1"]["per_mechanism"]
    mi = load("results/x/reach_ministral3-8b.json")["per_mechanism"]
    dose = {"qwen3-8b": load("results/dose/posthoc_qwen3-8b.json")["observations"],
            "ministral3-8b": load("results/dose_ministral/posthoc_ministral3-8b.json")["observations"]}
    dose_by = {n: {} for n in dose}
    for n, obs in dose.items():
        for o in obs:
            dose_by[n].setdefault(o["mechanism"], {})[str(o["seed"])] = (o["dose"], o["rl"])
    rows = []
    for t in sorted(matrix):
        m = matrix[t]
        row = {"mechanism": t, "category": atlas.get(t, {}).get("category"), "cluster": atlas.get(t, {}).get("cluster"), "fold": m["fold"],
               "base_outside_qwen3-8b": m["base_outside"], "pass64_qwen3-8b": m["pass@64"],
               "pass64_qwen35-9b": q35.get(t, {}).get("pass@64"), "pass64_ministral3-8b": mi.get(t, {}).get("pass@64"),
               "A_qwen3-8b": m["A"], "D_qwen3-8b": m["D"], "F_qwen3-8b": m["F"],
               "A_qwen35-9b": fmean(v for k, v in q35.get(t, {}).items() if k.startswith("A_s")) if any(k.startswith("A_s") for k in q35.get(t, {})) else None,
               "P": m["P"], "Pk2": m["Pk2"], "Pshuf": m["Pshuf"], "S": m["S"],
               "dose_qwen3-8b": dose_by["qwen3-8b"].get(t), "dose_ministral3-8b": dose_by["ministral3-8b"].get(t)}
        rows.append(row)
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")

    def f(x):
        return "-" if x is None else f"{100 * x:.0f}"

    def d(x):
        return "-" if not x else " / ".join(f"d{x[s][0]}:{100 * x[s][1]:.0f}" for s in ("42", "43") if s in x)

    lines = ["# Mechanism table (one row per mechanism; `explore/mechanism_table.py`)", "",
             "Percent. base = Qwen3-8B base outside accuracy; p64 = base pass@64 of each family; A/D/F = Qwen3-8B training-fold outside accuracy after RL "
             "(two seeds averaged); A35 = Qwen3.5-9B A arm; P/Pk2/Pshuf/S = OPD arms; DOSE = dose:RL outside accuracy per seed (42 / 43).", "",
             "| mechanism | category | cluster | fold | base | p64 Q3-8B | p64 Q3.5-9B | p64 Ministral | A | D | F | A35 | P | Pk2 | Pshuf | S | DOSE Q3-8B | DOSE Ministral |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['mechanism']} | {r['category']} | {r['cluster']} | {r['fold']} | {f(r['base_outside_qwen3-8b'])} | {f(r['pass64_qwen3-8b'])} | {f(r['pass64_qwen35-9b'])} | {f(r['pass64_ministral3-8b'])} | "
                     f"{f(r['A_qwen3-8b'])} | {f(r['D_qwen3-8b'])} | {f(r['F_qwen3-8b'])} | {f(r['A_qwen35-9b'])} | {f(r['P'])} | {f(r['Pk2'])} | {f(r['Pshuf'])} | {f(r['S'])} | {d(r['dose_qwen3-8b'])} | {d(r['dose_ministral3-8b'])} |")
    args.md.write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} mechanisms -> {args.out}, {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
