"""Per-mechanism reachability and given-row knowledge of a base model, from its archived X-style probes (third-family screen).

    python -m explore.family_reach --name ministral3-8b --kc experiments/_inputs/kc \
        --passk experiments/_x/ministral3-8b/passk_final_outside.x18.jsonl \
        --warm experiments/_x/ministral3-8b/warm_samples_F1.jsonl experiments/_x/ministral3-8b/warm_samples_F2.jsonl \
        --out results/x/reach_ministral3-8b.json

Reachability: unbiased pass@{1,8,64} per template over its outside rows of final.json (explore/x2_passk sample file, n = 64,
T 1.0, thinking off; x_report.passk_table). Knowledge: base accuracy on the template's given rows from the W_f warm samples
(8 completions per given row, T 1.0; graded at 2 % as in X, explore/opd_main_set). Sets: Z = pass@64 = 0 (X's P1 definition),
candidates = pass@64 <= 0.10 (PREREG_OPD's relaxed set, the DOSE screen of the third family); "knows" = given >= 0.5,
"proxy" = given >= 0.2 (the second-family feasibility note: about half of outside <= 0.10 & given >= 0.20 converted under DOSE).
Run on qwen3-8b it must reproduce the 14 Z mechanisms of results/x/report_qwen3-8b.json (--check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.reward import grade
from explore.x2_passk import load_jsonl, pass_at_k

TOL, KNOWS, PROXY, CAND = 0.02, 0.5, 0.2, 0.10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kc", type=Path, required=True)
    ap.add_argument("--passk", type=Path, required=True)
    ap.add_argument("--warm", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, help="X report whose P1.Z this run must reproduce")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    rows = {}
    for fold in ("F1", "F2"):
        for r in json.loads((args.kc / f"train_{fold}.json").read_text()):
            rows[r["id"]] = r
    shas = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.passk, *args.warm]}
    # reachability
    per = defaultdict(lambda: defaultdict(list))
    errors, n_rows = 0, 0
    for s in load_jsonl(args.passk):
        if "error" in s:
            errors += 1; continue
        if s.get("variant") != "outside":
            continue
        n_rows += 1
        for k in (1, 8, 64):
            per[s["template"]][k].append(pass_at_k(s["n"], s["c"], k))
    reach = {t: {f"pass@{k}": fmean(v) for k, v in d.items()} for t, d in per.items()}
    # knowledge from the warm samples
    recs = []
    for p in args.warm:
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert all(r["arm"] == "G" and rows[r["id"]]["variant"] == "given" for r in recs)
    empty = [r["id"] for r in recs if not r["completions"]]

    def one(rec):
        return rec["id"], [grade(c, rows[rec["id"]]["sub_answers"], tolerance=TOL).passed for c in rec["completions"]]
    with ThreadPoolExecutor(args.workers) as pool:
        verdicts = dict(pool.map(one, [r for r in recs if r["completions"]]))
    by_tpl = defaultdict(list)
    for i, v in verdicts.items():
        by_tpl[rows[i]["template"]].append((sum(v), len(v)))
    given = {t: {"rows": len(v), "given_acc": fmean(c / n for c, n in v), "pass@8": fmean(float(c > 0) for c, n in v)} for t, v in by_tpl.items()}
    fold_of = {r["template"]: r["cluster_fold"] for r in rows.values()}
    # warm-start pairs per template (explore.x_data.warm_rows: first program per given row right on its own problem, at the
    # inside value and on the outside twin; PREREG_DOSE §2 criterion 3 needs >= 4)
    from explore.x_data import fold_rows, warm_rows
    pairs = {}
    for p in args.warm:
        fold = "F1" if p.name.endswith("F1.jsonl") else "F2"
        _, counts = warm_rows(fold_rows(args.kc, fold), p)
        pairs.update(counts)
    table = {t: {"fold": fold_of.get(t), **reach.get(t, {}), **given.get(t, {}), "pairs": pairs.get(t, 0)}
             for t in sorted(set(reach) | set(given))}
    z = sorted(t for t, r in reach.items() if r["pass@64"] == 0)
    cand = sorted(t for t, r in reach.items() if r["pass@64"] <= CAND)
    knows = lambda t, th: t in given and given[t]["given_acc"] >= th
    out = {"family": args.name, "inputs_sha256": shas, "outside_rows": n_rows, "passk_error_rows": errors, "given_rows_without_samples": empty,
           "mean_pass@1": fmean(r["pass@1"] for r in reach.values()), "mean_pass@64": fmean(r["pass@64"] for r in reach.values()),
           "Z": z, "candidates_pass64_le_0.10": cand,
           "Z_knows_0.5": [t for t in z if knows(t, KNOWS)], "candidates_knows_0.5": [t for t in cand if knows(t, KNOWS)],
           "candidates_proxy_0.2": [t for t in cand if knows(t, PROXY)],
           "candidates_pairs_ge_4": [t for t in cand if table[t]["pairs"] >= 4], "per_mechanism": table}
    if args.check:
        want = json.loads(args.check.read_text())["P1"]["Z"]
        out["check_Z_matches_report"] = sorted(want) == z
        assert sorted(want) == z, (sorted(want), z)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{args.name}: {len(reach)} templates, outside rows {n_rows}, pass@1 mean {out['mean_pass@1']:.3f}, pass@64 mean {out['mean_pass@64']:.3f}")
    print(f"  Z (pass@64 = 0): {len(z)}; knows given >= 0.5: {len(out['Z_knows_0.5'])}")
    print(f"  candidates (pass@64 <= 0.10): {len(cand)}; knows >= 0.5: {len(out['candidates_knows_0.5'])}; proxy >= 0.2: {len(out['candidates_proxy_0.2'])}; pairs >= 4: {len(out['candidates_pairs_ge_4'])}")
    for t in cand:
        r = table[t]
        print(f"    {t:36s} {r['fold']} pass@64 {r.get('pass@64', float('nan')):.3f} pass@1 {r.get('pass@1', float('nan')):.3f} given {r.get('given_acc', float('nan')):.3f} pairs {r['pairs']:2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
