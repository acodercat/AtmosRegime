"""PREREG_DOSE (docs/PREREG_DOSE.md): the dose-gradient warm starts. Every unit mechanism of a fold gets 0, 1, 2 or 4 of its
verified warm-start pairs, independently per RL seed; everything else is the fold's W_f warm-start data, row for row.

    # 1. units and the random assignment (local, no GPU): writes results/dose/design.json
    python -m explore.dose design --report results/x/report_qwen3-8b.json \
        --warm-summary F1=sft_x-qwen3-8b-F1.summary.json F2=sft_x-qwen3-8b-F2.summary.json --out results/dose/design.json
    # 2. one warm-start parquet (server, needs the sandbox): the fold's W_f samples, only the unit pairs thinned
    python -m explore.dose build --design results/dose/design.json --fold F1 --seed 42 --kc $ATMOS_ROOT/experiments/_inputs/kc \
        --samples $ATMOS_ROOT/experiments/_x/qwen3-8b/warm_samples_F1.jsonl --raft $ATMOS_ROOT/data/sft_raft.parquet \
        --out $ATMOS_ROOT/data/sft_dose-qwen3-8b-F1-s42.parquet
    # check: --full keeps every pair; the parquet must equal the fold's W_f parquet (--expect), else exit 1
    python -m explore.dose build ... --full --expect $ATMOS_ROOT/data/sft_x-qwen3-8b-F1.parquet --out /tmp/x.parquet

A pair is one given row's first two-sided-correct base program, as an SFT row for the outside problem plus its inside twin
(explore/x_data.warm_rows, unchanged). Units (PREREG_DOSE §2): training-fold mechanisms of the fold without plume_mixing_lid,
base pass@64 <= 0.10 (the X-18 sample, as P1), A outside accuracy < 0.5 (mean of the seeds), at least 4 pairs in W_f.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

LEVELS = (0, 1, 2, 4)
MAX_PASS64 = 0.10
MAX_A = 0.5
MIN_PAIRS = max(LEVELS)
SEEDS = (42, 43)
FOLDS = ("F1", "F2")
EXCLUDED = ("plume_mixing_lid",)   # as x_report.P1_EXCLUDED


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def units(per_mechanism: dict, pairs: dict[str, dict], seeds=SEEDS) -> dict[str, list[str]]:
    """fold -> sorted unit mechanisms. per_mechanism: x_report P1 per_mechanism (fold, pass@64, A_s<seed>)."""
    out = {f: [] for f in FOLDS}
    for t, r in per_mechanism.items():
        if t in EXCLUDED or r.get("pass@64") is None:
            continue
        a = [r[f"A_s{s}"] for s in seeds if r.get(f"A_s{s}") is not None]
        if len(a) != len(seeds):
            raise SystemExit(f"{t}: A accuracy missing for a seed")
        if r["pass@64"] <= MAX_PASS64 and sum(a) / len(a) < MAX_A and pairs[r["fold"]].get(t, 0) >= MIN_PAIRS:
            out[r["fold"]].append(t)
    return {f: sorted(v) for f, v in out.items()}


def assign(mechanisms: list[str], fold: str, seed: int) -> dict[str, int]:
    """Balanced random doses: each level len // 4 times, the remainder levels drawn without replacement."""
    rng = random.Random(f"dose:{fold}:{seed}")
    n = len(mechanisms)
    levels = list(LEVELS) * (n // len(LEVELS)) + rng.sample(LEVELS, n % len(LEVELS))
    order = rng.sample(sorted(mechanisms), n)
    return dict(sorted(zip(order, levels)))


def design(args) -> int:
    report = json.loads(args.report.read_text())
    per = report["P1"]["per_mechanism"]
    pairs = {f: json.loads(p.read_text())["pairs_per_template"] for f, p in args.warm_summary.items()}
    u = units(per, pairs)
    doses = {f: {str(s): assign(u[f], f, s) for s in SEEDS} for f in FOLDS}
    out = {"levels": list(LEVELS), "rule": {"max_base_pass64": MAX_PASS64, "max_A_mean": MAX_A, "min_pairs": MIN_PAIRS,
                                            "excluded": list(EXCLUDED)},
           "inputs": {"report": str(args.report), "report_sha256": sha256(args.report),
                      **{f"warm_summary_{f}_sha256": sha256(p) for f, p in args.warm_summary.items()}},
           "units": u,
           "unit_facts": {t: {"fold": per[t]["fold"], "base_pass64": per[t]["pass@64"],
                              "A": [per[t][f"A_s{s}"] for s in SEEDS], "pairs": pairs[per[t]["fold"]][t]}
                          for f in FOLDS for t in u[f]},
           "doses": doses}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for f in FOLDS:
        print(f, len(u[f]), "units")
        for s in SEEDS:
            print(f"  s{s}", " ".join(f"{t}={d}" for t, d in doses[f][str(s)].items()))
    return 0


def pairs_of(sft: list[dict]) -> list[tuple[dict, dict]]:
    """x_data.warm_rows output -> its (outside, inside) pairs, in order."""
    if len(sft) % 2:
        raise SystemExit("odd number of warm rows")
    out = []
    for o, i in zip(sft[::2], sft[1::2]):
        if not (o["id"].endswith("_outside") and i["id"].endswith("_inside") and o["template"] == i["template"]
                and o["id"][:-len("_outside")] == i["id"][:-len("_inside")]):
            raise SystemExit(f"warm rows not in (outside, inside) pairs at {o['id']}")
        out.append((o, i))
    return out


def thin(sft: list[dict], doses: dict[str, int], fold: str, seed: int) -> tuple[list[dict], dict[str, list[str]]]:
    """Keep every pair of a non-unit template; of a unit template keep `dose` pairs drawn at random (order kept)."""
    pairs = pairs_of(sft)
    by_t: dict[str, list[str]] = {}
    for o, _ in pairs:
        by_t.setdefault(o["template"], []).append(o["id"])
    chosen = {}
    for t, d in sorted(doses.items()):
        have = sorted(by_t.get(t, []))
        if len(have) < d:
            raise SystemExit(f"{t}: dose {d} but only {len(have)} pairs")
        chosen[t] = sorted(random.Random(f"dose-pairs:{fold}:{seed}:{t}").sample(have, d))
    keep = {i for v in chosen.values() for i in v}
    out = [r for o, i in pairs if o["template"] not in doses or o["id"] in keep for r in (o, i)]
    return out, chosen


def build(args) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from explore.x_data import RAFT_SEED, fold_rows, warm_rows
    d = json.loads(args.design.read_text())
    doses = {} if args.full else d["doses"][args.fold][str(args.seed)]
    sft_full, counts = warm_rows(fold_rows(args.kc, args.fold), args.samples)
    sft, chosen = thin(sft_full, doses, args.fold, args.seed)
    raft = pq.read_table(args.raft).to_pylist()
    k = min(int(len(sft_full) * args.raft_ratio), len(raft))   # W_f's RAFT rows, whatever the dose
    extra = [{"messages": r["messages"], "id": r.get("id", ""), "kind": "raft", "template": ""}
             for r in random.Random(RAFT_SEED).sample(raft, k)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sft + extra), args.out)
    kept = {}
    for r in sft[::2]:
        kept[r["template"]] = kept.get(r["template"], 0) + 1
    summary = {"fold": args.fold, "seed": args.seed, "full": args.full, "doses": doses, "chosen_pairs": chosen,
               "pairs_per_template_w_f": counts, "pairs_per_template": kept, "template_rows": len(sft),
               "template_rows_w_f": len(sft_full), "raft_rows": len(extra), "design_sha256": sha256(args.design),
               "samples_sha256": sha256(args.samples), "raft_sha256": sha256(args.raft), "out": str(args.out),
               "out_sha256": sha256(args.out)}
    for t, n in doses.items():
        if kept.get(t, 0) != n:
            raise SystemExit(f"{t}: kept {kept.get(t, 0)} pairs, dose {n}")
    if any(kept.get(t) != n for t, n in counts.items() if t not in doses):
        raise SystemExit("a non-unit template lost pairs")
    if args.expect:
        same_bytes = summary["out_sha256"] == sha256(args.expect)
        same_rows = same_bytes or pq.read_table(args.expect).to_pylist() == pq.read_table(args.out).to_pylist()
        summary["expect"] = {"path": str(args.expect), "same_bytes": same_bytes, "same_rows": same_rows}
        if not same_rows:
            print(json.dumps(summary, indent=1))
            raise SystemExit("the rebuilt parquet differs from --expect")
    print(json.dumps(summary, indent=1))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


def folds_arg(items: list[str]) -> dict[str, Path]:
    out = dict(i.split("=", 1) for i in items)
    if sorted(out) != list(FOLDS):
        raise SystemExit(f"need F1=... F2=..., got {items}")
    return {f: Path(p) for f, p in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design"); d.add_argument("--report", type=Path, required=True)
    d.add_argument("--warm-summary", nargs=2, required=True, metavar="FOLD=PATH"); d.add_argument("--out", type=Path, required=True)
    d.set_defaults(fn=design)
    b = sub.add_parser("build"); b.add_argument("--design", type=Path, required=True); b.add_argument("--fold", required=True, choices=FOLDS)
    b.add_argument("--seed", type=int, required=True, choices=SEEDS); b.add_argument("--kc", type=Path, required=True)
    b.add_argument("--samples", type=Path, required=True); b.add_argument("--raft", type=Path, required=True)
    b.add_argument("--raft-ratio", type=float, default=2.0); b.add_argument("--out", type=Path, required=True)
    b.add_argument("--full", action="store_true", help="keep every pair (reproduces W_f)")
    b.add_argument("--expect", type=Path, help="fail unless the output equals this parquet")
    b.set_defaults(fn=build)
    args = ap.parse_args()
    if args.cmd == "design":
        args.warm_summary = folds_arg(args.warm_summary)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
