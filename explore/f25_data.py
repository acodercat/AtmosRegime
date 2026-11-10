"""PREREG_PAPER_ANALYSES §25-A (E5): the focused pool for prolonged RL — the rows of the A arm's fold-1 pool whose template
is one of the 8 target mechanisms, unchanged, plus the same filter applied to the fold's training-side probe; and the
final.json rows of those templates for the endpoint evaluation (all four variants) and for pass@64 (outside only).

    python -m explore.f25_data --pool $ATMOS_ROOT/data-x/F1-A --final experiments/_inputs/kc/final.json \
        --out-pool $ATMOS_ROOT/data-x/F1-A-f25 --out-inputs experiments/_inputs/f25

Rows are copied, never edited; the script checks that every kept row is byte-identical to its source row and that each
target template has 20 inside / 20 outside / 20 given rows in the pool. Prints the sha256 of every file it writes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

TARGETS = ("diffuse_transmission", "displacement_height_canopy", "freezing_point_depth", "line_broadening",
           "saturated_lapse", "henry_depletion", "unconfined_dupuit", "surface_multiple_reflection")


def info(extra) -> dict:
    return extra if isinstance(extra, dict) else ast.literal_eval(extra)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_parquet(src: Path, dst: Path) -> Counter:
    import pandas as pd
    d = pd.read_parquet(src)
    keep = [i for i, e in enumerate(d["extra_info"]) if info(e).get("template") in TARGETS]
    out = d.iloc[keep].reset_index(drop=True)
    for j, i in enumerate(keep):   # unchanged rows
        assert out.iloc[j].to_dict().__repr__() == d.iloc[i].to_dict().__repr__()
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst)
    return Counter((info(e)["template"], info(e)["variant"]) for e in out["extra_info"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, help="the A arm's fold-1 verl data dir (train.parquet, train_probe.parquet)")
    ap.add_argument("--out-pool", type=Path)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--out-inputs", type=Path, required=True)
    args = ap.parse_args()
    if args.pool:
        c = filter_parquet(args.pool / "train.parquet", args.out_pool / "train.parquet")
        for t in TARGETS:
            assert all(c[(t, v)] == 20 for v in ("inside", "outside", "given")), (t, c)
        assert sum(c.values()) == 480, sum(c.values())
        p = filter_parquet(args.pool / "train_probe.parquet", args.out_pool / "train_probe.parquet")
        print("pool rows", sum(c.values()), "probe rows", sum(p.values()))
        for f in ("train.parquet", "train_probe.parquet"):
            print(sha(args.out_pool / f), args.out_pool / f)
    rows = [r for r in json.loads(args.final.read_text()) if r["template"] in TARGETS]
    assert len(rows) == 640, len(rows)
    args.out_inputs.mkdir(parents=True, exist_ok=True)
    for name, sel in (("final_targets.json", rows), ("final_targets_outside.json", [r for r in rows if r["variant"] == "outside"])):
        f = args.out_inputs / name
        f.write_text(json.dumps(sel, indent=1, ensure_ascii=False) + "\n")
        print(sha(f), f, len(sel), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
