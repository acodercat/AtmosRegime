"""OPD training data (PREREG_OPD draft): the A arm's train.parquet unchanged, plus extra_info["teacher_prompt"] on every outside row =
the prompt of its `given` twin (same pair, in the same file). Every other row, and every row in --no-privilege mode, has no
teacher_prompt, so the teacher scores it under the student's own prompt.

    python train/opd/build_opd_data.py --src $R/data-x/F1-A --out $R/data-opd/F1-priv
    python train/opd/build_opd_data.py --src $R/data-x/F1-A --out $R/data-opd/F1-self --no-privilege
    python train/opd/build_opd_data.py --src $R/data-x/F1-A --out $R/data-opd/F1-privshuf --shuffle-hint F1   # OPD-13 control

--shuffle-hint TAG (PREREG_OPD OPD-13, arm Pshuf): every given twin is its outside row plus one inserted sentence (the
substitute relation). The teacher instead sees the outside row with ANOTHER template's sentence inserted at the same place;
templates are paired by a derangement drawn with random.Random(f"opd:shuffle-hint:{TAG}") over the sorted template list, so
no template keeps its own relation. Refuses if any given twin differs from its outside row by more than one insertion.

All other parquet files of --src are copied byte for byte. Prints and writes counts (out/opd_build.json); refuses to run if any
outside row lacks its given twin.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import shutil
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-privilege", action="store_true")
    ap.add_argument("--shuffle-hint", metavar="TAG", help="Pshuf control: insert another template's relation sentence")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    args.out.mkdir(parents=True)
    for p in sorted(args.src.glob("*.parquet")):
        if p.name != "train.parquet":
            shutil.copy2(p, args.out / p.name)
    df = pd.read_parquet(args.src / "train.parquet")
    given = {e["pair"]: df.iloc[i]["prompt"] for i, e in enumerate(df["extra_info"]) if e.get("variant") == "given"}
    n_priv, missing = 0, []
    if args.shuffle_hint:
        hint, where = {}, {}   # template -> its relation sentence; row index -> insertion offset in the user message
        for i, e in enumerate(df["extra_info"]):
            if e.get("variant") != "outside" or e["pair"] not in given:
                continue
            s, t = df.iloc[i]["prompt"][-1]["content"], given[e["pair"]][-1]["content"]
            ops = [o for o in difflib.SequenceMatcher(None, s, t, autojunk=False).get_opcodes() if o[0] != "equal"]
            if len(ops) != 1 or ops[0][0] != "insert":
                raise SystemExit(f"{e['id']}: given twin is not one insertion ({[o[0] for o in ops]})")
            where[i] = ops[0][1]
            hint.setdefault(e["template"], t[ops[0][3]:ops[0][4]])
        tpls = sorted(hint)
        rng = random.Random(f"opd:shuffle-hint:{args.shuffle_hint}")
        while True:
            perm = tpls[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(tpls, perm)):
                break
        donor = dict(zip(tpls, perm))
    extras = []
    for i, e in enumerate(df["extra_info"]):
        e = dict(e)
        if not args.no_privilege and e.get("variant") == "outside":
            if e["pair"] not in given:
                missing.append(e["id"])
            elif args.shuffle_hint:
                msgs = [dict(m) for m in df.iloc[i]["prompt"]]
                u = msgs[-1]["content"]
                msgs[-1]["content"] = u[:where[i]] + hint[donor[e["template"]]] + u[where[i]:]
                e["teacher_prompt"] = msgs
                e["shuffle_hint_donor"] = donor[e["template"]]
                n_priv += 1
            else:
                e["teacher_prompt"] = given[e["pair"]]
                n_priv += 1
        extras.append(e)
    if missing:
        raise SystemExit(f"{len(missing)} outside rows without a given twin, e.g. {missing[:3]}")
    df = df.copy()
    df["extra_info"] = extras
    df.to_parquet(args.out / "train.parquet")
    summary = {"src": str(args.src), "rows": len(df), "privileged_rows": n_priv, "no_privilege": args.no_privilege,
               "shuffle_hint": args.shuffle_hint, "donors": donor if args.shuffle_hint else None,
               "src_train_sha256": hashlib.sha256((args.src / "train.parquet").read_bytes()).hexdigest(),
               "out_train_sha256": hashlib.sha256((args.out / "train.parquet").read_bytes()).hexdigest()}
    (args.out / "opd_build.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
