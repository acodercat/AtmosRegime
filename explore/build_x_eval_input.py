"""PREREG_X §5 evaluation inputs (docs/PREREG_X.md), built once and deterministically; plus the completeness check
`explore/x_eval.sh` uses for resume.

    python -m explore.build_x_eval_input --final $ATMOS_ROOT/experiments/_inputs/kc/final.json --out-dir $ATMOS_ROOT/experiments/_inputs/x_eval
    python -m explore.build_x_eval_input --final ... --out-dir ... --check-run $ATMOS_ROOT/experiments/_x_eval/<label>   # exit 0 iff complete

- `judge.json`: the inside / outside rows of `final.json` with `pilot.JUDGE_INSTRUCTION` appended, ids `judge:<id>`
  (`build_detect_input.judged`, exactly the D-3 construction).
- `test_id.json`: the core problems of the frozen `test_id` split.
- `textbook.json`: the test-side textbook regime traps (`traps.arm_ids("regime")`, parent outside `train`) and their parents.
  `textbook_groups.json` names the trap ids of the arm and of the pilot's narrower set (`traps.family == "regime:recognise"`
  among `build_e1_input.build(families=True)`, what the pilot and D-3 reported), each trap with its parent.

`final.json` itself is evaluated as is. An existing file is never overwritten with different content.

`--think` (X-34, X-35) writes only `final_think.json`, the inside / outside rows of `final.json` in their order, for the
thinking-on evaluation; with `--check-run` it checks a thinking-on run against that file alone. No other input or the
manifest is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from atmoscoder_rl import pilot
from atmoscoder_rl.datasets import index_by_id, load_core, split_ids
from atmoscoder_rl.traps import arm_ids, family, trap_index
from explore.build_detect_input import judged
from explore.build_e1_input import build as build_families

#: run-file name (`--out-name` in x_eval.sh) -> input file; "final" is the kc bundle's final.json
EVALS = ("final", "judge", "test_id", "textbook")


def judge_rows(final: list[dict]) -> list[dict]:
    return [judged(r) for r in final if r["variant"] in pilot.JUDGE_LABEL]


def textbook() -> tuple[list[dict], dict]:
    # PREREG: "test-side textbook regime traps and their parents" -> the claim's arm traps.arm_ids("regime") (46, incl. regime twins);
    # the pilot / D-3 set (family regime:recognise of traps.json, 38) is a subset and reported alongside.
    index = trap_index()
    arm = arm_ids("regime", index, set(split_ids("train")))
    traps = [index[i][0] for i in arm]
    pilot_set = sorted(t["id"] for t in build_families(families=True)
                       if "parent_id" in t and family(t) == "regime:recognise")
    missing = set(pilot_set) - set(arm)
    if missing:
        raise SystemExit(f"pilot regime traps outside the arm: {sorted(missing)}")
    core = index_by_id(load_core())
    parents = sorted({t["parent_id"] for t in traps})
    groups = {"regime_arm": arm, "regime_recognise_pilot": pilot_set,
              "parent_of": {t["id"]: t["parent_id"] for t in traps}}
    return traps + [core[p] for p in parents], groups


def write(path: Path, obj) -> None:
    text = json.dumps(obj, indent=1, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with different content; not overwriting")
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def think_rows(final: list[dict]) -> list[dict]:
    return [r for r in final if r["variant"] in ("inside", "outside")]


def inputs(final_path: Path, out_dir: Path) -> dict[str, Path]:
    return {"final": final_path, **{k: out_dir / f"{k}.json" for k in EVALS[1:]}}


def complete(run: Path, paths: dict[str, Path]) -> list[str]:
    """The evaluations of one model directory that are missing or still hold rows without a response."""
    bad = []
    for name, src in paths.items():
        f = run / f"{name}.json"
        want = {r["id"] for r in json.loads(src.read_text())}
        try:
            have = {r["id"] for r in json.loads(f.read_text())["results"] if not r.get("error")}
        except (OSError, json.JSONDecodeError, KeyError):
            have = set()
        if not want <= have:
            bad.append(f"{name} ({len(want - have)} of {len(want)} rows missing)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--check-run", type=Path, help="a model's output directory: exit 0 iff all four evaluations are complete")
    ap.add_argument("--think", action="store_true", help="X-34 / X-35: final_think.json only (inside / outside rows)")
    args = ap.parse_args()
    if args.think:
        sub = args.out_dir / "final_think.json"
        if not args.check_run:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            write(sub, think_rows(json.loads(args.final.read_text())))
            print(f"final_think {len(json.loads(sub.read_text())):5d} rows  sha256 {hashlib.sha256(sub.read_bytes()).hexdigest()}")
            return 0
        bad = complete(args.check_run, {"final": sub})
        print("complete" if not bad else "incomplete: " + "; ".join(bad))
        return 0 if not bad else 1
    paths = inputs(args.final, args.out_dir)
    if args.check_run:
        bad = complete(args.check_run, paths)
        print("complete" if not bad else "incomplete: " + "; ".join(bad))
        return 0 if not bad else 1
    final = json.loads(args.final.read_text())
    tb, groups = textbook()
    core = index_by_id(load_core())
    files = {"judge": judge_rows(final), "test_id": [core[i] for i in split_ids("test_id")], "textbook": tb}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        write(paths[name], rows)
    write(args.out_dir / "textbook_groups.json", groups)
    manifest = {name: {"rows": len(json.loads(p.read_text())), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for name, p in paths.items()}
    write(args.out_dir / "manifest.json", manifest)
    for name, m in manifest.items():
        print(f"{name:9s} {m['rows']:5d} rows  sha256 {m['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
