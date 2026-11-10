"""CPU self-test of explore/x_think_report.py on x_report_selftest's fabricated family (qwen3-8b, seed 42). No model, no GPU.

    uv run python -m explore.x_think_report_selftest

The fabricated runs are cut to their inside / outside rows, as THINK=1 writes them; an R arm that solves nothing is added.
Expected values come from the fabrication itself, not from x_report: A and F solve the outside rows of the "known"
clusters of their training fold only, WA every training-fold row, base and R none; inside rows are always right. So on the
training fold A - base = F - base = the share of known clusters, WA = 1, R - base = 0; held out every outside difference is 0
and every inside difference 0; pass@64 > 0 exactly on the known mechanisms, so Spearman is 1. One label's outside rows are
set to 28,672 tokens (counted as truncated), another keeps 8,192-token rows (not counted). A row dropped from the R F2 run
makes that label, and only it, missing. X-36: in the A F1 run one right outside row of a known template becomes a
"prompt too long" error (counted wrong: that template's outside accuracy halves; the label stays complete), one row is
left ungradable after its attempts and one is graded from its reasoning text (both counted); another API error makes the
label missing; a missing pass@64 file is reported.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from explore import x_think_report as xt
from explore.x_report import Family
from explore.x_think_report import ThinkFamily
from explore.x_report_selftest import World, close

N = "qwen3-8b"


def cut(path: Path, tokens_outside: int | None = None, drop_last: bool = False) -> None:
    d = json.loads(path.read_text())
    rows = {r["id"]: r for r in json.loads((path.parents[2] / "final.json").read_text())}
    keep = [r for r in d["results"] if rows[r["id"]]["variant"] in ("inside", "outside")]
    if tokens_outside:
        for r in keep:
            if rows[r["id"]]["variant"] == "outside":
                r["attempts"][-1]["usage"]["completion_tokens"] = tokens_outside
    d["results"] = keep[:-1] if drop_last else keep
    path.write_text(json.dumps(d))


def run(w: World) -> dict:
    args = argparse.Namespace(name=N, seeds=[42], eval_dir=w.root / "eval", final=w.root / "final.json", inputs=w.root / "inputs")
    return xt.report(ThinkFamily(args), w.root / "passk.jsonl")


def main() -> int:
    fails = []
    def check(ok: bool, what: str) -> None:
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    tmp = Path(tempfile.mkdtemp(prefix="x_think_selftest_"))
    try:
        w = World(tmp / "w", N, [42])
        for f in ("F1", "F2"):
            w.specs[f"x-{N}-{f}-R-s42"] = {"out": lambda r: False}
            w.write(f"x-{N}-{f}-R-s42")
        for lab in xt.labels(Family(argparse.Namespace(name=N, seeds=[42], eval_dir=w.root / "eval",
                                                        final=w.root / "final.json", inputs=w.root / "inputs"))):
            cut(w.root / "eval" / lab / "final.json", tokens_outside=28672 if lab == f"x-{N}-F1-F-s42" else None)
        r = run(w)
        known = len({w.cluster[t] for t in w.known}) / len(set(w.cluster.values()))
        print(f"     expected share of known clusters: {known:.4f}")
        check(r["missing"] == [], "complete: no label missing")
        t = r["train_outside"]
        check(close(t["base"]["mean"], 0), "train: base 0")
        check(close(t["A-base"]["mean"], known) and close(t["F-base"]["mean"], known), "train: A - base = F - base = known share")
        check(close(t["WA"]["mean"], 1) and close(t["R-base"]["mean"], 0), "train: WA 1, R - base 0")
        check(all(close(r["held_outside"][f"{a}-base"]["mean"], 0) for a in ("A", "F", "R")), "held outside: A, F, R - base 0")
        check(all(close(r["held_inside"][f"{a}-base"]["mean"], 0) for a in xt.ARMS), "held inside: every arm - base 0")
        check(all(close(r["p1_spearman_pass64_vs_outside"][a]["rho"], 1) for a in ("A", "F")), "P1: Spearman 1 for A and F")
        n_io = sum(1 for x in w.rows if x["variant"] in ("inside", "outside"))
        n_out = sum(1 for x in w.rows if x["variant"] == "outside")
        rows = r["rows"]
        check(rows[f"x-{N}-F1-F-s42"]["truncated"] == n_out, "truncation: 28,672-token rows counted")
        check(rows[f"x-{N}-F2-F-s42"]["truncated"] == 0, "truncation: 8,192-token rows not counted")
        check(all(v["records"] == n_io for v in rows.values()), "every label holds the inside / outside rows only")

        # X-36: overflow counted wrong, ungradable / reasoning-graded counted, other errors -> missing, pass@64 file missing
        w3 = tmp / "w3"; shutil.copytree(w.root, w3)
        lab = f"x-{N}-F1-A-s42"; path = w3 / "eval" / lab / "final.json"
        rows_by_id = {x["id"]: x for x in w.rows}
        t = next(x for x in sorted(w.known) if w.fold_of[x] == "F1")
        d = json.loads(path.read_text()); res = d["results"]
        outs = [r for r in res if rows_by_id[r["id"]]["template"] == t and rows_by_id[r["id"]]["variant"] == "outside"]
        check(len(outs) == 2 and all(r["passed"] for r in outs), "fixture: the chosen known template has 2 right outside rows")
        i = res.index(outs[0])
        res[i] = {"id": outs[0]["id"], "passed": False, "num_attempts": 0, "error": "prompt too long: Error code: 400", "attempts": []}
        ins = [r for r in res if rows_by_id[r["id"]]["variant"] == "inside"]
        ins[0]["failed_reason"] = "no code in response"
        ins[1]["attempts"][-1]["reasoning"] = ins[1]["attempts"][-1]["response"]
        path.write_text(json.dumps(d))
        o3 = object.__new__(World); o3.__dict__ = {**w.__dict__, "root": w3}
        r3 = run(o3)
        args3 = argparse.Namespace(name=N, seeds=[42], eval_dir=w3 / "eval", final=w3 / "final.json", inputs=w3 / "inputs")
        check(close(ThinkFamily(args3).accuracy(lab, "outside")[t], .5), "overflow row counted wrong (template outside 1 -> 0.5)")
        check(close(Family(args3).accuracy(lab, "outside")[t], 1), "control: plain x_report Family would exclude it (stays 1)")
        check(r3["missing"] == [], "overflow row does not make the label missing")
        c = r3["rows"][lab]
        check((c["overflow_counted_wrong"], c["other_errors"], c["ungradable_after_attempts"], c["graded_from_reasoning"]) == (1, 0, 1, 1),
              "per-label counts: overflow 1, other errors 0, ungradable 1, graded from reasoning 1")
        res[i]["error"] = "connection reset"; path.write_text(json.dumps(d))
        check(run(o3)["missing"] == [lab], "another API error makes the label missing")
        args4 = argparse.Namespace(name=N, seeds=[42], eval_dir=w.root / "eval", final=w.root / "final.json", inputs=w.root / "inputs")
        r4 = xt.report(ThinkFamily(args4), w.root / "no_passk.jsonl")
        check(any("pass@64 file" in m for m in r4["missing"]), "missing pass@64 file reported")

        w2 = tmp / "w2"; shutil.copytree(w.root, w2)
        cut(w2 / "eval" / f"x-{N}-F2-R-s42" / "final.json", drop_last=True)
        o = object.__new__(World); o.__dict__ = {**w.__dict__, "root": w2}
        check(run(o)["missing"] == [f"x-{N}-F2-R-s42"], "one row dropped -> that label, and only it, missing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST PASS" if not fails else f"SELFTEST FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
