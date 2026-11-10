"""CPU self-test of explore/x_followup.py on x_report_selftest's fabricated family. No model, no GPU.

    uv run python -m explore.x_followup_selftest

The base solves inside pair 0 only; A solves every inside row. Scenarios: (1) no batch effect, R = base -> "no detectable effect"
and "from the correctness reward" (A - R inside = the gap HALF leaves, computed from the rows); (2) the new-batch base solves every inside
row -> "batch effect", A - R must be the gain difference (the gap), not the direct difference (0); (3) R solves every inside row ->
"the random reward also adds ... points"; (4) no R
evaluation -> incomplete. Plus the zero-hit table and the batch numbers.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from explore import x_followup, x_report
from explore.x_report_selftest import World, close

HALF = lambda r: r["p"] == 0


def half_gap(w: World) -> float:
    """Independent of x_report: what HALF misses on inside rows (1 - 1/pairs per template), template -> cluster -> mean.
    Every template is held out for exactly one fold, so the held-out set and the all-template set are the same."""
    pairs = {}
    for r in w.rows:
        if r["variant"] == "inside":
            pairs[r["template"]] = pairs.get(r["template"], 0) + 1
    per_cluster = {}
    for t, n in pairs.items():
        per_cluster.setdefault(w.cluster[t], []).append(1 - 1 / n)
    return sum(sum(v) / len(v) for v in per_cluster.values()) / len(per_cluster)


def build(root: Path, base_b2_ins, r_ins, with_r: bool = True) -> World:
    w = World(root, "qwen3-8b", [42, 43])
    w.write("x-qwen3-8b-base", ins=HALF)
    w.specs["x-qwen3-8b-base-b2"] = {}
    w.write("x-qwen3-8b-base-b2", ins=base_b2_ins)
    if with_r:
        for f in ("F1", "F2"):
            w.specs[f"x-qwen3-8b-{f}-R-s42"] = {}
            w.write(f"x-qwen3-8b-{f}-R-s42", ins=r_ins)
    return w


def run(w: World) -> dict:
    args = argparse.Namespace(name=w.name, seeds=w.seeds, eval_dir=w.root / "eval", final=w.root / "final.json",
                              inputs=w.root / "inputs")
    fam = x_report.Family(args)
    report = {"P1": {"Z": w.zero[:2]}}
    b = x_followup.batch_check(fam)
    return {"batch": b, "R": x_followup.r_arm(fam, [42], b["verdict"] == "无批次效应", report), "world": w}


def main() -> int:
    fails = []
    def check(name, ok):
        print(("ok   " if ok else "FAIL ") + name)
        if not ok:
            fails.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        r1 = run(build(tmp / "s1", HALF, HALF))
        check("1: batch verdict 无批次效应", r1["batch"]["verdict"] == "无批次效应")
        check("1: batch inside diff 0", close(r1["batch"]["starts"]["x-qwen3-8b-base"]["inside"]["mean"], 0))
        check("1: warm starts without -b2 are skipped", set(r1["batch"]["starts"]) == {"x-qwen3-8b-base"})
        hi = r1["R"]["held_inside"]
        check("1: R - base held inside 0", close(hi["R-base"]["mean"], 0))
        g = half_gap(r1["world"])
        print(f"     expected gap from the rows: {100 * g:.2f} points")
        check("1: A - R held inside = the gap, direct", close(hi["A-R"]["mean"], g) and hi["A-R_is"] == "direct")
        check("1: readings", r1["R"]["readings"] == ["随机奖励无可检出效应", "留出折原题的提高来自正确性奖励，不是随机奖励的效应"])
        w = r1["world"]
        t = w.zero[0]
        zt = r1["R"]["zero_hit_training_fold"][t]
        check("1: zero-hit table has A per seed and R", set(zt) == {"A_s42", "A_s43", "R_s42"})
        check("1: zero-hit R value is 0 (R solves no outside row)", close(zt["R_s42"], 0))

        r2 = run(build(tmp / "s2", lambda r: True, lambda r: True))
        check("2: batch verdict 有批次效应", r2["batch"]["verdict"] == "有批次效应")
        check("2: batch inside diff = the gap", close(r2["batch"]["starts"]["x-qwen3-8b-base"]["inside"]["mean"], g))
        hi = r2["R"]["held_inside"]
        check("2: R - new base held inside 0", close(hi["R-base"]["mean"], 0))
        check("2: A - R is the gain difference (the gap), not direct (0)",
              close(hi["A-R"]["mean"], g) and hi["A-R_is"] != "direct")

        r3 = run(build(tmp / "s3", HALF, lambda r: True))
        hi = r3["R"]["held_inside"]
        check("3: R - base held inside = the gap", close(hi["R-base"]["mean"], g))
        check("3: A - R held inside 0", close(hi["A-R"]["mean"], 0))
        check("3: reading 随机奖励也带来 … 点", r3["R"]["readings"] == [f"随机奖励也带来 R − 基座 {100 * g:.1f} 点（留出折原题）"])

        r4 = run(build(tmp / "s4", HALF, HALF, with_r=False))
        check("4: no R -> 不完整", r4["R"].get("verdict", "").startswith("不完整"))
    print("SELFTEST PASS" if not fails else f"SELFTEST FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
