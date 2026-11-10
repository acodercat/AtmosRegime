"""CPU self-test of explore/x_overlong_report.py on x_report_selftest's fabricated family (named qwen35-9b, seeds 42 and 43).
No model, no GPU.

    uv run python -m explore.x_overlong_report_selftest

FOL is the fabricated F arm except that it solves every outside row and is never truncated. Scenarios: (1) no batch
effect -> FOL - F held-out outside accuracy is the direct difference, 1.0 (F2-F solves no held-out outside row), H - F
unchanged (0); (2) the new-batch base solves every inside row where the old one solves pair 0 only -> batch effect, and
FOL - F held-out inside accuracy must be the gain difference with F seed 42 measured from the OLD base: 0 - (gap + 0) / 2,
the gap computed from the rows; the same-batch s43 difference stays 0; (3) the truncation sentence from the training side
(x_overlong_check.training replaced by a fixed value): under both thresholds -> no longer degenerates, one FOL seed at 0.5 -> still degenerates,
no logs -> incomplete; (4) FOL seed 43 not evaluated -> incomplete, rates still listed.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from explore import x_overlong_check, x_overlong_report as xo, x_report
from explore.x_report_selftest import World, close

HALF = lambda r: r["p"] == 0
N = "qwen35-9b"


def held_gap(w: World) -> float:
    """Independent of x_report: on F2 arms' held-out (F1) templates, the share of inside rows HALF misses, template ->
    cluster -> mean."""
    pairs = {}
    for r in w.rows:
        if r["variant"] == "inside" and r["cluster_fold"] == "F1":
            pairs[r["template"]] = pairs.get(r["template"], 0) + 1
    per_cluster = {}
    for t, n in pairs.items():
        per_cluster.setdefault(w.cluster[t], []).append(1 - 1 / n)
    return sum(sum(v) / len(v) for v in per_cluster.values()) / len(per_cluster)


def build(root: Path, base_ins, b2_ins, fol_seeds=(42, 43)) -> World:
    w = World(root, N, [42, 43])
    w.write(f"x-{N}-base", ins=base_ins)
    w.specs[f"x-{N}-base-b2"] = dict(w.specs[f"x-{N}-base"])
    w.write(f"x-{N}-base-b2", ins=b2_ins)
    for s in fol_seeds:
        w.specs[f"x-{N}-F2-FOL-s{s}"] = dict(w.specs[f"x-{N}-F2-F-s{s}"])
        w.write(f"x-{N}-F2-FOL-s{s}", out=lambda r: True, trunc=lambda r: False)
    return w


def run(w: World, logs=None) -> dict:
    args = argparse.Namespace(name=N, seeds=[42, 43], eval_dir=w.root / "eval", final=w.root / "final.json",
                              inputs=w.root / "inputs")
    return xo.report(x_report.Family(args), logs)


def main() -> int:
    fails = []
    def check(name, ok):
        print(("ok   " if ok else "FAIL ") + name)
        if not ok:
            fails.append(name)

    real_training = x_overlong_check.training
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            r1 = run(build(tmp / "s1", HALF, HALF))
            check("1: batch 无批次效应", r1["batch"] == "无批次效应")
            check("1: nothing missing", r1["missing"] == [])
            o = r1["sides"]["held"]["outside_acc"]
            check("1: F held outside 0, FOL 1", close(o["F"]["mean"], 0) and close(o["FOL"]["mean"], 1))
            check("1: FOL - F held outside = 1.0, direct", close(o["FOL-F"]["mean"], 1) and o["FOL-F_is"] == "direct")
            check("1: FOL s43 - F s43 = 1.0", close(o["FOL_s43-F_s43"]["mean"], 1))
            check("1: H - F unchanged", close(r1["sides"]["held"]["H-F"]["FOL-F"]["mean"], 0))
            check("1: training side reported", "outside_acc" in r1["sides"]["train"])
            t = r1["truncation"]
            check("1: FOL never truncated, F is", all(t[f"x-{N}-F2-FOL-s{s}"]["final_share"] == 0 for s in (42, 43))
                  and all(t[f"x-{N}-F2-F-s{s}"]["final_share"] > 0 for s in (42, 43)))
            check("1: no logs -> 不完整：训练日志缺失", r1["verdict"] == "不完整：训练日志缺失")

            w2 = build(tmp / "s2", HALF, lambda r: True)
            r2 = run(w2)
            g = held_gap(w2)
            print(f"     expected held-out gap from the rows: {100 * g:.2f} points")
            check("2: batch 有批次效应", r2["batch"] == "有批次效应")
            i = r2["sides"]["held"]["inside_acc"]
            check("2: FOL - F held inside = -(gap / 2), gain over own-batch start",
                  close(i["FOL-F"]["mean"], -g / 2) and i["FOL-F_is"] != "direct")
            check("2: direct difference would be 0", close(i["FOL_s43-F_s43"]["mean"], 0) and close(i["FOL"]["mean"], i["F"]["mean"]))

            fake = lambda clip: (lambda logs, run: {"clip_last20": clip(run)})
            x_overlong_check.training = fake(lambda run: .1 if "FOL" in run else .8)
            check("3: both FOL seeds under -> 未再退化", run(w2, Path("logs"))["verdict"] == xo.UNDER)
            x_overlong_check.training = fake(lambda run: .5 if run.endswith("FOL-s43") else .1)
            check("3: one FOL seed at 0.5 -> 仍退化", run(w2, Path("logs"))["verdict"] == xo.STILL)
            x_overlong_check.training = real_training

            r4 = run(build(tmp / "s4", HALF, HALF, fol_seeds=(42,)))
            check("4: FOL s43 missing -> 不完整", r4["verdict"].startswith("不完整") and r4["missing"] == [f"x-{N}-F2-FOL-s43"])
            check("4: rates still listed, no differences", set(r4["sides"]["held"]) == {"rates"}
                  and r4["sides"]["held"]["rates"]["F"]["rows"] > 0)
    finally:
        x_overlong_check.training = real_training
    print("SELFTEST PASS" if not fails else f"SELFTEST FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
