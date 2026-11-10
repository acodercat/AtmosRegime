"""PREREG_PAPER_ANALYSES §20: trained small models against frontier configurations on the same traps.

Reads the per-trap results of every X arm (``results/traps_rl/report_<family>.json``, ``arms.*.per_trap``)
and of the archived frontier configurations (``results/paper/traps_frontier.json``, ``configs.*.per_trap``)
and writes ``results/paper/frontier_paired.json``: for each family × arm × frontier configuration with at
least ``MIN_TRAPS`` traps in common, the two accuracy levels on those traps and the paired difference
arm − frontier with a 95 % percentile bootstrap over traps (``DRAWS`` draws, seed ``SEED``). The reading
is fixed in the pre-registration and applies to one cell only (Qwen3.5-9B arm A against DeepSeek V4 Flash
with thinking off); every other cell is numbers only.

    python -m explore.paper_frontier_paired --traps-report qwen3-8b=results/traps_rl/report_qwen3-8b.json \
        qwen35-9b=results/traps_rl/report_qwen35-9b.json --frontier results/paper/traps_frontier.json \
        --out results/paper/frontier_paired.json
    python -m explore.paper_frontier_paired --selftest
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

MIN_TRAPS = 30
DRAWS = 10_000
SEED = 20261008
READING_CELL = ("qwen35-9b", "A", "deepseek-v4-flash")


def paired(arm: dict[str, float], ext: dict[str, float], draws: int = DRAWS, seed: int = SEED) -> dict | None:
    """Levels and the paired difference arm − ext over the traps both have, bootstrapped over traps."""
    ids = sorted(set(arm) & set(ext))
    if len(ids) < MIN_TRAPS:
        return None
    a = [arm[i] for i in ids]
    e = [ext[i] for i in ids]
    d = [x - y for x, y in zip(a, e)]
    rng = random.Random(seed)
    n = len(ids)
    means = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(sum(d[j] for j in idx) / n)
    means.sort()
    lo = means[int(0.025 * draws)]
    hi = means[min(draws - 1, int(0.975 * draws))]
    return {
        "traps": n,
        "arm_acc": sum(a) / n,
        "frontier_acc": sum(e) / n,
        "diff": sum(d) / n,
        "diff_ci95": [lo, hi],
        "trap_ids": ids,
    }


def reading(cell: dict | None) -> str:
    if cell is None:
        return "重合不足，不读"
    lo, hi = cell["diff_ci95"]
    if lo > 0:
        return "训练后的 9B 在同一批陷阱上超过关思考的前沿模型"
    if hi < 0:
        return "训练后的 9B 在同一批陷阱上仍低于关思考的前沿模型"
    return "两者无可检出差别"


def build(reports: dict[str, Path], frontier: Path) -> dict:
    fr = json.loads(frontier.read_text(encoding="utf-8"))["configs"]
    ext = {m: {i: r["trap"] for i, r in v["per_trap"].items()} for m, v in fr.items()}
    out: dict = {
        "definition": {
            "unit": "trap; per-trap accuracy of an arm is the mean over its endpoints, of a frontier configuration the mean over its runs",
            "statistic": "mean over common traps of (arm − frontier)",
            "ci": f"percentile bootstrap over traps, {DRAWS} draws, seed {SEED}",
            "min_traps": MIN_TRAPS,
            "reading_cell": list(READING_CELL),
        },
        "families": {},
    }
    for fam, path in reports.items():
        rep = json.loads(path.read_text(encoding="utf-8"))
        fam_out: dict = {"traps": rep["traps"] if isinstance(rep["traps"], int) else len(rep["traps"]), "arms": {}}
        for arm, v in rep["arms"].items():
            per = {i: r["trap"] for i, r in v["per_trap"].items()}
            cells = {}
            for m in sorted(ext):
                c = paired(per, ext[m])
                if c is not None:
                    cells[m] = c
            fam_out["arms"][arm] = cells
        out["families"][fam] = fam_out
    fam, arm, m = READING_CELL
    cell = out["families"].get(fam, {}).get("arms", {}).get(arm, {}).get(m)
    out["reading"] = reading(cell)
    return out


def selftest() -> int:
    ids = [f"t{i}" for i in range(40)]
    arm = {i: 1.0 if k < 30 else 0.0 for k, i in enumerate(ids)}
    ext = {i: 1.0 if k < 20 else 0.0 for k, i in enumerate(ids)}
    c = paired(arm, ext)
    assert c["traps"] == 40 and abs(c["arm_acc"] - 0.75) < 1e-12 and abs(c["frontier_acc"] - 0.5) < 1e-12
    assert abs(c["diff"] - 0.25) < 1e-12 and c["diff_ci95"][0] > 0 and c["diff_ci95"][1] < 0.5, c
    assert paired(arm, ext) == c, "bootstrap must be deterministic"
    assert paired({i: arm[i] for i in ids[:29]}, ext) is None, "gate on common traps"
    same = {i: arm[i] for i in ids}
    c2 = paired(arm, same)
    assert c2["diff"] == 0.0 and c2["diff_ci95"] == [0.0, 0.0]
    assert reading(c) == "训练后的 9B 在同一批陷阱上超过关思考的前沿模型"
    assert reading(c2) == "两者无可检出差别"
    assert reading(paired(ext, arm)) == "训练后的 9B 在同一批陷阱上仍低于关思考的前沿模型"
    assert reading(None) == "重合不足，不读"
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--traps-report", nargs="*", default=[], help="family=results/traps_rl/report_<family>.json")
    ap.add_argument("--frontier", type=Path, default=Path("results/paper/traps_frontier.json"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    reports = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in args.traps_report}
    if not reports or args.out is None:
        ap.error("--traps-report and --out are required")
    out = build(reports, args.frontier)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for fam, f in out["families"].items():
        for arm, cells in f["arms"].items():
            for m, c in cells.items():
                print(f"{fam:9s} {arm:4s} vs {m:28s} n={c['traps']:2d} arm {100 * c['arm_acc']:5.1f} frontier {100 * c['frontier_acc']:5.1f} "
                      f"diff {100 * c['diff']:+5.1f} [{100 * c['diff_ci95'][0]:+.1f}, {100 * c['diff_ci95'][1]:+.1f}]")
    print("reading:", out["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
