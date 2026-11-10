"""PREREG_PAPER_ANALYSES §18: learning rate of answer-reward RL by bins of base reachability.

Reads the X reports (``results/x/report_<family>.json``, ``P1.per_mechanism``) and writes
``results/paper/pcurve.json``: for each family, the number of mechanisms, the number learned and
the learned share with a Clopper–Pearson 95 % interval in four fixed bins of base pass@1, plus
the reference curve g(p) = 1 − (1 − p)^8 − p^8 on a grid (the probability that a group of eight
rollouts contains both a right and a wrong answer, i.e. a non-zero advantage for that prompt).
Nothing is fitted or scaled.

    python -m explore.paper_pcurve --x-report qwen3-8b=results/x/report_qwen3-8b.json \
        qwen35-9b=results/x/report_qwen35-9b_s42-43.json --conditional results/paper/round4_conditional.json \
        --out results/paper/pcurve.json
    python -m explore.paper_pcurve --selftest
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BINS = [(0.0, 0.005), (0.005, 0.02), (0.02, 0.1), (0.1, 1.0 + 1e-9)]
LEARNED_AT = 0.5
GROUP = 8
GRID = [i / 1000 for i in range(0, 1001)]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    """Exact binomial interval via the beta quantile (bisection on the regularised incomplete beta)."""
    if n == 0:
        return None

    def betainc(a: float, b: float, x: float) -> float:
        # regularised incomplete beta by continued fraction (Numerical Recipes betacf)
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
        if x < (a + 1) / (a + b + 2):
            return math.exp(lbeta) * _betacf(a, b, x) / a
        return 1.0 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b

    def quantile(a: float, b: float, q: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if betainc(a, b, mid) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else quantile(k, n - k + 1, alpha / 2)
    upper = 1.0 if k == n else quantile(k + 1, n - k, 1 - alpha / 2)
    return [lower, upper]


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def g(p: float, n: int = GROUP) -> float:
    return 1.0 - (1.0 - p) ** n - p ** n


def learned(rec: dict) -> bool:
    ends = [rec[k] for k in rec if k.startswith("A_s")]
    return bool(ends) and sum(ends) / len(ends) >= LEARNED_AT


def eligible(per_mechanism: dict[str, dict], base_outside: dict[str, float]) -> dict[str, dict]:
    """Mechanisms the base model does not already solve outside the regime (outside accuracy < 0.5)."""
    return {m: r for m, r in per_mechanism.items() if m in base_outside and base_outside[m] < LEARNED_AT}


def bin_table(per_mechanism: dict[str, dict]) -> list[dict]:
    rows = []
    for lo, hi in BINS:
        members = {m: r for m, r in per_mechanism.items() if lo <= r["pass@1"] < hi}
        k = sum(learned(r) for r in members.values())
        n = len(members)
        ps = sorted(r["pass@1"] for r in members.values())
        rows.append(
            {
                "bin": [lo, min(hi, 1.0)],
                "n": n,
                "learned": k,
                "share": (k / n) if n else None,
                "ci95": clopper_pearson(k, n),
                "p_median": (ps[len(ps) // 2] if ps else None),
                "g_at_median": (g(ps[len(ps) // 2]) if ps else None),
            }
        )
    return rows


def reading(tables: dict[str, list[dict]]) -> str:
    ok = True
    for rows in tables.values():
        shares = [r["share"] for r in rows if r["share"] is not None]
        if any(b < a - 1e-12 for a, b in zip(shares, shares[1:])):
            ok = False
        low, mid = rows[0], rows[2]
        if low["ci95"] is None or mid["ci95"] is None or not (mid["ci95"][0] > low["ci95"][1]):
            ok = False
    return "学会率在 p 为百分之几处陡升，与组方差曲线同形" if ok else "只报数字"


def build(reports: dict[str, Path], conditional: Path) -> dict:
    out: dict = {
        "definition": {
            "p": "base pass@1 over 20 outside rows × 64 samples",
            "universe": f"mechanisms with base outside accuracy < {LEARNED_AT} (round4_conditional outside_start)",
            "learned": f"mean A-arm endpoint outside accuracy ≥ {LEARNED_AT}",
            "bins": BINS,
            "group_size": GROUP,
            "curve": "g(p) = 1 - (1-p)^n - p^n, unscaled",
        },
        "families": {},
    }
    cond = json.loads(Path(conditional).read_text(encoding="utf-8"))["families"]
    for fam, path in reports.items():
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        base_outside = {m: v["outside_start"] for m, v in cond[fam]["interventions"]["A"]["per_mechanism"].items()}
        per = eligible(rep["P1"]["per_mechanism"], base_outside)
        rows = bin_table(per)
        zero = [m for m, r in per.items() if r["pass@1"] == 0.0]
        out["families"][fam] = {
            "mechanisms_total": len(rep["P1"]["per_mechanism"]),
            "mechanisms_eligible": len(per),
            "bins": rows,
            "pass1_zero": {"n": len(zero), "learned": sum(learned(per[m]) for m in zero)},
        }
    out["curve"] = {"p": GRID, "g": [g(p) for p in GRID]}
    out["reading"] = reading({f: v["bins"] for f, v in out["families"].items()})
    return out


def selftest() -> int:
    assert abs(g(0.5) - (1 - 2 * 0.5 ** 8)) < 1e-12 and g(0.0) == 0.0 and g(1.0) == 0.0
    assert abs(g(0.02) - 0.1492) < 5e-4 and abs(g(0.1) - 0.5695) < 5e-4
    ci = clopper_pearson(0, 10)
    assert ci[0] == 0.0 and abs(ci[1] - 0.3085) < 1e-3, ci
    ci = clopper_pearson(7, 9)
    assert abs(ci[0] - 0.3999) < 1e-3 and abs(ci[1] - 0.9719) < 1e-3, ci
    per = {}
    for i in range(32):
        per[f"a{i}"] = {"pass@1": 0.001 * (i % 5), "A_s42": 1.0 if i < 6 else 0.0, "A_s43": 1.0 if i < 6 else 0.0}
    for i in range(14):
        per[f"b{i}"] = {"pass@1": 0.01, "A_s42": 1.0 if i < 10 else 0.0, "A_s43": 1.0 if i < 10 else 0.0}
    for i in range(9):
        per[f"c{i}"] = {"pass@1": 0.05, "A_s42": 1.0 if i < 7 else 0.0, "A_s43": 1.0 if i < 7 else 0.0}
    for i in range(14):
        per[f"d{i}"] = {"pass@1": 0.5, "A_s42": 1.0, "A_s43": 1.0}
    per["solved"] = {"pass@1": 0.9, "A_s42": 1.0, "A_s43": 1.0}
    base_outside = {m: 0.0 for m in per}
    base_outside["solved"] = 0.95
    per = eligible(per, base_outside)
    assert "solved" not in per and len(per) == 69
    rows = bin_table(per)
    assert [(r["n"], r["learned"]) for r in rows] == [(32, 6), (14, 10), (9, 7), (14, 14)], rows
    assert reading({"x": rows}) == "学会率在 p 为百分之几处陡升，与组方差曲线同形"
    rows[2]["ci95"] = [0.0, 0.5]
    assert reading({"x": rows}) == "只报数字"
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--x-report", nargs="*", default=[], help="family=results/x/report_<family>.json")
    ap.add_argument("--conditional", type=Path, default=Path("results/paper/round4_conditional.json"), help="per-mechanism base outside accuracy (outside_start)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    reports = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in args.x_report}
    if not reports or args.out is None:
        ap.error("--x-report and --out are required")
    out = build(reports, args.conditional)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    for fam, v in out["families"].items():
        print(fam, [(r["n"], r["learned"]) for r in v["bins"]], "pass1=0:", v["pass1_zero"])
    print("reading:", out["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
