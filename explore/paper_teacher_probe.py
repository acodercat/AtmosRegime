"""PREREG_PAPER_ANALYSES §24: summary of the external-teacher probes of PREREG_OPD §4.2.

Reads the per-row probe files ``experiments/_opd/teachers/passk_outside_n8.<teacher>.jsonl`` (one outside
row per line; ``n`` samples, ``c`` correct) and the zero-hit mechanisms ``P1.Z`` of
``results/x/report_qwen3-8b.json``, and writes ``results/paper/teacher_probe.json``: per teacher the
per-template pass@1 (mean over the template's rows of c/n), the mean over templates, the zero-hit
mechanisms with pass@1 >= THRESHOLD and their count, the largest pass@1 on a zero-hit mechanism, and
whether the pre-registered gate (at least GATE such mechanisms) is met. Numbers only.

    python -m explore.paper_teacher_probe --out results/paper/teacher_probe.json
    python -m explore.paper_teacher_probe --selftest
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

THRESHOLD = 0.5   # PREREG_OPD §4.2: a mechanism counts as "known" if the teacher's outside pass@1 >= 0.5
GATE = 6          # idem: arm T runs only if at least 6 of the zero-hit mechanisms are known
TEACHERS = {
    "qwen3-32b": {"label": "Qwen3-32B", "file": "passk_outside_n8.qwen3-32b.jsonl"},
    "qwen3-30b-a3b-2507": {"label": "Qwen3-30B-A3B-Instruct-2507", "file": "passk_outside_n8.qwen3-30b-a3b-2507.jsonl"},
}


def per_template_pass1(rows: list[dict]) -> dict[str, float]:
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("variant", "outside") != "outside":
            continue
        by_t[r["template"]].append(r["c"] / r["n"])
    return {t: fmean(v) for t, v in sorted(by_t.items())}


def summarise(p1: dict[str, float], zero_hit: list[str]) -> dict:
    z = {t: p1[t] for t in zero_hit if t in p1}
    known = sorted(t for t, v in z.items() if v >= THRESHOLD)
    return {
        "templates": len(p1),
        "mean_pass1": fmean(p1.values()),
        "zero_hit_mechanisms": len(z),
        "zero_hit_known": known,
        "n_zero_hit_known": len(known),
        "zero_hit_max_pass1": max(z.values()) if z else None,
        "gate_met": len(known) >= GATE,
        "per_template_pass1": p1,
        "zero_hit_pass1": z,
    }


def build(probe_dir: Path, report: Path) -> dict:
    zero_hit = json.loads(report.read_text(encoding="utf-8"))["P1"]["Z"]
    out: dict = {
        "analysis": "PREREG_PAPER_ANALYSES §24",
        "definition": {
            "pass1": "per template, mean over its outside rows of c/n (n = 8 samples, temperature 1.0)",
            "known": f"zero-hit mechanism with pass@1 >= {THRESHOLD}",
            "gate": f"arm T runs only if at least {GATE} zero-hit mechanisms are known (PREREG_OPD §4.2)",
            "zero_hit": "results/x/report_qwen3-8b.json P1.Z",
        },
        "zero_hit_mechanisms": sorted(zero_hit),
        "gate": GATE,
        "threshold": THRESHOLD,
        "teachers": {},
    }
    for key, meta in TEACHERS.items():
        rows = [json.loads(line) for line in (probe_dir / meta["file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        s = summarise(per_template_pass1(rows), zero_hit)
        s["label"] = meta["label"]
        s["rows"] = len(rows)
        out["teachers"][key] = s
    out["any_gate_met"] = any(t["gate_met"] for t in out["teachers"].values())
    return out


def selftest() -> int:
    rows = [{"template": "a", "variant": "outside", "n": 8, "c": 8}, {"template": "a", "variant": "outside", "n": 8, "c": 0},
            {"template": "b", "variant": "outside", "n": 8, "c": 2}, {"template": "b", "variant": "inside", "n": 8, "c": 8},
            {"template": "c", "variant": "outside", "n": 8, "c": 4}]
    p1 = per_template_pass1(rows)
    assert p1 == {"a": 0.5, "b": 0.25, "c": 0.5}, p1
    s = summarise(p1, ["a", "b", "zzz"])
    assert s["zero_hit_mechanisms"] == 2 and s["zero_hit_known"] == ["a"] and s["n_zero_hit_known"] == 1
    assert abs(s["zero_hit_max_pass1"] - 0.5) < 1e-12 and s["gate_met"] is False
    assert summarise({f"t{i}": 1.0 for i in range(6)}, [f"t{i}" for i in range(6)])["gate_met"] is True
    assert abs(s["mean_pass1"] - (0.5 + 0.25 + 0.5) / 3) < 1e-12
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe-dir", type=Path, default=Path("experiments/_opd/teachers"))
    ap.add_argument("--report", type=Path, default=Path("results/x/report_qwen3-8b.json"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.out is None:
        ap.error("--out is required")
    out = build(args.probe_dir, args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    for key, t in out["teachers"].items():
        print(f"{t['label']}: rows {t['rows']}, templates {t['templates']}, mean pass@1 {t['mean_pass1']:.3f}, "
              f"zero-hit known {t['n_zero_hit_known']}/{t['zero_hit_mechanisms']} (max {t['zero_hit_max_pass1']:.2f}), gate met {t['gate_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
