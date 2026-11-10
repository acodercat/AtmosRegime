"""Per-template knowledge measures of each family's base model, for the exploratory P1 block of `x_report` (PREREG_X X-17).

    python -m explore.x_knowledge --d-review qwen3-8b=experiments/d_execution_review/qwen3_8b.json \
        --d-review qwen35-9b=experiments/d_execution_review/qwen35_9b.json --out results/x/knowledge.json

Every number is copied from an existing report, none is recomputed:
  D_given, D_outside, D_given_minus_outside  D's template accuracies, think-off, 20 pairs per template (the D review report
                                             of the configuration, `templates.templates[t].accuracy`; no LLM judge)
  D_flag, D_flag_minus_outside, hint_share   X-20: the "flag" version (outside + "the standard treatment may not hold") and
                                             the share of the given - outside gap that the flag alone recovers, where that gap
                                             is >= 0.2 (absent otherwise): high = knowledge there but not used, low = missing
  K2_strict, K2_lenient                      X-7 / X-8 (`results/x7/k2_table_{strict,lenient}.json`; neither passed its 0.8 audit)
  K_any                                      X6 exploratory reading (`results/x6/report_kany.json`; audit 0.77, below 0.8)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from atmoscoder_rl.paths import REPO_ROOT

HINT_GAP = .2   # X-20: hint_share only where giving the relation recovers at least 4 of 20
D_KEY = {"qwen3-8b": "qwen3-8b.think-off", "qwen35-9b": "qwen3.5-9b.think-off"}
# where the D review reports live (results/d_execution/completed_configurations.json); --d-review takes a local copy
SERVER_REVIEW = "/path/to/workdir/experiments/d_execution/completed_review_20261011"
JUDGED = {"K2_strict": ("x7/k2_table_strict.json", "K2"), "K2_lenient": ("x7/k2_table_lenient.json", "K2"),
          "K_any": ("x6/report_kany.json", "K")}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def d_templates(path: Path, key: str) -> dict:
    """template -> D accuracies of the complete configuration `key` in a D review report."""
    conf = [c for c in json.loads(path.read_text())["configurations"] if c["key"] == key]
    if len(conf) != 1 or conf[0]["status"] != "complete":
        raise SystemExit(f"{path}: no single complete configuration {key!r}")
    out = {}
    for t, v in conf[0]["templates"]["templates"].items():
        a = v["accuracy"]
        out[t] = {"D_given": a["given"], "D_outside": a["outside"], "D_given_minus_outside": round(a["given"] - a["outside"], 6),
                  "D_flag": a["flag"], "D_flag_minus_outside": round(a["flag"] - a["outside"], 6)}
        gap = a["given"] - a["outside"]
        if gap >= HINT_GAP:
            out[t]["hint_share"] = round((a["flag"] - a["outside"]) / gap, 6)
    return out


def build(d_review: dict[str, Path], results: Path) -> dict:
    out = {"sources": {}, "families": {}}
    for fam, path in d_review.items():
        per = d_templates(path, D_KEY[fam])
        out["sources"][f"D:{fam}"] = {"path": f"{SERVER_REVIEW}/{path.name}", "configuration": D_KEY[fam], "sha256": sha256(path)}
        for name, (rel, field) in JUDGED.items():
            src = results / rel
            table = json.loads(src.read_text())[fam]["per_template"]
            out["sources"][name] = {"path": rel, "sha256": sha256(src)}
            for t, v in table.items():
                if t in per:   # X5 templates are not trained on
                    per[t][name] = v[field]
        out["families"][fam] = per
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d-review", action="append", required=True, metavar="FAMILY=PATH",
                    help=f"D review report per family ({', '.join(D_KEY)})")
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d_review = dict(s.split("=", 1) for s in args.d_review)
    unknown = set(d_review) - set(D_KEY)
    if unknown:
        raise SystemExit(f"unknown family {sorted(unknown)}")
    out = build({k: Path(v) for k, v in d_review.items()}, args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for fam, per in out["families"].items():
        print(fam, len(per), "templates;", {k: sum(k in v for v in per.values()) for k in ("K2_strict", "K2_lenient", "K_any")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
