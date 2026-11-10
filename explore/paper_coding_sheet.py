"""PREREG_PAPER_ANALYSES §10: the blind coding sheet for approximation types — physics descriptions only, no results.

    python -m explore.paper_coding_sheet --traps <Bench-Lab>/benchmark/traps.json --out results/paper/coding_sheet.md

Rows: the 72 atmospheric templates and 13 engineering templates (name, category, cluster, the docstring headline
"standard -> replacement", its "Parent's method" and "Where it fails" paragraphs) and the regime_boundary textbook traps
(id, category, the `mechanism` text, or the first sentence of the problem when there is none). The coder fills
results/paper/approx_types.json: {"templates": {name: type}, "traps": {id: type}, "merges": {...}, "coder": "...", "date": "..."}.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

from atmoscoder_rl.coverage_clusters import unit_of

ROOT = Path(__file__).resolve().parents[1]
TYPES = ("linearisation / local-slope extrapolation", "geometric idealisation (plane-parallel, flat, infinite)",
         "neglected process / missing term", "fixed reservoir / steady state / reactant in excess",
         "dimensionless-number regime crossing", "reference state / datum / boundary condition",
         "boundary / image source / finite domain", "instrument convention")


def paragraph(doc: str, label: str) -> str:
    m = re.search(rf"{label}:\s*(.*?)(?:\n\s*\n|\n[A-Z][A-Za-z' ()]{{2,40}}:|\Z)", doc, re.S)
    return " ".join(m.group(1).split()) if m else ""


def template_rows(pkg: str) -> list[dict]:
    rows = []
    for p in sorted((ROOT / "atmoscoder_rl" / pkg).glob("*.py")):
        if p.name.startswith(("_", "base")):
            continue
        src = p.read_text()
        doc = src.split('"""')[1] if '"""' in src else ""
        mod = importlib.import_module(f"atmoscoder_rl.{pkg}.{p.stem}")
        t = getattr(mod, "TEMPLATE", None)
        if t is None:
            continue
        headline = " ".join(doc.strip().split("\n\n")[0].split()) if doc.strip() else ""   # the first paragraph
        rows.append({"name": t.name, "category": t.category, "cluster": unit_of(t.name) if pkg == "templates" else "—",
                     "headline": headline, "parent": paragraph(doc, "Parent's method"), "fails": paragraph(doc, "Where it fails")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    atmos, eng = template_rows("templates"), template_rows("templates_x5")
    traps = [r for r in json.loads(args.traps.read_text()) if r.get("trap_type") == "regime_boundary"]
    lines = ["# Approximation-type coding sheet (PREREG_PAPER_ANALYSES §10)", "",
             "Physics descriptions only; no result of any model appears here. Code each row with ONE of:", ""]
    lines += [f"{i + 1}. {t}" for i, t in enumerate(TYPES)]
    lines += ["", "A type with fewer than 5 templates is merged into the nearest type; write the merge rule in approx_types.json.", "",
              f"## Atmospheric templates ({len(atmos)})", "", "| # | template | category | cluster | standard -> replacement | parent's method | where it fails |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(atmos, 1):
        lines.append(f"| {i} | `{r['name']}` | {r['category']} | {r['cluster']} | {r['headline']} | {r['parent']} | {r['fails']} |")
    lines += ["", f"## Engineering templates ({len(eng)})", "", "| # | template | category | standard -> replacement | parent's method | where it fails |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(eng, 1):
        lines.append(f"| {i} | `{r['name']}` | {r['category']} | {r['headline']} | {r['parent']} | {r['fails']} |")
    lines += ["", f"## Textbook regime traps ({len(traps)})", "", "| # | trap id | category | mechanism (or first sentence of the problem) |", "|---|---|---|---|"]
    for i, r in enumerate(traps, 1):
        text = r.get("mechanism") or (r["problem"].split(". ")[0] + ".")
        lines.append(f"| {i} | `{r['id']}` | {r['category']} | {' '.join(text.split())} |")
    skeleton = {"coder": "", "date": "", "types": list(TYPES), "merges": {},
                "templates": {r["name"]: "" for r in atmos + eng}, "traps": {r["id"]: "" for r in traps}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_name("approx_types.template.json").write_text(json.dumps(skeleton, indent=1) + "\n")
    print(f"{len(atmos)} atmospheric, {len(eng)} engineering, {len(traps)} regime traps -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
