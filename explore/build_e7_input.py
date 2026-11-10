"""E7 inputs (docs/PREREG_E7.md): the handbook arm, and the bare rows it is compared with where no run has them yet.

    python -m explore.build_e7_input --templates --out <path>.json   # bare e0_confirm rows + handbook inside/outside
    python -m explore.build_e7_input --families --out <path>.json    # handbook textbook traps + parents
    python -m explore.build_e7_input --single --out <path>.json      # only the relevant entry, template inside/outside
    python -m explore.build_e7_input --families-paired --out <path>.json  # bare + handbook traps/parents, one pass

`--templates`: the 17 E0-confirmation templates at seed 20261002 (inside / outside / given / flag, the same
ids as `e0_confirm`) plus the handbook version of their inside and outside rows. Qwen has no run on these
items yet, so the bare rows are the none and given arms of the same pass.
`--families`: the handbook version of every sound trap outside train and its parent (the rows of
`explore.build_e1_input --families`); their bare versions are already in `pilot_eval` for the same configs.
`--single` (review 2026-10-04): the template inside/outside rows with only the one handbook entry that covers the
template (`explore/handbook_map.json`), ids `hb1:` — separates "cannot pick the entry among 60 / long prompt"
from "cannot apply the generic entry". The same input carries the bare inside / outside / given rows and the
full-handbook `hb:` rows, so all four arms come from one pass.
`--families-paired` (review 2026-10-04): the bare and the handbook textbook rows in one input, so both arms come
from the same pass (a cross-run none arm moved by ~12 points between identical runs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl.build_templates import build
from explore.build_e1_input import build as build_families
from explore.build_handbook_input import build as with_handbook, handbook_text
from explore.build_pilot_eval import FLAG_PREFIX
from atmoscoder_rl import pilot

SEED = 20261002
NAMES = ("line_broadening,evaporative_cooling_limit,co2_forcing_log,ch4_forcing_sqrt,clausius_clapeyron_extrapolation,"
         "sun_photometer_airmass,displacement_height_canopy,finite_line_source,diffuse_transmission,henry_depletion,"
         "deposition_brownian,inertia_gravity_wave,raindrop_drag_regime,drop_ventilation,coriolis_neglect,"
         "anelastic_continuity,surface_multiple_reflection").split(",")


def template_rows() -> list[dict]:
    items = build(pilot.PAIRS, SEED, set(NAMES))
    flag = [{**i, "id": FLAG_PREFIX + i["id"], "problem": f"{i['problem']}\n\n{pilot.FLAG_INSTRUCTION}"}
            for i in items if i["variant"] == "outside"]
    bare = items + flag
    return bare + with_handbook(bare, handbook_text())


SINGLE = "hb1:"


def handbook_entries() -> dict[str, str]:
    """Entry title -> its block (heading line and body) in explore/handbook.md."""
    out, title, lines = {}, None, []
    for line in handbook_text().splitlines():
        if line.startswith("#"):
            if title:
                out[title] = "\n".join(lines).strip()
            title, lines = (line[4:].strip(), [line]) if line.startswith("### ") else (None, [])
        elif title:
            lines.append(line)
    if title:
        out[title] = "\n".join(lines).strip()
    return out


def single_rows() -> list[dict]:
    mapping = json.loads(Path(__file__).with_name("handbook_map.json").read_text(encoding="utf-8"))
    entries = handbook_entries()
    by_template = {t: e["title"] for e in mapping["entries"] for t in e["templates"]}
    items = build(pilot.PAIRS, SEED, set(NAMES))
    out = items + with_handbook(items, handbook_text())
    for i in items:
        if i["variant"] not in ("inside", "outside"):
            continue
        entry = entries[by_template[i["template"]]]
        out.append({**i, "id": SINGLE + i["id"],
                    "problem": f"{entry}\n\n---\nUse the reference note above only where it applies.\n\n{i['problem']}"})
    return out


def family_rows() -> list[dict]:
    return with_handbook(build_families(families=True), handbook_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--templates", action="store_true")
    group.add_argument("--families", action="store_true")
    group.add_argument("--single", action="store_true")
    group.add_argument("--families-paired", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.templates:
        rows = template_rows()
    elif args.single:
        rows = single_rows()
    elif args.families_paired:
        rows = build_families(families=True) + family_rows()
    else:
        rows = family_rows()
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate ids")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} rows ({sum(r['id'].startswith('hb:') for r in rows)} with the handbook)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
