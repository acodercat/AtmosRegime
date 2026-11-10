"""Reworded versions of three templates whose wording may define a quantity and then ask for it
(independent review 2026-10-04; docs/PREREG_REWORD.md). Same parameters as the E0 confirmation (seed 20261002),
ids `rw:<confirmation id>`, so each row pairs with its original wording on the same numbers.

    python -m explore.e0_reword --out experiments/_inputs/e0_reword.json

- line_broadening: "this width is proportional to pressure ... the half-width of the line" -> the scaling is
  attached to the pressure-broadened width and the question asks for the HWHM of the line's absorption
  profile as measured there (reviewer's wording).
- drop_ventilation: the stated equation becomes the growth equation "for a drop at rest relative to the air"
  and the question asks for the falling drop's rate.
- clausius_clapeyron_extrapolation: "increasing by about X hPa per additional kelvin" -> "currently increasing
  at about X hPa per kelvin" (a local rate, not a constant one).
Keys are unchanged (the template's `correct`); only the text changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl import pilot
from atmoscoder_rl.build_templates import build
from atmoscoder_rl.template_wording import text as reviewed_text

SEED = 20261002
NAMES = ("line_broadening", "drop_ventilation", "clausius_clapeyron_extrapolation")


def rows() -> list[dict]:
    out = []
    for item in build(pilot.PAIRS, SEED, set(NAMES)):
        text = reviewed_text(item["template"], item["params"], item["variant"])
        out.append({**item, "id": "rw:" + item["id"], "problem": text})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = rows()
    args.out.write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(r)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
