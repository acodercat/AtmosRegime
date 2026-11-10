"""Check every regime template offline: its physics, its matched texts, and how it grades.

    uv run python -m atmoscoder_rl.check_templates [--only name,name]

No model, no network, no output file. Two layers:

1. **Physics** — each template module's own `check()`: an independent recomputation of the
   key, gap bounds over the corners of the sampled ranges, the spread of published
   conventions, a literature anchor (references and review decisions are in each docstring).
2. **Texts and grading** — the same for every template, over 500 sampled pairs and the
   built rows: inside and outside differ in exactly one given and nothing else; the `given`
   text is the outside text plus one inserted sentence; rendered givens are the solver's
   inputs; the key passes, so does the key rounded to 2 decimals (rule 11), zero and its negative fail, the shortcut passes exactly inside, and
   the key still passes in a commensurate unit where the grader can convert one.
"""
from __future__ import annotations

import argparse
import math
import random
from difflib import SequenceMatcher

from .build_templates import TEMPLATE_SEED, build
from .grading import SI_FLOOR, TOLERANCE, si_magnitude, verify
from .templates import ALL_TEMPLATES as TEMPLATES, EXTENDED_MODULES, EXTENDED_TEMPLATES, MODULES as _ORIGINAL
MODULES = {**_ORIGINAL, **EXTENDED_MODULES}
from .templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, gap, num, pair

PAIRS = 500
ROWS_PER_TEMPLATE = 100

# A commensurate unit per answer unit, to check a correct answer still passes after conversion.
# Units the grader cannot convert (mol/L, sr, ...) have no entry and skip that one check.
CONVERTED = {"m/s": ("cm/s", 100), "s^-1": ("min^-1", 60), "km": ("m", 1000), "min": ("s", 60),
             "K/km": ("K/m", .001), "hPa": ("Pa", 100), "cm/s": ("m/s", .01), "m": ("km", .001),
             "h": ("min", 60), "cm^-1": ("m^-1", 100), "ug/m^3": ("mg/m^3", .001),
             "s": ("min", 1 / 60), "day": ("h", 24), "": ("%", 100), "um/s": ("mm/s", 1e-3), "mm/s": ("cm/s", .1),
             "deg^2": ("rad^2", (3.141592653589793 / 180) ** 2)}


def check_texts(name: str) -> None:
    """Replace the single differing given, not arbitrary matching numbers elsewhere."""
    template = TEMPLATES[name]
    rng = random.Random(f"review-{TEMPLATE_SEED}:{name}")   # "review-": the stream the 2026-09-24 review used
    for _ in range(PAIRS):
        inside, outside = pair(template, rng)
        diff = {k for k in inside if inside[k] != outside[k]}
        assert len(diff) == 1, (name, diff)
        key = diff.pop()
        a, b = template.text(inside, "inside"), template.text(outside, "outside")
        old, new = num(inside[key]), num(outside[key])
        # Some occurrence of the trigger, replaced, must make the full texts equal.
        assert b in [a[:i] + new + a[i + len(old):] for i in range(len(a)) if a.startswith(old, i)], name
        # Rendering must not depend on the role name.
        assert template.text(inside, "outside") == a and template.text(outside, "inside") == b
        ops = SequenceMatcher(None, b, template.text(outside, "given"), autojunk=False).get_opcodes()
        assert any(tag == "insert" for tag, *_ in ops)
        assert all(tag in {"equal", "insert"} for tag, *_ in ops), name
        for params in (inside, outside):             # the rendered givens are the actual inputs
            assert all(float(f"{x:.3g}") == x for x in params.values()), (name, params)
        assert gap(template.correct(inside), template.shortcut(inside)) <= INSIDE_MAX_GAP
        assert gap(template.correct(outside), template.shortcut(outside)) >= OUTSIDE_MIN_GAP


def check_grading(names: list[str]) -> int:
    rows = build(ROWS_PER_TEMPLATE, TEMPLATE_SEED, names)
    for i in range(0, len(rows), 3):
        inside, outside, given = rows[i:i + 3]
        assert outside["params"] == given["params"] and outside["sub_answers"] == given["sub_answers"]
        for row in (inside, outside, given):
            value = float(row["sub_answers"][0]["value"])
            assert math.isfinite(value) and abs(value) > 1e-8      # well above the grader's floor
            unit = row["sub_answers"][0]["unit"]
            magnitude = si_magnitude(value, unit)
            assert magnitude is None or magnitude >= SI_FLOOR, (
                f"{row['id']}: {value} {unit} is {magnitude:.1e} in SI, below the grader's 1e-15 floor "
                "(any small answer would pass after unit reconciliation; MEASUREMENT_NOTES §11)")

            def passes(v: float, u: str = unit) -> bool:
                return verify({"1": {"value": v, "unit": u}}, row["sub_answers"], TOLERANCE, units="benchmark")[0]

            assert passes(value)
            assert passes(round(value, 2)), (row["id"], value)   # rule 11: survives a solver's round(x, 2)
            assert not passes(0) and not passes(-value)
            assert passes(row["shortcut_value"]) == (row["variant"] == "inside"), row["id"]
            if unit in CONVERTED:
                other_unit, scale = CONVERTED[unit]
                assert passes(value * scale, other_unit), (row["id"], unit, other_unit)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated template names (default: all)")
    parser.add_argument("--extended", action="store_true", help="check the extended set (default: the original 30)")
    args = parser.parse_args()
    names = args.only.split(",") if args.only else (list(EXTENDED_TEMPLATES) if args.extended else list(_ORIGINAL))
    unknown = sorted(set(names) - set(TEMPLATES))
    if unknown:
        print(f"unknown templates: {unknown}")
        return 1
    for name in names:
        print(f"{name:<30} physics: {MODULES[name].check()}")
        check_texts(name)
    print(f"\n{len(names)} templates x {PAIRS} pairs: single-given texts, insertion-only given arm PASS")
    print(f"{check_grading(names)} built rows: key / shortcut / sign / zero / converted unit PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
