"""Offline checks of the X5 templates (`atmoscoder_rl.templates_x5`): each module's check(), the single-trigger text
rules of `check_templates.check_texts`, and that 20 distinct pairs can be drawn at the X5 seed.

    python -m atmoscoder_rl.check_templates_x5 [names ...]
"""
from __future__ import annotations

import random
import sys
from difflib import SequenceMatcher

from .build_templates import build
from .templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, gap, num, pair
from .templates_x5 import MODULES, TEMPLATES

X5_SEED = 20261013


def check_texts(name: str, pairs: int = 50) -> None:
    t = TEMPLATES[name]
    rng = random.Random(f"x5-texts:{name}")
    for _ in range(pairs):
        inside, outside = pair(t, rng)
        diff = {k for k in inside if inside[k] != outside[k]}
        assert len(diff) == 1, (name, diff)
        key = diff.pop()
        a, b = t.text(inside, "inside"), t.text(outside, "outside")
        old, new = num(inside[key]), num(outside[key])
        assert b in [a[:i] + new + a[i + len(old):] for i in range(len(a)) if a.startswith(old, i)], (name, old, new)
        assert t.text(inside, "outside") == a and t.text(outside, "inside") == b
        ops = SequenceMatcher(None, b, t.text(outside, "given"), autojunk=False).get_opcodes()
        assert any(tag == "insert" for tag, *_ in ops) and all(tag in {"equal", "insert"} for tag, *_ in ops), name
        for params in (inside, outside):
            assert all(float(f"{x:.3g}") == x for x in params.values()), (name, params)
        assert gap(t.correct(inside), t.shortcut(inside)) <= INSIDE_MAX_GAP
        assert gap(t.correct(outside), t.shortcut(outside)) >= OUTSIDE_MIN_GAP


def main() -> int:
    names = sys.argv[1:] or sorted(TEMPLATES)
    failed = 0
    for name in names:
        try:
            summary = MODULES[name].check()
            check_texts(name)
            build(20, X5_SEED, {name}, registry={name: TEMPLATES[name]})
            print(f"ok   {name:26s} {summary}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name:26s} {type(error).__name__}: {error}")
    print("PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
