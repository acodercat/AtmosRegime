"""Build matched inside / outside / given problems from the regime templates.

    uv run python -m atmoscoder_rl.build_templates --pairs 10 --out <path>.json   # runner input format
    uv run python -m atmoscoder_rl.build_templates --pairs 3 --show [--only a,b]  # read the texts

Every `outside` problem shares all givens with one `inside` problem except the trigger, and
every `given` problem is its `outside` twin with the replacement stated. `pair` links the three.
Within a template no inside or outside text repeats (a repeat is redrawn). Deterministic for a
fixed --seed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .templates import ALL_TEMPLATES, TEMPLATES
from .templates.base import VARIANTS, gap, pair

TEMPLATE_SEED = 20260924   # the pilot's seed; every template draws from its own seeded stream


MAX_REDRAWS = 1000


def build(n: int, seed: int, names=None, registry=None, *,
          exclude_texts: set[str] | None = None,
          redraw_counts: dict[str, int] | None = None) -> list[dict]:
    """`registry` defaults to the original 30; names from the extended set need ALL_TEMPLATES.

    `exclude_texts` reserves earlier inside/outside prompts (e.g. train before test).
    Rejected pairs consume the same per-template RNG stream. Without exclusions the
    historical output is unchanged. Optional `redraw_counts` records rejected pairs.
    """
    rows = []
    if registry is None:
        registry = ALL_TEMPLATES if names is not None and set(names) - set(TEMPLATES) else TEMPLATES
    for name, t in registry.items():
        if names is not None and name not in names:
            continue
        rng = random.Random(f"{seed}:{name}")
        seen: set[str] = set(exclude_texts or ())
        if redraw_counts is not None:
            redraw_counts[name] = 0
        for i in range(1, n + 1):
            # Redraw a pair whose inside or outside text already occurred for this template: a
            # repeated prompt is a correlated re-measurement, not a new item.
            for _ in range(MAX_REDRAWS):
                pin, pout = pair(t, rng)
                texts = {t.text(pin, "inside"), t.text(pout, "outside")}
                if not texts & seen:
                    break
                if redraw_counts is not None:
                    redraw_counts[name] += 1
            else:
                raise ValueError(f"{name}: cannot draw {n} distinct, unreserved pairs after {MAX_REDRAWS} redraws")
            seen |= texts
            for variant in VARIANTS:
                p = pin if variant == "inside" else pout
                c, s = t.correct(p), t.shortcut(p)
                rows.append({
                    # The seed is part of the id: two item sets built with different seeds must
                    # never share an id, or a resumed run would keep one set's results for the other.
                    "id": f"tpl_{seed}_{name}_{i:02d}_{variant}", "pair": f"{seed}_{name}_{i:02d}",
                    "seed": seed,
                    "template": name, "variant": variant, "category": t.category, "mechanism": t.mechanism,
                    "problem": t.text(p, variant),
                    "sub_answers": [{"sub": "1", "value": f"{c:.6g}", "unit": t.unit}],
                    "shortcut_value": float(f"{s:.6g}"), "shortcut_values": {"1": float(f"{s:.6g}")},
                    "gap": round(gap(c, s), 4), "params": p,
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int, default=10, help="matched pairs per template")
    parser.add_argument("--seed", type=int, default=TEMPLATE_SEED)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--show", action="store_true", help="print every problem text")
    parser.add_argument("--only", help="comma-separated template names (default: the original 30)")
    parser.add_argument("--extended", action="store_true", help="the extended set instead of the original 30")
    args = parser.parse_args()
    from .templates import EXTENDED_TEMPLATES
    names = args.only.split(",") if args.only else (list(EXTENDED_TEMPLATES) if args.extended else None)
    rows = build(args.pairs, args.seed, names)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    print(f"{'template':<22}{'pairs':>6}  inside gap (max)  outside gap (min .. max)")
    for name in dict.fromkeys(r["template"] for r in rows):
        gi = [r["gap"] for r in rows if r["template"] == name and r["variant"] == "inside"]
        go = [r["gap"] for r in rows if r["template"] == name and r["variant"] == "outside"]
        print(f"{name:<22}{len(gi):>6}  {max(gi):>16.4f}  {min(go):>8.3f} .. {max(go):.3f}")
    if args.show:
        for r in rows:
            print(f"\n[{r['id']}]  answer {r['sub_answers'][0]['value']} {r['sub_answers'][0]['unit']}"
                  f"  | shortcut {r['shortcut_value']}  | gap {r['gap']}\n{r['problem']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}: {len(rows)} problems ({len(rows)//3} pairs x 3 variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
