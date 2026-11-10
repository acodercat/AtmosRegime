"""PREREG_PAPER_ANALYSES §23: one concrete outside instance per mechanism, and the mechanisms that fail in
every local configuration.

Reads the template registry (``atmoscoder_rl.templates.ALL_TEMPLATES``), ``results/paper/atlas.json``
(``per_template.universality_local`` / ``configs_kept_local`` / ``deepseek_off``, §13),
``results/paper/round4_families.json`` (``S16a.zero_in_all``) and ``results/paper/cost.json``
(``templates.per_template.F_max``, §12), and writes ``results/paper/mechanism_examples.json``:

  universal_failures  templates kept in at least MIN_KEPT local configurations whose conditional outside
                      accuracy is at or below FAIL in every one of them (the rule of explore/paper_atlas.py)
  examples            per template, one (inside, outside) parameter pair from a fixed named generator, with the
                      standard-relation (shortcut) value, the correct value, their ratio and the template's F_max
  anchors             fixed-parameter instances named in the pre-registration, computed from the template functions

No model output is read. Descriptive; numbers only.

    python -m explore.paper_mechanism_examples --out results/paper/mechanism_examples.json
    python -m explore.paper_mechanism_examples --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from atmoscoder_rl.templates import ALL_TEMPLATES
from atmoscoder_rl.templates.base import Template, pair

FAIL = 0.5          # explore/paper_atlas.py
MIN_KEPT = 10       # idem
SEED_TAG = "20261108"
ANCHORS = {         # pre-registered fixed parameters (PREREG_PAPER_ANALYSES §23, S23-c)
    "co2_forcing_log": [{"dc": 280.0}],
    "line_broadening": [{"p": 0.4}, {"p": 0.1}],
    "sun_photometer_airmass": [{"theta": 87.0}],
}


def ratio(shortcut: float, correct: float) -> float:
    return max(abs(shortcut) / abs(correct), abs(correct) / abs(shortcut))


def instance(t: Template, p: dict) -> dict:
    s, c = t.shortcut(p), t.correct(p)
    return {"params": p, "shortcut": s, "correct": c, "unit": t.unit,
            "ratio": ratio(s, c), "relative_deviation": abs(s - c) / abs(c)}


def example(t: Template) -> dict:
    rng = random.Random(f"paper:examples:{t.name}:{SEED_TAG}")
    pin, pout = pair(t, rng)
    out = instance(t, pout)
    out["inside_params"] = pin
    return out


def universal_failures(per_template: dict) -> list[dict]:
    rows = []
    for name in sorted(per_template):
        v = per_template[name]
        if v["configs_kept_local"] >= MIN_KEPT and v["universality_local"] == v["configs_kept_local"]:
            rows.append({"template": name, "category": v["category"], "configs_kept_local": v["configs_kept_local"],
                         "mean_rec_outside_local": v["mean_rec_outside_local"], "deepseek_off": v["deepseek_off"]})
    return rows


def anchors(t: Template) -> list[dict]:
    """Fixed-parameter instances; base parameters, if the template has any, come from the named generator."""
    out = []
    for fixed in ANCHORS.get(t.name, []):
        rng = random.Random(f"paper:anchors:{t.name}:{SEED_TAG}")
        b = t.base(rng)
        out.append(instance(t, {**b, **fixed}))
    return out


def build(atlas: Path, families: Path, cost: Path) -> dict:
    per_t = json.loads(atlas.read_text(encoding="utf-8"))["per_template"]
    zero_all = json.loads(families.read_text(encoding="utf-8"))["S16a"]["zero_in_all"]
    fmax = json.loads(cost.read_text(encoding="utf-8"))["templates"]["per_template"]
    out: dict = {
        "analysis": "PREREG_PAPER_ANALYSES §23",
        "definition": {
            "universal_failure": f"configs_kept_local >= {MIN_KEPT} and conditional outside accuracy <= {FAIL} in every kept local configuration (paper_atlas.py)",
            "example": f"first admissible (inside, outside) pair from random.Random('paper:examples:<template>:{SEED_TAG}'); values at the outside parameters",
            "ratio": "max(shortcut/correct, correct/shortcut)",
            "F_max": "results/paper/cost.json templates.per_template.F_max (final set, 20 rows; training-study templates only)",
        },
        "universal_failures": universal_failures(per_t),
        "zero_in_all_base_models": sorted(zero_all),
        "examples": {},
        "anchors": {},
    }
    for name in sorted(ALL_TEMPLATES):
        t = ALL_TEMPLATES[name]
        ex = example(t)
        ex["category"] = t.category
        ex["mechanism"] = t.mechanism
        ex["F_max_final_set"] = fmax.get(name, {}).get("F_max")
        out["examples"][name] = ex
        if name in ANCHORS:
            out["anchors"][name] = anchors(t)
    out["n_templates"] = len(out["examples"])
    out["n_universal_failures"] = len(out["universal_failures"])
    return out


def selftest() -> int:
    t = ALL_TEMPLATES["co2_forcing_log"]
    a = instance(t, {"dc": 280.0})
    assert abs(a["correct"] - 3.71) < 0.01 and abs(a["shortcut"] - 5.348) < 0.001, a
    lb = ALL_TEMPLATES["line_broadening"]
    for p in (0.4, 0.1):
        b = lb.base(random.Random(f"paper:anchors:line_broadening:{SEED_TAG}"))
        r = instance(lb, {**b, "p": p})["ratio"]
        assert 19 <= r <= 85, (p, r)
    assert ratio(2.0, 1.0) == 2.0 and ratio(1.0, 2.0) == 2.0
    e1, e2 = example(t), example(t)
    assert e1 == e2, "example draw must be deterministic"
    assert e1["relative_deviation"] >= 0.15
    fake = {"a": {"category": "x", "configs_kept_local": 16, "universality_local": 16, "mean_rec_outside_local": 0.0, "deepseek_off": 0.0},
            "b": {"category": "x", "configs_kept_local": 16, "universality_local": 15, "mean_rec_outside_local": 0.1, "deepseek_off": 0.0},
            "c": {"category": "x", "configs_kept_local": 9, "universality_local": 9, "mean_rec_outside_local": 0.0, "deepseek_off": None}}
    assert [r["template"] for r in universal_failures(fake)] == ["a"]
    assert len(ALL_TEMPLATES) == 72
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--atlas", type=Path, default=Path("results/paper/atlas.json"))
    ap.add_argument("--families", type=Path, default=Path("results/paper/round4_families.json"))
    ap.add_argument("--cost", type=Path, default=Path("results/paper/cost.json"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.out is None:
        ap.error("--out is required")
    out = build(args.atlas, args.families, args.cost)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{out['n_universal_failures']} universal failures:", [r["template"] for r in out["universal_failures"]])
    for name, rows in out["anchors"].items():
        for r in rows:
            print(f"anchor {name} {r['params']}: shortcut {r['shortcut']:.4g} correct {r['correct']:.4g} {r['unit']} ratio {r['ratio']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
