"""E0d screening input for extended templates: inside / outside / given at the evaluation seed, plus the
outside rows with the flag sentence. Same ids as e0_templates, so a run merges into that experiment.

    python -m explore.build_screen_input --names a,b,c --out experiments/_inputs/e0_ext.json [--seed N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmoscoder_rl import pilot
from atmoscoder_rl.build_templates import build
from explore.build_pilot_eval import FLAG_PREFIX


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True, help="comma-separated template names, or ALL")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=pilot.PAIRS, help="pairs per template (5 for a pre-screen)")
    ap.add_argument("--seed", type=int, default=pilot.EVAL_SEED, help="parameter seed (a fresh one for a confirmation)")
    args = ap.parse_args()
    if args.names == "ALL":
        from atmoscoder_rl.templates import ALL_TEMPLATES
        names = set(ALL_TEMPLATES)
    else:
        names = set(args.names.split(","))
    items = build(args.pairs, args.seed, names)
    flag = [{**i, "id": FLAG_PREFIX + i["id"], "problem": f"{i['problem']}\n\n{pilot.FLAG_INSTRUCTION}"}
            for i in items if i["variant"] == "outside"]
    rows = items + flag
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
