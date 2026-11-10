"""PREREG_PAPER_ANALYSES §23 E2: the X textbook rows the diagnostic study's benchmark set lacks (8 regime traps and their 8
parents), for running D's comparison configurations on them in D's own harness.

    python -m explore.build_f23_inputs [--x-inputs experiments/_inputs/x_eval] [--d-input experiments/_inputs/families_v4/textbook.json]
                                       [--out experiments/_inputs/f23/textbook_missing.json]

Deterministic (rows in the X file's order); prints the output's sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(x_inputs: Path, d_input: Path) -> list[dict]:
    rows = json.loads((x_inputs / "textbook.json").read_text())
    groups = json.loads((x_inputs / "textbook_groups.json").read_text())
    have = {r["id"] for r in json.loads(d_input.read_text())}
    missing = {t for t in groups["regime_arm"] if t not in have}
    want = missing | {groups["parent_of"][t] for t in missing}
    out = [r for r in rows if r["id"] in want]
    assert len(missing) == 8 and len(out) == 16, (len(missing), len(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x-inputs", type=Path, default=ROOT / "experiments/_inputs/x_eval")
    ap.add_argument("--d-input", type=Path, default=ROOT / "experiments/_inputs/families_v4/textbook.json")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/_inputs/f23/textbook_missing.json")
    args = ap.parse_args()
    out = build(args.x_inputs, args.d_input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    args.out.write_text(data)
    print(hashlib.sha256(data.encode()).hexdigest(), args.out, len(out), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
