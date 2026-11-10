"""PREREG_DOSE_TEXT (docs/PREREG_DOSE_TEXT.md): the knowledge as a sentence instead of a program. The 17 DOSE units lose every
warm-start pair; a unit assigned to a seed gets 4 question/answer rows whose answer is its template's "given" clause, verbatim.

    # 1. design (local): the units of results/dose/design.json, the crossover assignment, the clauses and standard-approach names
    python -m explore.dose_text design --dose-design results/dose/design.json --out results/dose_text/design.json
    # 2. one warm-start parquet (server, needs the sandbox); the same arguments as explore.dose build
    python -m explore.dose_text build --design results/dose_text/design.json --fold F1 --seed 42 --kc $ATMOS_ROOT/experiments/_inputs/kc \
        --samples $ATMOS_ROOT/experiments/_x/qwen3-8b/warm_samples_F1.jsonl --raft $ATMOS_ROOT/data/sft_raft.parquet \
        --out $ATMOS_ROOT/data/sft_dtext-qwen3-8b-F1-s42.parquet
    # check: --full keeps every pair and adds no text; the parquet must equal the fold's W_f parquet (--expect), else exit 1

Assignment (§3): per fold, random.Random("dose-text:<fold>") shuffles the units; the first ceil(n / 2) carry the text in seed 42
and none in seed 43, the rest the other way round. Text rows (§3): system as the warm rows, user one of FRAMES with the unit's
standard-approach name, assistant the clause; placed where the unit's first pair stood.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

from atmoscoder_rl.prompts import SYSTEM
from explore.dose import FOLDS, SEEDS, pairs_of

N_ROWS = 4
FRAMES = (
    "In atmospheric-science calculations, the standard approach is {S}. Under what condition does it stop being valid, "
    "and what relation should be used instead?",
    "A problem is usually solved with {S}. When does this fail, and what replaces it?",
    "State the condition under which {S} is not valid, and give the relation that applies in that case.",
    "{S}: what is its range of validity, and what should replace it outside that range?",
)
STANDARD = {   # PREREG_DOSE_TEXT §3, fixed before any build
    "ch4_forcing_sqrt": "the linear local-slope approximation for the CH4 radiative forcing",
    "clausius_clapeyron_extrapolation": "linear extrapolation of the saturation vapour pressure with its local rate of change",
    "co2_forcing_log": "the linear local-slope approximation for the CO2 radiative forcing",
    "cunningham_slip": "the Stokes settling speed of a small particle",
    "diffuse_transmission": "Beer's-law beam transmittance applied to a diffuse flux",
    "displacement_height_canopy": "the neutral logarithmic wind profile referenced to the ground",
    "doppler_aliasing": "taking the Doppler radial velocity reported by the radar as the true velocity",
    "freezing_point_depth": "the freezing point of sea water from its salinity alone",
    "henry_depletion": "Henry's law with an unchanged gas reservoir",
    "inertia_gravity_wave": "the non-rotating gravity-wave dispersion relation",
    "line_broadening": "the pressure-scaled Lorentz half-width",
    "plume_ground_reflection": "the Gaussian plume formula without a ground",
    "snowpack_cold_content": "melt computed as absorbed energy divided by the latent heat of fusion",
    "spherical_slant_path": "the plane-parallel slant path H sec(theta)",
    "sun_photometer_airmass": "the plane-parallel air mass sec(theta)",
    "surface_multiple_reflection": "the single-pass transmitted flux absorbed by the surface",
    "unconfined_dupuit": "the Thiem drawdown formula with a fixed transmissivity",
}
TEMPLATES = Path(__file__).resolve().parents[1] / "atmoscoder_rl" / "templates"
_GIVEN = re.compile(r'given = \((.*?)if variant == "given" else ""\)', re.S)
_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def given_clause(template: str) -> str:
    """The template's "given" clause (text() of atmoscoder_rl/templates/<template>.py), whitespace normalised."""
    m = _GIVEN.search((TEMPLATES / f"{template}.py").read_text())
    if not m:
        raise SystemExit(f"{template}: no given clause found")
    text = " ".join(" ".join(_LITERAL.findall(m.group(1))).split())
    if not text:
        raise SystemExit(f"{template}: empty given clause")
    return text


def assign(units: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    """fold -> seed -> units that carry the text in that seed (each unit in exactly one seed)."""
    out = {}
    for fold, mechs in units.items():
        order = random.Random(f"dose-text:{fold}").sample(sorted(mechs), len(mechs))
        k = math.ceil(len(order) / 2)
        out[fold] = {str(SEEDS[0]): sorted(order[:k]), str(SEEDS[1]): sorted(order[k:])}
    return out


def design(args) -> int:
    dose = json.loads(args.dose_design.read_text())
    units = {f: sorted(dose["units"][f]) for f in FOLDS}
    mechs = sorted(sum(units.values(), []))
    missing = [m for m in mechs if m not in STANDARD]
    if missing:
        raise SystemExit(f"no standard-approach name for {missing}")
    d = {"rule": "PREREG_DOSE_TEXT: DOSE units; text in one seed per unit (random.Random('dose-text:<fold>'), first ceil(n/2) in seed 42)",
         "dose_design_sha256": sha256(args.dose_design), "units": units, "text": assign(units), "rows_per_unit": N_ROWS,
         "frames": list(FRAMES), "standard": {m: STANDARD[m] for m in mechs}, "clauses": {m: given_clause(m) for m in mechs}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: d[k] for k in ("units", "text")}, indent=1))
    return 0


def text_rows(mechanism: str, standard: str, clause: str) -> list[dict]:
    return [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": frame.format(S=standard)},
                          {"role": "assistant", "content": clause}],
             "id": f"dose-text:{mechanism}:{i}", "kind": "knowledge_text", "template": mechanism}
            for i, frame in enumerate(FRAMES)]


def inject(sft: list[dict], units: list[str], injected: list[str], standard: dict, clauses: dict) -> tuple[list[dict], dict]:
    """Drop every pair of a unit template; where an injected unit's first pair stood, put its text rows. Order otherwise kept."""
    pairs = pairs_of(sft)
    unknown = [u for u in injected if u not in units]
    if unknown:
        raise SystemExit(f"injected but not a unit: {unknown}")
    out, placed = [], {}
    for o, i in pairs:
        t = o["template"]
        if t in units:
            if t in injected and t not in placed:
                rows = text_rows(t, standard[t], clauses[t])
                placed[t] = len(out)
                out += rows
            continue
        out += [o, i]
    absent = [u for u in injected if u not in placed]
    if absent:
        raise SystemExit(f"injected units without a pair in W_f (nowhere to place the text): {absent}")
    return out, placed


def build(args) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from explore.x_data import RAFT_SEED, fold_rows, warm_rows
    d = json.loads(args.design.read_text())
    units = d["units"][args.fold]
    injected = [] if args.full else d["text"][args.fold][str(args.seed)]
    sft_full, counts = warm_rows(fold_rows(args.kc, args.fold), args.samples)
    sft, placed = (sft_full, {}) if args.full else inject(sft_full, units, injected, d["standard"], d["clauses"])
    raft = pq.read_table(args.raft).to_pylist()
    k = min(int(len(sft_full) * args.raft_ratio), len(raft))   # W_f's RAFT rows, whatever the injection
    extra = [{"messages": r["messages"], "id": r.get("id", ""), "kind": "raft", "template": ""}
             for r in random.Random(RAFT_SEED).sample(raft, k)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sft + extra), args.out)
    kept, text = {}, {}
    for r in sft:
        if r["kind"] == "knowledge_text":
            text[r["template"]] = text.get(r["template"], 0) + 1
        elif r["id"].endswith("_outside"):
            kept[r["template"]] = kept.get(r["template"], 0) + 1
    summary = {"fold": args.fold, "seed": args.seed, "full": args.full, "units": units, "injected": injected, "text_rows": text,
               "text_position": placed, "pairs_per_template_w_f": counts, "pairs_per_template": kept, "template_rows": len(sft),
               "template_rows_w_f": len(sft_full), "raft_rows": len(extra), "design_sha256": sha256(args.design),
               "samples_sha256": sha256(args.samples), "raft_sha256": sha256(args.raft), "out": str(args.out),
               "out_sha256": sha256(args.out)}
    if not args.full:
        if any(kept.get(u, 0) for u in units):
            raise SystemExit("a unit kept a pair")
        if any(text.get(u, 0) != N_ROWS for u in injected) or set(text) != set(injected):
            raise SystemExit(f"text rows {text} do not match the injected units {injected}")
        if any(kept.get(t) != n for t, n in counts.items() if t not in units):
            raise SystemExit("a non-unit template lost pairs")
    if args.expect:
        same_bytes = summary["out_sha256"] == sha256(args.expect)
        same_rows = same_bytes or pq.read_table(args.expect).to_pylist() == pq.read_table(args.out).to_pylist()
        summary["expect"] = {"path": str(args.expect), "same_bytes": same_bytes, "same_rows": same_rows}
        if not same_rows:
            print(json.dumps(summary, indent=1))
            raise SystemExit("the rebuilt parquet differs from --expect")
    print(json.dumps(summary, indent=1))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design"); d.add_argument("--dose-design", type=Path, required=True); d.add_argument("--out", type=Path, required=True)
    d.set_defaults(fn=design)
    b = sub.add_parser("build"); b.add_argument("--design", type=Path, required=True); b.add_argument("--fold", required=True, choices=FOLDS)
    b.add_argument("--seed", type=int, required=True, choices=SEEDS); b.add_argument("--kc", type=Path, required=True)
    b.add_argument("--samples", type=Path, required=True); b.add_argument("--raft", type=Path, required=True)
    b.add_argument("--raft-ratio", type=float, default=2.0); b.add_argument("--out", type=Path, required=True)
    b.add_argument("--full", action="store_true", help="keep every pair, add no text (reproduces W_f)")
    b.add_argument("--expect", type=Path, help="fail unless the output equals this parquet")
    b.set_defaults(fn=build)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
