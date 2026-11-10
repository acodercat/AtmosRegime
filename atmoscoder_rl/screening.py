"""Template screening (confirmatory admission rule) and the exploratory A/B contrast —
docs/PREREG_TEMPLATE_SCREENING.md Parts 1 and 2, in code.

    uv run python -m atmoscoder_rl.screening --items <items.json> --run <run file> [--out <report.json>]

`--items` is the file `build_templates` wrote for the screening seed; `--run` is the calibration
model's run file over it. Nothing here is a free parameter: every threshold below is the one the
pre-registration fixes, and changing one after the data exist is an amendment, not an edit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .capture import shortcut_match
from .grading import TOLERANCE
from .runs import Record, is_gradable, load_reps
from .stats import fisher_greater_p, rank_sum_greater_p

# ---- Part 1: admission (shares of gradable items) ------------------------------------------------
PAIRS = 20                 # matched pairs per template
MIN_GRADABLE = 16          # per arm; fewer and the template is "not screened", neither admitted nor failed
INSIDE_MIN = 0.7           # (a) the parent method is within reach
CAPTURE_MIN = 0.3          # (b) outside answers reproducing the shortcut, among pairs whose inside is right
MIN_CONDITIONING = 10      # (b) is undetermined with fewer such pairs
GIVEN_MIN = 0.6            # (c) the replacement is computable when stated

# ---- Part 2 (exploratory): the classification registered in MECHANISMS.md (commit c141f17) --------
KIND = {
    **dict.fromkeys(("mie_geometric", "line_broadening", "kelvin_curvature", "coriolis_neglect",
                     "doppler_aliasing", "henry_depletion", "effective_henry", "log_wind_stability",
                     "plume_mixing_lid", "finite_line_source", "radar_beam_height",
                     "taylor_dispersion"), "A"),
    **dict.fromkeys(("planck_linearization", "hydrostatic_constant_density", "weak_acid_dissociation",
                     "solid_angle", "optically_thin_emission", "inertia_gravity_wave",
                     "anelastic_continuity", "mixed_layer_response", "ppm_conversion_pressure",
                     "pseudo_first_order"), "B"),
    **dict.fromkeys(("gradient_wind_curvature", "rossby_deformation"), "borderline"),
    # Added 2026-09-27, after the classification was registered: screened like every template,
    # outside the A/B contrast (MECHANISMS.md, "Three templates added").
    **dict.fromkeys(("evaporative_cooling_limit", "ice_saturation", "lognormal_mass"), "post-reg"),
}
#: The author's prior "expected lure: strong" (MECHANISMS.md status table), for the pre-registered
#: stratified secondary: A and B cannot be told apart from design effort without it.
#: As registered on 2026-09-26; the 2026-09-27 domain review lowered several of these priors
#: (MECHANISMS.md), which is one more reason the contrast is exploratory.
PRIOR_STRONG = {"coriolis_neglect", "mie_geometric", "line_broadening", "doppler_aliasing",
                "henry_depletion", "plume_mixing_lid", "finite_line_source",
                "hydrostatic_constant_density", "planck_linearization"}

#: Templates sharing a mechanism (MECHANISMS.md, 2026-09-27): per-mechanism statistics are also
#: reported with each cluster as one unit, and a later trained / held-out split moves whole clusters.
CLUSTERS = {
    "rotation": ("coriolis_neglect", "inertia_gravity_wave"),
    "rotating_balance": ("gradient_wind_curvature", "rossby_deformation"),
    "density_with_height": ("hydrostatic_constant_density", "anelastic_continuity", "ppm_conversion_pressure"),
    "spherical_geometry": ("radar_beam_height", "solid_angle", "spherical_slant_path"),
    "aqueous_equilibria": ("henry_depletion", "effective_henry", "weak_acid_dissociation"),
    "dispersion": ("plume_mixing_lid", "finite_line_source", "taylor_dispersion"),
    "radiative_linearisation": ("planck_linearization", "optically_thin_emission"),
    # ice_saturation joins kelvin_curvature: both ask for the equilibrium vapour pressure of a surface
    # that is not the flat water the stated e_s describes (review 2026-09-27).
    "size_vs_length_scale": ("mie_geometric", "kelvin_curvature", "cunningham_slip", "ice_saturation"),
}
CLUSTER_OF = {t: c for c, members in CLUSTERS.items() for t in members}

# ---- grading details fixed with the pre-registration --------------------------------------------
#: Declared molar-concentration units are rescaled to the key's before grading and capture (the
#: grader cannot convert them).
_MOLAR = {"mol/l": 1.0, "m": 1.0, "mmol/l": 1e-3, "mm": 1e-3, "umol/l": 1e-6, "um": 1e-6,
          "nmol/l": 1e-9, "nm": 1e-9}
RESCALED_TEMPLATES = {"henry_depletion", "effective_henry", "weak_acid_dissociation"}
#: line_broadening: an outside answer at 1.15-1.20 x the key used the 1/e Doppler width. Reported,
#: never a capture (it cannot match the Lorentz shortcut) and never a pass.
ONE_OVER_E_BAND = (1.15, 1.20)


def _molar_factor(unit: str) -> float | None:
    u = unit.strip().lower().replace("μ", "u").replace("µ", "u").replace(" ", "")
    u = u.replace("l^-1", "/l").replace("l-1", "/l").replace("dm^-3", "/l")
    return _MOLAR.get(u)


def _rescale(record: Record, key_unit: str) -> Record:
    """The record with molar concentrations converted into the key's unit, re-graded."""
    key_factor = _molar_factor(key_unit)
    details = []
    for d in record.get("details") or []:
        factor = _molar_factor(d.get("actual_unit") or "")
        if key_factor and factor and isinstance(d.get("actual"), (int, float)):
            d = {**d, "actual": d["actual"] * factor / key_factor, "actual_unit": key_unit}
            expected = float(d["expected"][0])
            d["passed"] = abs(d["actual"] - expected) <= TOLERANCE * abs(expected)
        details.append(d)
    if not details:
        return record
    return {**record, "details": details, "passed": all(d.get("passed") for d in details)}


def screen(items: list[dict[str, Any]], rep: dict[str, Record]) -> dict[str, dict[str, Any]]:
    """Per template: the three criteria, their denominators, the verdict and what is reported."""
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in items:
        by_pair[item["pair"]][item["variant"]] = item
    out: dict[str, dict[str, Any]] = {}
    templates = sorted({item["template"] for item in items})
    for name in templates:
        pairs = [v for p, v in sorted(by_pair.items()) if next(iter(v.values()))["template"] == name]
        records: dict[str, list[tuple[dict, Record]]] = defaultdict(list)
        for pair in pairs:
            for variant, item in pair.items():
                record = rep.get(item["id"])
                if record is None or not is_gradable(record):
                    continue
                if name in RESCALED_TEMPLATES:
                    record = _rescale(record, item["sub_answers"][0]["unit"])
                records[variant].append((item, record))
        graded = {v: len(records[v]) for v in ("inside", "outside", "given")}
        passed = {v: sum(bool(r["passed"]) for _, r in records[v]) for v in graded}
        inside_ok = {item["pair"] for item, r in records["inside"] if r["passed"]}
        conditioned = [(item, r) for item, r in records["outside"] if item["pair"] in inside_ok]
        captured = sum(shortcut_match(r, item) for item, r in conditioned)
        entry: dict[str, Any] = {
            "kind": KIND.get(name, "pilot"), "prior_strong": name in PRIOR_STRONG,
            "gradable": graded, "correct": passed,
            "conditioning_pairs": len(conditioned), "captured": captured,
            "inside_share": passed["inside"] / graded["inside"] if graded["inside"] else None,
            "given_share": passed["given"] / graded["given"] if graded["given"] else None,
            "capture_share": captured / len(conditioned) if len(conditioned) >= MIN_CONDITIONING else None,
            "gap_pairs": _discordant(records),
        }
        if name == "line_broadening":
            lo, hi = ONE_OVER_E_BAND
            entry["one_over_e_width"] = sum(
                1 for item, r in records["outside"]
                for d in r.get("details") or [] if isinstance(d.get("actual"), (int, float))
                and lo <= d["actual"] / float(item["sub_answers"][0]["value"]) <= hi)
        if min(graded.values()) < MIN_GRADABLE:
            entry["verdict"] = "not screened"
        elif entry["capture_share"] is None:
            entry["verdict"] = "failed (a)" if entry["inside_share"] < INSIDE_MIN else "undetermined (b)"
        else:
            failing = [c for c, ok in (("a", entry["inside_share"] >= INSIDE_MIN),
                                       ("b", entry["capture_share"] >= CAPTURE_MIN),
                                       ("c", entry["given_share"] >= GIVEN_MIN)) if not ok]
            entry["verdict"] = "admitted" if not failing else "failed (" + ", ".join(failing) + ")"
        out[name] = entry
    return out


def _discordant(records: dict[str, list[tuple[dict, Record]]]) -> tuple[int, int]:
    """Inside right & outside wrong vs the reverse, over pairs with both gradable (reported only)."""
    inside = {item["pair"]: bool(r["passed"]) for item, r in records["inside"]}
    outside = {item["pair"]: bool(r["passed"]) for item, r in records["outside"]}
    both = inside.keys() & outside.keys()
    return (sum(inside[p] and not outside[p] for p in both), sum(outside[p] and not inside[p] for p in both))


def ab_test(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Part 2, exploratory, on the 12 A and 10 B templates (pilots, borderline and post-reg
    excluded): the capture share of A against B's, exact one-sided rank-sum test; the share with capture >= 0.3,
    one-sided Fisher; the rank-sum within the author's prior-strong and not-strong strata; and the
    rank-sum with each single-label cluster collapsed to its mean (mixed-label clusters dropped).
    Reported as estimates of a hypothesis, never as a confirmed or refuted claim."""
    def shares(kind: str, strong: bool | None = None) -> list[float]:
        return [e["capture_share"] for e in table.values()
                if e["kind"] == kind and e["capture_share"] is not None
                and (strong is None or e["prior_strong"] == strong)]

    a, b = shares("A"), shares("B")
    undetermined = {k: sum(1 for e in table.values() if e["kind"] == k and e["capture_share"] is None)
                    for k in ("A", "B")}
    result: dict[str, Any] = {
        "n": {"A": len(a), "B": len(b)}, "undetermined": undetermined,
        "primary_rank_sum_p": rank_sum_greater_p(a, b) if a and b else None,
        "secondary_fisher": None, "strata": {},
    }
    if a and b:
        pa, pb = sum(s >= CAPTURE_MIN for s in a), sum(s >= CAPTURE_MIN for s in b)
        result["secondary_fisher"] = {"A": f"{pa}/{len(a)}", "B": f"{pb}/{len(b)}",
                                      "p": fisher_greater_p(pa, len(a), pb, len(b))}
    for label, strong in (("prior strong", True), ("prior not strong", False)):
        sa, sb = shares("A", strong), shares("B", strong)
        result["strata"][label] = {"A": len(sa), "B": len(sb),
                                   "p": rank_sum_greater_p(sa, sb) if sa and sb else None}
    collapsed: dict[str, list[float]] = {"A": [], "B": []}
    units: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, e in table.items():
        if e["kind"] in ("A", "B") and e["capture_share"] is not None:
            units[CLUSTER_OF.get(name, name)][e["kind"]].append(e["capture_share"])
    for kinds in units.values():
        if len(kinds) == 1:                      # the cluster is single-label
            (kind, values), = kinds.items()
            collapsed[kind].append(sum(values) / len(values))
    result["cluster_collapsed"] = {"A": len(collapsed["A"]), "B": len(collapsed["B"]),
                                   "p": rank_sum_greater_p(collapsed["A"], collapsed["B"])
                                   if collapsed["A"] and collapsed["B"] else None}
    result["status"] = "exploratory"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, required=True, help="the screening items file")
    parser.add_argument("--run", type=Path, required=True, help="the calibration model's run file")
    parser.add_argument("--out", type=Path, help="also write the report as JSON to this file")
    args = parser.parse_args()
    items = json.loads(args.items.read_text())
    reps = load_reps(args.run.parent, "", args.run.name)
    if len(reps) != 1:
        print(f"expected one run file, found {len(reps)}")
        return 1
    table = screen(items, reps[0])
    print(f"{'template':<30}{'kind':<11}{'inside':>8}{'capture':>10}{'given':>7}  verdict")
    for name, e in table.items():
        cap = "—" if e["capture_share"] is None else f"{e['captured']}/{e['conditioning_pairs']}"
        print(f"{name:<30}{e['kind']:<11}{e['correct']['inside']:>4}/{e['gradable']['inside']:<3}"
              f"{cap:>10}{e['correct']['given']:>4}/{e['gradable']['given']:<3} {e['verdict']}")
    ab = ab_test(table)
    print(f"\nPart 2 (exploratory): A {ab['n']['A']} vs B {ab['n']['B']} templates with a capture share "
          f"(undetermined A {ab['undetermined']['A']}, B {ab['undetermined']['B']}); "
          f"rank-sum p = {ab['primary_rank_sum_p']} (exploratory)")
    print(f"  cluster-collapsed: {ab['cluster_collapsed']}")
    print(f"  secondary Fisher: {ab['secondary_fisher']}")
    print(f"  strata: {ab['strata']}")
    if args.out:
        args.out.write_text(json.dumps({"templates": table, "part2": ab}, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
