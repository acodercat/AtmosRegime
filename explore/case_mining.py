"""Case mining for the paper: rule-selected, verbatim examples from the stored runs (no model call, no hand-picking before the
rules). Writes results/cases/candidates.json; results/cases/README.md curates from it and names the rule each case came from.

    python -m explore.case_mining --bench <Bench-Lab> --dose-eval experiments/dose/eval --dose-sft experiments/dose/sft_data \
        --x-textbook experiments/x_textbook --final experiments/_inputs/kc/final.json --dose-posthoc results/dose/posthoc_qwen3-8b.json \
        --out results/cases/candidates.json

Rules (each candidate records the rule, the ids and the verbatim texts):
 A  frontier silent failure: deepseek-v4-flash thinking-off fails a textbook regime trap in >= 2 of 3 runs with the shortcut value
    while solving the parent in >= 2 of 3.
 B  knows-but-uses: deepseek-v4-flash-reasoning fails a trap with the shortcut value and its reasoning text contains a validity
    word (regime, valid, assum, neglect, approximation, applies / applicable, correction, breaks down).
 C  thinking rescues: deepseek thinking-off fails (>= 2/3, shortcut) and thinking-on passes (>= 2/3) the same trap.
 D  DOSE flip: a unit whose dose-0 seed's RL is 0 and dose->=1 seed's RL is >= 0.95 — one outside row: warm-start response and RL
    response in the dosed seed, RL response in the dose-0 seed.
 E  DOSE stiff: a unit never learned (RL 0 in both seeds) at dose >= 2 — RL response and one injected warm-start program.
 F  RL unification: a unit whose warm start gave the shortcut on <= 2/20 outside rows and whose RL endpoint gives it on 20/20 —
    two warm-start responses and the RL response on the same rows.
 G  textbook trap learned by RL: Qwen3-8B base fails a trap, every A endpoint passes; and one never learned by any endpoint with
    the shortcut in every endpoint — base and A responses.
 H  comment-level knows-but-uses (thinking off): a DOSE model's outside answer equals the shortcut while its code mentions the
    replacement concept (per-unit keyword list below).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

KEYWORDS = {
    "cunningham_slip": r"cunningham|slip|knudsen|mean free path", "displacement_height_canopy": r"displacement|zero[- ]plane",
    "plume_ground_reflection": r"image|reflect", "henry_depletion": r"deplet|conserv|closed|mass balance",
    "line_broadening": r"voigt|doppler", "sun_photometer_airmass": r"kasten|refraction|curvature", "doppler_aliasing": r"nyquist|alias|fold",
    "clausius_clapeyron_extrapolation": r"clausius|exponential", "co2_forcing_log": r"logarithm|\blog\b|saturat", "ch4_forcing_sqrt": r"square root|sqrt|overlap",
    "inertia_gravity_wave": r"coriolis|inertia|rotation", "diffuse_transmission": r"diffus|1\.66|hemispher", "freezing_point_depth": r"pressure",
    "snowpack_cold_content": r"cold content|warm the|below 0|heat the", "spherical_slant_path": r"curvature|spherical|shell",
    "surface_multiple_reflection": r"multiple|series|back", "unconfined_dupuit": r"dupuit|unconfined|water table",
}
VALIDITY = re.compile(r"regime|\bvalid|assum|neglect|approximat|applicab|applies|correction|breaks? down|no longer", re.I)


def close(a, b, tol=0.02):
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1e-12)
    except (TypeError, ValueError):
        return False


def captured(rec, shortcut):
    actual = {d.get("sub"): d.get("actual") for d in (rec.get("details") or []) if isinstance(d, dict)}
    return bool(shortcut) and all(s in actual and close(actual[s], v) for s, v in shortcut.items())


def resp(rec):
    return (rec.get("attempts") or [{}])[-1].get("response", "")


def reasoning(rec):
    return (rec.get("attempts") or [{}])[-1].get("reasoning", "")


def snippet(text, pattern, width=160):
    m = re.search(pattern, text, re.I)
    if not m:
        return ""
    s, e = max(0, m.start() - width), min(len(text), m.end() + width)
    return text[s:e].replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--dose-eval", type=Path, required=True)
    ap.add_argument("--dose-sft", type=Path, required=True)
    ap.add_argument("--x-textbook", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--dose-posthoc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    traps = {r["id"]: r for r in json.loads((args.bench / "benchmark" / "traps.json").read_text())}
    core = {r["id"]: r for r in json.loads((args.bench / "benchmark" / "core.json").read_text())}
    final = {r["id"]: r for r in json.loads(args.final.read_text())}
    cases = defaultdict(list)

    def runs(model, exp):
        return [{r["id"]: r for r in json.loads(Path(p).read_text())["results"]} for p in sorted(glob.glob(str(args.bench / "experiments" / exp / f"{model}.run*.json")))]
    ds_off, ds_on = runs("deepseek-v4-flash", "trap"), runs("deepseek-v4-flash-reasoning", "trap")
    core_off = runs("deepseek-v4-flash", "core_code")
    gpt = runs("gpt55-reasoning", "trap")
    regime = [t for t in traps.values() if t["trap_type"] == "regime_boundary" and (t.get("review") or {}).get("measures_regime_judgement")]
    for t in regime:
        i, sv = t["id"], t.get("shortcut_values") or {}
        off = [r[i] for r in ds_off if i in r and "details" in r[i]]
        par = [r[t["control_id"]] for r in core_off if t["control_id"] in r and "details" in r[t["control_id"]]]
        on = [r[i] for r in ds_on if i in r and "details" in r[i]]
        off_fail_cap = [r for r in off if not r["passed"] and captured(r, sv)]
        if len(off) >= 2 and len(off_fail_cap) >= 2 and par and sum(p["passed"] for p in par) >= 2:
            cases["A_frontier_silent_failure"].append({"trap": i, "category": t["category"], "mechanism": t.get("mechanism"), "problem": t["problem"],
                                                       "parent": t["control_id"], "parent_problem": core[t["control_id"]]["problem"], "expected": t["sub_answers"],
                                                       "shortcut_values": sv, "runs_failed_captured": f"{len(off_fail_cap)}/{len(off)}",
                                                       "parent_passed": f"{sum(p['passed'] for p in par)}/{len(par)}", "response": resp(off_fail_cap[0]), "details": off_fail_cap[0]["details"]})
            if len(on) >= 2 and sum(r["passed"] for r in on) >= 2:
                ok = next(r for r in on if r["passed"])
                cases["C_thinking_rescues"].append({"trap": i, "category": t["category"], "mechanism": t.get("mechanism"), "off_response": resp(off_fail_cap[0]),
                                                    "on_reasoning_excerpt": snippet(reasoning(ok), VALIDITY.pattern, 300), "on_response": resp(ok)})
        for r in on:
            if not r["passed"] and captured(r, sv) and VALIDITY.search(reasoning(r)):
                cases["B_knows_but_uses_reasoning"].append({"trap": i, "category": t["category"], "mechanism": t.get("mechanism"), "problem": t["problem"],
                                                            "reasoning_excerpt": snippet(reasoning(r), VALIDITY.pattern, 300), "reasoning_chars": len(reasoning(r)),
                                                            "response": resp(r), "expected": t["sub_answers"], "details": r["details"]})
                break
        g = [r[i] for r in gpt if i in r and "details" in r[i]]
        if g and sum(not r["passed"] for r in g) >= 2:
            cases["A2_gpt55_reasoning_failures"].append({"trap": i, "category": t["category"], "mechanism": t.get("mechanism"), "failed": f"{sum(not r['passed'] for r in g)}/{len(g)}",
                                                         "captured": sum(captured(r, sv) for r in g if not r["passed"]), "response": resp(next(r for r in g if not r["passed"]))})

    # ---- DOSE (D, E, F, H) ---------------------------------------------------------------------------------------------
    post = json.loads(args.dose_posthoc.read_text())["observations"]
    by_m = defaultdict(dict)
    for o in post:
        by_m[o["mechanism"]][o["seed"]] = o
    def label(kind, fold, seed):
        return f"sft-dose-qwen3-8b-{fold}-s{seed}" if kind == "warm" else f"dose-qwen3-8b-{fold}-WA-s{seed}"
    def load(lab):
        return {r["id"]: r for r in json.loads((args.dose_eval / lab / "final.json").read_text())["results"]}
    cache = {}
    def rec(lab, i):
        if lab not in cache:
            cache[lab] = load(lab)
        return cache[lab].get(i)
    for m, by_seed in by_m.items():
        fold = next(iter(by_seed.values()))["fold"]
        rows = [i for i, r in final.items() if r["template"] == m and r["variant"] == "outside"]
        seeds = sorted(by_seed)
        zero = [s for s in seeds if by_seed[s]["dose"] == 0 and by_seed[s]["rl"] == 0]
        hit = [s for s in seeds if by_seed[s]["dose"] >= 1 and by_seed[s]["rl"] >= 0.95]
        if zero and hit:
            s0, s1 = zero[0], hit[0]
            i = next((i for i in rows if rec(label("rl", fold, s1), i)["passed"] and not rec(label("warm", fold, s1), i)["passed"]), rows[0])
            cases["D_dose_flip"].append({"mechanism": m, "fold": fold, "dose_seed": s1, "dose": by_seed[s1]["dose"], "zero_seed": s0, "row": i, "problem": final[i]["problem"],
                                         "expected": final[i]["sub_answers"], "shortcut_values": final[i].get("shortcut_values"),
                                         "warm_response_dosed_seed": resp(rec(label("warm", fold, s1), i)), "rl_response_dosed_seed": resp(rec(label("rl", fold, s1), i)),
                                         "rl_response_zero_seed": resp(rec(label("rl", fold, s0), i)), "warm_p_dosed": by_seed[s1]["warm_p"], "warm_p_zero": by_seed[s0]["warm_p"]})
        if all(by_seed[s]["rl"] == 0 for s in seeds) and max(by_seed[s]["dose"] for s in seeds) >= 2:
            s = max(seeds, key=lambda s: by_seed[s]["dose"])
            i = rows[0]
            cases["E_dose_stiff"].append({"mechanism": m, "fold": fold, "seed": s, "dose": by_seed[s]["dose"], "warm_p": by_seed[s]["warm_p"], "row": i, "problem": final[i]["problem"],
                                          "expected": final[i]["sub_answers"], "shortcut_values": final[i].get("shortcut_values"), "rl_response": resp(rec(label("rl", fold, s), i)),
                                          "shortcut_rl_rows": by_seed[s]["shortcut_rl"], "shortcut_warm_rows": by_seed[s]["shortcut_warm"]})
        for s in seeds:
            o = by_seed[s]
            if o["shortcut_warm"] <= 2 and o["shortcut_rl"] >= 20:
                w, r_ = label("warm", fold, s), label("rl", fold, s)
                ex = [i for i in rows if not rec(w, i)["passed"]][:2]
                cases["F_rl_unification"].append({"mechanism": m, "fold": fold, "seed": s, "dose": o["dose"], "rows": ex, "problem": final[ex[0]]["problem"],
                                                  "warm_responses": [resp(rec(w, i)) for i in ex], "rl_response": resp(rec(r_, ex[0])),
                                                  "warm_actual": [[d.get("actual") for d in rec(w, i).get("details") or []] for i in ex], "expected": [final[i]["sub_answers"] for i in ex]})
        if m in KEYWORDS:
            for s in seeds:
                for kind in ("warm", "rl"):
                    lab = label(kind, fold, s)
                    for i in rows:
                        r = rec(lab, i)
                        if r is None or r["passed"]:
                            continue
                        code = resp(r)
                        comments = [l.strip() for l in code.splitlines() if l.strip().startswith("#") or '"""' in l]
                        hits = [l for l in comments if re.search(KEYWORDS[m], l, re.I)]
                        sv = final[i].get("shortcut_values") or {}
                        if hits and captured(r, sv):
                            cases["H_comment_knows_but_uses"].append({"mechanism": m, "label": lab, "row": i, "comment_lines": hits[:4], "response": code, "expected": final[i]["sub_answers"], "shortcut_values": sv})
                            break

    # ---- textbook traps after RL (G) ----------------------------------------------------------------------------------
    tb_in = {r["id"]: r for r in json.loads((args.x_textbook / "inputs" / "textbook.json").read_text())}
    groups = json.loads((args.x_textbook / "inputs" / "textbook_groups.json").read_text())
    def tb(lab):
        p = args.x_textbook / "eval" / lab / "textbook.json"
        return {r["id"]: r for r in json.loads(p.read_text())["results"]} if p.exists() else {}
    base = tb("x-qwen3-8b-base")
    a_labels = [f"x-qwen3-8b-{f}-A-s{s}" for f in ("F1", "F2") for s in (42, 43)]
    arms = {l: tb(l) for l in a_labels}
    for i in groups["regime_arm"]:
        t, sv = tb_in[i], tb_in[i].get("shortcut_values") or {}
        b, ends = base.get(i), [arms[l].get(i) for l in a_labels]
        if b is None or any(e is None for e in ends):
            continue
        if not b["passed"] and all(e["passed"] for e in ends):
            cases["G_textbook_learned_by_rl"].append({"trap": i, "category": t["category"], "mechanism": t.get("mechanism"), "problem": t["problem"], "expected": t["sub_answers"],
                                                      "shortcut_values": sv, "base_response": resp(b), "base_captured": captured(b, sv), "a_response": resp(ends[0]), "a_label": a_labels[0]})
        if not b["passed"] and all(not e["passed"] and captured(e, sv) for e in ends):
            cases["G2_textbook_never_learned"].append({"trap": i, "category": t["category"], "mechanism": t.get("mechanism"), "problem": t["problem"], "expected": t["sub_answers"],
                                                      "shortcut_values": sv, "base_response": resp(b), "a_response": resp(ends[0]), "a_label": a_labels[0]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n")
    for k, v in cases.items():
        print(f"{k}: {len(v)}")
        for c in v[:12]:
            key = c.get("trap") or c.get("mechanism")
            extra = c.get("category") or f"dose {c.get('dose')}"
            print(f"   {key:40s} {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
