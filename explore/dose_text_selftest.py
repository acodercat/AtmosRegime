"""CPU self-test of explore/dose_text.py (clause extraction, assignment, text rows, injection) and explore/dose_text_report.py
(on x_report_selftest's fabricated family).

    uv run python -m explore.dose_text_selftest

Report world: units are the first zero-pass training mechanisms of each fold; the text goes to one seed per unit. A warm start
samples a unit (c = 3 of 64, p = 0.047) and the RL endpoint solves it iff the text was injected in that seed; the warm start
itself never solves a unit. So H1-text = 1.0 (holds), every injected observation has p >= 0.02 and succeeds, and the
p < 0.005 group is empty (H2-p untestable). A second world puts half the injected units at p = 0 and failing: H2-p holds.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from explore import dose_text as dt, dose_text_report as dtr
from explore.x_report_selftest import World, close


def main() -> int:
    failures = []

    def check(cond: bool, what: str) -> None:
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            failures.append(what)

    # ---- clause and assignment (real templates, the prereg's table) ----------------------------------------------------
    c = dt.given_clause("displacement_height_canopy")
    check(c.startswith("Because the mast is less than two canopy heights") and "two-thirds" in c and "  " not in c,
          "given clause: the template's given sentence, whitespace normalised")
    dose = json.loads(Path("results/dose/design.json").read_text())
    a = dt.assign(dose["units"])
    check(a["F1"]["42"] == ["diffuse_transmission", "freezing_point_depth", "henry_depletion", "surface_multiple_reflection"]
          and a["F1"]["43"] == ["displacement_height_canopy", "line_broadening", "unconfined_dupuit"]
          and a["F2"]["42"] == ["ch4_forcing_sqrt", "cunningham_slip", "doppler_aliasing", "inertia_gravity_wave", "plume_ground_reflection"]
          and a["F2"]["43"] == ["clausius_clapeyron_extrapolation", "co2_forcing_log", "snowpack_cold_content", "spherical_slant_path",
                                "sun_photometer_airmass"], "assignment equals PREREG_DOSE_TEXT §3's table")
    check(all(sorted(a[f]["42"] + a[f]["43"]) == sorted(dose["units"][f]) for f in a), "every unit in exactly one seed")
    check(a == dt.assign({f: list(reversed(v)) for f, v in dose["units"].items()}), "assignment does not depend on the input order")
    check(all(m in dt.STANDARD for m in sum(dose["units"].values(), [])), "every unit has a standard-approach name")

    # ---- text rows and injection ----------------------------------------------------------------------------------------
    rows = dt.text_rows("u", "the usual way", "Because X, use Y.")
    check(len(rows) == dt.N_ROWS and all(r["kind"] == "knowledge_text" and r["template"] == "u" for r in rows)
          and all([m["role"] for m in r["messages"]] == ["system", "user", "assistant"] for r in rows)
          and all("the usual way" in r["messages"][1]["content"] and "{S}" not in r["messages"][1]["content"] for r in rows)
          and all(r["messages"][2]["content"] == "Because X, use Y." for r in rows)
          and len({r["messages"][1]["content"] for r in rows}) == dt.N_ROWS, "4 text rows: 4 frames with the name, the clause verbatim")

    def pair(t, k):
        return [{"id": f"{t}_{k}_outside", "template": t, "kind": "warmstart"}, {"id": f"{t}_{k}_inside", "template": t, "kind": "warmstart"}]
    sft = [r for t, n in (("keep1", 2), ("u", 3), ("keep2", 1), ("v", 2)) for k in range(n) for r in pair(t, k)]
    out, placed = dt.inject(sft, ["u", "v"], ["v"], {"v": "S_v"}, {"v": "clause v"})
    check(not any(r["template"] == "u" for r in out) and sum(r["template"] == "v" and r["kind"] == "knowledge_text" for r in out) == 4
          and sum(r["kind"] == "warmstart" for r in out) == 6, "units lose every pair; the injected unit gets 4 text rows")
    check(placed == {"v": 6} and out[6]["kind"] == "knowledge_text", "text rows stand where the unit's first pair stood")
    check([r["id"] for r in out if r["kind"] == "warmstart"] == [r["id"] for r in sft if r["template"].startswith("keep")],
          "non-unit rows keep their order")
    for bad, kw, what in ((["w"], {"w": "S"}, "an injected non-unit"), (["u"], {"u": "S"}, "an injected unit without a pair")):
        try:
            dt.inject([r for r in sft if r["template"] != "u"], ["u", "v"], bad, kw, {k: "c" for k in kw}); check(False, what + " is an error")
        except SystemExit:
            check(True, what + " is an error")

    # ---- report ---------------------------------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        w = World(root / "w", "qwen3-8b", [42, 43])
        units = {f: [t for t in w.zero if w.fold_of[t] == f][:6] for f in ("F1", "F2")}
        design = {"units": units, "text": dt.assign(units), "standard": {}, "clauses": {}}
        (root / "design.json").write_text(json.dumps(design))
        pk = root / "passk"; pk.mkdir()

        def write_world(solve, sample):
            for f in units:
                for s in (42, 43):
                    inj = design["text"][f][str(s)]
                    wa = w.specs[f"x-qwen3-8b-{f}-WA-s{s}"]["out"]
                    for lab, out in ((dtr.warm_label("qwen3-8b", f, s), lambda r, f=f: w.trained(r, f) and r["template"] not in units[f]),
                                     (dtr.rl_label("qwen3-8b", f, s),
                                      lambda r, f=f, inj=inj, wa=wa: solve(r, inj) if r["template"] in units[f] else wa(r))):
                        w.specs[lab] = {}
                        w.write(lab, out=out)
                    with open(pk / f"passk_final_outside.{dtr.warm_label('qwen3-8b', f, s)}.jsonl", "w") as fh:
                        for r in w.rows:
                            if r["variant"] == "outside":
                                fh.write(json.dumps({"id": r["id"], "template": r["template"], "variant": "outside", "n": 64,
                                                     "c": sample(r, inj)}) + "\n")
        write_world(lambda r, inj: r["template"] in inj, lambda r, inj: 3 if r["template"] in inj else 0)
        args = argparse.Namespace(design=root / "design.json", name="qwen3-8b", passk_dir=pk, eval_dir=root / "w" / "eval",
                                  final=root / "w" / "final.json", inputs=root / "w" / "inputs", seeds=[42, 43], dose_posthoc=None)
        r = dtr.report(args)
        obs = r["observations"]
        check(r["complete"] and not r["missing"], "complete world reads as complete")
        check(len(obs) == 2 * sum(len(v) for v in units.values()) and sum(o["injected"] for o in obs) == len(obs) // 2,
              f"one observation per unit and seed, half injected ({len(obs)})")
        h = r["H1_text"]
        check(close(h["value"], 1.0) and h["ci95"][0] > 0 and h["verdict"] == "成立" and close(h["unpaired"]["value"], 1.0),
              "H1-text: paired difference 1.0, holds; unpaired 1.0")
        g = r["H2_p"]
        check(g["p>=0.02"]["n"] == len(obs) // 2 and g["p>=0.02"]["successes"] == g["p>=0.02"]["n"] and g["p<0.005"]["n"] == 0
              and g["verdict"] == "不可检验" and g["untestable_groups"] == ["p<0.005"], "H2-p: every injected observation at p 0.047 and solved; p < 0.005 group empty -> untestable")
        s = r["secondary"]
        check(close(s["groups"]["not_injected"]["rl"], 0) and s["groups"]["not_injected"]["p_at_or_above_0.02"] == 0
              and s["groups"]["injected"]["p_at_or_above_0.02"] == len(obs) // 2, "groups: not injected RL 0 and p 0; injected p >= 0.02")
        check(all(close(p["p_text"], 3 / 64) and close(p["p_none"], 0) for p in s["p_text_vs_none"]) and len(s["p_text_vs_none"]) == len(obs) // 2,
              "p text vs none per mechanism")
        check(close(s["rl-warm"]["injected"]["value"], 1.0) and close(s["not_injected_rl-A"]["value"], 0)
              and close(s["non_unit_new_rl-WA"]["value"], 0), "RL - warm 1.0 on injected; not injected = A; non-unit = WA")
        check(all(o["shortcut_warm"] is not None and o["shortcut_rl"] is not None for o in obs), "shortcut counts read (0 without details)")

        # H2-p holds: half the injected units unreachable (p = 0) and unsolved
        half = {f: set(sorted(units[f])[::2]) for f in units}
        write_world(lambda r, inj: r["template"] in inj and r["template"] not in half[w.fold_of[r["template"]]],
                    lambda r, inj: 3 if r["template"] in inj and r["template"] not in half[w.fold_of[r["template"]]] else 0)
        r2 = dtr.report(args)
        g2 = r2["H2_p"]
        check(g2["verdict"] == "成立" and g2["p<0.005"]["successes"] == 0 and g2["p>=0.02"]["successes"] == g2["p>=0.02"]["n"]
              and g2["p<0.005"]["n"] + g2["p>=0.02"]["n"] == len(obs) // 2, "H2-p holds when the unreachable half fails and the reachable half succeeds")
        check(close(r2["H1_text"]["value"], 0.5) and r2["H1_text"]["verdict"] == "成立", "H1-text 0.5 there, still holds")

        # H2-p fails: the unreachable half (p = 0) is solved all the same
        write_world(lambda r, inj: r["template"] in inj,
                    lambda r, inj: 3 if r["template"] in inj and r["template"] not in half[w.fold_of[r["template"]]] else 0)
        r2b = dtr.report(args)
        g2b = r2b["H2_p"]
        check(g2b["verdict"] == "不成立" and g2b["p<0.005"]["successes"] == g2b["p<0.005"]["n"] > 0
              and g2b["p>=0.02"]["successes"] == g2b["p>=0.02"]["n"] > 0, "H2-p does not hold when the p = 0 half is solved too")

        # H1-text fails when RL ignores the text (solves every unit)
        write_world(lambda r, inj: True, lambda r, inj: 3 if r["template"] in inj else 0)
        r3 = dtr.report(args)
        check(close(r3["H1_text"]["value"], 0) and r3["H1_text"]["verdict"] == "不成立", "RL solving every unit: H1-text = 0, does not hold")

        # incomplete: one pass@64 file missing
        (pk / f"passk_final_outside.{dtr.warm_label('qwen3-8b', 'F2', 43)}.jsonl").unlink()
        r4 = dtr.report(args)
        check(not r4["complete"] and r4["H1_text"]["verdict"] == "不完整" and r4["H2_p"]["verdict"] == "不完整"
              and any("F2-s43" in m for m in r4["missing"]), "a missing pass@64 file: 不完整, named in missing")
        dtr.show(r)

    print("SELFTEST " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
