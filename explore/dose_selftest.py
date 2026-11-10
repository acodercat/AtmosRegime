"""CPU self-test of explore/dose.py (unit rule, assignment, thinning) and explore/dose_report.py (on x_report_selftest's
fabricated family: real templates and folds).

    uv run python -m explore.dose_selftest

Report world: units are the first zero-pass training mechanisms of each fold. Warm starts and RL endpoints solve a unit's
outside rows iff its dose is >= 2; the warm start's pass@64 is > 0 iff the dose is >= 2. So dose 0 and 1 stay at 0 and
dose 2 and 4 reach 1: H1 = share of dose >= 1 observations with dose >= 2, H2 holds (0 of the pass@64 = 0 group, all of
the pass@64 > 0 group). Non-unit training mechanisms: the new RL endpoints copy WA's verdicts.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from explore import dose, dose_report as dr
from explore.x_report_selftest import World, close


def main() -> int:
    failures = []

    def check(cond: bool, what: str) -> None:
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            failures.append(what)

    # ---- unit rule ------------------------------------------------------------------------------------------------------
    per = {"in_all": {"fold": "F1", "pass@64": 0.10, "A_s42": 0.4, "A_s43": 0.58},      # pass@64 at the bound, A mean 0.49
           "pass_hi": {"fold": "F1", "pass@64": 0.11, "A_s42": 0.0, "A_s43": 0.0},
           "a_hi": {"fold": "F2", "pass@64": 0.0, "A_s42": 0.5, "A_s43": 0.5},
           "few_pairs": {"fold": "F2", "pass@64": 0.0, "A_s42": 0.0, "A_s43": 0.0},
           "plume_mixing_lid": {"fold": "F2", "pass@64": 0.0, "A_s42": 0.0, "A_s43": 0.0},
           "no_pass": {"fold": "F2", "pass@64": None, "A_s42": 0.0, "A_s43": 0.0},
           "f2_ok": {"fold": "F2", "pass@64": 0.0, "A_s42": 0.0, "A_s43": 0.0}}
    pairs = {"F1": {"in_all": 4, "pass_hi": 20}, "F2": {"a_hi": 20, "few_pairs": 3, "plume_mixing_lid": 20, "no_pass": 20, "f2_ok": 4}}
    u = dose.units(per, pairs)
    check(u == {"F1": ["in_all"], "F2": ["f2_ok"]},
          "unit rule: pass@64 <= 0.10, A mean < 0.5, >= 4 pairs, plume_mixing_lid and no pass@64 out")
    try:
        dose.units({"x": {"fold": "F1", "pass@64": 0.0, "A_s42": 0.0}}, {"F1": {"x": 9}})
        check(False, "a missing A seed is an error")
    except SystemExit:
        check(True, "a missing A seed is an error")

    # ---- assignment -----------------------------------------------------------------------------------------------------
    mech = [f"m{i}" for i in range(10)]
    a = dose.assign(mech, "F2", 42)
    c = Counter(a.values())
    check(sorted(a) == sorted(mech) and set(c) <= set(dose.LEVELS) and all(c[lv] >= 2 for lv in dose.LEVELS)
          and max(c.values()) - min(c.values()) <= 1, f"10 units: every level 2 or 3 times ({dict(c)})")
    check(a == dose.assign(list(reversed(mech)), "F2", 42), "assignment does not depend on the input order")
    check(a != dose.assign(mech, "F2", 43) and a != dose.assign(mech, "F1", 42), "independent per seed and fold")
    seven = Counter(dose.assign(mech[:7], "F1", 42).values())
    check(sorted(seven.values()) == [1, 2, 2, 2], f"7 units: three levels twice, one once ({dict(seven)})")

    # ---- thinning -------------------------------------------------------------------------------------------------------
    def pair(t, k):
        return [{"id": f"{t}_{k}_outside", "template": t}, {"id": f"{t}_{k}_inside", "template": t}]
    sft = [r for t, n in (("u", 6), ("v", 3), ("keep", 5)) for k in range(n) for r in pair(t, k)]
    out, chosen = dose.thin(sft, {"u": 2, "v": 0}, "F1", 42)
    ids = [r["id"] for r in out]
    check(len(chosen["u"]) == 2 and chosen["v"] == [] and sum(r["template"] == "keep" for r in out) == 10
          and sum(r["template"] == "u" for r in out) == 4 and not any(r["template"] == "v" for r in out),
          "dose 2 keeps 2 pairs (4 rows), dose 0 none, a non-unit template all")
    check(ids == [r["id"] for r in sft if r["id"] in set(ids)], "kept rows keep their order")
    check(all(f"{i[:-len('_outside')]}_inside" in ids for i in chosen["u"]), "a kept pair keeps its inside twin")
    check(dose.thin(sft, {"u": 2}, "F1", 42)[1] == {"u": chosen["u"]} and dose.thin(sft, {"u": 2}, "F1", 43)[1] != {"u": chosen["u"]},
          "pair choice fixed per (fold, seed, template), differs between seeds")
    check(dose.thin(sft, {}, "F1", 42)[0] == sft, "no doses (--full) keeps every row")
    for bad, what in (({"v": 4}, "a dose above the available pairs"),):
        try:
            dose.thin(sft, bad, "F1", 42); check(False, what + " is an error")
        except SystemExit:
            check(True, what + " is an error")
    try:
        dose.thin(sft[1:] + sft[:1], {"u": 1}, "F1", 42); check(False, "rows out of (outside, inside) order are an error")
    except SystemExit:
        check(True, "rows out of (outside, inside) order are an error")

    # ---- DOSE-2 within-dose H2, on hand-made observations ---------------------------------------------------------------
    def ob(t, d, p, rl):
        return {"mechanism": t, "fold": "F1", "seed": 42, "dose": d, "warm_pass64": p, "rl": rl}
    hand = [ob("a", 1, 0.0, 0.0), ob("b", 1, 0.1, 1.0), ob("c", 2, 0.0, 0.0), ob("d", 2, 0.2, 1.0), ob("e", 2, 0.3, 0.0),
            ob("f", 4, 0.5, 1.0), ob("z", 0, 0.0, 0.0)]
    check(close(dr.within_dose_stat(hand), (2 * 1.0 + 3 * 0.5) / 5),
          "within-dose difference: levels with both groups only, weighted by the level's observations (0.70, not 0.75)")
    lv = dr.exploratory(hand)["h2_within_dose"]["levels"]
    check(sorted(lv) == [1, 2, 4] and lv[2] == {"pass64_0": {"n": 1, "successes": 0}, "pass64_pos": {"n": 2, "successes": 1}},
          "within-dose counts per level, dose 0 left out")

    # ---- report ---------------------------------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        w = World(root / "w", "qwen3-8b", [42, 43])
        units = {f: [t for t in w.zero if w.fold_of[t] == f][:6] for f in ("F1", "F2")}
        doses = {f: {str(s): dose.assign(units[f], f, s) for s in (42, 43)} for f in units}
        design = {"levels": list(dose.LEVELS), "doses": doses}
        (root / "design.json").write_text(json.dumps(design))
        pk = root / "passk"; pk.mkdir()
        for f in units:
            for s in (42, 43):
                d = doses[f][str(s)]
                wa = w.specs[f"x-qwen3-8b-{f}-WA-s{s}"]["out"]
                hit = lambda r, d=d: d.get(r["template"], 0) >= 2
                for lab, out in ((dr.warm_label("qwen3-8b", f, s), lambda r, f=f, hit=hit: w.trained(r, f) and hit(r)),
                                 (dr.rl_label("qwen3-8b", f, s),
                                  lambda r, f=f, d=d, hit=hit, wa=wa: hit(r) if r["template"] in d else wa(r))):
                    w.specs[lab] = {}
                    w.write(lab, out=out)
                with open(pk / f"passk_final_outside.{dr.warm_label('qwen3-8b', f, s)}.jsonl", "w") as fh:
                    for r in w.rows:
                        if r["variant"] == "outside":
                            fh.write(json.dumps({"id": r["id"], "template": r["template"], "variant": "outside", "n": 64,
                                                 "c": 3 if hit(r) else 0}) + "\n")
        args = argparse.Namespace(design=root / "design.json", name="qwen3-8b", passk_dir=pk, eval_dir=root / "w" / "eval",
                                  final=root / "w" / "final.json", inputs=root / "w" / "inputs", seeds=[42, 43])
        r = dr.report(args)
        obs = r["observations"]
        dosed = [o for o in obs if o["dose"] >= 1]
        want_h1 = sum(o["dose"] >= 2 for o in dosed) / len(dosed)
        check(r["complete"] and not r["missing"], "complete world reads as complete")
        check(len(obs) == 2 * sum(len(v) for v in units.values()), f"one observation per unit and seed ({len(obs)})")
        check(close(r["H1"]["value"], want_h1) and r["H1"]["verdict"] == "成立", f"H1 = {want_h1:.3f}, holds")
        check(r["H1"]["ci95"][0] > 0 and r["H1"]["draws_used"] > 9000, "H1 interval above 0, draws without a group dropped")
        h2 = r["H2"]
        check(h2["pass64_0"]["successes"] == 0 and h2["pass64_pos"]["successes"] == h2["pass64_pos"]["n"]
              and h2["pass64_0"]["n"] == sum(o["dose"] == 1 for o in obs) and h2["verdict"] == "成立",
              "H2: dose 1 = pass@64 0, none succeed; dose 2 / 4 all succeed; holds")
        pd = r["secondary"]["per_dose"]
        check(close(pd[0]["rl"], 0) and close(pd[1]["rl"], 0) and close(pd[2]["rl"], 1) and close(pd[4]["rl"], 1)
              and pd[4]["warm_pass64_pos"] == pd[4]["n"] and pd[1]["warm_pass64_pos"] == 0, "per dose: RL 0 / 0 / 1 / 1")
        s = r["secondary"]
        check(close(s["rl-warm"]["all_dosed"]["value"], 0) and s["rl-warm"]["dosed_warm_below_50_pass64_pos"]["observations"] == 0,
              "RL - warm 0 when the warm start already solves; no 'low but sampled' observation")
        check(close(s["dose0_spillover"]["rl-A"]["value"], 0) and close(s["dose0_spillover"]["warm-base"]["value"], 0),
              "dose 0: RL = A = 0, warm = base = 0 (no spillover)")
        check(close(s["non_unit_new_rl-WA"]["value"], 0), "non-unit mechanisms: new RL copies WA, difference 0")
        e = r["exploratory"]
        check(e["h2_within_dose"]["pooled_difference"]["value"] is None
              and all(min(v["pass64_0"]["n"], v["pass64_pos"]["n"]) == 0 for v in e["h2_within_dose"]["levels"].values()),
              "DOSE-2: pass@64 > 0 iff dose >= 2, so no level has both groups and the pooled difference is None")
        runs = e["h1_per_run"]
        check(sorted(runs) == ["F1-s42", "F1-s43", "F2-s42", "F2-s43"]
              and sum(v["n_dosed"] + v["n_dose0"] for v in runs.values()) == len(obs)
              and all(close(v["difference"], dr.h1_stat([o for o in obs if f"{o['fold']}-s{o['seed']}" == k]))
                      and close(v["rl_dose0"], 0) for k, v in runs.items()),
              "DOSE-2: H1 per run, four runs covering every observation, dose 0 at 0")

        # H2 untestable: every dose >= 1 observation sampled by the warm start
        for f in units:
            for s_ in (42, 43):
                d = doses[f][str(s_)]
                with open(pk / f"passk_final_outside.{dr.warm_label('qwen3-8b', f, s_)}.jsonl", "w") as fh:
                    for row in w.rows:
                        if row["variant"] == "outside":
                            fh.write(json.dumps({"id": row["id"], "template": row["template"], "variant": "outside", "n": 64,
                                                 "c": 3 if d.get(row["template"], 0) >= 1 else 0}) + "\n")
        r2 = dr.report(args)
        check(r2["H2"]["pass64_0"]["n"] == 0 and r2["H2"]["verdict"] == "不可检验", "H2 with an empty pass@64 = 0 group: 不可检验")
        check(r2["H1"]["verdict"] == "成立", "H1 unaffected by pass@64")

        # H1 fails when RL ignores the dose (solves every unit)
        for f in units:
            for s_ in (42, 43):
                w.write(dr.rl_label("qwen3-8b", f, s_), out=lambda row, f=f: w.trained(row, f))
        r3 = dr.report(args)
        check(close(r3["H1"]["value"], 0) and r3["H1"]["verdict"] == "不成立", "RL solving every unit: H1 = 0, does not hold")

        # incomplete: one RL endpoint without its final.json
        (root / "w" / "eval" / dr.rl_label("qwen3-8b", "F1", 43) / "final.json").unlink()
        r4 = dr.report(args)
        check(not r4["complete"] and r4["H1"]["verdict"] == "不完整" and r4["H2"]["verdict"] == "不完整"
              and any("F1-WA-s43" in m for m in r4["missing"]), "a missing endpoint: 不完整, named in missing")
        dr.show(r)

    print("SELFTEST " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
