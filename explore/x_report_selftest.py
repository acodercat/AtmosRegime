"""CPU self-test of explore/x_report.py: fabricate a family's evaluations with known answers in a temp dir, check the numbers,
the fold mapping and the verdicts. No model, no GPU, no server.

    uv run python -m explore.x_report_selftest

Templates, clusters and folds are the real ones (results/kc/split.json); 2 pairs per template, except one zero-pass template
(T1) with 4 pairs, so row and template weighting differ. Mechanisms in every third cluster are "known" (base pass@64 > 0).
Base, W_F1 and W_F2 differ on outside rows (W_F1 solves all F2 rows, W_F2 pair 0 of F1 rows), so a wrong start for WA / WD
shows up as transfer. Seed 43 solves only pair 0 where seed 42 solves both. F answers every judge-run row wrong (accuracy
must come from final.json), and its final inside pair-1 responses carry a "regime" key with the runner's stored verdict
set to fail (accuracy must come from the regrade). The D-3 file is fabricated. Scenarios: main; P1 at the exact 50 %
boundary on both seeds (not met) and on one seed only (otherwise); incomplete; family 2 with the base already detecting (gain
over the start, not over A); P2 not met / parse-rate / accuracy / all-zero; infrastructure errors (denominators, P2b
within-seed pairing); the X-17 knowledge block (fabricated knowledge file); the judge-row cache (read back, used, replaced on change).
"""
from __future__ import annotations

import argparse
import functools
import json
import shutil
import tempfile
from pathlib import Path

from atmoscoder_rl import pilot
from atmoscoder_rl.paths import REPO_ROOT
from explore import x_report

SUB = [{"sub": "1", "value": "10", "unit": "m"}]
TEST_IDS = [f"core_{i}" for i in range(10)]
TRAPS = {f"trap_{i}": f"parent_{i}" for i in range(4)}
D3 = {"qwen3-8b.think-off": False, "qwen3.5-9b.think-off": True}


def response(right: bool, judgement: int | None = None, regime: bool = False) -> str:
    extra = "" if judgement is None and not regime else f', "regime": {0 if judgement is None else judgement}'
    return f'```python\ndef solve():\n    return {{"1": {{"value": {10 if right else 20}, "unit": "m"}}{extra}}}\n```'


def record(i: str, right: bool, *, judgement=None, regime=False, stored=None, tokens=500, error=False) -> dict:
    if error:
        return {"id": i, "passed": False, "num_attempts": 0, "error": "connection reset", "attempts": []}
    return {"id": i, "passed": right if stored is None else stored, "num_attempts": 1,
            "details": [{"sub": "1", "expected": 10, "actual": 10 if right else 20}],
            "attempts": [{"prompt": "", "response": response(right, judgement, regime), "outcome": "graded",
                          "usage": {"prompt_tokens": 100, "completion_tokens": tokens, "total_tokens": 100 + tokens}}]}


class World:
    def __init__(self, root: Path, name: str, seeds: list[int]) -> None:
        self.root, self.name, self.seeds = root, name, seeds
        split = json.loads((REPO_ROOT / "results" / "kc" / "split.json").read_text())
        self.split = split
        self.cluster = {t: c for c, ts in split["clusters"].items() for t in ts}
        self.fold_of = {t: f for f, v in split["folds"].items() for t in v["trained_templates"]}
        order = sorted(set(self.cluster.values()))
        self.known = {t for t in self.fold_of if order.index(self.cluster[t]) % 3 == 0}
        big = [c for c in order if order.index(c) % 3 and len(split["clusters"][c]) >= 2
               and all(self.fold_of[t] == "F2" for t in split["clusters"][c])]
        self.t1 = sorted(split["clusters"][big[0]])[0]              # 4 pairs, zero-pass, held out for F1 arms
        self.t2 = sorted(split["clusters"][big[1]])[0]              # infrastructure-error scenario
        pairs = lambda t: 4 if t == self.t1 else 2
        self.rows = [{"id": f"r_{t}_{p}_{v}", "template": t, "cluster": self.cluster[t], "cluster_fold": self.fold_of[t],
                      "variant": v, "p": p, "problem": "x", "sub_answers": SUB}
                     for t in sorted(self.fold_of) for p in range(pairs(t)) for v in ("inside", "outside", "given", "flag")]
        self.zero = sorted(t for t in self.fold_of if t not in self.known and t not in x_report.P1_EXCLUDED)
        inputs = root / "inputs"; inputs.mkdir(parents=True)
        (root / "final.json").write_text(json.dumps(self.rows))
        (inputs / "judge.json").write_text(json.dumps([{"id": "judge:" + r["id"]} for r in self.rows if r["variant"] in pilot.JUDGE_LABEL]))
        (inputs / "test_id.json").write_text(json.dumps([{"id": i} for i in TEST_IDS]))
        (inputs / "textbook.json").write_text(json.dumps([{"id": i} for i in [*TRAPS, *TRAPS.values()]]))
        (inputs / "textbook_groups.json").write_text(json.dumps({"regime_arm": sorted(TRAPS),
                                                                 "regime_recognise_pilot": sorted(TRAPS)[:2], "parent_of": TRAPS}))
        (root / "d3.json").write_text(json.dumps({k: {"can_detect": v, "templates": {"parse_rate": .9, "discrimination": {}}}
                                                  for k, v in D3.items()}))
        with open(root / "passk.jsonl", "w") as fh:
            for r in self.rows:
                if r["variant"] == "outside":
                    fh.write(json.dumps({"id": r["id"], "template": r["template"], "variant": "outside", "n": 64,
                                         "c": 5 if r["template"] in self.known else 0}) + "\n")
        # X-17 knowledge: "known" mechanisms carry no D gap and full K; the first 3 zero-pass ones have K (K2 both, K_any) but
        #   a gap; D_given alternates 1.0 / 0.5 so the given-readable subset is a known count
        self.k_in_z = self.zero[:3]
        self.knowledge = {t: {"D_given": 1.0 if i % 2 == 0 else .5,
                              "D_given_minus_outside": 0.0 if t in self.known else (1.0 if i % 2 == 0 else .5),
                              **{k: 1.0 if t in self.known or t in self.k_in_z else 0.0 for k in ("K2_strict", "K2_lenient", "K_any")},
                              "D_flag_minus_outside": 0.0, **({"hint_share": 0.0} if i % 2 == 0 else {})}
                          for i, t in enumerate(sorted(self.fold_of))}
        (root / "knowledge.json").write_text(json.dumps({"families": {name: self.knowledge}}))
        self.specs = self.default_specs()
        for label in self.specs:
            self.write(label)

    def trained(self, r, f): return r["cluster_fold"] == f
    def sharpen(self, r, f, s): return self.trained(r, f) and r["template"] in self.known and (s == 42 or r["p"] == 0)
    def warm_out(self, f): return (lambda r: r["cluster_fold"] == "F2") if f == "F1" else (lambda r: r["cluster_fold"] == "F1" and r["p"] == 0)

    def default_specs(self) -> dict:
        n, sp = self.name, {}
        sp[f"x-{n}-base"] = {}
        for f in ("F1", "F2"):
            sp[f"sft-x-{n}-{f}"] = {"out": self.warm_out(f)}
            for s in self.seeds:
                sp[f"x-{n}-{f}-A-s{s}"] = {"out": lambda r, f=f, s=s: self.sharpen(r, f, s)}
                sp[f"x-{n}-{f}-D-s{s}"] = {"out": lambda r, f=f, s=s: self.sharpen(r, f, s) or (not self.trained(r, f) and r["p"] == 0)}
                sp[f"x-{n}-{f}-F-s{s}"] = {"out": lambda r, f=f, s=s: self.sharpen(r, f, s),
                                           "judge": lambda r: pilot.JUDGE_LABEL[r["variant"]], "jright": lambda r: False,
                                           "regime": lambda r: r["variant"] == "inside" and r["p"] == 1,
                                           "trunc": lambda r: r["variant"] == "inside" and r["p"] == 1}
                w = self.warm_out(f)
                sp[f"x-{n}-{f}-WA-s{s}"] = {"out": lambda r, f=f, w=w: self.trained(r, f) or w(r)}
                sp[f"x-{n}-{f}-WD-s{s}"] = {"out": lambda r, f=f, w=w: self.trained(r, f) or w(r), "test": 6}
        return sp

    def write(self, label: str, **change) -> None:
        sp = self.specs[label] = {**self.specs[label], **change}
        get = lambda k, default: sp.get(k, default)
        out, ins, judge, jright = get("out", lambda r: False), get("ins", lambda r: True), get("judge", lambda r: 1), \
            get("jright", lambda r: r["variant"] == "inside")
        infra, regime, trunc = get("infra", lambda kind, r: False), get("regime", lambda r: False), get("trunc", lambda r: False)
        d = self.root / "eval" / label; d.mkdir(parents=True, exist_ok=True)
        final = []
        for r in self.rows:
            right = out(r) if r["variant"] == "outside" else ins(r) if r["variant"] == "inside" else r["variant"] == "given"
            final.append(record(r["id"], right, regime=regime(r), stored=False if regime(r) else None,
                                tokens=8192 if trunc(r) else 500, error=infra("final", r)))
        jud = [record("judge:" + r["id"], jright(r), judgement=judge(r), error=infra("judge", r))
               for r in self.rows if r["variant"] in pilot.JUDGE_LABEL]
        test = [record(i, k < get("test", 8)) for k, i in enumerate(TEST_IDS)]
        tb = [record(i, False) for i in TRAPS] + [record(p, True) for p in TRAPS.values()]
        for kind, recs in (("final", final), ("judge", jud), ("test_id", test), ("textbook", tb)):
            (d / f"{kind}.json").write_text(json.dumps({"metrics": {}, "results": recs}))

    def copy(self, root: Path) -> "World":
        shutil.copytree(self.root, root)
        other = object.__new__(World)
        other.__dict__ = {**self.__dict__, "root": root, "specs": dict(self.specs)}
        return other

    def run(self, d3: Path | None = None, d3_key: str | None = None, knowledge: Path | None = None) -> dict:
        args = argparse.Namespace(name=self.name, seeds=self.seeds, eval_dir=self.root / "eval", final=self.root / "final.json",
                                  inputs=self.root / "inputs", passk=self.root / "passk.jsonl", passk_sens=None,
                                  knowledge=knowledge or self.root / "knowledge.json",
                                  d3=d3 or self.root / "d3.json", d3_key=d3_key or x_report.D3_KEY[self.name],
                                  k2_key=x_report.K2_KEY[self.name], k2_dir=REPO_ROOT / "results" / "x7")
        return x_report.report(args)


def close(a, b, eps=1e-9) -> bool:
    return a is not None and abs(a - b) < eps


def main() -> int:
    grade, regrade = pilot.grade_regime, x_report.regrade_response
    memo_g = functools.lru_cache(maxsize=None)(lambda text, subs, label: grade(text, json.loads(subs), label))
    memo_r = functools.lru_cache(maxsize=None)(lambda text, subs: regrade(text, json.loads(subs)))
    pilot.grade_regime = lambda text, subs, label, **kw: memo_g(text, json.dumps(subs), label)   # unique responses only
    x_report.regrade_response = lambda text, subs: memo_r(text, json.dumps(subs))
    fails = []

    def check(cond: bool, what: str) -> None:
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            fails.append(what)

    tmp = Path(tempfile.mkdtemp(prefix="x_report_selftest_"))
    try:
        # ---- 1. main ------------------------------------------------------------------------------------------------
        w = World(tmp / "s1", "qwen3-8b", [42, 43])
        r = w.run()
        check(r["complete"], "1: complete")
        fm = r["fold_mapping"]
        check(fm["F1"]["training_fold_templates"] == sorted(w.split["folds"]["F1"]["trained_templates"])
              and fm["F1"]["held_out_templates"] == sorted(w.split["folds"]["F2"]["trained_templates"])
              and len(fm["F1"]["training_fold_templates"]) == 29, "1: fold mapping (F1 arm: trains on F1's 29, held out on F2's 41)")
        p = r["P1"]
        check(p["Z"] == w.zero and all(close(p["spearman_pass64_vs_outside"][a]["rho"], 1.0) for a in "ADF")
              and p["verdict"] == "P1 成立", f"1: Z ({len(p['Z'])}), Spearman 1, {p['verdict']}")
        wr = r["warm_x_reward"]
        n_known = len({w.cluster[t] for t in w.known})
        check(close(wr["A"]["mean"], .75 * n_known / 32) and wr["A"]["units"] == 32,
              "1: seeds averaged within cluster before resampling (A = 0.75 x known share, 32 units)")
        check(wr["reading"].startswith("训练机制上的成功由热身注入解释"), "1: warm-start reading fires")
        t = r["transfer"]
        k = len(w.split["clusters"][w.cluster[w.t1]])
        expect_d = (31 * .5 + (.25 + .5 * (k - 1)) / k) / 32          # template-weighted within T1's cluster
        check(close(t["D"]["endpoint-start"]["mean"], expect_d) and t["D"]["verdict"] == "迁移",
              f"1: D transfer {expect_d:.4f} (template-, not row-weighted)")
        check(close(t["WA"]["start"]["mean"], .75) and all(close(t[a]["endpoint-start"]["mean"], 0) for a in ("A", "F", "WA", "WD"))
              and all(t[a]["verdict"] == "只报数字" for a in ("A", "F", "WA", "WD")), "1: WA / WD measured against their own fold's W_f")
        q = r["P2"]
        check(close(q["F-A_H-F_held"]["mean"], 1.0) and q["threshold_points"] == 20 and q["checks"]["gain_is"] == "F - A",
              "1: P2 gain F - A = 100, threshold 20")
        check(q["held_out_accuracy_no_judge_prompt"]["inside"] == {"A": 1.0, "F": 1.0},
              "1: accuracy from the regraded final.json (not the judge run, not the stored verdict)")
        check(q["verdict"] == "P2 成立" and q["F_arm_failure"]["flag"] is None, f"1: {q['verdict']}, no F failure")
        check(r["final_responses_with_regime_key"]["x-qwen3-8b-F1-F-s42"] == 70
              and r["final_responses_with_regime_key"]["x-qwen3-8b-F1-A-s42"] == 0, "1: regime keys counted per label")
        check(r["P2b"]["verdict"].startswith("只报数字"), "1: P2b numbers only")
        notes = {k_: v["note"] for k_, v in r["safety"].items()}
        check(all((v == "能力代价") == ("-WD-" in k_) for k_, v in notes.items()), "1: safety, only WD carries 能力代价")
        check(close(r["truncation"]["by_arm_final"]["F"]["share"], 70 / len(w.rows)), "1: truncation share of F")
        a = r["audit_sample"]
        check(len(a) == 40 and all("-F-" in x["label"] and x["id"].endswith("_outside") for x in a)
              and a == w.run()["audit_sample"], "1: audit sample, 40 held-out outside F rows, deterministic")
        check(r["textbook"]["x-qwen3-8b-base"]["regime_arm"]["trap_right_given_parent_right"] == [0, 4], "1: textbook cells")
        x_report.show(r)

        # ---- 2. P1 branches -----------------------------------------------------------------------------------------
        cheat = w.zero[:2]
        w2 = w.copy(tmp / "s2a")
        for f in ("F1", "F2"):
            for s in w.seeds:   # exactly 50 % (pair 0 of 2) on both seeds
                w2.write(f"x-qwen3-8b-{f}-A-s{s}", out=lambda r, f=f, s=s: w.sharpen(r, f, s)
                         or (w.trained(r, f) and r["template"] in cheat and r["p"] == 0))
        r = w2.run()
        check(r["P1"]["Z_at_or_above_50_every_seed"]["A"] == cheat and r["P1"]["verdict"].startswith("P1 不成立"),
              f"2a: 50 % boundary on both seeds -> {r['P1']['verdict']}")
        w2 = w.copy(tmp / "s2b")
        for f in ("F1", "F2"):
            w2.write(f"x-qwen3-8b-{f}-A-s42", out=lambda r, f=f: w.sharpen(r, f, 42) or (w.trained(r, f) and r["template"] in cheat))
        r = w2.run()
        check(r["P1"]["verdict"] == "其余：只报数字，不下 P1 结论", f"2b: Z hit on one seed only -> {r['P1']['verdict']}")

        # ---- 3. incomplete --------------------------------------------------------------------------------------------
        w3 = w.copy(tmp / "s3")
        (w3.root / "eval" / "x-qwen3-8b-F2-WD-s43" / "test_id.json").unlink()
        r = w3.run()
        check(not r["complete"] and r["missing"]["F2"] == ["x-qwen3-8b-F2-WD-s43/test_id"] and not r["missing"]["F1"]
              and r["P1"]["verdict"] == r["P2"]["verdict"] == r["transfer"]["D"]["verdict"] == "不完整：不作确证性判定",
              "3: missing file named, no confirmatory verdicts")

        # ---- 4. family 2, base already detects (fabricated D-3) ---------------------------------------------------------
        w4 = World(tmp / "s4", "qwen35-9b", [42])
        w4.write("x-qwen35-9b-base", judge=lambda r: 0 if r["variant"] == "outside" and r["p"] == 0 else 1)
        r = w4.run()
        q = r["P2"]
        check(q["threshold_points"] == 15 and q["checks"]["gain_is"] == "F - start" and close(q["F-A_H-F_held"]["mean"], 1.0)
              and q["F-start_H-F_held"]["mean"] < .6 and q["verdict"].startswith("P2（改报）成立"),
              f"4a: gain over the base {q['F-start_H-F_held']['mean']:.3f} >= 15 -> {q['verdict']}")
        w4.write("x-qwen35-9b-base", judge=lambda r: pilot.JUDGE_LABEL[r["variant"]])
        r = w4.run()
        check(close(r["P2"]["F-start_H-F_held"]["mean"], 0) and r["P2"]["verdict"] == "P2 不成立",
              "4b: base judges as well as F: F - A = 100 but F - start = 0 -> P2 不成立")
        check(r["P1"]["verdict"] == "P1 成立" and r["transfer"]["D"]["verdict"] == "迁移", "4: P1 and transfer with one seed")
        for bad in (tmp / "nope.json", None):
            try:
                w4.run(d3=bad, d3_key=None if bad else "no-such-key")
                check(False, "4c: missing D-3 raises")
            except SystemExit:
                check(True, f"4c: missing D-3 {'file' if bad else 'key'} raises")

        # ---- 5. P2 failure branches and infrastructure errors -------------------------------------------------------
        def f_arm(world, **change):
            for f in ("F1", "F2"):
                for s in world.seeds:
                    world.write(f"x-{world.name}-{f}-F-s{s}", **change)
        w5 = w.copy(tmp / "s5a"); f_arm(w5, judge=lambda r: 1)
        r = w5.run()
        check(r["P2"]["verdict"] == "P2 不成立", f"5a: F judges like A -> {r['P2']['verdict']}")
        w5 = w.copy(tmp / "s5b")
        f_arm(w5, judge=lambda r: None if r["variant"] == "outside" and r["p"] == 1 else pilot.JUDGE_LABEL[r["variant"]])
        r = w5.run(); q = r["P2"]
        check(q["rates_held_out"]["F"]["parse_rate"] < .8 and close(q["by_arm"]["F"]["held"]["H-F"]["mean"], 1.0)
              and q["verdict"] == "其余只报数字" and q["F_arm_failure"]["flag"] == "F 臂失败",
              f"5b: unparsable rows out of H's denominator; parse {q['rates_held_out']['F']['parse_rate']:.3f} -> {q['verdict']}, F 臂失败")
        w5 = w.copy(tmp / "s5c"); f_arm(w5, ins=lambda r: r["p"] == 0, regime=lambda r: False)
        r = w5.run()
        check(not r["P2"]["checks"]["accuracy_ok"] and r["P2"]["verdict"] == "其余只报数字", "5c: F loses inside accuracy -> 其余只报数字")
        w5 = w.copy(tmp / "s5d"); f_arm(w5, judge=lambda r: 0)
        r = w5.run()
        check(r["P2"]["F_arm_failure"]["all_zero"] and r["P2"]["verdict"] == "P2 不成立", "5d: F judges all 0 -> F 臂失败, P2 不成立")
        w5 = w.copy(tmp / "s5e")
        t2 = w.t2
        w5.write("x-qwen3-8b-F1-A-s43", infra=lambda kind, r: (kind == "final" and r["template"] == t2 and r["variant"] == "outside")
                 or (kind == "judge" and r["id"] == f"r_{t2}_0_inside"))
        w5.write("x-qwen3-8b-F1-F-s42", out=lambda r: w.sharpen(r, "F1", 42) or r["template"] == t2)
        r = w5.run()
        check(r["infra_errors"]["x-qwen3-8b-F1-A-s43"] == {"final": 2, "judge": 1, "test_id": 0, "textbook": 0}
              and not r["complete"], "5e: infrastructure errors counted per evaluation; run incomplete")
        check(close(r["P2b"]["strict"]["F-A"]["per_template"][t2], 1.0),
              "5e: P2b pairs within seed (seed 43 has no A value for the template: 1.0, not 0.5)")
        check(close(r["transfer"]["A"]["held_out"]["mean"], 0.0), "5e: infrastructure rows stay out of the denominators")

        # ---- 6. X-17 knowledge block (exploratory) ------------------------------------------------------------------
        r = w.run()
        k = r["P1_knowledge"]
        mech = [t for t in w.fold_of if t not in x_report.P1_EXCLUDED]
        check(k["verdict"].startswith("探索性") and k["Z"] == r["P1"]["Z"]
              and all(k["predictors"][n]["templates"] == len(mech) for n in x_report.KNOWLEDGE if n != "hint_share")
              and k["predictors"]["hint_share"]["templates"] == sum("hint_share" in w.knowledge[t] for t in mech),
              "6: block present, numbers only, same Z as P1, every predictor on the mechanisms that have it")
        check(k["predictors"]["D_given_minus_outside|D_given>=0.75"]["templates"]
              == sum(w.knowledge[t]["D_given"] >= .75 for t in mech), "6: given-readable subset uses D_given >= 0.75")
        ka = k["predictors"]["K_any"]["A"]
        check(ka["all"]["rho"] is not None and ka["all"]["rho"] > 0 and ka["all"]["ci95"]
              and k["predictors"]["D_given_minus_outside"]["A"]["all"]["rho"] < 0,
              f"6: direction (K_any rho {ka['all']['rho']:.3f} > 0, D gap rho < 0)")
        check(ka["Z"]["units"] == len(r["P1"]["Z"]) and ka["Z"]["rho"] is None,
              "6: within Z the outcome has no variance -> rho undefined (None), not an error")
        z = k["Z_x_knowledge"]
        check(z["K2_strict_and_lenient_high"]["Z_with_knowledge"] == sorted(w.k_in_z) == z["K_any_high"]["Z_with_knowledge"]
              and close(z["K_any_high"]["outside"]["A"]["with"], 0.0)
              and z["K_any_high"]["Z_without_knowledge"] == len(r["P1"]["Z"]) - len(w.k_in_z),
              "6: Z x knowledge cross-tab lists the zero-pass mechanisms that have knowledge")
        check(w.run(knowledge=tmp / "none.json")["P1_knowledge"] == {"status": "missing"}, "6: missing knowledge file -> missing, no error")

        # ---- 7. judge-row cache (<label>/judge_regrade.json) ----------------------------------------------------------
        fam = lambda world: x_report.Family(argparse.Namespace(name=world.name, seeds=world.seeds, eval_dir=world.root / "eval",
                                                               final=world.root / "final.json", inputs=world.root / "inputs"))
        lab = "x-qwen3-8b-F1-F-s42"
        w7 = w.copy(tmp / "s7")
        cache = w7.root / "eval" / lab / "judge_regrade.json"
        cache.unlink(missing_ok=True)
        f7 = fam(w7)
        rows = [{**f7.rows[i[len(x_report.PREFIX):]], "id": i} for i in f7.inputs["judge"]]
        fresh = x_report.score(rows, f7.records(lab, "judge"), lambda r_: pilot.JUDGE_LABEL[r_["variant"]])
        check(f7.judged(lab) == fresh and cache.exists(), "7: first read scores every row and writes the cache, same list as score()")
        check(fam(w7).judged(lab) == fresh, "7: a new Family reads the cache back, same list")
        got = json.loads(cache.read_text())
        got["scored"][0][1] = None if got["scored"][0][1] is not None else 1
        cache.write_text(json.dumps(got))
        check(fam(w7).judged(lab)[0]["judgement"] == got["scored"][0][1], "7: the cached verdict is what is read (no re-run)")
        w7.write(lab, judge=lambda r_: 0)
        check(all(x["judgement"] == 0 for x in fam(w7).judged(lab))
              and json.loads(cache.read_text())["source_sha256"] != got["source_sha256"], "7: judge.json changed -> rescored, cache replaced")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        pilot.grade_regime, x_report.regrade_response = grade, regrade
    print(f"\n{'SELFTEST PASS' if not fails else f'SELFTEST FAIL ({len(fails)})'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
