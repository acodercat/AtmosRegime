"""CPU self-test of explore/x_warm_knowledge.py on x_report_selftest's fabricated family (real templates, folds, clusters).

    uv run python -m explore.x_warm_knowledge_selftest

Injected = every "known" mechanism and every other zero-pass one (2 pairs each in the warm summaries). W_f solves pair 0 of
injected outside rows (accuracy 0.5) and has W_f pass@64 > 0 exactly there; WA solves every injected outside row and nothing
else; WD solves every training-fold row (so the not-injected zero-pass mechanisms are WD's >= 50 % hits); A and D are
x_report_selftest's (known mechanisms only, seed 43 half of seed 42); the base solves nothing.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from explore import x_warm_knowledge as xw
from explore.x_report import P1_EXCLUDED
from explore.x_report_selftest import World, close


def main() -> int:
    failures = []

    def check(cond: bool, what: str) -> None:
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            failures.append(what)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        w = World(root / "w", "qwen3-8b", [42, 43])
        zero = [t for t in w.zero]
        injected = set(w.known) | set(zero[::2])
        z_inj = sorted(t for t in zero[::2] if t not in P1_EXCLUDED)
        z_not = sorted(t for t in zero[1::2] if t not in P1_EXCLUDED)
        for f in ("F1", "F2"):
            w.write(f"sft-x-qwen3-8b-{f}", out=lambda r, f=f: w.trained(r, f) and r["template"] in injected and r["p"] == 0)
            for s in (42, 43):
                w.write(f"x-qwen3-8b-{f}-WA-s{s}", out=lambda r, f=f: w.trained(r, f) and r["template"] in injected)
                w.write(f"x-qwen3-8b-{f}-WD-s{s}", out=lambda r, f=f: w.trained(r, f))
        wf, summ = {}, {}
        for f in ("F1", "F2"):
            wf[f] = root / f"wf-{f}.jsonl"
            with open(wf[f], "w") as fh:
                for r in w.rows:
                    if r["variant"] == "outside":
                        fh.write(json.dumps({"id": r["id"], "template": r["template"], "variant": "outside", "n": 64,
                                             "c": 3 if r["template"] in injected else 0}) + "\n")
            summ[f] = root / f"summary-{f}.json"
            summ[f].write_text(json.dumps({"pairs_per_template": {t: 2 for t in injected if w.fold_of[t] == f}}))
        args = argparse.Namespace(name="qwen3-8b", seeds=[42, 43], eval_dir=root / "w" / "eval", final=root / "w" / "final.json",
                                  inputs=root / "w" / "inputs", passk=root / "w" / "passk.jsonl", passk_wf=wf, warm_summary=summ)
        r = xw.report(args)
        a, rows = r["analysis"], r["per_mechanism"]
        g = a["groups"]

        check(not set(P1_EXCLUDED) & set(rows), "plume_mixing_lid excluded, as in P1")
        check(len(rows) == len([t for t in w.fold_of if t not in P1_EXCLUDED]), "every training-fold mechanism has a row")
        check(all(rows[t]["fold"] == w.fold_of[t] for t in rows), "each mechanism scored by the arm trained on its fold")
        check(g["Z_injected"]["mechanisms"] == z_inj and g["Z_not_injected"]["mechanisms"] == z_not,
              f"Z split by injection ({len(z_inj)} / {len(z_not)})")
        check(sorted(a["Z"]) == sorted(z_inj + z_not), "Z = base pass@64 = 0")
        check(all(rows[t]["pairs"] == 2 for t in z_inj) and all(rows[t]["pairs"] == 0 for t in z_not), "pairs read from the summaries")
        ai, an = g["Z_injected"]["accuracy"], g["Z_not_injected"]["accuracy"]
        wf_want = sum(.25 if t == w.t1 else .5 for t in z_inj) / len(z_inj)   # T1 has 4 pairs: pair 0 is 1 / 4 of its rows
        check(w.t1 in z_inj and close(ai["W_f"]["mean"], wf_want), "Z injected: W_f per template 1 / pairs (T1 0.25), then over mechanisms")
        check(close(ai["base"]["mean"], 0) and close(ai["WA"]["mean"], 1) and close(ai["A"]["mean"], 0), "Z injected: base 0, WA 1, A 0")
        check(close(an["W_f"]["mean"], 0) and close(an["WA"]["mean"], 0) and close(an["WD"]["mean"], 1), "Z not injected: W_f 0, WA 0, WD 1")
        di = g["Z_injected"]["differences"]
        check(close(di["W_f-base"]["mean"], wf_want) and close(di["WA-W_f"]["mean"], 1 - wf_want) and close(di["WA-A"]["mean"], 1)
              and close(di["A-base"]["mean"], 0), "Z injected differences: W_f-base, WA-W_f = 1 - W_f, WA-A 1, A-base 0")
        check(di["WA-W_f"]["units"] == len(z_inj), "differences are over mechanisms")
        check(g["Z_injected"]["at_or_above_50"]["WA"] == z_inj
              and g["Z_injected"]["at_or_above_50"]["W_f"] == [t for t in z_inj if t != w.t1],
              "W_f at exactly 50 % counts as >= 50 %, T1 at 25 % does not")
        rest = g["rest"]["accuracy"]
        check(close(rest["A"]["mean"], .75) and close(rest["WA"]["mean"], 1), "rest: A seed mean (1 + 0.5) / 2, WA 1")
        p = a["p1_warm_start"]
        check(sorted(p["Z_wf"]) == z_not, "Z_wf = W_f pass@64 = 0 = the not-injected zero-pass mechanisms")
        check(p["Z_wf_WA_at_or_above_50_any_seed"] == [] and p["Z_wf_WD_at_or_above_50_every_seed"] == z_not,
              "Z_wf hits: WA none, WD all (every seed)")
        check(close(p["spearman_wf_pass64_vs_WA"]["rho"], 1.0), "Spearman(W_f pass@64, WA) = 1 when WA follows W_f pass@64")
        check(p["spearman_wf_pass64_vs_WD"]["rho"] is None, "Spearman with a constant WD (1 everywhere) is undefined, reported as None")
        xw.show(r)

    print("SELFTEST " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
