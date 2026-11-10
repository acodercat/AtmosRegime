"""Stage-1 judge labels (explore/results/judge_e1_regime.json: A kept the parent method, B recognised
the change but got the replacement wrong, C other) on E1 regime traps, among pairs whose parent is
right in the same repetition; for all regime traps and for the 13 whose parent the base solves.
"A exact" counts A answers that also match the shortcut numerically. Exploratory, 2026-09-28.
Run where `experiments/explore_e1` is:

    python -m explore.judge_conditional
"""
import json
from collections import Counter
from atmoscoder_rl.capture import shortcut_match
from atmoscoder_rl.datasets import split_ids
from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.runs import is_gradable, load_numbered_reps
from atmoscoder_rl.traps import arm_ids, trap_index
D = json.load(open("explore/results/judge_e1_regime.json"))
labels = next(v for k, v in D.items() if k != "_note")
cfgs_judged = sorted({k.split("|")[0] for k in labels})
index = trap_index(); ids = arm_ids("regime", index, set(split_ids("train")))
runs = load_numbered_reps(EXPERIMENTS_DIR, "explore_e1")
base = runs["qwen3-8b.think-off"]
known = [t for t in ids if sum(bool(r.get(index[t][0]["parent_id"], {}).get("passed")) for r in base.values()) >= 2]
def lab(v): return v["label"] if isinstance(v, dict) else v
for cfg in cfgs_judged:
    for name, subset in (("all regime", ids), ("base-known 13", known)):
        c = Counter(); exact = Counter()
        for k, rep in runs.get(cfg, {}).items():
            for tid in subset:
                trap = index[tid][0]; t, p = rep.get(tid), rep.get(trap["parent_id"])
                if not (is_gradable(t) and is_gradable(p) and p["passed"]): continue
                if t["passed"]: c["right"] += 1; continue
                l = labels.get(f"{cfg}|{k}|{tid}")
                l = lab(l) if l is not None else "unjudged"
                c[l] += 1; exact[l] += shortcut_match(t, trap)
        n = sum(c.values())
        print(f"{cfg:28s} {name:14s} parent-right pairs {n:3d}: " + ", ".join(f"{x} {c[x]}" for x in ("right", "A", "B", "C", "unjudged")) + f"   (exact shortcut among A {exact['A']})")
