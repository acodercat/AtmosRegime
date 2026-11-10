"""Stage 1, E1 regime arm (textbook traps): shortcut capture and recognition conditioned on the
parent being right in the same repetition, for every arm, and on the parents the base already
solves (>= 2 of 3 repetitions). Exploratory, 2026-09-27. Run where `experiments/explore_e1` is:

    python -m explore.e1_conditional
"""
from atmoscoder_rl.capture import shortcut_match
from atmoscoder_rl.datasets import split_ids
from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.runs import is_gradable, load_runs
from atmoscoder_rl.traps import arm_ids, trap_index
index = trap_index(); train = set(split_ids("train"))
ids = arm_ids("regime", index, train)
runs = load_runs(EXPERIMENTS_DIR, "explore_e1")
for cfg in ("qwen3-8b.think-off", "rl-s120.think-off", "seed1-s120.think-off", "seed2-s120.think-off",
            "seed3-s120.think-off", "ctrl-random-s120.think-off", "sft-gold.think-off", "sft-raft.think-off"):
    par_ok = cond_cap = wrong = cap_wrong = n = 0
    for rep in runs.get(cfg, []):
        for tid in ids:
            trap = index[tid][0]; t, p = rep.get(tid), rep.get(trap["parent_id"])
            if not (is_gradable(t) and is_gradable(p)): continue
            n += 1; sc = (not t["passed"]) and shortcut_match(t, trap)
            if not t["passed"]: wrong += 1; cap_wrong += sc
            if p["passed"]: par_ok += 1; cond_cap += sc
    if n: print(f"{cfg:28s} pairs {n:3d}  parent right {par_ok:3d}  capture|parent right {cond_cap:3d} ({cond_cap/max(par_ok,1):.0%})   capture|trap wrong {cap_wrong}/{wrong} ({cap_wrong/max(wrong,1):.0%})")

# Items whose parent the base already solves (>= 2 of 3 repetitions): the template primary's analogue.
base_reps = runs["qwen3-8b.think-off"]
known = [tid for tid in ids
         if sum(bool(r.get(index[tid][0]["parent_id"], {}).get("passed")) for r in base_reps) >= 2]
print(f"\nbase-known parents: {len(known)} of {len(ids)} regime traps")
for cfg in ("qwen3-8b.think-off", "rl-s120.think-off", "seed1-s120.think-off", "seed2-s120.think-off",
            "seed3-s120.think-off", "ctrl-random-s120.think-off"):
    par_ok = cond_cap = trap_ok = n = 0
    for rep in runs.get(cfg, []):
        for tid in known:
            trap = index[tid][0]; t, p = rep.get(tid), rep.get(trap["parent_id"])
            if not (is_gradable(t) and is_gradable(p)): continue
            n += 1; trap_ok += bool(t["passed"])
            if p["passed"]:
                par_ok += 1; cond_cap += (not t["passed"]) and shortcut_match(t, trap)
    print(f"{cfg:28s} pairs {n:3d} parent right {par_ok:3d} trap right {trap_ok:3d}  capture|parent right {cond_cap} ({cond_cap/max(par_ok,1):.0%})")

print("\nrecognition = trap right | parent right (all regime traps / base-known subset)")
for cfg in ("qwen3-8b.think-off", "rl-s120.think-off", "seed1-s120.think-off", "seed2-s120.think-off",
            "seed3-s120.think-off", "ctrl-random-s120.think-off", "sft-gold.think-off", "sft-raft.think-off"):
    out = []
    for subset in (ids, known):
        pr = tr_ = 0
        for rep in runs.get(cfg, []):
            for tid in subset:
                trap = index[tid][0]; t, p = rep.get(tid), rep.get(trap["parent_id"])
                if is_gradable(t) and is_gradable(p) and p["passed"]:
                    pr += 1; tr_ += bool(t["passed"])
        out.append(f"{tr_}/{pr} ({tr_/max(pr,1):.0%})")
    print(f"{cfg:28s} all {out[0]:14s} base-known {out[1]}")
