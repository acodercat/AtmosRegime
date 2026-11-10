"""Blind sample of judged E1 regime answers for auditing the judge (A0, 2026-09-28): 8 A + 8 B from
the base and rl-s120, thinking off, seed 7; run where `experiments/explore_e1` is (paths below are the server's).
Writes the blind file (no label) and the key separately."""
import json, random, re
from atmoscoder_rl.runs import load_numbered_reps
from atmoscoder_rl.paths import EXPERIMENTS_DIR
from atmoscoder_rl.traps import trap_index
D = json.load(open("explore/results/judge_e1_regime.json"))
labels = next(v for k, v in D.items() if k != "_note")
index = trap_index(); runs = load_numbered_reps(EXPERIMENTS_DIR, "explore_e1")
rng = random.Random(7)
pool = {"A": [], "B": []}
for key, v in labels.items():
    cfg, run, tid = key.split("|")
    if cfg in ("qwen3-8b.think-off", "rl-s120.think-off") and not v["passed"] and v["label"] in pool:
        pool[v["label"]].append(key)
sample = rng.sample(pool["A"], 8) + rng.sample(pool["B"], 8); rng.shuffle(sample)
blind, key_out = [], []
for n, key in enumerate(sample, 1):
    cfg, run, tid = key.split("|"); t = index[tid][0]
    rec = runs[cfg][int(run)][tid]
    code = rec["attempts"][-1]["response"]
    m = re.findall(r"```python(.*?)```", code, re.S); code = (m[-1] if m else code)[-1500:]
    blind.append(f"### S{n:02d}\nPROBLEM: {t['problem'][:900]}\nWHAT CHANGED (expert note): {t.get('shortcut') or ''} {t.get('mechanism') or ''}\nCODE (tail):\n{code}\n")
    key_out.append({"sid": f"S{n:02d}", "key": key, "label": labels[key]["label"], "evidence": labels[key]["evidence"]})
open("/path/to/workdir/logs/judge_blind.md", "w").write("\n".join(blind))
json.dump(key_out, open("/path/to/workdir/logs/judge_key.json", "w"), indent=1)
print("pool sizes", {k: len(v) for k, v in pool.items()})
