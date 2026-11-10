"""verl entry point for the training pilot (docs/PILOT_DESIGN.md): answer reward plus, on rows that
carry a label, a reward for judging whether the standard treatment is valid.

Both pilot arms train with this file. A row's `ground_truth` is either the plain `sub_answers` list
(every standard row, and all of arm A) — scored by `reward.grade` exactly, as `verl_reward.py` does —
or `{"sub_answers": [...], "regime": 0|1}` (arm B's inside / outside template rows, and the
`probe_tpl_heldout_judge` validation file), scored by `pilot.grade_regime` — with the truth's `answer_weight`
when it carries one (PREREG_X arm F: 0.8). Every row returns the same
metric keys, so verl can stack them: `answer`, `judge/labelled`, `judge/correct`, `judge/missing`,
`cf/twinned`, `cf/mapped`, `cf/right` besides `reward.grade`'s outcome breakdown.

A truth carrying a `twin` (arm D of docs/REWARD_V2.md) is scored by `pilot.grade_cf`, the
counterfactual-execution reward: 1 if the solver is also right with the trigger moved to the twin's value.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections import Counter
from typing import Any

from atmoscoder_rl.pilot import answer_weight_of, decode_truth, grade_cf, grade_regime
from atmoscoder_rl.sandbox import counts as sandbox_counts


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    sub_answers, label, twin = decode_truth(ground_truth)
    if twin is not None:
        result = grade_cf(solution_str, sub_answers, twin)
    else:
        result = grade_regime(solution_str, sub_answers, label, answer_weight=answer_weight_of(ground_truth))
    _diag(result, solution_str, extra_info)
    return result


# PREREG_X X-15: per-minute counts of every reward call (outcome, score level, response length), written next to
# the run's logs, to see what the groups dynamic sampling evicts look like. Read-only with respect to the reward:
# it never changes `result`, and any failure here is swallowed. X-16 adds `sandbox`: bwrap start failures and
# fallbacks to the unprivileged layers (`sandbox.counts`) since the previous row. Off unless TENSORBOARD_DIR (set by
# train/dapo_qwen3_8b.sh) is in the environment.
_DIAG_LOCK = threading.Lock()
_DIAG: dict[str, Any] = {"t0": time.time(), "n": 0, "outcome": Counter(), "score": Counter(), "length": Counter(),
                         "variant": Counter(), "examples": [], "exec_error_texts": []}


def _diag(result: dict[str, Any], solution_str: str, extra_info: dict[str, Any] | None) -> None:
    try:
        tb = os.environ.get("TENSORBOARD_DIR")
        if not tb:
            return
        outcome = next((k.split("/", 1)[1] for k, v in result.items() if k.startswith("outcome/") and v), "none")
        score = float(result.get("score", 0.0))
        n = len(solution_str)
        length = "0" if n == 0 else "<50" if n < 50 else "<200" if n < 200 else "<1000" if n < 1000 else ">=1000"
        variant = str((extra_info or {}).get("variant", "standard"))
        with _DIAG_LOCK:
            d = _DIAG
            d["n"] += 1
            d["outcome"][outcome] += 1
            d["score"]["0" if score == 0 else "1" if score == 1 else "partial"] += 1
            d["length"][length] += 1
            d["variant"][f"{variant}:{outcome}"] += 1
            if outcome != "passed" and len(d["examples"]) < 3 and n < 200:
                d["examples"].append(solution_str[:200])
            # X-23: the exact text an exec_error was scored on, to re-run offline against the stored rollouts
            if outcome == "exec_error" and len(d["exec_error_texts"]) < 2:
                d["exec_error_texts"].append(solution_str[:6000])
            now = time.time()
            if now - d["t0"] < 60:
                return
            row = {"t": time.strftime("%F %T"), "secs": round(now - d["t0"]), "n": d["n"],
                   **{k: dict(d[k]) for k in ("outcome", "score", "length", "variant")}, "examples": d["examples"],
                   "exec_error_texts": d["exec_error_texts"],
                   "sandbox": sandbox_counts(reset=True)}
            d.update(t0=now, n=0, outcome=Counter(), score=Counter(), length=Counter(), variant=Counter(), examples=[],
                     exec_error_texts=[])
        out = os.path.join(os.path.dirname(os.path.dirname(tb)), "x", "reward_diag")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"{os.path.basename(tb)}.{socket.gethostname()}.{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass
