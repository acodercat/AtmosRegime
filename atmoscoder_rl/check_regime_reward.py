"""Checks for the pilot's judgement reward (`pilot.grade_regime`, `verl_reward_regime`).

    uv run python -m atmoscoder_rl.check_regime_reward

Runs real solvers through the sandbox. Every case states the score it must get; exit 1 on any miss.
"""
from __future__ import annotations

import json

from . import verl_reward, verl_reward_regime
from .pilot import encode_truth, parse_judgement

SUBS = [{"sub": "1", "value": "10.0", "unit": "m/s"}]


def completion(body: str) -> str:
    return f"```python\ndef solve(x=10.0):\n    return {body}\n```"


RIGHT, WRONG = '{"1": {"value": x, "unit": "m/s"}', '{"1": {"value": 2 * x, "unit": "m/s"}'
CASES = [
    # (name, completion, label, expected score)
    ("unlabelled right", completion(RIGHT + "}"), None, 1.0),
    ("unlabelled wrong", completion(WRONG + "}"), None, 0.0),
    ("unlabelled, stray regime key is graded as before", completion(RIGHT + ', "regime": 0}'), None, 1.0),
    ("right answer, right judgement", completion(RIGHT + ', "regime": 1}'), 1, 1.0),
    ("right answer, wrong judgement", completion(RIGHT + ', "regime": 0}'), 1, 0.5),
    ("wrong answer, right judgement", completion(WRONG + ', "regime": 0}'), 0, 0.5),
    ("wrong answer, wrong judgement", completion(WRONG + ', "regime": 1}'), 0, 0.0),
    ("judgement in rule-2 shape", completion(RIGHT + ', "regime": {"value": 1, "unit": ""}}'), 1, 1.0),
    ("judgement as bool", completion(RIGHT + ', "regime": True}'), 1, 1.0),
    ("judgement missing", completion(RIGHT + "}"), 1, 0.5),
    ("judgement malformed (0.5)", completion(RIGHT + ', "regime": 0.5}'), 0, 0.5),
    ("judgement malformed (string)", completion(RIGHT + ', "regime": "no"}'), 0, 0.5),
    # The key is removed before grading: placed first, it must not be matched positionally.
    ("regime key first, answer under another key",
     '```python\ndef solve(x=10.0):\n    return {"regime": 0, "v": {"value": x, "unit": "m/s"}}\n```', 0, 1.0),
    ("only the judgement", completion('{"regime": 0}'), 0, 0.5),
    ("exec error", "```python\ndef solve(:\n```", 0, 0.0),
    ("no code", "", 1, 0.0),
]


# Counterfactual reward (pilot.grade_cf). Physics of the toy: y = x for x < 50 (the "inside" regime),
# y = 2x beyond it. The row is x = 10 (answer 10); its twin is x = 100 (answer 200).
TWIN = {"sub_answers": [{"sub": "1", "value": "200", "unit": "m"}], "trigger": [10.0, 100.0]}
CF_SUBS = [{"sub": "1", "value": "10", "unit": "m"}]


def cf_solver(signature: str, body: str) -> str:
    return f"```python\ndef solve({signature}):\n{body}\n```"


CF_CASES = [
    ("robust: branches on the regime", cf_solver("x=10.0", "    return {'1': {'value': x if x < 50 else 2 * x, 'unit': 'm'}}"), 1.0),
    ("shortcut: right here, wrong at the twin", cf_solver("x=10.0", "    return {'1': {'value': x, 'unit': 'm'}}"), 0.5),
    ("wrong on its own problem", cf_solver("x=10.0", "    return {'1': {'value': 3 * x, 'unit': 'm'}}"), 0.0),
    ("robust, trigger in other units (km default)",
     cf_solver("x_km=0.01", "    x = x_km * 1000\n    return {'1': {'value': x if x < 50 else 2 * x, 'unit': 'm'}}"), 1.0),
    ("robust, trigger as a 273.15 offset",
     cf_solver("t_k=283.15", "    x = t_k - 273.15\n    return {'1': {'value': x if x < 50 else 2 * x, 'unit': 'm'}}"), 1.0),
    ("unmapped: value hard-coded, no parameter", cf_solver("", "    x = 10.0\n    return {'1': {'value': x, 'unit': 'm'}}"), 0.5),
    ("unmapped: two parameters hold the value",
     cf_solver("x=10.0, y=10.0", "    return {'1': {'value': x if x < 50 else 2 * x, 'unit': 'm'}}"), 0.5),
    ("twin run crashes", cf_solver("x=10.0", "    assert x < 50\n    return {'1': {'value': x, 'unit': 'm'}}"), 0.5),
]


def main() -> int:
    failures = 0
    keys = None
    for name, text, label, want in CASES:
        out = verl_reward_regime.compute_score("atmoscoder/train", text, encode_truth(SUBS, label))
        ok = abs(out["score"] - want) < 1e-9
        keys = keys or set(out)
        if set(out) != keys:
            ok = False; print(f"  key set differs: {sorted(set(out) ^ keys)}")
        if label is None:   # must equal the answer-only entry point exactly
            ref = verl_reward.compute_score("atmoscoder/train", text, json.dumps(SUBS))
            ok &= all(out[k] == ref[k] for k in ref)
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: score {out['score']} (want {want})"
              f"  answer {out['answer']} judge/correct {out['judge/correct']} judge/missing {out['judge/missing']}")
    for name, text, want in CF_CASES:
        out = verl_reward_regime.compute_score("atmoscoder/train", text, encode_truth(CF_SUBS, twin=TWIN))
        ok = abs(out["score"] - want) < 1e-9 and set(out) == keys
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} cf, {name}: score {out['score']} (want {want})"
              f"  mapped {out['cf/mapped']} right-at-twin {out['cf/right']}")
    for entry, want in ((1, 1), (0.0, 0), (True, 1), ({"value": 0}, 0), (2, None), ("1", 1), (None, None)):
        if parse_judgement(entry) != want:
            failures += 1; print(f"  FAIL parse_judgement({entry!r}) = {parse_judgement(entry)!r}, want {want!r}")
    print("PASS" if not failures else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
