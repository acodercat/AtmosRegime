"""Control arm: a reward carrying no information about correctness.

Spurious Rewards (arXiv 2506.10947) found that Qwen-family models gain more than 20
points on MATH-500 from RLVR even when the reward is random, while Llama and OLMo gain
nothing — the effect is a property of that model family, not of the training signal. Our
policy is Qwen3-8B, so every gain reported for the real reward has to be measured against
this arm before it can be attributed to learning the physics.

`compute_score` returns an independent fair coin flip per rollout. It still executes the
solver and grades it, but only to report `true_score` in the metrics: that lets us plot
the model's real accuracy over a run whose gradient never saw it. verl forms advantages
from `score`, so the graded value cannot leak into training.

One consequence to expect, not a misconfiguration: with a random score, a group's rollouts
almost never share a value, so DAPO's dynamic sampling filters nothing and each step
consumes one generation batch instead of the up-to-ten the real reward needed. The config
is identical; the throughput difference is caused by the reward itself.
"""

from __future__ import annotations

import json
import random
from typing import Any

from atmoscoder_rl.reward import grade

_rng = random.Random()


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    result = grade(solution_str, json.loads(ground_truth))
    return {"score": float(_rng.randint(0, 1)),   # the gradient sees only this
            "true_score": result.reward,          # logged, never used to learn
            **result.trainer_metrics()}
