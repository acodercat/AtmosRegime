"""verl entry point: `custom_reward_function.path` pointing at this file.

verl loads the file by path and calls `compute_score(data_source, solution_str,
ground_truth, extra_info)` per rollout, so this module uses absolute imports and expects
`atmoscoder_rl` on `PYTHONPATH` (the training `env.sh` sets it). `ground_truth` arrives
as a string: the dataset export serialises the problem's `sub_answers` list as JSON there.

The return is a dict rather than a bare score so the outcome breakdown reaches verl's
metrics: `exec_error` share not falling means the policy is not learning the solver
contract; `wrong_answer` share is the physics signal. Collapsed into one scalar, neither
is visible.
"""

from __future__ import annotations

import json
from typing import Any

from atmoscoder_rl.reward import grade


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    result = grade(solution_str, json.loads(ground_truth))
    return {"score": result.reward, **result.trainer_metrics()}
