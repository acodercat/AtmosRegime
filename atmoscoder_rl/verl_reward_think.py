"""verl entry point for a policy trained with thinking on (docs/THINK_RL.md).

With `enable_thinking=true` the response is `<think>...</think>` followed by the answer. The
evaluation runner grades only the answer (vLLM's reasoning parser strips the trace), so the reward
must do the same: `extract_code` would otherwise take the first fenced block, which is often a
draft inside the thinking. The text after the last `</think>` is graded with `reward.grade`, exactly
as `verl_reward.py` grades a thinking-off response. A response whose thinking never closes (the
length cap hit mid-thought) has no answer and scores 0 as `no_code`, which is also how the runner
would see it. `think/closed` and `think/tokens_share` are logged so truncation is visible.
"""

from __future__ import annotations

from typing import Any

from atmoscoder_rl.pilot import decode_truth
from atmoscoder_rl.reward import RewardResult, grade

CLOSE = "</think>"


def split_thinking(text: str) -> tuple[str, str, bool]:
    """(thinking, answer, closed). No `<think>` at all: the whole text is the answer."""
    if CLOSE in text:
        head, _, tail = text.rpartition(CLOSE)
        return head, tail, True
    if "<think>" in text:
        return text, "", False
    return "", text, True


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    thinking, answer, closed = split_thinking(solution_str or "")
    if not closed:
        result = RewardResult(0.0, False, "no_code", error="thinking not closed")
    else:
        # A validation file may carry a judgement label or a twin in its truth (pilot.encode_truth); only
        # the answers are graded here. (The c0think run's workers loaded the earlier version, which
        # raised on such truths: its probe_tpl_heldout_judge validation metric is 0 for that reason.)
        sub_answers, _, _ = decode_truth(ground_truth)
        result = grade(answer, sub_answers)
    total = len(solution_str or "") or 1
    return {"score": result.reward, **result.trainer_metrics(),
            "think/closed": float(closed), "think/chars_share": len(thinking) / total}
