"""Unit test for _atmos_realign (train/opd/patch_verl_privileged.py) against verl's slicing convention (OPD-5).

    $R/verl/.venv/bin/python train/opd/test_realign.py       # anywhere torch is installed; prints OK

The fake teacher output follows verl (pinned 5b79827b): vllm_rollout/utils.py extract_prompt_logprobs drops the first prompt
position and appends a dummy last row, so for a sequence of L tokens row i (0 <= i < L - 1) is the top-k distribution for the
token at position i + 1 and row L - 1 is the dummy (ids 0, logprob 0). The loss reads rows prompt_len - 1 .. L - 2 of the
student's sequence (workers/utils/padding.py: values[seq_offset - resp_len - 1 : seq_offset - 1]), i.e. the distribution for
each response token. After realignment those rows must equal the teacher's rows for the same response tokens, computed under
the teacher's (longer) prompt, and no placeholder row may be among them. The v1 realignment is reproduced and shown to fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_verl_privileged as patch  # noqa: E402

ns: dict = {"torch": torch}
exec(patch.HELPER, ns)
realign = ns["_atmos_realign"]
K, PAD = 4, 151643


def fake_teacher_rows(seq_len: int, tag: float):
    """Row i: top-k ids tag*1000 + (i+1)*10 + rank, logprobs summing to mass 0.9 (rank-weighted); dummy last row per verl."""
    ids = torch.zeros((seq_len, K), dtype=torch.int32)
    lp = torch.zeros((seq_len, K), dtype=torch.float32)
    w = torch.tensor([0.5, 0.25, 0.1, 0.05])
    for i in range(seq_len - 1):
        ids[i] = torch.tensor([int(tag * 1000 + (i + 1) * 10 + r) for r in range(K)], dtype=torch.int32)
        lp[i] = w.log()
    return ids, lp


def loss_rows(t: torch.Tensor, prompt_len: int, response_len: int):
    """The rows verl's loss reads for the response tokens (padding.py left-shift slice)."""
    seq_offset = prompt_len + response_len
    return t[seq_offset - response_len - 1 : seq_offset - 1]


def v1_realign(teacher_ids, teacher_logprobs, teacher_prompt_len, prompt_len, response_len, pad_id):
    resp_ids, resp_lp = teacher_ids[teacher_prompt_len:], teacher_logprobs[teacher_prompt_len:]
    head_ids = torch.full((prompt_len, *resp_ids.shape[1:]), pad_id, dtype=resp_ids.dtype)
    head_lp = torch.zeros((prompt_len, *resp_lp.shape[1:]), dtype=resp_lp.dtype)
    return torch.cat([head_ids, resp_ids], 0), torch.cat([head_lp, resp_lp], 0)


def main() -> int:
    prompt_len, teacher_prompt_len, response_len = 7, 11, 5
    t_ids, t_lp = fake_teacher_rows(teacher_prompt_len + response_len, tag=1)
    # what the loss should see: the teacher's rows for the response tokens, i.e. rows teacher_prompt_len-1 .. L_t-2
    want_ids = loss_rows(t_ids, teacher_prompt_len, response_len)
    want_lp = loss_rows(t_lp, teacher_prompt_len, response_len)
    assert want_ids.shape[0] == response_len and int(want_ids[0, 0]) == 1000 + teacher_prompt_len * 10   # token at index tp
    out_ids, out_lp = realign(t_ids, t_lp, teacher_prompt_len, prompt_len, response_len, PAD)
    assert out_ids.shape == (prompt_len + response_len, K) and out_lp.shape == out_ids.shape, out_ids.shape
    got_ids, got_lp = loss_rows(out_ids, prompt_len, response_len), loss_rows(out_lp, prompt_len, response_len)
    assert torch.equal(got_ids, want_ids) and torch.equal(got_lp, want_lp), (got_ids[:, 0], want_ids[:, 0])
    assert not (got_ids == PAD).any(), "a placeholder row is read by the loss"
    assert float(got_lp.exp().sum(-1).max()) <= 1.0 + 1e-5
    assert torch.equal(out_ids[-1], t_ids[-1]) and torch.equal(out_lp[-1], t_lp[-1])   # dummy last row stays last
    assert (out_ids[: prompt_len - 1] == PAD).all() and (out_lp[: prompt_len - 1] == 0).all()
    # same prompt (no privilege): the realignment is not applied by the patch; applied anyway it must be the identity
    s_ids, s_lp = fake_teacher_rows(prompt_len + response_len, tag=2)
    o_ids, o_lp = realign(s_ids, s_lp, prompt_len, prompt_len, response_len, PAD)
    assert torch.equal(loss_rows(o_ids, prompt_len, response_len), loss_rows(s_ids, prompt_len, response_len))
    # v1 reproduces OPD-5: the first response token reads a placeholder (pad ids, mass K)
    b_ids, b_lp = v1_realign(t_ids, t_lp, teacher_prompt_len, prompt_len, response_len, PAD)
    b_got_ids, b_got_lp = loss_rows(b_ids, prompt_len, response_len), loss_rows(b_lp, prompt_len, response_len)
    assert (b_got_ids[0] == PAD).all() and float(b_got_lp[0].exp().sum()) == K and torch.equal(b_got_ids[1:], want_ids[1:])
    # the mass guard: a row with logprobs 0 inside the kept range is refused
    bad_lp = t_lp.clone(); bad_lp[teacher_prompt_len] = 0.0
    try:
        realign(t_ids, bad_lp, teacher_prompt_len, prompt_len, response_len, PAD)
    except RuntimeError as e:
        assert "top-k mass" in str(e)
    else:
        raise AssertionError("mass guard did not fire")
    print("OK: realign v2 feeds the loss the teacher's response-token rows; v1 fed a placeholder at the first response token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
