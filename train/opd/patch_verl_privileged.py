"""Privileged-teacher patch for verl's on-policy distillation (PREREG_OPD draft): the teacher scores the student's response under a
different prompt — the `given` twin of an outside row, which states the replacement relation — while the student sees only the
outside prompt. Rows without extra_info["teacher_prompt"] are scored under the student's own prompt (unchanged verl behaviour).

    python train/opd/patch_verl_privileged.py <verl checkout>      # a separate worktree, never the shared $R/verl a running job imports

Edits verl/experimental/agent_loop/agent_loop.py (the teacher prompt) and verl/trainer/distillation/losses.py (the top-k loss
accepts strategy fsdp2 on the fsdp path, as it already does for veomni) by exact replacement; each edit is skipped if present.
The teacher's logprobs are computed on teacher_prompt_ids + response_ids; the rows the loss reads (shift-aligned: from the last
prompt row on, see _atmos_realign) are kept and the rows before them are replaced by len(prompt_ids) - 1 placeholder rows (pad
id, logprob 0), so the returned tensors have the student's shape and the downstream padding / response mask are untouched.
train/opd/test_realign.py checks the alignment against verl's slicing convention and runs anywhere torch is installed. On the first call per worker the
helper renders the student's own raw_prompt with the same Continuous-Token builder and asserts it equals the student's prompt_ids, so a
template mismatch fails loudly; with ATMOS_OPD_PRIVILEGED=1 a run without any teacher_prompt also fails instead of silently degrading.
"""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = """            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs
"""

REPLACEMENT = """            # ATMOSCODER privileged teacher (train/opd/patch_verl_privileged.py)
            teacher_prompt_ids = await self._atmos_teacher_prompt_ids(prompt_ids, sample_kwargs)
            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=teacher_prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
            )
            if teacher_prompt_ids is not prompt_ids:
                teacher_ids, teacher_logprobs = _atmos_realign(teacher_ids, teacher_logprobs, len(teacher_prompt_ids),
                                                               len(prompt_ids), len(response_ids), self.tokenizer.pad_token_id)
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs

    async def _atmos_teacher_prompt_ids(self, prompt_ids, sample_kwargs):
        \"\"\"The teacher's prompt ids: extra_info["teacher_prompt"] rendered with the student's own prompt builder, else the student's.\"\"\"
        import json
        import os
        extra = (sample_kwargs or {}).get("extra_info")
        extra = extra.item() if hasattr(extra, "item") else extra
        required = os.environ.get("ATMOS_OPD_PRIVILEGED") == "1"
        diag = os.environ.get("ATMOS_OPD_DIAG")   # file-based diagnostics (worker log lines do not reach the driver log reliably)
        if not getattr(self, "_atmos_seen_first", False):   # one diagnostic record per worker: what the teacher hook actually receives
            self._atmos_seen_first = True
            rec = {"event": "first_call", "pid": os.getpid(), "module": __file__, "privileged_env": os.environ.get("ATMOS_OPD_PRIVILEGED"),
                   "kwargs_keys": sorted((sample_kwargs or {}).keys()), "extra_info_type": type(extra).__name__,
                   "has_teacher_prompt": bool(isinstance(extra, dict) and extra.get("teacher_prompt") is not None)}
            logger.warning("privileged teacher hook: %s", rec)
            if diag:
                with open(diag, "a") as fh:
                    fh.write(json.dumps(rec) + "\\n")
        if not isinstance(extra, dict):
            if required:
                raise RuntimeError("privileged teacher: extra_info is not in the sample kwargs")
            return prompt_ids
        messages = extra.get("teacher_prompt")
        counts = self.__dict__.setdefault("_atmos_counts", {"privileged": 0, "plain": 0})
        if (counts["plain"] + counts["privileged"]) % 256 == 255:
            logger.warning("privileged teacher hook counts: %s", counts)
            if diag:
                with open(diag, "a") as fh:
                    fh.write(json.dumps({"event": "counts", "pid": os.getpid(), **counts}) + "\\n")
        if messages is None:
            counts["plain"] += 1
            if required and counts["plain"] + counts["privileged"] == 2048 and counts["privileged"] == 0:
                raise RuntimeError("privileged teacher: no teacher_prompt in the first 2048 samples")
            return prompt_ids
        counts["privileged"] += 1
        if not hasattr(self, "_atmos_builder"):
            self._atmos_builder = create_continuous_token_builder(
                self.tokenizer, hf_model_type=self.hf_model_type,
                chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}) or {},
                mm_processor_kwargs=self.config.data.get("mm_processor_kwargs", {}) or {}, processor=self.processor)
            own = self._atmos_builder.build_initial_tokens(list(sample_kwargs["raw_prompt"]))
            if list(own) != list(prompt_ids):
                raise RuntimeError("privileged teacher: the student's prompt re-rendered by this builder differs from the rollout's prompt_ids")
        loop = asyncio.get_running_loop()
        ids = await loop.run_in_executor(None, lambda: self._atmos_builder.build_initial_tokens(list(messages)))
        if counts["privileged"] == 1:
            logger.warning("privileged teacher active: first privileged sample, teacher prompt %d ids vs student %d", len(ids), len(prompt_ids))
            if diag:
                with open(diag, "a") as fh:
                    fh.write(json.dumps({"event": "first_privileged", "pid": os.getpid(), "id": extra.get("id"), "teacher_prompt_ids": len(ids),
                                         "student_prompt_ids": len(prompt_ids)}) + "\\n")
        return list(ids)
"""

HELPER_ANCHOR = "class AgentLoopWorker:"
HELPER = '''def _atmos_realign(teacher_ids, teacher_logprobs, teacher_prompt_len, prompt_len, response_len, pad_id):
    """Realign the teacher's rows (computed over teacher_prompt + response) to the student's sequence (prompt + response).

    verl's teacher rows are shift-aligned: vllm_rollout/utils.py extract_prompt_logprobs drops the first prompt position and
    appends a dummy last row (ids 0, logprob 0), so row i holds the teacher's top-k for the token at i + 1; the loss reads rows
    prompt_len - 1 .. L - 2 (workers/utils/padding.py, the "left-shift by one token" slice). Row teacher_prompt_len - 1 is
    therefore the teacher's distribution for the FIRST response token. Keep rows teacher_prompt_len - 1 .. end (response_len + 1
    rows, the dummy last row included) and put prompt_len - 1 placeholder rows (pad id, logprob 0) in front: the result has the
    student's length and the loss reads exactly the teacher's response-token rows, never a placeholder.
    v2 (OPD-5): v1 kept rows teacher_prompt_len .. end, so the loss read one placeholder row — 32 pad ids at probability 1 — at
    the first response token of every privileged sample (teacher_mass_max 32, loss_max 1476.5, collapse at step 5).
    """
    assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == teacher_prompt_len + response_len
    keep_ids, keep_lp = teacher_ids[teacher_prompt_len - 1:], teacher_logprobs[teacher_prompt_len - 1:]
    mass = keep_lp[:-1].exp().sum(-1)   # every row the loss can read; the dummy last row (logprob 0) is never read
    if mass.numel() and float(mass.max()) > 1.001:
        raise RuntimeError(f"privileged teacher: top-k mass {float(mass.max()):.3f} > 1 among the rows kept for the loss")
    head_ids = torch.full((prompt_len - 1, *keep_ids.shape[1:]), pad_id, dtype=keep_ids.dtype)
    head_lp = torch.zeros((prompt_len - 1, *keep_lp.shape[1:]), dtype=keep_lp.dtype)
    out_ids, out_lp = torch.cat([head_ids, keep_ids], 0), torch.cat([head_lp, keep_lp], 0)
    assert out_ids.shape[0] == out_lp.shape[0] == prompt_len + response_len
    return out_ids, out_lp


'''
PATCH_VERSION_MARK = "v2 (OPD-5)"


LOSS_ANCHOR = """        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses"""
LOSS_REPLACEMENT = """        case "fsdp" | "fsdp2" | "veomni":   # ATMOSCODER: our actor is fsdp2; the loss reads student_logits the same way (OPD-2)
            import verl.trainer.distillation.fsdp.losses as fsdp_losses"""


def main() -> int:
    root = Path(sys.argv[1])
    rc = 0
    path = root / "verl" / "experimental" / "agent_loop" / "agent_loop.py"
    src = path.read_text()
    if "ATMOSCODER privileged teacher" in src and PATCH_VERSION_MARK in src:
        print("agent_loop already patched (v2)")
    elif "ATMOSCODER privileged teacher" in src:
        print("agent_loop carries the v1 patch (wrong row kept, OPD-5) — restore it with `git checkout -- verl/experimental/agent_loop/agent_loop.py` and rerun"); rc = 1
    elif src.count(ANCHOR) != 1 or src.count(HELPER_ANCHOR) != 1 or "\nimport torch\n" not in src or "\nimport asyncio\n" not in src \
            or "create_continuous_token_builder" not in src or "logger = " not in src:
        print("agent_loop anchor not found exactly once — verl changed, review the patch"); rc = 1
    else:
        path.write_text(src.replace(ANCHOR, REPLACEMENT).replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR)); print(f"patched {path}")
    path = root / "verl" / "trainer" / "distillation" / "losses.py"
    src = path.read_text()
    if "ATMOSCODER: our actor is fsdp2" in src:
        print("losses already patched")
    elif src.count(LOSS_ANCHOR) != 1:
        print("losses anchor not found exactly once — verl changed, review the patch"); rc = 1
    else:
        path.write_text(src.replace(LOSS_ANCHOR, LOSS_REPLACEMENT)); print(f"patched {path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
