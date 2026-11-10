"""SFT dataset that tokenises each conversation once, as the policy later sees it (PREREG_X amendment X-10).

verl's MultiTurnSFTDataset tokenises every message on its own and concatenates the pieces. That breaks in two ways here:
- Qwen3-8B: a lone assistant turn renders without the empty `<think>\\n\\n</think>\\n\\n` block that the RL rollout and the
  evaluation prompt (thinking off) end with, so with `ignore_input_ids_mismatch=True` the warm start learned a format the
  policy never sees;
- Qwen3.5: its template refuses to render a system or assistant message alone ("No user query found in messages").

Here the whole (system, user, assistant) row is rendered once with thinking off; the loss covers only the tokens after the
generation prompt, i.e. exactly what the model generates at rollout time. The parent's sanity check then compares against the
same whole render, so `ignore_input_ids_mismatch` can stay False and any mismatch fails loudly.

    data.custom_cls.path=$ATMOS_REPO/atmoscoder_rl/sft_dataset.py data.custom_cls.name=WholeConversationSFTDataset
"""
from __future__ import annotations

import torch
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.tokenizer.chat_template import apply_chat_template


class WholeConversationSFTDataset(MultiTurnSFTDataset):
    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        if index != 0:  # everything is produced at index 0; the parent concatenates empty pieces for the rest
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty.clone(), empty.clone(), {}
        if [m["role"] for m in full_message] not in (["system", "user", "assistant"], ["user", "assistant"]):
            raise ValueError(f"expected (system,) user, assistant; got {[m['role'] for m in full_message]}")
        processor = self.processor if self.processor is not None else self.tokenizer
        kw = {**self.apply_chat_template_kwargs, "enable_thinking": False}
        whole = dict(apply_chat_template(processor, messages=full_message, tools=tools, add_generation_prompt=False,
                                         tokenize=True, return_dict=True, return_tensors="pt", **kw))
        prompt = dict(apply_chat_template(processor, messages=full_message[:-1], tools=tools, add_generation_prompt=True,
                                          tokenize=True, return_dict=True, return_tensors="pt", **kw))
        input_ids, attention_mask = whole.pop("input_ids")[0], whole.pop("attention_mask")[0]
        n = prompt["input_ids"].shape[1]
        if not torch.equal(input_ids[:n], prompt["input_ids"][0]):
            raise ValueError("the generation prompt is not a token prefix of the whole conversation")
        if n >= input_ids.shape[0]:
            raise ValueError("no assistant tokens after the generation prompt")
        loss_mask = torch.ones_like(attention_mask)
        loss_mask[:n] = 0
        return input_ids, loss_mask, attention_mask, whole

    def sanity_check(self, input_ids, messages, tools, enable_thinking):
        return super().sanity_check(input_ids, messages, tools, False)
