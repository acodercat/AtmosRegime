"""Diagnostic (PREREG_DOSE_MINISTRAL DM-4): do the FSDP-side Hugging Face forward and vLLM give the same token log-probs?

The Ministral RL runs showed rollout/actor probability correlation 0.57 at step 1 (Qwen: 0.999). This generates greedy
responses with vLLM for a few training prompts (logprob of every generated token), then teacher-forces the same token
ids through the Hugging Face model three ways: one sequence at a time (padding-free, plain), all sequences packed into
one row with restarting position_ids (what verl's use_remove_padding feeds the model), and left-padded batch.

    python explore/diag_logprob_match.py --model <merged warm start> --data data-x/F1-A/train.parquet --n 4 --out x.json

DM-8 (the residual mismatch with packing off): `--temperature 1.0` samples as the RL rollout does, `--ref-attn eager --ref-dtype
float32` adds a float32 eager Hugging Face forward as the reference both bf16 sides are compared with, and every comparison is
also split by response position (`by_position`).
"""
from __future__ import annotations

import argparse
import json
import math

import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-attn", default=None, help="also score with this attention implementation as the reference")
    ap.add_argument("--ref-dtype", default="float32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = pq.read_table(a.data).slice(0, a.n).to_pylist()
    prompts = [list(r["prompt"]) for r in rows]

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.45, enforce_eager=True)
    tok = llm.get_tokenizer()
    ids = [tok.apply_chat_template(p, add_generation_prompt=True, tokenize=True) for p in prompts]
    ids = [list(i["input_ids"]) if hasattr(i, "keys") else list(i) for i in ids]   # transformers 5 returns a dict
    outs = llm.generate([{"prompt_token_ids": i} for i in ids],
                        SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens, logprobs=0, seed=a.seed))
    seqs, vlp = [], []
    for i, o in zip(ids, outs):
        c = o.outputs[0]
        seqs.append((list(i), list(c.token_ids)))
        vlp.append([lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)])
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=a.attn).cuda().eval()

    def score(logits, full, start, n):
        lp = torch.log_softmax(logits.float(), -1)
        return [lp[start + k - 1, full[start + k]].item() for k in range(n)]

    single, packed = [], []
    with torch.no_grad():
        for p, r in seqs:
            full = p + r
            lg = model(torch.tensor([full], device="cuda")).logits[0]
            single.append(score(lg, full, len(p), len(r)))
        cat, pos, offs = [], [], []
        for p, r in seqs:
            offs.append(len(cat)); cat += p + r; pos += list(range(len(p) + len(r)))
        lg = model(torch.tensor([cat], device="cuda"), position_ids=torch.tensor([pos], device="cuda")).logits[0]
        for (p, r), o in zip(seqs, offs):
            full = cat[o:o + len(p) + len(r)]
            packed.append(score(lg[o:o + len(full)], full, len(p), len(r)))

    ref = None
    if a.ref_attn:
        del model
        gc.collect(); torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.ref_dtype),
                                                     attn_implementation=a.ref_attn).cuda().eval()
        ref = []
        with torch.no_grad():
            for p, r in seqs:
                full = p + r
                ref.append(score(model(torch.tensor([full], device="cuda")).logits[0], full, len(p), len(r)))

    BUCKETS = ((0, 256), (256, 1024), (1024, 1 << 30))

    def cmp(x, y):
        fx = [v for s in x for v in s]; fy = [v for s in y for v in s]
        d = [abs(u - v) for u, v in zip(fx, fy)]
        out = {"tokens": len(d), "nll_a": -sum(fx) / len(fx), "nll_b": -sum(fy) / len(fy),
               "ppl_a": math.exp(-sum(fx) / len(fx)), "ppl_b": math.exp(-sum(fy) / len(fy)),
               "mean_abs_logprob_diff": sum(d) / len(d), "max_abs_logprob_diff": max(d),
               "mean_signed_diff_b_minus_a": sum(v - u for u, v in zip(fx, fy)) / len(d), "by_position": {}}
        for lo, hi in BUCKETS:
            dd = [abs(u - v) for s, t in zip(x, y) for k, (u, v) in enumerate(zip(s, t)) if lo <= k < hi]
            if dd:
                out["by_position"][f"{lo}-{hi if hi < 1 << 30 else 'end'}"] = {"tokens": len(dd), "mean_abs": sum(dd) / len(dd)}
        return out

    res = {"model": a.model, "attn": a.attn, "n": a.n, "temperature": a.temperature, "response_tokens": [len(r) for _, r in seqs],
           "vllm_vs_hf_single": cmp(vlp, single), "vllm_vs_hf_packed": cmp(vlp, packed), "hf_single_vs_packed": cmp(single, packed),
           **({"ref": f"{a.ref_attn}/{a.ref_dtype}", "ref_vs_vllm": cmp(ref, vlp), "ref_vs_hf_single": cmp(ref, single)} if ref else {}),
           "first_response": tok.decode(seqs[0][1])[:600]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "first_response"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
