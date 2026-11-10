#!/usr/bin/env python
"""Text-only extraction of a Ministral 3 checkpoint (PREREG_DOSE_MINISTRAL, gate 1).

Ministral-3-8B-Instruct-2512 ships as Mistral3ForConditionalGeneration (Pixtral vision tower + projector + a
Ministral3 language model).  verl's DAPO trainer loads the whole wrapper into FSDP but its vLLM rollout builds the
text model only, so the first weight sync dies on `vision_tower.patch_conv.weight` (smoke of 2026-10-30, 114).
The study only ever sends text, so the vision tower is dead weight: `convert` writes the language model alone as a
Ministral3ForCausalLM checkpoint (transformers 5.9 and vLLM 0.24 both register that architecture), keeping every
tokenizer / chat-template / generation file byte-identical, and `check` verifies on text prompts that the two
checkpoints give the same logits and the same greedy continuations.

  python explore/ministral_text_only.py convert --src <Mistral3 dir> --dst <new dir>
  python explore/ministral_text_only.py check --src <Mistral3 dir> --dst <new dir> --final experiments/_inputs/kc/final.json \
      --out results/x/ministral_text_only_check.json [--n 16] [--device cuda:0]
  python explore/ministral_text_only.py fix-yarn --params <Mistral3 dir>/params.json <dir> [<dir> ...]

fix-yarn (PREREG_DOSE_MINISTRAL DM-9): without `consolidated*.safetensors` + `params.json` vLLM 0.24 reads the HF config, whose
YaRN branch ignores mscale / mscale_all_dim and multiplies cos / sin by yarn_get_mscale(16) = 1.277 unless
`rope_parameters.apply_yarn_scaling` is false; the official params.json says `yarn.apply_scale: false` and transformers already
computes attention_factor 1. fix-yarn adds the key to each directory's config.json (the original kept as config.json.pre-yarnfix;
a directory already carrying it is left alone) and checks that transformers' AutoConfig keeps the key through a load / save
round trip, so verl's checkpoints and merged endpoints inherit it.
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

PREFIX = "language_model."
COPY = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
        "generation_config.json", "tekken.json", "SYSTEM_PROMPT.txt"]
PER_LAYER = ("input_layernorm.weight", "post_attention_layernorm.weight", "mlp.down_proj.weight", "mlp.gate_proj.weight",
             "mlp.up_proj.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight", "self_attn.q_proj.weight",
             "self_attn.v_proj.weight")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_keys(num_layers: int) -> set[str]:
    keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(num_layers):
        keys |= {f"model.layers.{i}.{t}" for t in PER_LAYER}
    return keys


def convert(src: Path, dst: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = json.load(open(src / "config.json"))
    assert cfg["architectures"] == ["Mistral3ForConditionalGeneration"], cfg["architectures"]
    text = dict(cfg["text_config"])
    assert text["model_type"] == "ministral3", text["model_type"]
    assert not cfg["tie_word_embeddings"]
    text.update(architectures=["Ministral3ForCausalLM"], dtype=cfg.get("dtype", "bfloat16"),
                tie_word_embeddings=False, transformers_version=cfg.get("transformers_version"))
    text["rope_parameters"] = dict(text["rope_parameters"], apply_yarn_scaling=False)   # DM-9: see fix-yarn
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} is not empty")
    dst.mkdir(parents=True, exist_ok=True)

    index = json.load(open(src / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    shards = [s for s in sorted(set(weight_map.values()))
              if any(k.startswith(PREFIX) for k, v in weight_map.items() if v == s)]
    kept, dropped, total = {}, [], 0
    for i, shard in enumerate(shards, 1):
        out = {}
        with safe_open(src / shard, "pt") as f:
            for k in f.keys():
                if k.startswith(PREFIX):
                    t = f.get_tensor(k)
                    out[k[len(PREFIX):]] = t
                    total += t.numel() * t.element_size()
                else:
                    dropped.append(k)
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(out, dst / name, metadata={"format": "pt"})
        kept.update({k: name for k in out})
        print(f"{shard} -> {name}: kept {len(out)}")
    for k in weight_map:  # keys of shards without language weights are dropped too
        if not k.startswith(PREFIX) and k not in dropped:
            dropped.append(k)
    want = expected_keys(text["num_hidden_layers"])
    assert set(kept) == want, (sorted(set(kept) ^ want))[:10]
    json.dump({"metadata": {"total_size": total}, "weight_map": kept}, open(dst / "model.safetensors.index.json", "w"), indent=2)
    json.dump(text, open(dst / "config.json", "w"), indent=2)

    copied = {}
    for name in COPY:
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
            copied[name] = sha256(src / name)
    tok_cfg = json.load(open(dst / "tokenizer_config.json"))
    tok_cfg.pop("processor_class", None)  # no Pixtral processor for a text model
    # transformers 5.9's Llama tokenizer class mangles this byte-level (tekken) tokenizer.json — it drops whitespace on encode
    # ("def solve" -> "def", "s", "olve") and emits "Ġ"/"Ċ" literally on decode — so verl's rewards saw no runnable code (smoke of
    # 2026-10-30, attempt 4). The generic fast class goes straight through tokenizer.json and round-trips exactly.
    tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    json.dump(tok_cfg, open(dst / "tokenizer_config.json", "w"), indent=2)
    prefixes = sorted({".".join(k.split(".")[:2]) for k in dropped})
    json.dump({"source": str(src), "source_config_sha256": sha256(src / "config.json"), "kept_tensors": len(kept),
               "dropped_tensors": len(dropped), "dropped_prefixes": prefixes, "copied_files_sha256": copied,
               "tokenizer_config_change": "processor_class removed; tokenizer_class -> PreTrainedTokenizerFast"},
              open(dst / "MANIFEST.text_only.json", "w"), indent=2)
    print(f"kept {len(kept)} tensors ({total / 2**30:.2f} GiB), dropped {len(dropped)} ({', '.join(prefixes)}) -> {dst}")


def check(src: Path, dst: Path, final: Path, out: Path, n: int, device: str, max_new: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration

    rows = json.load(open(final))
    sample = random.Random("ministral-text-only:20261030").sample(rows, n)
    from tokenizers import Tokenizer

    tok_dst, raw = AutoTokenizer.from_pretrained(dst), Tokenizer.from_file(str(dst / "tokenizer.json"))
    prompts = [tok_dst.apply_chat_template([{"role": "user", "content": r["problem"]}], tokenize=False,
                                           add_generation_prompt=True) for r in sample]
    for p in prompts + ["def solve(T=283):\n    x = T * 2\n    return {\"a\": x}"]:   # the text-side tokenizer must be exact
        ids = tok_dst(p, add_special_tokens=False).input_ids
        assert ids == raw.encode(p, add_special_tokens=False).ids, p[:60]
        assert tok_dst.decode(ids, skip_special_tokens=False) == p, p[:60]   # prompts carry <s> and [INST] tokens: keep them
    full = Mistral3ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16, device_map=device).eval()
    text = AutoModelForCausalLM.from_pretrained(dst, dtype=torch.bfloat16, device_map=device).eval()
    assert type(text).__name__ == "Ministral3ForCausalLM", type(text).__name__
    res = []
    with torch.no_grad():
        for r, p in zip(sample, prompts):
            ids = tok_dst(p, return_tensors="pt", add_special_tokens=False).to(device)
            lf = full(**ids).logits.float()
            lt = text(**ids).logits.float()
            gf = full.generate(**ids, max_new_tokens=max_new, do_sample=False)[0, ids.input_ids.shape[1]:]
            gt = text.generate(**ids, max_new_tokens=max_new, do_sample=False)[0, ids.input_ids.shape[1]:]
            res.append({"id": r["id"], "prompt_tokens": int(ids.input_ids.shape[1]),
                        "max_abs_logit_diff": float((lf - lt).abs().max()),
                        "argmax_agreement": float((lf.argmax(-1) == lt.argmax(-1)).float().mean()),
                        "greedy_tokens": int(gf.shape[0]), "greedy_identical": bool(gf.shape == gt.shape and bool((gf == gt).all()))})
            print(res[-1])
    summary = {"n": n, "max_new_tokens": max_new, "max_abs_logit_diff": max(x["max_abs_logit_diff"] for x in res),
               "min_argmax_agreement": min(x["argmax_agreement"] for x in res),
               "greedy_identical": sum(x["greedy_identical"] for x in res), "src": str(src), "dst": str(dst),
               "dst_manifest": json.load(open(dst / "MANIFEST.text_only.json")), "rows": res}
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "dst_manifest")}))


def fix_yarn(params: Path, dirs: list[Path]) -> None:
    import tempfile
    from transformers import AutoConfig

    yarn = json.load(open(params))["yarn"]
    assert yarn.get("apply_scale") is False, yarn   # the official setting this key reproduces
    for d in dirs:
        cfg_path = d / "config.json"
        cfg = json.load(open(cfg_path))
        rp = cfg["rope_parameters"]
        assert rp["rope_type"] == "yarn" and rp["mscale"] == rp["mscale_all_dim"] == 1.0, rp
        if rp.get("apply_yarn_scaling") is False:
            state = "already set"
        else:
            assert "apply_yarn_scaling" not in rp, rp
            backup = d / "config.json.pre-yarnfix"
            assert not backup.exists(), backup
            shutil.copyfile(cfg_path, backup)
            rp["apply_yarn_scaling"] = False
            tmp = cfg_path.with_suffix(".json.tmp")
            json.dump(cfg, open(tmp, "w"), indent=2)
            tmp.replace(cfg_path)
            state = "fixed"
        with tempfile.TemporaryDirectory() as t:
            AutoConfig.from_pretrained(d).save_pretrained(t)
            kept = json.load(open(Path(t) / "config.json"))["rope_parameters"].get("apply_yarn_scaling")
        assert kept is False, f"{d}: transformers drops apply_yarn_scaling on save ({kept})"
        print(f"{d}: {state}; round trip keeps apply_yarn_scaling false; config sha256 {sha256(cfg_path)[:12]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert"); c.add_argument("--src", type=Path, required=True); c.add_argument("--dst", type=Path, required=True)
    k = sub.add_parser("check"); k.add_argument("--src", type=Path, required=True); k.add_argument("--dst", type=Path, required=True)
    k.add_argument("--final", type=Path, required=True); k.add_argument("--out", type=Path, required=True)
    k.add_argument("--n", type=int, default=16); k.add_argument("--device", default="cuda:0"); k.add_argument("--max-new", type=int, default=64)
    y = sub.add_parser("fix-yarn"); y.add_argument("--params", type=Path, required=True); y.add_argument("dirs", type=Path, nargs="+")
    a = ap.parse_args()
    if a.cmd == "fix-yarn":
        fix_yarn(a.params, a.dirs)
    elif a.cmd == "convert":
        convert(a.src, a.dst)
    else:
        check(a.src, a.dst, a.final, a.out, a.n, a.device, a.max_new)


if __name__ == "__main__":
    main()
