#!/usr/bin/env bash
# PREREG_X amendment X-10: end-to-end smoke of one family on 8 GPUs before its chain — everything the chain does, in miniature:
#   1. warm-start SFT (whole-conversation dataset, seed 42) for 2 steps, saved and merged;
#   2. the merged SFT model renders the same prompt as the base and answers through vLLM (thinking off);
#   3. 2 DAPO steps of arm F at production settings (128 x 8, 3,072 tokens, train_probe validation) starting from that merged
#      model (the WA / WD path), saving every step, merged, and served the same way.
#   NAME=qwen35-9b MODEL=/path/to/shared/Qwen3.5-9B setsid nohup bash explore/x10_smoke.sh > $ATMOS_ROOT/logs/x10/smoke-qwen35-9b.log 2>&1 < /dev/null &
# Prints SMOKE10 PASS or SMOKE10 FAIL <stage>. Leaves nothing but logs: its checkpoints and merged models are removed at the end.
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
: "${NAME:?}" "${MODEL:?}"
R=$ATMOS_ROOT; P=$R/verl/.venv/bin/python; L=$R/logs/x10; mkdir -p $L
say(){ echo "[$(date +%F' '%T)] $*"; }
fail(){ say "SMOKE10 FAIL $*"; exit 1; }
cd $ATMOS_REPO
S=smoke10-sft-$NAME; T=smoke10-rl-$NAME
rm -rf $R/checkpoints/$S $R/checkpoints/$T $R/merged/$S $R/merged/$T

merge(){  # exp -> merged dir on stdout
  local src=$(ls -d $R/checkpoints/$1/global_step_* | sort -V | tail -1) M=$R/merged/$1/merged
  [ -d "$src/actor" ] && src=$src/actor
  (cd $R/verl && CUDA_VISIBLE_DEVICES= $P -m verl.model_merger merge --backend fsdp --local_dir $src --target_dir $M > $L/merge-$1.log 2>&1) || return 1
  echo $M
}

serve_check(){  # model_dir label: same rendered prompt as the base, loads in vLLM, answers with a code block
  local M=$1
  $P - "$MODEL" "$M" <<'PY' || return 1
import sys, json
from transformers import AutoTokenizer
base, new = (AutoTokenizer.from_pretrained(p) for p in sys.argv[1:3])
m = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
a = base.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
b = new.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("template same:", a == b, repr(b[-60:]))
cfg = json.load(open(sys.argv[2] + "/config.json")); print("architectures:", cfg.get("architectures"))
sys.exit(0 if a == b else 1)
PY
  CUDA_VISIBLE_DEVICES=0 setsid $P -m vllm.entrypoints.openai.api_server --model $M --served-model-name m --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.85 --generation-config vllm --host 127.0.0.1 --port 8299 > $L/serve-$(basename $(dirname $M)).log 2>&1 < /dev/null &
  local pg=$! ok=1
  for _ in $(seq 90); do curl -sf http://127.0.0.1:8299/v1/models >/dev/null && break; kill -0 $pg 2>/dev/null || break; sleep 10; done
  $P - <<'PY' && ok=0
import json, urllib.request
q = json.load(open("/path/to/workdir/experiments/_inputs/kc/final.json"))[0]["problem"]
from atmoscoder_rl.prompts import chat
body = {"model": "m", "messages": chat(q), "temperature": 0, "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8299/v1/chat/completions", json.dumps(body).encode(),
                                                  {"Content-Type": "application/json"}), timeout=600)
t = json.load(r)["choices"][0]["message"]["content"] or ""
print("reply head:", repr(t[:120])); print("has code block:", "```python" in t, "think tag:", "<think>" in t)
raise SystemExit(0 if "```python" in t and "<think>" not in t else 1)
PY
  kill -TERM -- -$pg 2>/dev/null; sleep 15; kill -KILL -- -$pg 2>/dev/null
  return $ok
}

say "=== 1. SFT 2 steps ==="
MODEL_PATH=$MODEL SOURCE=x-$NAME-F1 EXP=$S EPOCHS=1 LR=5e-6 SEED=42 SAVE_FREQ=2 bash train/sft_qwen3_8b.sh \
  trainer.total_training_steps=2 > $L/$S.nohup 2>&1 || fail "sft (see $L/$S.nohup)"
grep -o "train/loss[^ ,]*[ :][0-9.]*" $R/logs/$S.log | head -2
SM=$(merge $S) || fail "sft merge"
say "merged SFT: $SM"; ls $SM
say "=== 2. serve merged SFT ==="
serve_check $SM || fail "serve merged sft"

say "=== 3. DAPO 2 steps, arm F, from the merged SFT model ==="
SEED=42 MODEL_PATH=$SM DATA=$R/data-x/F1-F REWARD=regime EXP=$T TOTAL_STEPS=2 TEST_FREQ=2 SAVE_FREQ=1 KEEP_CKPT=1 \
  VAL_FILES="['$R/data-x/F1-F/train_probe.parquet']" VAL_TEMP=0.3 bash train/dapo_qwen3_8b.sh > $L/$T.nohup 2>&1 || fail "dapo (see $L/$T.nohup)"
grep -o "step:2 .*" $R/logs/$T.log | tr ' ' '\n' | grep -E "critic/score/mean|response_length/clip_ratio|response_length/mean|val-aux.*answer|timing_s/step" | head -12
TM=$(merge $T) || fail "rl merge"
say "merged RL: $TM"
serve_check $TM || fail "serve merged rl"

rm -rf $R/checkpoints/$S $R/checkpoints/$T $R/merged/$S $R/merged/$T
say "SMOKE10 PASS"
