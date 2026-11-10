#!/usr/bin/env bash
# PREREG_X amendment X-10: the warm start's RAFT background from the family's own base model (prereg §3: "the base's own verified
# solutions"). Same recipe as data/sft_raft.parquet (Qwen3-8B, 2026-09-15): the 1,140 frozen-train problems x 8 samples at the RL
# rollout settings (local-raft: T 1.0, 3,072 tokens, thinking off), at most 2 verified completions per problem.
#   NAME=qwen35-9b MODEL=/path/to/shared/Qwen3.5-9B setsid nohup bash explore/x10_raft.sh > $ATMOS_ROOT/logs/x10/raft-qwen35-9b.log 2>&1 < /dev/null &
# Output: $ATMOS_ROOT/data/raft_$NAME/{raft_verified.json,sft_raft.parquet}. Prints RAFTDONE at the end.
# GPUS (default all 8) serves the model data-parallel over those GPUs; SECTION (default local-raft) is the models.toml section the
# sampler calls — local-raft-plain for Mistral-tokenizer servers, which reject chat_template_kwargs (third family, Ministral 3).
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
: "${NAME:?}" "${MODEL:?}"
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; SECTION=${SECTION:-local-raft}; NG=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
R=$ATMOS_ROOT; OUT=$R/data/raft_$NAME; mkdir -p $OUT $R/logs/x10
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1; export PYTHONPATH=$PWD   # this checkout's sampler (ATMOS_NO_CHAT_TEMPLATE_KWARGS), not the shared one
CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 setsid "$R/verl/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name local --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.88 --data-parallel-size $NG \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --generation-config vllm --host 127.0.0.1 --port 8077 > $R/logs/x10/server-raft-$NAME.log 2>&1 < /dev/null &
SERVER=$!
until curl -sf http://127.0.0.1:8077/v1/models >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; exit 1; }; sleep 10; done
echo "server up"
$R/eval-venv/bin/python -m atmoscoder_rl.sample_raft --out $OUT/raft_verified.json --samples 8 --cap-per-problem 2 --model $SECTION
rc=$?
kill -- -$SERVER 2>/dev/null; sleep 20
[ $rc = 0 ] || { echo "sample_raft failed ($rc)"; exit 1; }
$R/verl/.venv/bin/python -m atmoscoder_rl.export_sft --out-dir $OUT --source raft --raft-input $OUT/raft_verified.json || exit 1
echo RAFTDONE
