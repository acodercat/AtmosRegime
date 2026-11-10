#!/usr/bin/env bash
# PREREG_X §4 pass@k on final.json outside rows for one model (X-18: the base re-sample, the W_f samples, the pass@32 T 0.7
# sensitivity), served as in explore/x_prep.sh. Writes OUT and OUT's .texts.jsonl (the sampled texts); resume-safe.
#   GPU=1 PORT=8241 NAME=qwen3-8b MODEL=/path/to/shared/Qwen3-8B OUT=$ROOT/experiments/_x/qwen3-8b/passk_final_outside.x18.jsonl bash explore/x_passk.sh
#   ... N=32 TEMP=0.7 OUT=.../passk_final_outside.sens.jsonl   (sensitivity)
#   ... GPU=0,1 TP=2 N=8 ...                                        (a teacher probe, PREREG_OPD §4)
source /path/to/workdir/env.sh   # before set -u: the system profile it pulls in reads unset variables
set -euo pipefail
ROOT=/path/to/workdir
N=${N:-64}; TEMP=${TEMP:-1.0}
LOG=$ROOT/logs/x; mkdir -p "$LOG" "$(dirname "$OUT")"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 setsid "$ROOT/verl/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "x-$NAME" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.88 --tensor-parallel-size "${TP:-1}" \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --generation-config vllm --host 127.0.0.1 --port "$PORT" > "$LOG/server-passk-$NAME.log" 2>&1 &
SERVER=$!
trap 'kill -- -$SERVER 2>/dev/null || kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; exit 1; }
  sleep 10
done
cd "$(dirname "$0")/.."
"$ROOT/eval-venv/bin/python" -m explore.x2_passk sample --base-url "http://127.0.0.1:$PORT/v1" --model "x-$NAME" \
  --input "$ROOT/experiments/_inputs/kc/final.json" --variants outside --n "$N" --temperature "$TEMP" --workers 4 --out "$OUT"
echo PASSKDONE
