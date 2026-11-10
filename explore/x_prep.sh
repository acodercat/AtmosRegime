#!/usr/bin/env bash
# PREREG_X preparation on one base model: serve, warm-start samples for both folds, base pass@64 on final.json outside rows, stop.
#   GPU=1 PORT=8241 NAME=qwen3-8b MODEL=/path/to/shared/Qwen3-8B bash explore/x_prep.sh
# Runs explore.* from the checkout this script lives in (a worktree at the launched commit, as x_passk.sh), and names the
# pass@64 file as X-18 does (passk_final_outside.x18.jsonl, the sandbox-fixed sampling x_report reads by default).
source /path/to/workdir/env.sh
set -euo pipefail
ROOT=/path/to/workdir
LOG=$ROOT/logs/x; mkdir -p "$LOG"
OUT=$ROOT/experiments/_x/$NAME; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 setsid "$ROOT/verl/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "x-$NAME" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.88 \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --generation-config vllm --host 127.0.0.1 --port "$PORT" > "$LOG/server-$NAME.log" 2>&1 &
SERVER=$!
trap 'kill -- -$SERVER 2>/dev/null || kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; exit 1; }
  sleep 10
done
cd "$(dirname "$0")/.."
P=$ROOT/eval-venv/bin/python
for f in F1 F2; do
  $P -m explore.x_data warm-sample --fold $f --kc $ROOT/experiments/_inputs/kc --base-url "http://127.0.0.1:$PORT/v1" --model "x-$NAME" --out "$OUT/warm_samples_$f.jsonl"
done
$P -m explore.x2_passk sample --base-url "http://127.0.0.1:$PORT/v1" --model "x-$NAME" --input $ROOT/experiments/_inputs/kc/final.json --variants outside --n 64 --workers 4 --out "$OUT/passk_final_outside.x18.jsonl"
echo PREPDONE
