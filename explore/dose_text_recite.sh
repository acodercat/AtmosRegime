#!/usr/bin/env bash
# PREREG_DOSE_TEXT DT-3: serve one text warm start (as explore/x_passk.sh) and sample the four training questions of every
# unit of its fold, 16 answers each (explore.dose_text_recite sample). Resume-safe.
#   GPU=0 PORT=8241 LABEL=sft-dtext-qwen3-8b-F1-s42 MODEL=/path/to/local/dose/merged/sft-dtext-qwen3-8b-F1-s42/global_step_53 \
#     bash explore/dose_text_recite.sh        # writes $ROOT/experiments/_dose/qwen3-8b/recite.$LABEL.jsonl
source /path/to/workdir/env.sh   # before set -u
set -euo pipefail
ROOT=/path/to/workdir
OUT=${OUT:-$ROOT/experiments/_dose/qwen3-8b/recite.$LABEL.jsonl}
LOG=$ROOT/logs/x; mkdir -p "$LOG" "$(dirname "$OUT")"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 setsid "$ROOT/verl/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name x-recite --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.88 \
  --max-num-seqs 128 --generation-config vllm --host 127.0.0.1 --port "$PORT" > "$LOG/server-recite-$LABEL.log" 2>&1 &
SERVER=$!
trap 'kill -- -$SERVER 2>/dev/null || kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; exit 1; }
  sleep 10
done
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD
"$ROOT/eval-venv/bin/python" -m explore.dose_text_recite sample --design results/dose_text/design.json --label "$LABEL" \
  --base-url "http://127.0.0.1:$PORT/v1" --model x-recite --out "$OUT"
echo "RECITEDONE $LABEL"
