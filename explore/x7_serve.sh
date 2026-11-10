#!/usr/bin/env bash
# X-7 on one local model (thinking off): serve it, ask the K2 probe, run X7 judge-then-act, stop the server.
#   GPU=2 PORT=8261 OUT=qwen3-8b MODEL=/path/to/shared/Qwen3-8B [TOPUP=1] bash explore/x7_serve.sh [extra vLLM args]
source /path/to/workdir/env.sh   # before set -u
set -euo pipefail
ROOT=/path/to/workdir
X=$ROOT/experiments/_x7; mkdir -p "$X" "$ROOT/logs/x7"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 setsid "$ROOT/verl/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "x7-$OUT" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --generation-config vllm --host 127.0.0.1 --port "$PORT" "$@" > "$ROOT/logs/x7/server-$OUT.log" 2>&1 &
SERVER=$!
trap 'kill -- -$SERVER 2>/dev/null || kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; exit 1; }
  sleep 10
done
cd $ROOT/analysis-repo
P=$ROOT/eval-venv/bin/python
if [ "${TOPUP:-0}" = 1 ]; then   # PREREG_X X-9: T replies for turn-1 answers re-classified not_valid
  $P -m explore.x7_judge_act topup --input "$X/x7_items.json" --run "$X/x7_$OUT.jsonl" --base-url "http://127.0.0.1:$PORT/v1" --model "x7-$OUT" --thinking-off --workers 16
  echo TOPUPDONE; exit 0
fi
$P -m explore.x6_probe ask --input "$X/k2_probe.json" --base-url "http://127.0.0.1:$PORT/v1" --model "x7-$OUT" --thinking-off --workers 16 --out "$X/k2_answers_$OUT.jsonl"
echo K2ASKDONE
$P -m explore.x7_judge_act run --input "$X/x7_items.json" --base-url "http://127.0.0.1:$PORT/v1" --model "x7-$OUT" --thinking-off --workers 16 --out "$X/x7_$OUT.jsonl"
echo X7DONE
