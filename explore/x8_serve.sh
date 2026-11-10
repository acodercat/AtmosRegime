#!/usr/bin/env bash
# X8 elicit-then-solve on a local model (docs/PREREG_X.md X-20 §2: "serving settings as in D"): serve the configuration with the vLLM
# arguments it had in D (results/d5/serving.json; port unchanged), run explore.x8_elicit with thinking off, stop the server.
#   GPUS=0,1,2,3 bash explore/x8_serve.sh qwen3.5-27b.think-off
# Resume-safe: x8_elicit skips every row already in the output and leaves API failures for the next run of this command.
source /path/to/workdir/env.sh   # before set -u
set -euo pipefail
CONFIG=$1
ROOT=/path/to/workdir
REPO=$(cd "$(dirname "$0")/.." && pwd)
INPUT=$ROOT/experiments/_x8/x8_items.json
OUT=$ROOT/experiments/_x8/x8_${CONFIG%.think-off}-off.jsonl
LOG=$ROOT/logs/x8; mkdir -p "$LOG"

case $CONFIG in *.think-off) ;; *) echo "X8 is thinking off only"; exit 1;; esac
read -r PORT NGPU WORKERS < <(python3 -c '
import json, sys
v = json.load(open(sys.argv[1]))[sys.argv[2]]
print(v["port"], v["gpus"], v["workers"])' "$REPO/results/d5/serving.json" "$CONFIG")
mapfile -t ARGV < <(python3 -c '
import json, sys
print("\n".join(json.load(open(sys.argv[1]))[sys.argv[2]]["vllm_arguments"]))' "$REPO/results/d5/serving.json" "$CONFIG")
NAME=$(python3 -c 'import sys; a = sys.argv[1:]; print(a[a.index("--served-model-name") + 1])' "${ARGV[@]}")
[ "$(tr ',' '\n' <<< "$GPUS" | grep -c .)" = "$NGPU" ] || { echo "$CONFIG needs $NGPU GPUs, got GPUS=$GPUS"; exit 1; }
[ -e "${ARGV[1]}/config.json" ] || { echo "weights ${ARGV[1]} not on this machine"; exit 1; }
[ "$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$INPUT")" = 850 ] || { echo "$INPUT is not the 850-row X8 input"; exit 1; }

CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  setsid "$ROOT/verl/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${ARGV[@]}" > "$LOG/server-$CONFIG.log" 2>&1 &
SERVER=$!
trap 'kill -- -$SERVER 2>/dev/null || kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server exited (log $LOG/server-$CONFIG.log)"; exit 1; }
  sleep 10
done
cd "$REPO"
# x8_elicit grades message.content only: with thinking off the whole reply must arrive there, not in a reasoning field.
"$ROOT/eval-venv/bin/python" - "$PORT" "$NAME" <<'PY'
import sys, openai
port, name = sys.argv[1:]
m = openai.OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{port}/v1").chat.completions.create(
    model=name, messages=[{"role": "user", "content": "What is 17 + 25? Answer with the number."}], temperature=0.0,
    max_tokens=2048, extra_body={"chat_template_kwargs": {"enable_thinking": False}}).choices[0].message
reasoning = (getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None) or "").strip()
content = (m.content or "").strip()
print(f"thinking-off check: content {content[:40]!r}, reasoning {len(reasoning)} chars")
sys.exit(0 if content and not reasoning else 1)
PY
"$ROOT/eval-venv/bin/python" -m explore.x8_elicit run --input "$INPUT" --base-url "http://127.0.0.1:$PORT/v1" --model "$NAME" \
  --thinking-off --workers "$WORKERS" --out "$OUT"
python3 -c '
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
empty = sum(not (r.get(k) or "").strip() for r in rows for k in ("elicit", "C_text", "H_text", "E_text"))
print(f"{sys.argv[2]}: {len(rows)} / 850 rows, {empty} empty replies")' "$OUT" "$CONFIG"
echo "$CONFIG x8 done"
