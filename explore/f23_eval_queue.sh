#!/usr/bin/env bash
# PREREG_PAPER_ANALYSES §23 E1: evaluate stored X / DOSE / OPD models, one after another on one GPU group, on the D-5
# benchmark set (experiments/_inputs/d5/textbook.json, 207 rows: 54 regime and 56 geometry traps with their 97 parents,
# none of them in the X textbook set's 46 regime traps) in the X evaluation harness — explore/x_eval.sh's server flags and
# runner settings (atmoscoder_rl.evaluation.runner, code protocol, models.toml x-eval-<SLOT>: T 0.3, 8,192 tokens, thinking
# off; 5 % tolerance, at most 5 content attempts, one sample per row, a second pass for rows left as API errors).
# Each label's model is the checkpoint its X evaluation used ($ATMOS_ROOT/experiments/_x_eval/<label>/serving.json);
# a label whose checkpoint is not on this machine is reported MISSING and skipped. Output:
# $ATMOS_ROOT/experiments/_x_eval_d5/<label>/textbook_d5.json + serving.json. Resume-safe; a complete label is skipped.
#
#   cd <worktree> && LABELS="x-qwen35-9b-base x-qwen35-9b-F1-A-s42 ..." GPUS=0,1,2,3 SLOT=0 \
#     setsid nohup bash explore/f23_eval_queue.sh > $ATMOS_ROOT/logs/f23/queue-<host>-<slot>.log 2>&1 < /dev/null &
# Prints EVALDONE / EVALINCOMPLETE / MISSING per label and QUEUEDONE at the end.
source /path/to/workdir/env.sh >/dev/null 2>&1   # before set -u
set -uo pipefail
: "${LABELS:?}" "${GPUS:?}"
SLOT=${SLOT:-0}; PORT=$((8270 + SLOT)); SECTION=x-eval-$SLOT
R=$ATMOS_ROOT; P=$R/eval-venv/bin/python
INPUT=$R/experiments/_inputs/d5/textbook.json; SHA=cc3d59e2ac1b661ab14dd154897bf9b303e2e44d2648119dd11f08acaaa27906
OUT=$R/experiments/_x_eval_d5; ROWS=207
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .); WORKERS=${WORKERS:-$((16 * NGPU))}
mkdir -p $R/logs/f23 $OUT
say(){ echo "[$(date +%F' '%T)] $*"; }
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1; export PYTHONPATH=$PWD
[ "$(sha256sum "$INPUT" | cut -d' ' -f1)" = "$SHA" ] || { say "input $INPUT is not the sealed D-5 file"; exit 1; }

complete(){  # rows with a gradable record (no API error) == ROWS
  $P - "$OUT/$1/textbook_d5.json" "$ROWS" <<'PY'
import json, sys
from pathlib import Path
f = Path(sys.argv[1])
if not f.exists():
    sys.exit(1)
rs = json.loads(f.read_text())["results"]
ok = {r["id"] for r in rs if not r.get("error") and r.get("attempts") and r.get("num_attempts", 0) > 0}
sys.exit(0 if len(ok) == int(sys.argv[2]) else 1)
PY
}
model_of(){ $P -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$R/experiments/_x_eval/$1/serving.json" 2>/dev/null; }

for LABEL in $LABELS; do
  if complete $LABEL; then say "EVALDONE $LABEL (already complete)"; continue; fi
  MODEL=$(model_of $LABEL)
  [ -n "$MODEL" ] && [ -f "$MODEL/config.json" ] || { say "MISSING $LABEL (${MODEL:-no serving.json}) on $(hostname)"; continue; }
  CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 setsid "$R/verl/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name x-eval --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --data-parallel-size $NGPU --generation-config vllm --host 127.0.0.1 --port $PORT > $R/logs/f23/server-$LABEL.log 2>&1 < /dev/null &
  SERVER=$!
  up=0
  for _ in $(seq 1 1080); do   # up to 3 h: on 2026-11-08 the venv imports from /path/to/shared alone took 27 min
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { up=1; break; }
    kill -0 $SERVER 2>/dev/null || break
    sleep 10
  done
  if [ $up = 1 ]; then
    say "serving $MODEL as $LABEL on GPUs $GPUS (dp $NGPU), port $PORT"
    mkdir -p $OUT/$LABEL
    $P -c 'import json, sys; print(json.dumps(dict(zip(("label", "model", "host", "gpus", "section", "commit", "input_sha256"), sys.argv[1:8]))))' \
      "$LABEL" "$MODEL" "$(hostname)" "$GPUS" "$SECTION" "$(git rev-parse HEAD)" "$SHA" > $OUT/$LABEL/serving.json
    for pass in 1 2; do
      $P -m atmoscoder_rl.evaluation.runner --model $SECTION --input $INPUT --experiments $OUT --exp-id $LABEL --out-name textbook_d5 \
        --tolerance 0.05 --max-attempts 5 --workers $WORKERS | grep -v '^\[' || true
      complete $LABEL && break
    done
  else
    say "server for $LABEL did not come up (logs/f23/server-$LABEL.log)"
  fi
  kill -TERM -- -$SERVER 2>/dev/null; sleep 20; kill -KILL -- -$SERVER 2>/dev/null || true
  for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || break; sleep 5; done
  for g in ${GPUS//,/ }; do for _ in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 4096 ] && break; sleep 5; done; done
  if complete $LABEL; then say "EVALDONE $LABEL"; else say "EVALINCOMPLETE $LABEL"; fi
done
say "QUEUEDONE $(hostname) slot $SLOT"
