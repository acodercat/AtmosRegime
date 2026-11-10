#!/usr/bin/env bash
# PREREG_X §5 (docs/PREREG_X.md, X-10 "evaluate on the machine that holds the model"): evaluate ONE model on this machine — serve it with
# data-parallel vLLM over GPUS (the x7_serve.sh flags but a 32k context so a self-repair prompt carrying an 8,192-token failed reply still fits (X-11), --generation-config vllm; thinking off is in the
# request), then run four evaluations through atmoscoder_rl.evaluation.runner (code protocol, T 0.3, 5 % tolerance, at most
# 5 content attempts, 1 sample per row; models.toml x-eval-<SLOT>) into $ATMOS_ROOT/experiments/_x_eval/<LABEL>/:
#   final.json     every row of experiments/_inputs/kc/final.json (5,600)
#   judge.json     its inside / outside rows with pilot.JUDGE_INSTRUCTION (2,800; explore/build_x_eval_input.py)
#   test_id.json   the test_id split (64)
#   textbook.json  test-side textbook regime traps and their parents
# Resume-safe: the runner keeps completed rows; a model whose four files are complete is skipped without starting a server.
# Labels (explore/x_report.py reads them): base x-<name>-base, warm start sft-x-<name>-<fold>, endpoints the chain's run
# names x-<name>-<fold>-<arm>-s<seed>. Extra arguments go to vLLM and must be the same for every model of a family.
#
#   cd $ATMOS_REPO && LABEL=x-qwen3-8b-F1-A-s42 MODEL=/path/to/local/x/merged/x-qwen3-8b-F1-A-s42/global_step_120 GPUS=0,1,2,3 \
#     [SLOT=0|1|2] setsid nohup bash explore/x_eval.sh > $ATMOS_ROOT/logs/x_eval/x-qwen3-8b-F1-A-s42.log 2>&1 < /dev/null &
# Prints EVALDONE <label> at the end (EVALINCOMPLETE <label> if rows are still missing after two passes).
# THINK=1 (X-34, X-35, X-36): thinking on (models.toml x-eval-think-<SLOT>, vLLM --reasoning-parser qwen3, context 40,960
# as D), only the inside / outside rows of final.json (final_think.json, saved as final.json), into
# experiments/_x_eval_think/<LABEL>/; the server log is server-<LABEL>-think.log (the label is the thinking-off one).
source /path/to/workdir/env.sh >/dev/null 2>&1   # before set -u
set -uo pipefail
: "${LABEL:?}" "${MODEL:?}" "${GPUS:?}"
SLOT=${SLOT:-0}; PORT=$((8270 + SLOT)); SECTION=x-eval-$SLOT
R=$ATMOS_ROOT; P=$R/eval-venv/bin/python
FINAL=$R/experiments/_inputs/kc/final.json; IN=$R/experiments/_inputs/x_eval; OUT=$R/experiments/_x_eval
EVALS="final judge test_id textbook"; THINKARG=(); PARSER=(); MAXLEN=32768; LOGTAG=
if [ -n "${THINK:-}" ]; then
  SECTION=x-eval-think-$SLOT; OUT=$R/experiments/_x_eval_think; EVALS="final"; THINKARG=(--think); PARSER=(--reasoning-parser qwen3)
  MAXLEN=40960; LOGTAG=-think
fi
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
WORKERS=${WORKERS:-$((16 * NGPU))}
mkdir -p $R/logs/x_eval $OUT/$LABEL
say(){ echo "[$(date +%F' '%T)] $*"; }
# the code beside this script (env.sh pins ATMOS_REPO to the shared checkout, so a clone must not rely on it)
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1; export PYTHONPATH=$PWD
[ -f "$MODEL/config.json" ] || { say "no model at $MODEL"; exit 1; }
$P -m explore.build_x_eval_input --final $FINAL --out-dir $IN "${THINKARG[@]}" || { say "inputs failed"; exit 1; }
check(){ $P -m explore.build_x_eval_input --final $FINAL --out-dir $IN "${THINKARG[@]}" --check-run $OUT/$LABEL; }
if check; then echo "EVALDONE $LABEL"; exit 0; fi

CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 setsid "$R/verl/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name x-eval --dtype bfloat16 --max-model-len $MAXLEN --gpu-memory-utilization 0.85 \
  --data-parallel-size $NGPU --generation-config vllm --host 127.0.0.1 --port $PORT "${PARSER[@]}" "$@" > $R/logs/x_eval/server-$LABEL$LOGTAG.log 2>&1 < /dev/null &
SERVER=$!
trap 'kill -TERM -- -$SERVER 2>/dev/null; sleep 15; kill -KILL -- -$SERVER 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { say "server exited (see logs/x_eval/server-$LABEL$LOGTAG.log)"; exit 1; }
  sleep 10
done
say "serving $MODEL as $LABEL on GPUs $GPUS (dp $NGPU), port $PORT"
$P -c 'import json, sys; print(json.dumps(dict(zip(("label", "model", "host", "gpus", "section", "commit"), sys.argv[1:7]), vllm_extra=sys.argv[7:])))' \
  "$LABEL" "$MODEL" "$(hostname)" "$GPUS" "$SECTION" "$(git rev-parse HEAD)" "${PARSER[@]}" "$@" > $OUT/$LABEL/serving.json

for pass in 1 2; do   # pass 2 re-runs only rows left as API errors (the runner drops them on resume)
  for name in $EVALS; do
    src=$IN/$name.json; [ $name = final ] && src=$FINAL; [ -n "${THINK:-}" ] && src=$IN/final_think.json
    $P -m atmoscoder_rl.evaluation.runner --model $SECTION --input $src --experiments $OUT --exp-id $LABEL --out-name $name \
      --tolerance 0.05 --max-attempts 5 --workers $WORKERS | grep -v '^\[' || true
  done
  check && { echo "EVALDONE $LABEL"; exit 0; }
done
echo "EVALINCOMPLETE $LABEL"; exit 1
