#!/usr/bin/env bash
# PREREG_PAPER_ANALYSES §21: frontier models in the X evaluation harness (atmoscoder_rl.evaluation.runner, code protocol,
# T 0.3, 5 % tolerance, at most 5 content attempts, one sample per row; models.toml frontier21-*), run locally against the
# API gateways. Inputs are the X evaluation files: textbook.json (46 regime traps + 46 parents) three times for every
# configuration, final_think.json (inside / outside rows of the final set, 2,800) once for the thinking-off configurations.
# Pass 2 re-runs only rows left as API errors (the runner drops them on resume). Prints F21DONE <config> per configuration.
#
#   GATEWAY_B_API_KEY=... GATEWAY_A_API_KEY=... setsid nohup bash explore/frontier21.sh > experiments/frontier21/launch.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
: "${GATEWAY_B_API_KEY:?}" "${GATEWAY_A_API_KEY:?}"
IN=experiments/_inputs/x_eval; OUT=experiments/frontier21; WORKERS=${WORKERS:-12}
OFF="frontier21-deepseek-v4-flash-off frontier21-gpt-5.5-off frontier21-gpt-5.4-off frontier21-minimax-m3-off"
ON="frontier21-deepseek-v4-flash-on frontier21-gpt-5.5-on"
mkdir -p $OUT/logs
sha256sum $IN/textbook.json $IN/final_think.json > $OUT/INPUTS.sha256
say(){ echo "[$(date +%F' '%T)] $*"; }
run(){ local model=$1 name=$2 src=$3; shift 3
  uv run --extra eval python -m atmoscoder_rl.evaluation.runner --model "$model" --input "$src" --experiments $OUT --exp-id "$model" \
    --out-name "$name" --tolerance 0.05 --max-attempts 5 --workers $WORKERS "$@" 2>&1 | grep -v '^\[' || true; }
one_model(){ local m=$1
  for pass in 1 2; do
    say "$m pass $pass"
    for r in 1 2 3; do run "$m" textbook $IN/textbook.json --run $r; done
    case " $OFF " in *" $m "*) run "$m" final $IN/final_think.json;; esac
  done
  say "F21DONE $m"; }
for m in $OFF $ON; do one_model "$m" > $OUT/logs/$m.log 2>&1 & done
wait
say "F21ALLDONE"
