#!/usr/bin/env bash
# PREREG_PAPER_ANALYSES §23 on one 4-GPU group, one stage after another: E1 (explore/f23_eval_queue.sh over LABELS), E2
# (explore/f23_d_serve.sh: the two thinking-off comparison configurations on the 16 missing rows x 3 and the final-set rows
# x 1, then the two thinking-on ones on the 16 rows x 3), E4 (explore/x_passk.sh, Qwen3.5-27B base, n 64, T 1.0, TP 4).
# STAGES selects (default "E1 E2 E4"). A failed step is logged and the chain goes on; every step is resume-safe.
#   cd <worktree> && GPUS=4,5,6,7 SLOT=1 LABELS="..." setsid nohup bash explore/f23_chain.sh > $ATMOS_ROOT/logs/f23/chain-<host>.log 2>&1 < /dev/null &
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
: "${GPUS:?}"
R=$ATMOS_ROOT; STAGES=${STAGES:-E1 E2 E4}; SLOT=${SLOT:-1}
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
mkdir -p $R/logs/f23
say(){ echo "[$(date +%F' '%T)] $*"; }
wait_free(){ for g in ${GPUS//,/ }; do until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 4096 ]; do sleep 30; done; done; }
MISSING=$R/experiments/_inputs/f23/textbook_missing.json; FINAL=$R/experiments/_inputs/x_eval/final_think.json
d(){  # config name input runs
  wait_free; say "=== E2 $1 $2 x$4 ==="
  GPUS=$GPUS bash explore/f23_d_serve.sh "$1" "$2" "$3" "$4" > $R/logs/f23/d-$1-$2.log 2>&1 && say "E2 $1 $2 done" || say "E2 $1 $2 FAILED (logs/f23/d-$1-$2.log)"
}
for stage in $STAGES; do
  case $stage in
    E1) : "${LABELS:?}"; wait_free; say "=== E1 queue ==="
        LABELS="$LABELS" GPUS=$GPUS SLOT=$SLOT bash explore/f23_eval_queue.sh;;
    E2) d qwen3.5-27b.think-off missing $MISSING 3; d qwen3.5-27b.think-off final $FINAL 1
        d gemma4-31b.think-off missing $MISSING 3;  d gemma4-31b.think-off final $FINAL 1
        d qwen3.5-27b.think-on missing $MISSING 3;  d gemma4-31b.think-on missing $MISSING 3;;
    E4) wait_free; say "=== E4 Qwen3.5-27B pass@64 ==="
        GPU=$GPUS TP=4 PORT=8243 NAME=qwen35-27b MODEL=/path/to/shared/Qwen3.5-27B \
          OUT=$R/experiments/_x/qwen35-27b/passk_final_outside.x18.jsonl bash explore/x_passk.sh > $R/logs/f23/passk-qwen35-27b.log 2>&1 \
          && say "E4 done" || say "E4 FAILED (logs/f23/passk-qwen35-27b.log)";;
  esac
done
say "F23 CHAIN DONE $(hostname)"
