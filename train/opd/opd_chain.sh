#!/usr/bin/env bash
# PREREG_OPD §5 (OPD-8): the remaining OPD runs back to back on one machine. For each ARM:FOLD:SEED — train with
# train/opd/run_opd.sh (GPUs 0-3 student + 7 teacher), check the finished run's log (rc 0, step 120, top-k masses <= 1.001,
# per-token loss_max < 100 — the OPD-5 gates), merge step 120, evaluate with explore/x_eval.sh on GPUs 4-6 while the next run
# trains. A run whose evaluation is complete is skipped; one still training outside the chain is waited for, not restarted.
#   RUNS="S:F1:42 S:F2:42 Pk3:F1:42 Pk3:F2:42 P:F1:43 P:F2:43" setsid nohup bash train/opd/opd_chain.sh > $ATMOS_ROOT/logs/opd/chain.log 2>&1 < /dev/null &
#   CHECK_ONLY=1 RUNS="P:F1:42 P:F2:42" bash train/opd/opd_chain.sh        # only print the gate verdicts
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
: "${RUNS:?ARM:FOLD:SEED ...}"
R=$ATMOS_ROOT; P=$R/verl/.venv/bin/python; PE=$R/eval-venv/bin/python; CK=/path/to/local/opd/checkpoints; MG=/path/to/local/opd/merged
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1; export PYTHONPATH=$PWD
say(){ echo "[$(date +%F' '%T)] $*"; }
clean(){ sed "s/\x1b\[[0-9;]*m//g" "$1"; }

gate(){  # exp wrapper-log: the OPD-5 gates over the whole training log
  local exp=$1 w=$2 L=$R/logs/$1.log rc steps mass lmax
  rc=$(grep -ao "OPD $exp exited [0-9]*" "$w" | tail -1 | awk '{print $NF}')
  steps=$(clean "$L" | grep -ao "step:[0-9]* - " | grep -o "[0-9]*" | sort -n | tail -1)
  mass=$(clean "$L" | grep -ao "teacher_mass_max:[0-9.e+-]*\|student_mass_max:[0-9.e+-]*" | cut -d: -f2 | sort -g | tail -1)
  lmax=$(clean "$L" | grep -ao "distillation/loss_max:[0-9.e+-]*" | cut -d: -f2 | sort -g | tail -1)
  local ok=1
  # OPD-13: the top-K masses exist only for the top-K loss (Pk3 / Pk2 log none, OPD-10); Pk2's k2 is unclamped by design,
  # so its loss_max is logged, not gated (any NaN / inf still fails); every other arm keeps the OPD-5 gates
  case $ARM in Pk3|Pk2) mass=${mass:-0};; esac
  [ "$ARM" = Pk2 ] && { clean "$L" | grep -aq "distillation/loss[a-z_]*:\(nan\|inf\)" && ok=0; lmax_gate=1e30; } || lmax_gate=100
  [ "${rc:-1}" = 0 ] || ok=0; [ "${steps:-0}" -ge 120 ] || ok=0
  awk -v m="${mass:-9}" -v l="${lmax:-9999}" -v g="$lmax_gate" 'BEGIN{exit !(m <= 1.001 && l < g)}' || ok=0
  say "$exp gate: rc ${rc:-?} steps ${steps:-?} mass_max ${mass:-?} loss_max ${lmax:-?} -> $([ $ok = 1 ] && echo PASS || echo FAIL)"
  [ $ok = 1 ]
}
wait_free(){ for g in "$@"; do until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 4096 ]; do sleep 60; done; done; }
evaluated(){ $PE -m explore.build_x_eval_input --final $R/experiments/_inputs/kc/final.json --out-dir $R/experiments/_inputs/x_eval --check-run $R/experiments/_x_eval/$1 >/dev/null 2>&1; }

EVAL_PID=
for run in $RUNS; do
  IFS=: read -r ARM FOLD SEED <<< "$run"; EXP=opd-qwen3-8b-$FOLD-$ARM-s$SEED; W=$R/logs/opd/run-$ARM-$FOLD-s$SEED.log
  if [ -n "${CHECK_ONLY:-}" ]; then gate $EXP $W; continue; fi
  if evaluated $EXP; then say "$EXP already evaluated"; continue; fi
  if [ ! -f $MG/$EXP/global_step_120/config.json ]; then
    if [ -f "$W" ] && ! grep -q "OPD $EXP exited" "$W"; then
      say "$EXP is training outside the chain, waiting"; until grep -q "OPD $EXP exited" "$W"; do sleep 60; done
    elif [ "$(cat $CK/$EXP/latest_checkpointed_iteration.txt 2>/dev/null)" != 120 ]; then
      wait_free 0 1 2 3 7; say "=== train $EXP ==="
      ARM=$ARM FOLD=$FOLD SEED=$SEED GPUS=0,1,2,3,7 bash train/opd/run_opd.sh > "$W" 2>&1
    fi
    gate $EXP "$W" || { say "$EXP FAILED the gate: not merged, not evaluated"; continue; }
    mkdir -p $MG/$EXP
    (cd $R/verl && CUDA_VISIBLE_DEVICES= $P -m verl.model_merger merge --backend fsdp --local_dir $CK/$EXP/global_step_120/actor \
       --target_dir $MG/$EXP/global_step_120 > $R/logs/opd/merge_$EXP.log 2>&1) || { say "$EXP merge failed"; continue; }
    rm -f $CK/$EXP/global_step_120/actor/optim_*.pt; say "$EXP merged"
  fi
  [ -n "$EVAL_PID" ] && wait $EVAL_PID
  wait_free 4 5 6; say "=== evaluate $EXP (GPUs 4-6) ==="
  ( LABEL=$EXP MODEL=$MG/$EXP/global_step_120 GPUS=4,5,6 SLOT=1 bash explore/x_eval.sh > $R/logs/opd/post-$ARM-$FOLD-s$SEED.log 2>&1 ) &
  EVAL_PID=$!
done
[ -n "$EVAL_PID" ] && wait $EVAL_PID
say "OPD CHAIN DONE"
