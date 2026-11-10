#!/usr/bin/env bash
# PREREG_X X-29 section 3, only if explore/x_overlong_check.py says seed-43 Qwen3.5 F2-F degenerated: F2-F seeds 42 and 43
# again with DAPO's overlong penalty (x_chain.sh EXTRA_ARMS=FOL -> x-qwen35-9b-F2-FOL-s42/s43), then their evaluation.
#   DRY=1 bash explore/x_overlong_run.sh                  # print the plan
#   setsid nohup bash explore/x_overlong_run.sh > $ATMOS_ROOT/logs/x/overlong-$(hostname -s).log 2>&1 < /dev/null &
# Runs on a machine holding the W_f sft-x-qwen35-9b-F2 (x_chain.sh would otherwise build and train one). The chain must not
# train anything else: every standard arm of F2 seeds 42 / 43 not merged here gets a SKIP marker (STOPPED, never on a
# merged run). Waits for no x_chain, no follow-up queue and idle GPUs, then a 2-step smoke with OVERLONG=1 (stop if it fails), then the chain.
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
R=${ATMOS_ROOT:-/path/to/workdir}; M=/path/to/local/x/merged; ST=/path/to/local/x
NAME=qwen35-9b; MODEL=/path/to/shared/Qwen3.5-9B; FOLD=F2; SEEDS="42 43"
say(){ echo "[$(date '+%F %T')] $*"; }
run(){ if [ -n "${DRY:-}" ]; then echo "  DRY: $*"; else eval "$@"; fi; }
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1   # the checkout this script sits in (env.sh pins ATMOS_REPO)
say "overlong check on $(hostname -s) at $(git log --oneline -1)"
w=$(ls -d $M/sft-x-$NAME-$FOLD/global_step_*/config.json 2>/dev/null | head -1)
[ -n "$w" ] || { say "ABORT: no W_f sft-x-$NAME-$FOLD on this machine"; exit 1; }
grep -q '" FOL "' explore/x_chain.sh || { say "ABORT: this checkout's x_chain.sh has no FOL arm"; exit 1; }
grep -q 'overlong_buffer_cfg.enable=True' train/dapo_qwen3_8b.sh || { say "ABORT: dapo_qwen3_8b.sh has no OVERLONG"; exit 1; }
for s in $SEEDS; do e=x-$NAME-$FOLD-FOL-s$s
  if ls $M/$e/global_step_*/config.json >/dev/null 2>&1 || [ -f $ST/checkpoints/$e/STOPPED ]; then say "ABORT: $e already exists here"; exit 1; fi; done
if [ -z "${DRY:-}" ]; then
  until ! pgrep -f "[x]_chain.sh|[x]_followup_run" >/dev/null &&   # a follow-up queue idles GPUs between its stages
         [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 120; done
fi
smoke=smoke-x29-FOL-$(hostname -s)
say "smoke $smoke (2 steps, OVERLONG=1, REWARD=regime, $FOLD-F pool)"
run "env SMOKE=1 OVERLONG=1 REWARD=regime EXP=$smoke MODEL_PATH=$MODEL DATA=$R/data-x/$FOLD-F VAL_FILES=\"['$R/data-x/$FOLD-F/train_probe.parquet']\" bash train/dapo_qwen3_8b.sh > $R/logs/x/$smoke.nohup 2>&1 < /dev/null"
rc=$?
if [ -z "${DRY:-}" ]; then
  grep -aq "step:2 - " $R/logs/x/$smoke.nohup || { say "SMOKE FAILED (rc $rc, no step 2)"; exit 1; }
  # the dapo manager logs overlong_reward per row; validation logs every per-row key as val-aux (the config dump does not)
  grep -aq "val-aux/[^ :]*overlong" $R/logs/x/$smoke.nohup || { say "SMOKE FAILED: no val-aux overlong metric, the penalty is not on"; exit 1; }
  say "smoke ok (rc $rc)"
fi
skip=""
for s in $SEEDS; do for a in A D F WA WD; do e=x-$NAME-$FOLD-$a-s$s
  ls $M/$e/global_step_*/config.json >/dev/null 2>&1 && continue
  skip="$skip $e"
  run "mkdir -p $ST/checkpoints/$e && { [ -f $ST/checkpoints/$e/STOPPED ] || echo 'split: run on another machine' > $ST/checkpoints/$e/STOPPED; }"
done; done
say "train x-$NAME-$FOLD-FOL-s42/s43 (skip markers:${skip:- none})"
run "env NAME=$NAME MODEL=$MODEL SEEDS=\"$SEEDS\" FOLDS=$FOLD EXTRA_ARMS=FOL X_STORE=$ST bash explore/x_chain.sh > $R/logs/x/chain-$NAME-$FOLD-FOL.log 2>&1 < /dev/null"
say "== evaluation"
lane(){ local slot=$1 gpus=$2 label=$3 model
  model=$(ls -d $M/$label/global_step_* 2>/dev/null | tail -1)
  [ -n "${DRY:-}" ] || [ -f "$model/config.json" ] || { say "$label: no model"; return; }
  say "eval $label ($model) slot $slot GPUs $gpus"
  run "LABEL=$label MODEL=$model GPUS=$gpus SLOT=$slot bash explore/x_eval.sh > $R/logs/x_eval/$label.log 2>&1 < /dev/null"
  [ -n "${DRY:-}" ] || say "$(tail -1 $R/logs/x_eval/$label.log)"; }
lane 0 0,1,2,3 x-$NAME-$FOLD-FOL-s42 & lane 1 4,5,6,7 x-$NAME-$FOLD-FOL-s43 & wait
say "OVERLONG DONE"
