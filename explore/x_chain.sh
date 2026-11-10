#!/usr/bin/env bash
# PREREG_X (docs/PREREG_X.md): for one family, both folds — warm start W_f (build + SFT + merge), then the five arms
# A / D / F (from the base) and WA / WD (from W_f) for each RL seed, each merged at its last checkpoint. Unattended,
# resume-safe (a merged model is skipped). Needs the fold pools (explore/x_data.py export) and the warm-start samples
# (explore/x_prep.sh) already on disk. Takes the whole machine while an RL run is on.
#
#   cd $ATMOS_REPO && NAME=qwen3-8b MODEL=/path/to/shared/Qwen3-8B SEEDS="42 43" \
#     setsid nohup bash explore/x_chain.sh > $ATMOS_ROOT/logs/x/chain-qwen3-8b.log 2>&1 &
# EXTRA_ARMS="R" (X-29 §2) adds, after WD, the random-reward control R: from the base, the A pool, REWARD=random (a fair
# coin per rollout); the five arms above are unchanged. Unset = the five arms only.
# EXTRA_ARMS="FOL" (X-29 §3, only if seed-43 F2-F degenerates) adds F with DAPO's overlong penalty: from the base, the F pool,
# the regime reward, OVERLONG=1 (verl's dapo reward manager; group filtering still reads the unpenalised score).
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
P=$ATMOS_ROOT/verl/.venv/bin/python
say(){ echo "[$(date +%F' '%T)] $*"; }
: "${NAME:?}" "${MODEL:?}"
SEEDS=${SEEDS:-42}
FOLDS=${FOLDS:-"F1 F2"}
STEPS=${STEPS:-120}
MAX_HOURS=${MAX_HOURS:-24}   # PREREG_X amendment X-4: wall-clock cap per RL run (summed over attempts, X-10)
RAFT=${RAFT:-$ATMOS_ROOT/data/sft_raft.parquet}   # X-10: the family's own base model's verified solutions
ST=${X_STORE:-$ATMOS_ROOT}   # X-10: checkpoints and merged models on this machine's local disk (/path/to/local/x); logs stay shared
KC=$ATMOS_ROOT/experiments/_inputs/kc
X=$ATMOS_ROOT/experiments/_x/$NAME
cd $ATMOS_REPO
mkdir -p $ST/checkpoints $ST/merged
say "chain $NAME folds '$FOLDS' seeds '$SEEDS' on $(hostname), store $ST"

wait_free(){ while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${used:-0}" -lt 4096 ] && return 0; sleep 60; done; }

merge(){  # exp step  (actor checkpoints under checkpoints/$exp/global_step_$step/actor; SFT under .../global_step_$step)
  local exp=$1 step=$2 src=$ST/checkpoints/$1/global_step_$2 M=$ST/merged/$1/global_step_$2
  [ -d "$src/actor" ] && src=$src/actor
  [ -f "$M/config.json" ] && return 0
  wait_free; mkdir -p $ST/merged/$exp
  (cd $ATMOS_ROOT/verl && CUDA_VISIBLE_DEVICES= $P -m verl.model_merger merge --backend fsdp \
     --local_dir $src --target_dir $M > $ATMOS_ROOT/logs/x/merge_$exp.log 2>&1) || { say "$exp: merge failed"; return 1; }
  rm -f $ST/checkpoints/$exp/global_step_*/actor/optim_*.pt $ST/checkpoints/$exp/global_step_*/optim_*.pt
  say "$exp merged at step $step"
}

latest(){ cat $ST/checkpoints/$1/latest_checkpointed_iteration.txt 2>/dev/null || echo 0; }

run_guarded(){  # log cmd...: run in its own process group under the bwrap/load watchdog; returns the command's rc (99 = watchdog)
  local log=$1; shift
  setsid "$@" > "$log" 2>&1 < /dev/null &
  local pg=$! tripped=0
  while kill -0 $pg 2>/dev/null; do
    b=$(pgrep -c bwrap); l=$(cut -d. -f1 /proc/loadavg)
    if [ "${b:-0}" -gt 300 ] || [ "${l:-0}" -gt 1000 ]; then
      say "WATCHDOG bwrap=$b load=$l: stopping process group $pg"; tripped=1
      kill -TERM -- -$pg 2>/dev/null; sleep 30; kill -KILL -- -$pg 2>/dev/null
      "$ATMOS_ROOT/verl/.venv/bin/ray" stop --force > /dev/null 2>&1; break
    fi
    sleep 20
  done
  wait $pg; local rc=$?
  [ $tripped = 1 ] && return 99
  return $rc
}

train(){  # exp data start seed [reward, default regime] [overlong: 1 adds OVERLONG=1, unset leaves the command as before]
  local exp=$1 data=$2 start=$3 seed=$4 reward=${5:-regime} overlong=${6:-} C=$ST/checkpoints/$1
  if ls $ST/merged/$exp/global_step_*/config.json >/dev/null 2>&1; then say "$exp: merged, skipping"; return 0; fi
  mkdir -p $C
  # X-10: a run ends at STEPS, at the X-4 cap (wall clock summed over attempts) or after a second crash; one same-seed resume
  # (verl resume_mode auto) is allowed after an infrastructure crash or a watchdog stop.
  while [ "$(latest $exp)" -lt "$STEPS" ] && [ ! -f $C/STOPPED ]; do
    local used=$(cat $C/WALL_SECONDS 2>/dev/null || echo 0) tries=$(cat $C/ATTEMPTS 2>/dev/null || echo 0)
    local left=$(( MAX_HOURS * 3600 - used ))
    if [ $left -le 0 ]; then say "$exp: wall-clock cap already used"; echo cap > $C/STOPPED; break; fi
    echo $((tries + 1)) > $C/ATTEMPTS
    say "=== train $exp (seed $seed, from $start, $data, attempt $((tries + 1)), ${left}s left) ==="
    wait_free
    local t0=$(date +%s)
    # X-5: validate only on the fold's fixed 256-row training-side probe (step 0 and every 20 steps), T = 0.3, for curves
    run_guarded $ATMOS_ROOT/logs/x/$exp.attempt$((tries + 1)).nohup env CKPT_ROOT=$ST/checkpoints SEED=$seed MODEL_PATH=$start DATA=$data REWARD=$reward ${overlong:+OVERLONG=$overlong} EXP=$exp \
      TOTAL_STEPS=$STEPS TEST_FREQ=20 SAVE_FREQ=5 KEEP_CKPT=1 VAL_FILES="['$data/train_probe.parquet']" VAL_TEMP=0.3 \
      timeout --signal=TERM --kill-after=10m ${left}s bash train/dapo_qwen3_8b.sh
    local rc=$?
    echo $(( used + $(date +%s) - t0 )) > $C/WALL_SECONDS
    if [ $rc = 124 ] || [ $rc = 137 ]; then          # X-4: wall-clock cap reached -> early termination, last checkpoint
      say "$exp hit the ${MAX_HOURS} h cap (rc $rc); stopping its processes by PID"
      for pid in $(ps -eo pid,args | grep "[t]rainer.experiment_name=$exp " | awk '{print $1}'); do kill $pid 2>/dev/null; done
      "$ATMOS_ROOT/verl/.venv/bin/ray" stop --force > /dev/null 2>&1
      sleep 30
      echo cap > $C/STOPPED
    elif [ $rc = 0 ]; then
      [ "$(latest $exp)" -ge "$STEPS" ] || { say "$exp: trainer exited 0 before step $STEPS"; echo early-exit > $C/STOPPED; }
    else
      "$ATMOS_ROOT/verl/.venv/bin/ray" stop --force > /dev/null 2>&1; sleep 30
      [ $((tries + 1)) -ge 2 ] && echo "crash rc $rc" > $C/STOPPED
    fi
    say "$exp attempt exited (rc $rc) at step $(latest $exp); wall clock $(( $(cat $C/WALL_SECONDS) / 60 )) min in total; mean clip ratio $(grep -o 'response_length/clip_ratio:[0-9.]*' $ATMOS_ROOT/logs/$exp.log | awk -F: '{s+=$2; n++} END {if (n) printf "%.3f over %d steps", s/n, n}'); evicted samples per step (the V1 trainer's only trace of extra generation batches): $(grep -o 'filter_groups/evicted_samples:[0-9.]*' $ATMOS_ROOT/logs/$exp.log | awk -F: '{s+=$2; n++} END {if (n) printf "mean %.0f over %d steps", s/n, n}')"
  done
  local step=$(latest $exp)
  [ "$step" -gt 0 ] || { say "$exp: no checkpoint ($(cat $C/STOPPED 2>/dev/null))"; return 1; }
  merge $exp $step && rm -rf $C/global_step_*   # X-10: keep the merged model only (disk); logs and metrics stay
}

for fold in $FOLDS; do
  W=sft-x-$NAME-$fold
  if ! ls $ST/merged/$W/global_step_*/config.json >/dev/null 2>&1; then
    if [ ! -f $ATMOS_ROOT/data/sft_x-$NAME-$fold.parquet ]; then
      say "=== warm-start data $fold ==="
      ATMOSCODER_SANDBOX_CONCURRENCY=32 $P -m explore.x_data warm-build --fold $fold --kc $KC --samples $X/warm_samples_$fold.jsonl \
        --raft $RAFT --raft-ratio 2 --out $ATMOS_ROOT/data/sft_x-$NAME-$fold.parquet > $ATMOS_ROOT/logs/x/warm-build-$NAME-$fold.log 2>&1 || { say "warm-build failed"; exit 1; }
    fi
    if [ "$(latest $W)" = 0 ]; then
      say "=== warm-start SFT $W (1 epoch, lr 5e-6) ==="
      wait_free
      CKPT_ROOT=$ST/checkpoints MODEL_PATH=$MODEL SOURCE=x-$NAME-$fold EXP=$W EPOCHS=1 LR=5e-6 SEED=42 bash train/sft_qwen3_8b.sh > $ATMOS_ROOT/logs/x/$W.nohup 2>&1
      say "SFT exited ($?) at step $(latest $W)"
    fi
    merge $W $(latest $W) || exit 1
    rm -rf $ST/checkpoints/$W/global_step_*   # X-10: keep the merged warm start only
  fi
  WM=$(ls -d $ST/merged/$W/global_step_* | head -1)
  for seed in $SEEDS; do
    train x-$NAME-$fold-A-s$seed  $ATMOS_ROOT/data-x/$fold-A $MODEL $seed
    train x-$NAME-$fold-D-s$seed  $ATMOS_ROOT/data-x/$fold-D $MODEL $seed
    train x-$NAME-$fold-F-s$seed  $ATMOS_ROOT/data-x/$fold-F $MODEL $seed
    train x-$NAME-$fold-WA-s$seed $ATMOS_ROOT/data-x/$fold-A $WM $seed
    train x-$NAME-$fold-WD-s$seed $ATMOS_ROOT/data-x/$fold-D $WM $seed
    case " ${EXTRA_ARMS:-} " in *" R "*) train x-$NAME-$fold-R-s$seed $ATMOS_ROOT/data-x/$fold-A $MODEL $seed random;; esac
    case " ${EXTRA_ARMS:-} " in *" FOL "*) train x-$NAME-$fold-FOL-s$seed $ATMOS_ROOT/data-x/$fold-F $MODEL $seed regime 1;; esac
  done
done
say "=== X CHAIN $NAME DONE ==="
