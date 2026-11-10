#!/usr/bin/env bash
# PREREG_DOSE (docs/PREREG_DOSE.md): one fold of the dose-gradient experiment on this machine, in two stages so the warm
# starts can be checked before any RL starts.
#   STAGE=prep: the W_f rebuild check (explore.dose build --full must reproduce the fold's W_f parquet), then per seed the
#               dose warm-start data, SFT (as W_f: 1 epoch, lr 5e-6, SFT seed 42) and merge; then, in parallel, each warm
#               start's evaluation (explore/x_eval.sh, 3 GPUs) and pass@64 on final.json outside rows (explore/x_passk.sh, 1 GPU).
#   STAGE=rl:   per seed the RL run from that seed's warm start, exactly as x_chain.sh's WA (the fold's A pool, the regime
#               reward = the answer reward on these rows, 120 steps, X-4 / X-10 cap and resume rules), merged at its last
#               checkpoint; then both endpoints evaluated in parallel (4 GPUs each).
# PREREG_DOSE_TEXT reuses this chain with PREFIX=dtext BUILDER=explore.dose_text DESIGN=<worktree>/results/dose_text/design.json.
# Training runs from TRAIN_REPO (default the shared checkout $ATMOS_REPO, as every X run; a family whose training needs a newer
# train/ script, e.g. MAX_MODEL_LEN for Ministral 3, sets TRAIN_REPO to this worktree); the dose code, x_eval.sh and x_passk.sh
# from the checkout this script is in. Resume-safe: built data, merged models and complete evaluations are skipped.
#
#   cd <worktree> && STAGE=prep NAME=qwen3-8b MODEL=/path/to/shared/Qwen3-8B FOLD=F1 X_STORE=/path/to/local/dose \
#     setsid nohup bash explore/dose_chain.sh > $ATMOS_ROOT/logs/dose/prep-F1.log 2>&1 < /dev/null &
# DRY=1 prints every command that would run (nothing is built, trained or served).
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
P=$ATMOS_ROOT/verl/.venv/bin/python
say(){ echo "[$(date +%F' '%T)] $*"; }
: "${STAGE:?prep or rl}" "${NAME:?}" "${MODEL:?}" "${FOLD:?}" "${X_STORE:?}"
SEEDS=${SEEDS:-"42 43"}
STEPS=${STEPS:-120}
MAX_HOURS=${MAX_HOURS:-24}
WT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
ST=$X_STORE
KC=$ATMOS_ROOT/experiments/_inputs/kc
PREFIX=${PREFIX:-dose}          # label / data prefix: dose (PREREG_DOSE) or dtext (PREREG_DOSE_TEXT)
BUILDER=${BUILDER:-explore.dose} # the module whose `build` makes the warm-start parquet
DESIGN=${DESIGN:-$WT/results/dose/design.json}
TRAIN_REPO=${TRAIN_REPO:-$ATMOS_REPO}   # where train/sft_qwen3_8b.sh and train/dapo_qwen3_8b.sh are run from
RAFT=${RAFT:-$ATMOS_ROOT/data/sft_raft.parquet}   # the family's own RAFT background (X-10; explore/x10_raft.sh writes data/raft_<NAME>/sft_raft.parquet)
EXPECT=${EXPECT-$ATMOS_ROOT/data/sft_x-$NAME-$FOLD.parquet}   # W_f parquet the --full rebuild must equal; a family without W_f (no X chain) sets EXPECT= and the --full build is only run
OUTX=$ATMOS_ROOT/experiments/_dose/$NAME
LOGS=$ATMOS_ROOT/logs/dose
mkdir -p $LOGS $OUTX $ATMOS_ROOT/logs/x_eval
DRY=${DRY:-}
run(){ if [ -n "$DRY" ]; then echo "DRY $*"; else "$@"; fi; }
log(){ if [ -n "$DRY" ]; then echo /dev/stdout; else echo $1; fi; }   # DRY prints the commands instead of logging them
say "dose chain stage $STAGE $NAME $FOLD prefix $PREFIX builder $BUILDER design $DESIGN seeds '$SEEDS' on $(hostname), code $WT at $(git -C $WT rev-parse --short HEAD), training code $TRAIN_REPO at $(git -C $TRAIN_REPO rev-parse --short HEAD), store $ST"
[ -f "$DESIGN" ] || { say "no $DESIGN"; exit 1; }
if [ -z "$DRY" ]; then
  exec 9> $LOGS/.lock-$(hostname)
  flock -n 9 || { say "another dose chain is running here; not starting"; exit 1; }
  if pgrep -f "[x]_chain.sh|[x]_eval.sh|[x]_passk.sh|[x]_think_queue|[x]_followup_run" >/dev/null; then
    say "an X chain, evaluation or pass@k is running here; not starting"; exit 1; fi
fi
mkdir -p $ST/checkpoints $ST/merged

wait_free(){ [ -n "$DRY" ] && return 0; while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${used:-0}" -lt 4096 ] && return 0; sleep 60; done; }

merge(){  # exp step (as x_chain.sh)
  local exp=$1 step=$2 src=$ST/checkpoints/$1/global_step_$2 M=$ST/merged/$1/global_step_$2
  [ -d "$src/actor" ] && src=$src/actor
  [ -f "$M/config.json" ] && return 0
  [ -n "$DRY" ] && { echo "DRY merge $src -> $M"; return 0; }
  wait_free; mkdir -p $ST/merged/$exp
  (cd $ATMOS_ROOT/verl && CUDA_VISIBLE_DEVICES= $P -m verl.model_merger merge --backend fsdp \
     --local_dir $src --target_dir $M > $LOGS/merge_$exp.log 2>&1) || { say "$exp: merge failed"; return 1; }
  rm -f $ST/checkpoints/$exp/global_step_*/actor/optim_*.pt $ST/checkpoints/$exp/global_step_*/optim_*.pt
  say "$exp merged at step $step"
}

latest(){ cat $ST/checkpoints/$1/latest_checkpointed_iteration.txt 2>/dev/null || echo 0; }

run_guarded(){  # as x_chain.sh
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

train(){  # exp data start seed: x_chain.sh's train() with the default (regime) reward, no overlong shaping
  local exp=$1 data=$2 start=$3 seed=$4 C=$ST/checkpoints/$1
  if ls $ST/merged/$exp/global_step_*/config.json >/dev/null 2>&1; then say "$exp: merged, skipping"; return 0; fi
  if [ -n "$DRY" ]; then
    echo "DRY train $exp: cd $TRAIN_REPO; CKPT_ROOT=$ST/checkpoints SEED=$seed MODEL_PATH=$start DATA=$data REWARD=regime EXP=$exp TOTAL_STEPS=$STEPS TEST_FREQ=20 SAVE_FREQ=5 KEEP_CKPT=1 VAL_FILES=['$data/train_probe.parquet'] VAL_TEMP=0.3 timeout ${MAX_HOURS}h bash train/dapo_qwen3_8b.sh"
    return 0
  fi
  mkdir -p $C
  while [ "$(latest $exp)" -lt "$STEPS" ] && [ ! -f $C/STOPPED ]; do
    local used=$(cat $C/WALL_SECONDS 2>/dev/null || echo 0) tries=$(cat $C/ATTEMPTS 2>/dev/null || echo 0)
    local left=$(( MAX_HOURS * 3600 - used ))
    if [ $left -le 0 ]; then say "$exp: wall-clock cap already used"; echo cap > $C/STOPPED; break; fi
    echo $((tries + 1)) > $C/ATTEMPTS
    say "=== train $exp (seed $seed, from $start, $data, attempt $((tries + 1)), ${left}s left) ==="
    wait_free
    local t0=$(date +%s)
    (cd $TRAIN_REPO && run_guarded $LOGS/$exp.attempt$((tries + 1)).nohup env CKPT_ROOT=$ST/checkpoints SEED=$seed MODEL_PATH=$start DATA=$data REWARD=regime EXP=$exp \
      TOTAL_STEPS=$STEPS TEST_FREQ=20 SAVE_FREQ=5 KEEP_CKPT=1 VAL_FILES="['$data/train_probe.parquet']" VAL_TEMP=0.3 \
      timeout --signal=TERM --kill-after=10m ${left}s bash train/dapo_qwen3_8b.sh)
    local rc=$?
    echo $(( used + $(date +%s) - t0 )) > $C/WALL_SECONDS
    if [ $rc = 124 ] || [ $rc = 137 ]; then
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
    say "$exp attempt exited (rc $rc) at step $(latest $exp); wall clock $(( $(cat $C/WALL_SECONDS) / 60 )) min in total; mean clip ratio $(grep -o 'response_length/clip_ratio:[0-9.]*' $ATMOS_ROOT/logs/$exp.log | awk -F: '{s+=$2; n++} END {if (n) printf "%.3f over %d steps", s/n, n}')"
  done
  local step=$(latest $exp)
  [ "$step" -gt 0 ] || { say "$exp: no checkpoint ($(cat $C/STOPPED 2>/dev/null))"; return 1; }
  merge $exp $step && rm -rf $C/global_step_*
}

warm(){ echo sft-$PREFIX-$NAME-$FOLD-s$1; }
merged(){ ls -d $ST/merged/$1/global_step_* 2>/dev/null | head -1; }

evaluate(){  # label model gpus slot
  local log=$ATMOS_ROOT/logs/x_eval/$1.log
  run env LABEL=$1 MODEL=$2 GPUS=$3 SLOT=$4 bash $WT/explore/x_eval.sh > $(log $log) 2>&1
}

if [ "$STAGE" = prep ]; then
  check=$ATMOS_ROOT/scratch/dose/check-$PREFIX-$NAME-$FOLD.parquet; mkdir -p $(dirname $check)
  if [ ! -f $check.ok ]; then
    say "=== W_f rebuild check $FOLD ==="
    (cd $WT && run env PYTHONPATH=$WT ATMOSCODER_SANDBOX_CONCURRENCY=32 $P -m $BUILDER build --design $DESIGN --fold $FOLD --seed 42 --full \
       --kc $KC --samples $ATMOS_ROOT/experiments/_x/$NAME/warm_samples_$FOLD.jsonl --raft $RAFT \
       ${EXPECT:+--expect $EXPECT} --out $check > $(log $LOGS/check-$NAME-$FOLD.log) 2>&1) \
      || { say "W_f rebuild check FAILED (see $LOGS/check-$NAME-$FOLD.log)"; exit 1; }
    [ -n "$DRY" ] || touch $check.ok
    say "W_f rebuild check passed"
  fi
  for seed in $SEEDS; do
    W=$(warm $seed); DATA=$ATMOS_ROOT/data/sft_$PREFIX-$NAME-$FOLD-s$seed.parquet
    if [ -n "$(merged $W)" ]; then say "$W: merged, skipping"; continue; fi
    if [ ! -f $DATA ]; then
      say "=== dose data $W ==="
      (cd $WT && run env PYTHONPATH=$WT ATMOSCODER_SANDBOX_CONCURRENCY=32 $P -m $BUILDER build --design $DESIGN --fold $FOLD --seed $seed \
         --kc $KC --samples $ATMOS_ROOT/experiments/_x/$NAME/warm_samples_$FOLD.jsonl --raft $RAFT \
         --out $DATA > $(log $LOGS/build-$W.log) 2>&1) || { say "dose build failed"; exit 1; }
    fi
    if [ "$(latest $W)" = 0 ]; then
      say "=== warm-start SFT $W (1 epoch, lr 5e-6, SFT seed 42) ==="
      wait_free
      (cd $TRAIN_REPO && run env CKPT_ROOT=$ST/checkpoints MODEL_PATH=$MODEL SOURCE=$PREFIX-$NAME-$FOLD-s$seed EXP=$W EPOCHS=1 LR=5e-6 SEED=42 \
         bash train/sft_qwen3_8b.sh > $(log $LOGS/$W.nohup) 2>&1)
      say "SFT exited ($?) at step $(latest $W)"
    fi
    merge $W $(latest $W) || exit 1
    [ -n "$DRY" ] || rm -rf $ST/checkpoints/$W/global_step_*
  done
  say "=== warm-start evaluation and pass@64 ==="
  set -- $SEEDS
  W1=$(warm $1); W2=$(warm $2); M1=$(merged $W1); M2=$(merged $W2)
  [ -n "$DRY" ] && { M1=${M1:-$ST/merged/$W1/global_step_N}; M2=${M2:-$ST/merged/$W2/global_step_N}; }
  [ -n "$M1" ] && [ -n "$M2" ] || { say "a warm start is not merged"; exit 1; }
  evaluate $W1 $M1 0,1,2 0 & e1=$!
  evaluate $W2 $M2 3,4,5 1 & e2=$!
  run env GPU=6 PORT=8241 NAME=$W1 MODEL=$M1 OUT=$OUTX/passk_final_outside.$W1.jsonl bash $WT/explore/x_passk.sh > $(log $LOGS/passk-$W1.log) 2>&1 & p1=$!
  run env GPU=7 PORT=8242 NAME=$W2 MODEL=$M2 OUT=$OUTX/passk_final_outside.$W2.jsonl bash $WT/explore/x_passk.sh > $(log $LOGS/passk-$W2.log) 2>&1 & p2=$!
  rc=0; for p in $e1 $e2 $p1 $p2; do wait $p || rc=1; done
  [ -n "$DRY" ] || { tail -1 $ATMOS_ROOT/logs/x_eval/$W1.log; tail -1 $ATMOS_ROOT/logs/x_eval/$W2.log; tail -1 $LOGS/passk-$W1.log; tail -1 $LOGS/passk-$W2.log; }
  say "=== DOSE PREP $NAME $FOLD DONE (rc $rc) ==="
elif [ "$STAGE" = rl ]; then
  for seed in $SEEDS; do
    W=$(warm $seed); WM=$(merged $W)
    [ -n "$DRY" ] && WM=${WM:-$ST/merged/$W/global_step_N}
    [ -n "$WM" ] || { say "$W is not merged; run STAGE=prep first"; exit 1; }
    train $PREFIX-$NAME-$FOLD-WA-s$seed $ATMOS_ROOT/data-x/$FOLD-A $WM $seed
  done
  say "=== endpoint evaluation ==="
  set -- $SEEDS
  E1=$PREFIX-$NAME-$FOLD-WA-s$1; E2=$PREFIX-$NAME-$FOLD-WA-s$2
  M1=$(merged $E1); M2=$(merged $E2)
  [ -n "$DRY" ] && { M1=${M1:-$ST/merged/$E1/global_step_N}; M2=${M2:-$ST/merged/$E2/global_step_N}; }
  pids=()
  [ -n "$M1" ] && { evaluate $E1 $M1 0,1,2,3 0 & pids+=($!); } || say "$E1 has no merged model"
  [ -n "$M2" ] && { evaluate $E2 $M2 4,5,6,7 1 & pids+=($!); } || say "$E2 has no merged model"
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  say "=== DOSE RL $NAME $FOLD DONE (rc $rc) ==="
else
  say "STAGE must be prep or rl"; exit 1
fi
