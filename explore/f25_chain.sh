#!/usr/bin/env bash
# PREREG_PAPER_ANALYSES §25-A (E5): prolonged, focused RL on Qwen3-8B fold 1 seed 42 — from the A arm's step-120 merged
# model, on the A pool's rows of the 8 target mechanisms (explore/f25_data.py), A's settings except 32 prompts x 32 samples
# per step, mini-batch 8, dynamic-sampling filter off, 4 GPUs; up to 200 steps or 36 h; checkpoint every 20 steps.
# Stages (STAGES, default "data smoke"; the train stage is started only after the smoke has been read):
#   data   build the focused pool on this machine and check the input hashes
#   smoke  2 steps with the production settings (TOTAL_STEPS=2), then stop: read the gate metrics before training
#   train  the run (one same-seed resume after an infrastructure crash), then merge the last checkpoint
#   eval   X's harness on final_targets.json (640 rows) for the endpoint; pass@64 on the 160 outside rows for the start
#          (A step 120) and the endpoint
# Every step's rollouts (prompt, response, score) are dumped to /path/to/local/x/rollouts/<exp>/ (F25-2: the per-step count of
# samples with a non-zero advantage, and their total on the target outside rows, are reported).
# Never calls `ray stop` (114's GPUs 0-3 run another service); stops only its own process groups.
#   cd <worktree> && GPUS=4,5,6,7 STAGES="data smoke" setsid nohup bash explore/f25_chain.sh > $ATMOS_ROOT/logs/f25/chain.log 2>&1 < /dev/null &
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
: "${GPUS:?}"
R=$ATMOS_ROOT; P=$R/verl/.venv/bin/python; EP=$R/eval-venv/bin/python
STAGES=${STAGES:-data smoke}
START=/path/to/local/x/merged/x-qwen3-8b-F1-A-s42/global_step_120
EXP=f25-qwen3-8b-F1-A-s42-ext
CK=/path/to/local/x/checkpoints; MG=/path/to/local/x/merged
DATA=$R/data-x/F1-A-f25
IN=$R/experiments/_inputs/f25; OUT=$R/experiments/_x_f25
STEPS=200; MAX_HOURS=36
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
OVER=(algorithm.filter_groups.enable=False data.train_batch_size=32 data.gen_batch_size=32
      actor_rollout_ref.rollout.n=32 actor_rollout_ref.actor.ppo_mini_batch_size=8
      trainer.n_gpus_per_node=$NGPU trainer.save_freq=20)
SHA_TARGETS=36045f5b8f91b6bdb6364a8fbe296833508374c25dd37fef78ef720e59b2f64d
SHA_OUTSIDE=ded5b2184f01d9ed27fc2cd380d1256a0c6ac323eea5a5e5ee560319884db77b
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1; export PYTHONPATH=$PWD
mkdir -p $R/logs/f25 $OUT
say(){ echo "[$(date +%F' '%T)] $*"; }
wait_free(){ for g in ${GPUS//,/ }; do until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 4096 ]; do sleep 60; done; done; }
latest(){ cat $CK/$1/latest_checkpointed_iteration.txt 2>/dev/null || echo 0; }
stop_group(){ kill -TERM -- -$1 2>/dev/null; sleep 30; kill -KILL -- -$1 2>/dev/null || true; }

run_train(){  # exp total_steps logfile
  local exp=$1 steps=$2 log=$3
  CUDA_VISIBLE_DEVICES=$GPUS CKPT_ROOT=$CK SEED=42 MODEL_PATH=$START DATA=$DATA REWARD=regime EXP=$exp \
    TOTAL_STEPS=$steps TEST_FREQ=20 KEEP_CKPT=1 VAL_FILES="['$DATA/train_probe.parquet']" VAL_TEMP=0.3 \
    setsid timeout --signal=TERM --kill-after=10m ${4:-$((MAX_HOURS * 3600))}s bash train/dapo_qwen3_8b.sh "${OVER[@]}" trainer.rollout_data_dir=/path/to/local/x/rollouts/$exp > $log 2>&1 < /dev/null &
  local pg=$!
  while kill -0 $pg 2>/dev/null; do   # the X chains' bwrap / load watchdog
    b=$(pgrep -c bwrap); l=$(cut -d. -f1 /proc/loadavg)
    if [ "${b:-0}" -gt 300 ] || [ "${l:-0}" -gt 1000 ]; then say "WATCHDOG bwrap=$b load=$l: stopping $exp"; stop_group $pg; return 99; fi
    sleep 20
  done
  wait $pg
}

merge(){  # exp step
  local src=$CK/$1/global_step_$2/actor M=$MG/$1/global_step_$2
  [ -f $M/config.json ] && return 0
  mkdir -p $MG/$1
  (cd $R/verl && CUDA_VISIBLE_DEVICES= $P -m verl.model_merger merge --backend fsdp --local_dir $src --target_dir $M > $R/logs/f25/merge-$1.log 2>&1) \
    && say "$1 merged at step $2" || { say "$1: merge failed"; return 1; }
}

serve(){  # model port -> sets SERVER; returns 0 when up
  CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 setsid $P -m vllm.entrypoints.openai.api_server \
    --model "$1" --served-model-name x-eval --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --data-parallel-size $NGPU --generation-config vllm --host 127.0.0.1 --port $2 > $R/logs/f25/server-$(basename $(dirname $1)).log 2>&1 < /dev/null &
  SERVER=$!
  for _ in $(seq 1 1080); do
    curl -sf "http://127.0.0.1:$2/v1/models" > /dev/null && return 0
    kill -0 $SERVER 2>/dev/null || return 1
    sleep 10
  done
  return 1
}

for stage in $STAGES; do
  case $stage in
  data)
    $P -m explore.f25_data --pool $R/data-x/F1-A --out-pool $DATA --final $R/experiments/_inputs/kc/final.json --out-inputs $IN || { say "data failed"; exit 1; }
    [ "$(sha256sum $IN/final_targets.json | cut -d' ' -f1)" = $SHA_TARGETS ] && [ "$(sha256sum $IN/final_targets_outside.json | cut -d' ' -f1)" = $SHA_OUTSIDE ] \
      || { say "input hashes differ from §25"; exit 1; }
    [ -f $START/config.json ] || { say "no start model $START"; exit 1; }
    say "data ok";;
  smoke)
    wait_free; say "=== smoke (2 steps, production settings) ==="
    rm -rf $CK/$EXP-smoke
    run_train $EXP-smoke 2 $R/logs/f25/smoke.log 7200; say "smoke exited ($?) at step $(latest $EXP-smoke)"
    grep -o "training/rollout_actor_probs_pearson_corr:[-0-9.e]*\|critic/score/mean:[-0-9.e]*\|response_length/clip_ratio:[-0-9.e]*\|perf/max_memory_allocated_gb:[-0-9.e]*\|timing_s/step:[-0-9.e]*" $R/logs/f25/smoke.log | tail -12
    rm -rf $CK/$EXP-smoke; say "SMOKEDONE";;
  train)
    C=$CK/$EXP; mkdir -p $C
    while [ "$(latest $EXP)" -lt $STEPS ] && [ ! -f $C/STOPPED ]; do
      used=$(cat $C/WALL_SECONDS 2>/dev/null || echo 0); tries=$(cat $C/ATTEMPTS 2>/dev/null || echo 0)
      left=$(( MAX_HOURS * 3600 - used ))
      [ $left -le 0 ] && { echo cap > $C/STOPPED; break; }
      echo $((tries + 1)) > $C/ATTEMPTS
      wait_free; say "=== train $EXP attempt $((tries + 1)), ${left}s left, from step $(latest $EXP) ==="
      t0=$(date +%s)
      run_train $EXP $STEPS $R/logs/f25/$EXP.attempt$((tries + 1)).log $left; rc=$?
      echo $(( used + $(date +%s) - t0 )) > $C/WALL_SECONDS
      if [ $rc = 124 ] || [ $rc = 137 ]; then echo cap > $C/STOPPED
      elif [ $rc = 0 ]; then [ "$(latest $EXP)" -ge $STEPS ] || echo early-exit > $C/STOPPED
      else [ $((tries + 1)) -ge 2 ] && echo "crash rc $rc" > $C/STOPPED; sleep 60; fi
      say "$EXP attempt exited (rc $rc) at step $(latest $EXP); wall clock $(( $(cat $C/WALL_SECONDS) / 60 )) min"
    done
    step=$(latest $EXP)
    [ "$step" -gt 0 ] && merge $EXP $step && say "TRAINDONE $EXP step $step ($(cat $C/STOPPED 2>/dev/null || echo complete))";;
  eval)
    END=$(ls -d $MG/$EXP/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
    [ -n "$END" ] || { say "no merged endpoint"; exit 1; }
    for pair in "a120:$START" "ext:$END"; do
      tag=${pair%%:*}; M=${pair#*:}; mkdir -p $OUT/$tag
      wait_free
      serve $M 8271 || { say "server for $tag did not come up"; stop_group $SERVER; continue; }
      say "serving $M as $tag"
      echo "{\"tag\": \"$tag\", \"model\": \"$M\", \"host\": \"$(hostname)\", \"gpus\": \"$GPUS\", \"commit\": \"$(git rev-parse HEAD)\"}" > $OUT/$tag/serving.json
      if [ $tag = ext ]; then
        for pass in 1 2; do
          $EP -m atmoscoder_rl.evaluation.runner --model x-eval-1 --input $IN/final_targets.json --experiments $OUT --exp-id $tag --out-name final_targets \
            --tolerance 0.05 --max-attempts 5 --workers $((16 * NGPU)) | grep -v '^\[' || true
        done
      fi
      $EP -m explore.x2_passk sample --base-url http://127.0.0.1:8271/v1 --model x-eval --input $IN/final_targets_outside.json \
        --variants outside --n 64 --temperature 1.0 --workers 16 --out $OUT/$tag/passk_targets_outside.jsonl
      stop_group $SERVER
      say "EVALDONE $tag"
    done;;
  esac
done
say "F25 CHAIN DONE $(hostname) ($STAGES)"
