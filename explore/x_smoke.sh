#!/usr/bin/env bash
# PREREG_X section 3: family-2 technical smoke (2 DAPO steps on fold-F1 arm-A data, training-side only).
#   MODEL=/path/to/shared/Qwen3.5-9B NAME=qwen35-9b GPUS=0,1,2,3 [ARM=A|D|F] bash explore/x_smoke.sh
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
say(){ echo "[$(date +%F' '%T)] $*"; }
IFS=, read -ra G <<< "$GPUS"; N=${#G[@]}
say "waiting for GPUs $GPUS to be free"
while true; do
  busy=0
  for g in "${G[@]}"; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "${u:-99999}" -gt 4096 ] && busy=1; done
  [ $busy = 0 ] && break; sleep 60
done
say "=== smoke $NAME arm ${ARM:-A} on $N GPUs ==="
HERE=$(cd "$(dirname "$0")/.." && pwd)   # this checkout's dapo script (MAX_MODEL_LEN-aware), not the shared $ATMOS_REPO one
cd $ATMOS_REPO
CUDA_VISIBLE_DEVICES=$GPUS SMOKE=1 MODEL_PATH=$MODEL DATA=$ATMOS_ROOT/data-x/F1-${ARM:-A} REWARD=regime EXP=smoke-x-$NAME \
  bash $HERE/train/dapo_qwen3_8b.sh trainer.n_gpus_per_node=$N > $ATMOS_ROOT/logs/x/smoke-$NAME.nohup 2>&1
say "smoke exited $? ; steps: $(grep -c 'step:' $ATMOS_ROOT/logs/smoke-x-$NAME.log 2>/dev/null)"
grep -o "step:[0-9]*.*reward[^ ]* [0-9.]*" $ATMOS_ROOT/logs/smoke-x-$NAME.log | tail -3
echo SMOKEDONE
