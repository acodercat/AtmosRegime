#!/bin/bash
# Move every artifact of one X run aside before it is rerun under the same name, so the rerun starts
# from nothing and overwrites, appends to or mixes with none of the old run's files (PREREG_X X-24).
# Run on the machine that holds the run's local store; refuses while any process of the run is alive.
#   bash explore/x_supersede.sh x-qwen3-8b-F1-WA-s43 x24
# Moves (to $ATMOS_ROOT/superseded-<tag>/ and $X_STORE/superseded-<tag>/, same relative paths):
#   local:  checkpoints/<exp> (STOPPED, ATTEMPTS, WALL_SECONDS included), merged/<exp>
#   shared: checkpoints/<exp> (val_generations), logs/tensorboard/<exp>, logs/<exp>.log,
#           logs/x/<exp>.attempt*.nohup, logs/x/merge_<exp>.log, logs/x/reward_diag/<exp>.*.jsonl
set -eu
exp=$1 tag=$2
ATMOS_ROOT=${ATMOS_ROOT:-/path/to/workdir}; ST=${X_STORE:-/path/to/local/x}
case $exp in x-*-s4[0-9]) ;; *) echo "not an X run name: $exp" >&2; exit 1;; esac
if ps -eo args= | grep -q "[t]rainer.experiment_name=$exp "; then echo "$exp is still running" >&2; exit 1; fi
move(){  # root relpath...
  local root=$1; shift
  for rel in "$@"; do
    [ -e "$root/$rel" ] || continue
    local dst=$root/superseded-$tag/$rel
    [ -e "$dst" ] && { echo "already there: $dst" >&2; exit 1; }
    mkdir -p "$(dirname "$dst")"; mv "$root/$rel" "$dst"; echo "moved $root/$rel"
  done
}
move $ST checkpoints/$exp merged/$exp
cd $ATMOS_ROOT
move $ATMOS_ROOT checkpoints/$exp logs/tensorboard/$exp logs/$exp.log logs/x/merge_$exp.log \
  $(ls logs/x/$exp.attempt*.nohup logs/x/reward_diag/$exp.*.jsonl 2>/dev/null)
