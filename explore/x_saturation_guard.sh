#!/bin/bash
# PREREG_X X-13: end a run as dynamic-sampling saturated when its global step has not advanced for LIMIT seconds.
# One per machine, beside explore/x_chain.sh. It writes the run's STOPPED (reason "saturated") first, so the chain
# does not resume it, then sends TERM to that run's process group only; the chain merges the last checkpoint.
#   setsid nohup bash explore/x_saturation_guard.sh > $ATMOS_ROOT/logs/x/guard-$(hostname -s).log 2>&1 < /dev/null &
ATMOS_ROOT=${ATMOS_ROOT:-/path/to/workdir}
L=$ATMOS_ROOT/logs/x; ST=${X_STORE:-/path/to/local/x}; LIMIT=${LIMIT:-7200}; POLL=${POLL:-300}
declare -A seen since
say() { echo "[$(date '+%F %T')] $*"; }
say "guard on $(hostname -s): limit ${LIMIT}s, poll ${POLL}s"
while true; do
  while read -r pgid exp; do
    log=$(ls -t $L/$exp.attempt*.nohup 2>/dev/null | head -1)
    step=$(grep -ao 'training/global_step:[0-9]*' "$log" 2>/dev/null | tail -1 | cut -d: -f2)
    [ -n "$step" ] || continue                        # still starting up: never judged
    # the metrics line reaches the log in batches, hours late; a saved checkpoint is also progress
    ck=$(cat $ST/checkpoints/$exp/latest_checkpointed_iteration.txt 2>/dev/null)
    [ -n "$ck" ] && [ "$ck" -gt "$step" ] 2>/dev/null && step=$ck
    key="$exp/$pgid"
    if [ "${seen[$key]}" != "$step" ]; then seen[$key]=$step; since[$key]=$(date +%s); continue; fi
    idle=$(( $(date +%s) - ${since[$key]} ))
    if [ $idle -gt $LIMIT ] && [ ! -f $ST/checkpoints/$exp/STOPPED ]; then
      echo "saturated: global step $step did not advance for ${idle}s (X-13 limit ${LIMIT}s)" > $ST/checkpoints/$exp/STOPPED
      say "$exp: step $step unchanged for ${idle}s; STOPPED written, TERM to process group $pgid"
      kill -TERM -- -"$pgid"
    fi
  done < <(ps -eo pgid=,args= | grep '[m]ain_ppo' | sed -n 's/^ *\([0-9]*\) .*trainer.experiment_name=\([^ ]*\).*/\1 \2/p' | sort -u)
  sleep $POLL
done
