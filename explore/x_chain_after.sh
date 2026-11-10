#!/bin/bash
# Start explore/x_chain.sh once another chain on this machine has finished, so the GPUs do not sit idle between them.
#   AFTER_LOG=<chain log to wait on> SKIP="<exp> ..." <x_chain.sh env> setsid nohup bash explore/x_chain_after.sh > <log> 2>&1 < /dev/null &
# AFTER_LOG must contain "X CHAIN ... DONE". SKIP names runs another machine does: each gets a STOPPED marker in this
# machine's store before the chain starts, so the chain skips it (it logs "no checkpoint" and moves on).
# SUPERSEDE names runs to redo from scratch: explore/x_supersede.sh moves their files to superseded-$SUPERSEDE_TAG first (X-24).
set -uo pipefail
ATMOS_ROOT=${ATMOS_ROOT:-/path/to/workdir}; ST=${X_STORE:-$ATMOS_ROOT}
say() { echo "[$(date '+%F %T')] $*"; }
[ -z "${SUPERSEDE:-}" ] || : "${SUPERSEDE_TAG:?SUPERSEDE needs SUPERSEDE_TAG}"
say "waiting for $AFTER_LOG to report DONE on $(hostname -s)"
until grep -aq "X CHAIN .* DONE" "$AFTER_LOG" 2>/dev/null; do sleep 120; done
for exp in ${SKIP:-}; do
  mkdir -p $ST/checkpoints/$exp
  [ -f $ST/checkpoints/$exp/STOPPED ] || echo "split: run on another machine" > $ST/checkpoints/$exp/STOPPED
  say "skip $exp"
done
for exp in ${SUPERSEDE:-}; do
  bash explore/x_supersede.sh $exp $SUPERSEDE_TAG || { say "supersede $exp failed; not starting"; exit 1; }
done
say "starting x_chain.sh (NAME=${NAME:-} FOLDS=${FOLDS:-} SEEDS=${SEEDS:-})"
exec bash explore/x_chain.sh
