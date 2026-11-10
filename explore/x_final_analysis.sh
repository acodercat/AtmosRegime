#!/usr/bin/env bash
# PREREG_X after the seed-43 and R-arm follow-up (X-28, X-29): every CPU-only read, in one place, into one directory.
#   bash explore/x_final_analysis.sh                       # refuses while a chain or follow-up queue still runs
#   FORCE=1 OUT=$ATMOS_ROOT/scratch/dry bash explore/x_final_analysis.sh   # rehearsal on incomplete data
# Checks first (X-25, X-30), all before any accuracy is read:
# 1. every seed-43, -b2 and R evaluation complete (build_x_eval_input --check-run);
# 2. evaluation execution failures re-executed twice (no accuracy is read);
# 3. training runs: step signature, persistent exec_error, validation exec_error rows dumped to val_exec.jsonl
#    (regrade those locally: python -m explore.x_run_checks regrade --rows val_exec.jsonl --out regrade.json);
# then the reads, in X-29 section 1's order:
# 4. x_followup per family (batch check first, then the R readings), with the family's seed-42 report for P1's Z;
# 5. x_report: Qwen3-8B seeds 42 43 as before; Qwen3.5-9B seed 42 (unchanged reference) and seeds 42 43, the X-28
#    replication, which X-29 section 1 lets stand only if its batch verdict is "no batch effect" (printed at the end);
# 6. X-29 section 3 overlong trigger on Qwen3.5 F2-F seed 43 (seed 42 alongside as the reference);
# 7. X-26 curve sources manifest for s42, s43 and R runs (export runs locally from the manifest).
# Nothing here writes into results/; the outputs are read and reviewed before anything is committed.
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
R=${ATMOS_ROOT:-/path/to/workdir}; E=$R/experiments
OUT=${OUT:-$R/scratch/final-$(date +%m%d-%H%M)}
PY="env PYTHONPATH=. $R/eval-venv/bin/python"
say(){ echo "[$(date '+%F %T')] $*"; }
# the code beside this script (env.sh pins ATMOS_REPO to the shared checkout, so a clone must not rely on it)
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
if [ -z "${FORCE:-}" ] && pgrep -f "[x]_chain.sh|[x]_followup_run" >/dev/null; then
  say "a chain or follow-up queue is still running on $(hostname -s); FORCE=1 to rehearse anyway"; exit 1; fi
mkdir -p $OUT
say "final analysis at $(git log --oneline -1) -> $OUT"
step(){ local name=$1; shift; say "$name"; "$@" > $OUT/$name.log 2>&1; local rc=$?; say "$name rc $rc"; tail -8 $OUT/$name.log; }
COMMON="--eval-dir $E/_x_eval --final $E/_inputs/kc/final.json --inputs $E/_inputs/x_eval"

# the 18 new labels: 10 Qwen3.5 seed-43 endpoints, 4 -b2 starts (X-29 section 1), 4 R endpoints (section 2)
LABELS="$(for f in F1 F2; do for a in A D F WA WD; do echo x-qwen35-9b-$f-$a-s43; done; done)
x-qwen35-9b-base-b2 sft-x-qwen35-9b-F1-b2 sft-x-qwen35-9b-F2-b2 x-qwen3-8b-base-b2
$(for n in qwen3-8b qwen35-9b; do for f in F1 F2; do echo x-$n-$f-R-s42; done; done)"
check_runs(){ local bad=0 lab out
  for lab in $LABELS; do
    out=$($PY -m explore.build_x_eval_input --final $E/_inputs/kc/final.json --out-dir $E/_inputs/x_eval \
          --check-run $E/_x_eval/$lab 2>&1) || bad=1
    echo "$lab: $out"
  done
  echo "$(echo $LABELS | wc -w) labels, $([ $bad = 0 ] && echo all complete || echo SOME INCOMPLETE)"; return $bad; }
step check_run check_runs
step exec_recheck $PY -m explore.x_eval_exec_recheck --root $E/_x_eval --out $OUT/exec_recheck.json
step run_checks $PY -m explore.x_run_checks scan --root $R --dump-val $OUT/val_exec.jsonl --out $OUT/run_checks.json

step report-qwen35-9b-s42 $PY -m explore.x_report --name qwen35-9b --seeds 42 $COMMON \
  --passk $E/_x/qwen35-9b/passk_final_outside.x18.jsonl --passk-sens $E/_x/qwen35-9b/passk_final_outside.sens.jsonl --out $OUT/report_qwen35-9b_s42.json
step report-qwen3-8b $PY -m explore.x_report --name qwen3-8b --seeds 42 43 $COMMON \
  --passk $E/_x/qwen3-8b/passk_final_outside.x18.jsonl --passk-sens $E/_x/qwen3-8b/passk_final_outside.sens.jsonl --out $OUT/report_qwen3-8b.json
step followup-qwen3-8b $PY -m explore.x_followup --name qwen3-8b --seeds 42 43 $COMMON \
  --report $OUT/report_qwen3-8b.json --out $OUT/followup_qwen3-8b.json
step followup-qwen35-9b $PY -m explore.x_followup --name qwen35-9b --seeds 42 43 $COMMON \
  --report $OUT/report_qwen35-9b_s42.json --out $OUT/followup_qwen35-9b.json
step report-qwen35-9b-s42-43 $PY -m explore.x_report --name qwen35-9b --seeds 42 43 $COMMON \
  --passk $E/_x/qwen35-9b/passk_final_outside.x18.jsonl --passk-sens $E/_x/qwen35-9b/passk_final_outside.sens.jsonl --out $OUT/report_qwen35-9b.json

step overlong $PY -m explore.x_overlong_check --logs $R/logs --report $OUT/report_qwen35-9b.json \
  --runs x-qwen35-9b-F2-F-s43 x-qwen35-9b-F2-F-s42 --out $OUT/overlong.json
step curve_sources $PY -m explore.x_curve_sources --logs $R/logs --glob 'x-*-s4[23]' --out $OUT/x_sources.json
for n in qwen3-8b qwen35-9b; do
  v=$($PY -c "import json; print(json.load(open('$OUT/followup_$n.json'))['batch']['verdict'])" 2>/dev/null)
  say "$n batch verdict: ${v:-missing}"
done
say "X-29 §1: Qwen3.5 seeds 42 43 (report_qwen35-9b.json) stands only if its verdict is 无批次效应; otherwise read report_qwen35-9b_s42.json and the -b2 comparison"
say "ALL DONE"
