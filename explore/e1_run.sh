#!/usr/bin/env bash
# D confirmation has its own frozen two-instrument schedule. Dispatch before the legacy
# server setup so an offline plan neither touches GPUs nor looks in another repository.
# explore/families_run.py was removed in 52c64cf (D finished); to replay this branch, check it out from 52c64cf~1.
if [ "${FAMILIES:-0}" = 1 ]; then
  exec "${FAMILIES_PYTHON:-python3}" -m explore.families_run \
    --input-dir "${FAMILIES_INPUT_DIR:-experiments/_inputs/families_v4}" \
    --record-dir "${FAMILIES_RECORD_DIR:-results/families_v4}" "$@"
fi
# E1 (TODO.md, stage 1, exploratory): the three trap arms and their parents, across model
# size and thinking mode. Inference only; no training.
#
#   cd $ATMOS_REPO && nohup bash explore/e1_run.sh > $ATMOS_ROOT/logs/e1.log 2>&1 &
#   ONLY=qwen3-8b REPS=1 N=8 EXP=explore_e1_smoke bash explore/e1_run.sh   # smoke: 8 items
#   HINT=1 ONLY=qwen3-8b bash explore/e1_run.sh     # E2: same items with the validity hint
#   E3=1 bash explore/e1_run.sh                      # E3: knowledge probes (explore/e3_probe.py)
#   TPL=1 ONLY="qwen3-8b rl-s120" bash explore/e1_run.sh   # regime-template pilot (atmoscoder_rl/templates)
#   FAM=1 bash explore/e1_run.sh      # docs/PREREG_TEMPLATE_SCREENING.md Part 3: every trap family, pinned configs
#   TPLRL=1 bash explore/e1_run.sh    # docs/PREREG_TEMPLATE_RL.md: all 30 templates on the base, RL and random-reward checkpoints
#   GIVEN=1 bash explore/e1_run.sh    # docs/PREREG_GIVEN_VERSIONS.md: regime traps, their replacement-given versions, parents
#   PILOT=1 bash explore/e1_run.sh    # docs/PILOT_DESIGN.md: base, C0 and the two pilot arms on the pilot's evaluation file
#   PILOT=judge bash explore/e1_run.sh  # the same four on the held-out rows with the judgement instruction (1 rep)
#   PILOT=trained bash explore/e1_run.sh  # post hoc: the same four on the trained templates at the evaluation seed (1 rep)
#   PILOT=flag ONLY="qwen3-8b rl-s120" bash explore/e1_run.sh      # flag-only control; MODES=on for thinking
#   PILOT=1 MODES=on ONLY="qwen3-8b rl-s120" bash explore/e1_run.sh  # the pilot file with thinking on
#   E0Q=1 bash explore/e1_run.sh      # the 17 templates that fool deepseek, on base / C0 / C0-think (s60), off and on
#   E0E=1 bash explore/e1_run.sh      # Qwen3.6-27B on the E0 confirmation items (seed 20261002), off and on
#   CURVE=1 bash explore/e1_run.sh    # C0 step curve: base and rl-s20..s200, thinking on, the textbook traps, 3 reps
#   SIZE=1 bash explore/e1_run.sh     # Qwen3 4B/8B/14B/32B + Llama-3.1-8B on the textbook traps, off and on, 3 reps
#   E7T=1 bash explore/e1_run.sh; E7B=1 bash explore/e1_run.sh   # E7 handbook arm (docs/PREREG_E7.md): templates, textbook traps
#
# Resume-safe: the runner (atmoscoder_rl.evaluation) skips every completed id, so re-running continues.
# Results: $ATMOS_REPO/experiments/$EXP/<name>.think-<mode>.run<N>.json
# Uses the models.toml entries local-e1-off, local-e1-on, local-e1-on-32k, local-e4-off
# (model_id "local"), in this repository. ENTRY overrides the entry, MODES restricts the thinking modes (E4: MODES=off).
# EXTRA configs run only when named in ONLY, never by default.
source /path/to/workdir/env.sh >/dev/null 2>&1
set -uo pipefail
unset UV_PYTHON; export UV_PROJECT_ENVIRONMENT=$ATMOS_ROOT/eval-venv   # uv sync --extra eval, Python 3.12
VENV=$ATMOS_ROOT/verl/.venv; export PATH=$VENV/bin:$HOME/.local/bin:$PATH
X=$ATMOS_REPO/experiments
EVAL=(uv run --python 3.12 --extra eval python)   # model calls: the eval venv above, never verl's
M=/path/to/shared
PORT=8077
REPS=${REPS:-3}
ONLY=${ONLY:-}
ENTRY=${ENTRY:-}
MODES=${MODES:-}
HINT=${HINT:-0}
E3=${E3:-0}
TPL=${TPL:-0}
FAM=${FAM:-0}
TPLRL=${TPLRL:-0}
GIVEN=${GIVEN:-0}
PILOT=${PILOT:-0}
TPLRL_SEED=20260928   # docs/PREREG_TEMPLATE_RL.md; fixed at sealing
if [ "$PILOT" != 0 ]; then   # docs/PILOT_DESIGN.md §4: thinking off; 3 reps on the standard prompt, 1 with the judgement
  ONLY=${ONLY:-"qwen3-8b rl-s120 pilot-A-s120 pilot-B-s120"}; MODES=${MODES:-off}
  if [ "$PILOT" = judge ]; then REPS=1; EXP=pilot_judge
  elif [ "$PILOT" = trained ]; then REPS=1; EXP=pilot_trained   # post hoc: the trained templates, new numbers
  elif [ "$PILOT" = flag ]; then REPS=${REPS:-3}; EXP=pilot_flag  # flag-only control (TODO.md 2026-09-29)
  elif [ "$PILOT" = flagtraps ]; then REPS=${REPS:-3}; EXP=pilot_flag_traps  # the same control on the textbook traps
  else REPS=${REPS:-3}; EXP=pilot_eval; fi
fi
if [ "$FAM" = 1 ]; then   # pre-registered: these configurations, thinking off, 3 reps, a fresh experiment id
  ONLY="qwen3-8b rl-s120 seed1-s120 seed2-s120 seed3-s120 phi-4"; MODES=off; REPS=3; EXP=prereg_families
fi
if [ "$GIVEN" = 1 ]; then   # pre-registered: base and the four RL checkpoints, thinking off, 3 reps, a fresh experiment id
  ONLY="qwen3-8b rl-s120 seed1-s120 seed2-s120 seed3-s120"; MODES=off; REPS=3; EXP=prereg_given
fi
if [ "$TPLRL" = 1 ]; then   # pre-registered: these configurations, thinking off, one pass each, a fresh experiment id
  ONLY="qwen3-8b rl-s120 seed1-s120 seed2-s120 seed3-s120 ctrl-random-s120"; MODES=off; REPS=1; EXP=prereg_tpl_rl
fi
E0Q=${E0Q:-0}
# E0d follow-up (TODO.md): the 17 templates that fool deepseek (14 strong, 3 moderate), on Qwen base, C0
# and C0-think at the same step, thinking off and on, one pass. Same ids and texts as e0_templates.
E0Q_NAMES=line_broadening,evaporative_cooling_limit,co2_forcing_log,ch4_forcing_sqrt,clausius_clapeyron_extrapolation,sun_photometer_airmass,displacement_height_canopy,finite_line_source,diffuse_transmission,henry_depletion,deposition_brownian,inertia_gravity_wave,raindrop_drag_regime,drop_ventilation,coriolis_neglect,anelastic_continuity,surface_multiple_reflection
if [ "$E0Q" = 1 ]; then
  ONLY=${ONLY:-"qwen3-8b rl-s60 c0think-s60"}; MODES=${MODES:-"off on"}; REPS=${E0Q_REPS:-1}; EXP=e0_qwen
fi
E0E=${E0E:-0}; CURVE=${CURVE:-0}; E7T=${E7T:-0}; E7B=${E7B:-0}; REWORD=${REWORD:-0}; E7S=${E7S:-0}; E7P=${E7P:-0}; AUDIT=${AUDIT:-0}; E7C=${E7C:-0}; VARP=${VARP:-0}; GATE=${GATE:-0}; SIZE=${SIZE:-0}
if [ "$E0E" = 1 ]; then   # E0e (PROPOSAL §5): a second large model on the E0 confirmation items (seed 20261002)
  ONLY=${ONLY:-qwen3.6-27b}; MODES=${MODES:-"off on"}; REPS=1; EXP=e0_confirm
fi
if [ "$REWORD" = 1 ]; then  # docs/PREREG_REWORD.md: three reworded templates, same parameters as the E0 confirmation
  ONLY=${ONLY:-qwen3.6-27b}; MODES=${MODES:-"off on"}; REPS=1; EXP=e0_reword
fi
if [ "$SIZE" = 1 ]; then   # size curve + second family (TODO.md): every sound trap outside train and its parent
  ONLY=${ONLY:-"qwen3-4b qwen3-8b qwen3-14b qwen3-32b llama3.1-8b"}; MODES=${MODES:-"off on"}; REPS=3; EXP=size_curve
fi
if [ "$GATE" = 1 ]; then   # docs/PREREG_COVERAGE.md gate 4: every template on the 8B base, thinking off, screening seed
  ONLY=${ONLY:-rl-s120}; MODES=${MODES:-off}; REPS=1; EXP=coverage_gate   # C0 since amendment 1 (PREREG_COVERAGE.md)
fi
if [ "$E7C" = 1 ]; then    # PREREG_E7 second addendum: position, selection vs length, BM25 retrieval, one pass
  ONLY=${ONLY:-"qwen3-8b rl-s120"}; MODES=${MODES:-on}; REPS=1; EXP=e7_checks
fi
if [ "$VARP" = 1 ]; then   # PREREG_E7 second addendum, check 3: E4b test-set precision
  ONLY=${ONLY:-"qwen3-8b rl-s120"}; MODES=${MODES:-on}; REPS=2; EXP=e4b_precision
fi
if [ "$AUDIT" = 1 ]; then  # E4b second fidelity audit by a different model (results/e4b/README.md)
  ONLY=${ONLY:-qwen3.6-27b}; MODES=${MODES:-on}; REPS=1; EXP=e4b_audit
fi
if [ "$E7S" = 1 ]; then    # PREREG_E7 addendum: one relevant entry vs 60 vs none vs given, one pass
  ONLY=${ONLY:-"qwen3-8b rl-s120"}; MODES=${MODES:-on}; REPS=1; EXP=e7_single
fi
if [ "$E7P" = 1 ]; then    # PREREG_E7 addendum: textbook none and handbook arms in the same pass
  ONLY=${ONLY:-"qwen3-8b rl-s120"}; MODES=${MODES:-on}; REPS=3; EXP=e7_handbook_traps_paired
fi
if [ "$E7T" = 1 ]; then    # E7 (docs/PREREG_E7.md): handbook arm on the E0 confirmation templates, with their bare rows
  ONLY=${ONLY:-"qwen3-8b rl-s120"}; MODES=${MODES:-"off on"}; REPS=1; EXP=e7_handbook
fi
if [ "$E7B" = 1 ]; then    # E7: handbook arm on the textbook traps and parents (bare arm: pilot_eval)
  ONLY=${ONLY:-"qwen3-8b rl-s120"}; MODES=${MODES:-"off on"}; REPS=3; EXP=e7_handbook_traps
fi
if [ "$CURVE" = 1 ]; then  # E1b (PROPOSAL §5): C0 checkpoints along training, thinking on, every sound trap outside train
  ONLY=${ONLY:-"qwen3-8b rl-s20 rl-s60 rl-s100 rl-s120 rl-s140 rl-s200"}; MODES=${MODES:-on}; REPS=3; EXP=c0_curve
fi
D=$ATMOS_ROOT/data/explore
if [ "$E0Q" = 1 ]; then IN=$D/e0_qwen.json; BUILD="explore.build_screen_input --names $E0Q_NAMES --out $IN"
elif [ "$E0E" = 1 ]; then IN=$D/e0_confirm.json; BUILD="explore.build_screen_input --names $E0Q_NAMES --seed 20261002 --out $IN"
elif [ "$CURVE" = 1 ]; then IN=$D/c0_curve.json; BUILD="explore.build_e1_input --out $IN --families"
elif [ "$REWORD" = 1 ]; then IN=$D/e0_reword.json; BUILD="explore.e0_reword --out $IN"
elif [ "$SIZE" = 1 ]; then IN=$D/size_curve.json; BUILD="explore.build_e1_input --out $IN --families"
elif [ "$GATE" = 1 ]; then IN=$D/coverage_gate.json; BUILD="explore.build_screen_input --names ${GATE_NAMES:-ALL} --seed 20261006 --pairs 20 --out $IN"
elif [ "$E7C" = 1 ]; then IN=$D/e7_checks.json; BUILD="explore.build_e7_checks --templates --out $IN"
elif [ "$VARP" = 1 ]; then IN=$D/e4b_precision.json; BUILD="explore.build_e7_checks --variants --out $IN"
elif [ "$E7S" = 1 ]; then IN=$D/e7_single.json; BUILD="explore.build_e7_input --single --out $IN"
elif [ "$E7P" = 1 ]; then IN=$D/e7_paired.json; BUILD="explore.build_e7_input --families-paired --out $IN"
elif [ "$E7T" = 1 ]; then IN=$D/e7_templates.json; BUILD="explore.build_e7_input --templates --out $IN"
elif [ "$E7B" = 1 ]; then IN=$D/e7_families.json; BUILD="explore.build_e7_input --families --out $IN"
elif [ "$PILOT" != 0 ]; then IN=$D/$EXP.json; BUILD="explore.build_pilot_eval --out-dir $D"
elif [ "$FAM" = 1 ]; then IN=$D/prereg_families.json; BUILD="explore.build_e1_input --out $IN --families"
elif [ "$GIVEN" = 1 ]; then IN=$D/prereg_given.json; BUILD="explore.build_given_input --out $IN"
elif [ "$TPLRL" = 1 ]; then IN=$D/prereg_tpl_rl.json; BUILD="atmoscoder_rl.build_templates --pairs 20 --seed $TPLRL_SEED --out $IN"
elif [ "$TPL" = 1 ]; then IN=$D/tpl_pilot.json; EXP=${EXP:-explore_tpl}; BUILD="atmoscoder_rl.build_templates --pairs ${TPL_N:-10} --out $IN"
elif [ "$HINT" = 1 ]; then IN=$D/e2_input.json; EXP=${EXP:-explore_e2}; BUILD="explore.build_e1_input --out $IN --hint"
else IN=$D/e1_input.json; EXP=${EXP:-explore_e1}; BUILD="explore.build_e1_input --out $IN"; fi
N=${N:-}
say(){ echo "[$(date +%F' '%T)] $*"; }

# name | weights | tensor-parallel | data-parallel | thinking modes the model supports
CONFIGS=(
  "qwen3-8b|$M/Qwen3-8B|1|8|off on"
  "rl-s120|$ATMOS_ROOT/merged/dapo-qwen3-8b/global_step_120|1|8|off on"
  "qwen3-14b|$M/Qwen3-14B|1|8|off on"
  "qwen3-4b-2507|$M/Qwen3-4B-Instruct|1|8|off"
  "qwen3-30b-a3b-2507|$M/Qwen3-30B-A3B-Instruct-2507|2|4|off"
  "llama3.1-8b|$M/Llama-3.1-8B-Instruct|1|8|off"
  "qwen3.5-9b|$M/Qwen3.5-9B|1|8|off on"
  "qwen3.6-27b|$M/Qwen3.6-27B|2|4|off on"
)
# Run only when named in ONLY. Sixth field: options (maxlen=, parser=, entry_on=, entry_off=).
R=$ATMOS_ROOT/merged
EXTRA=(
  "seed1-s120|$R/seed1/global_step_120|1|8|off"
  "seed2-s120|$R/seed2/global_step_120|1|8|off"
  "seed3-s120|$R/seed3/global_step_120|1|8|off"
  "ctrl-random-s120|$R/ctrl-random/global_step_120|1|8|off"
  "pilot-A-s120|$R/pilot-A/global_step_120|1|8|off"
  "pilot-B-s120|$R/pilot-B/global_step_120|1|8|off"
  "pilot-D-s120|$R/pilot-D/global_step_120|1|8|off"
  "pilot-D-s80|$R/pilot-D/global_step_80|1|8|off"     # D was stopped at step 101 (2026-09-30); step 80 is its last checkpoint
  "sft-gold|$R/sft-gold/global_step_175|1|8|off"
  "sft-raft|$R/sft-raft/global_step_165|1|8|off"
  "phi-4|$M/phi-4|1|8|off|maxlen=16384"
  "phi-4-reasoning|$M/Phi-4-reasoning|1|8|on|maxlen=32768 parser=deepseek_r1 entry_on=local-e1-on-32k"
)
W3=$(ls -d $R/sft-warm3/global_step_* 2>/dev/null | head -1)
[ -n "$W3" ] && EXTRA+=("sft-warm3|$W3|1|8|off")
for st in 20 40 60 80 100 140 160 180 200; do EXTRA+=("rl-s$st|$R/dapo-qwen3-8b/global_step_$st|1|8|off on"); done
for st in 20 40 60 80 100 120; do EXTRA+=("c0think-s$st|$R/c0think/global_step_$st|1|8|off on"); done   # docs/THINK_RL.md
for sd in 2 3; do   # docs/PREREG_REPLICATION.md: the pilot arms with more seeds (seed 1 = the original runs)
  EXTRA+=("c0-seed$sd|$R/c0-seed$sd/global_step_120|1|8|off" "pilot-A-seed$sd|$R/pilot-A-seed$sd/global_step_120|1|8|off"
          "pilot-D-seed$sd|$R/pilot-D-seed$sd/global_step_80|1|8|off")
done
EXTRA+=("qwen3-4b|$M/Qwen3-4B|1|8|off on" "qwen3-32b|$M/Qwen3-32B|2|4|off on")   # size curve (TODO.md)

VLLM_PID=
reap_orphans(){  # vLLM processes of ours whose parent is gone (ppid 1): nothing else looks like that
  local o; o=$(ps -u "$(id -un)" -o pid=,ppid=,args= | awk '$2 == 1 && $3 ~ /^VLLM::/ {print $1}')
  [ -n "$o" ] && { say "  reaping orphaned vLLM processes: $o"; kill -KILL $o 2>/dev/null; sleep 5; }
}
wait_gpus_free(){  # up to 15 min for every GPU to drop below 4 GiB used
  # Seen twice: a start fails because one GPU is still held (a straggler from the previous config,
  # or someone else on this shared box) and frees up minutes later. Waiting beats skipping.
  for i in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt 4096 ] && return 0
    [ $i = 1 ] && say "  waiting for GPUs to free up (max used ${used} MiB)"
    sleep 10
  done
  say "  GPUs still busy after 15 min (max used ${used} MiB); trying anyway"
}
start_vllm(){  # weights tp dp mode name maxlen parser
  reap_orphans; wait_gpus_free
  local parser=(); [ "$4" = on ] && parser=(--reasoning-parser $7)
  # setsid: vLLM and its workers get their own process group, so stop_vllm kills exactly them
  setsid $VENV/bin/python -m vllm.entrypoints.openai.api_server --model "$1" \
    --served-model-name local --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size $2 --data-parallel-size $3 --gpu-memory-utilization 0.85 \
    --max-model-len $6 "${parser[@]}" > $ATMOS_ROOT/logs/vllm_e1_$5.log 2>&1 &
  VLLM_PID=$!
  for i in $(seq 1 90); do
    curl -s -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && return 0
    kill -0 $VLLM_PID 2>/dev/null || return 1
    sleep 10
  done
  return 1
}
stop_vllm(){
  if [ -n "$VLLM_PID" ]; then
    kill -TERM -- -$VLLM_PID 2>/dev/null
    for i in $(seq 1 30); do pgrep -g $VLLM_PID >/dev/null || break; sleep 2; done
    # Seen: one worker ignored SIGTERM, outlived the group leader and held 72 GiB for an hour,
    # which made every later start fail. Nothing in the group may survive a stop.
    if pgrep -g $VLLM_PID >/dev/null; then
      say "  vLLM group $VLLM_PID ignored SIGTERM; SIGKILL"; kill -KILL -- -$VLLM_PID 2>/dev/null; sleep 5
    fi
  fi
  for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 3; done
  sleep 5; VLLM_PID=
}
trap stop_vllm EXIT

# Stage-1 results were written under the Bench checkout before this repository ran its own
# evaluations. Resuming here without moving them would silently start every run from scratch.
OLD=/path/to/AtmosCoder-Bench/experiments
XP=$([ "$E3" = 1 ] && echo explore_e3 || echo $EXP)
if [ -d "$OLD/$XP" ] && [ ! -d "$X/$XP" ]; then
  say "refusing: $XP exists under $OLD but not under $X; move it first (docs/IMPORTED_RUNS.md)"; exit 1
fi

[ "$E3" = 1 ] || [ "$AUDIT" = 1 ] || [ "${DRYRUN:-0}" = 1 ] || { cd $ATMOS_REPO && $VENV/bin/python -m $BUILD || exit 1; }

ALL=("${CONFIGS[@]}"); [ -n "$ONLY" ] && ALL+=("${EXTRA[@]}")
for c in "${ALL[@]}"; do
  IFS='|' read -r name weights tp dp modes opts <<< "$c"
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && continue   # ONLY: space-separated names
  maxlen=40960; parser=qwen3; entry_on=local-e1-on; entry_off=local-e1-off
  for kv in $opts; do case $kv in
    maxlen=*) maxlen=${kv#*=};; parser=*) parser=${kv#*=};;
    entry_on=*) entry_on=${kv#*=};; entry_off=*) entry_off=${kv#*=};; esac; done
  [ -n "$MODES" ] && modes=$(for m in $modes; do [[ " $MODES " == *" $m "* ]] && echo -n "$m "; done)
  [ -d "$weights" ] || [ "${DRYRUN:-0}" = 1 ] || { say "$name: no weights at $weights, skipping"; continue; }
  for mode in $modes; do
    tag=$name.think-$mode
    entry=${ENTRY:-$([ $mode = on ] && echo $entry_on || echo $entry_off)}
    if [ "${DRYRUN:-0}" = 1 ]; then echo "would run: $tag entry=$entry maxlen=$maxlen parser=$parser tp=$tp dp=$dp exp=$EXP reps=$REPS"; continue; fi
    say "=== $tag (tp=$tp dp=$dp) ==="
    stop_vllm
    # One retry: a start can fail on a transient cause (seen: EADDRINUSE on vLLM's random
    # distributed-init port), and a skipped config costs a whole arm of the experiment.
    start_vllm "$weights" $tp $dp $mode $tag $maxlen $parser || {
      say "$tag: vLLM failed to start, retrying once"; stop_vllm
      start_vllm "$weights" $tp $dp $mode $tag $maxlen $parser; } || {
      say "$tag: VLLM NOT READY, skipping (see logs/vllm_e1_$tag.log)"; stop_vllm; continue; }
    if [ "$AUDIT" = 1 ]; then
      (cd $ATMOS_REPO && "${EVAL[@]}" -m explore.audit_trap_variants --model $entry --workers 32 \
         --out $X/_inputs/trap_variants_audit_$tag.jsonl) 2>&1 | tail -1 | sed "s/^/  /"
      continue
    fi
    if [ "$E3" = 1 ]; then
      (cd $ATMOS_REPO && "${EVAL[@]}" -m explore.e3_probe --tag $tag --thinking $mode \
         --reps $REPS --out-dir $X/explore_e3) 2>&1 | tail -1 | sed "s/^/  /"
      continue
    fi
    workers=$([ $mode = on ] && echo 64 || echo 32)
    for r in $(seq 1 $REPS); do
      out=$(cd $ATMOS_REPO && "${EVAL[@]}" -m atmoscoder_rl.evaluation.runner --model $entry --input $IN \
            --exp-id $EXP --out-name $tag --run $r --workers $workers ${N:+--sample $N} 2>&1)
      say "  $tag run$r: $(echo "$out" | grep -E 'Accuracy:' | tail -1)"
    done
  done
done
stop_vllm
say "=== E1 DONE ==="
