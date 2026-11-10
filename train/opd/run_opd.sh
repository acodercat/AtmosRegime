#!/usr/bin/env bash
# PREREG_OPD: one OPD training run (or its 2-step smoke) on top of train/dapo_qwen3_8b.sh — the A arm's settings plus the
# distillation block, dynamic sampling off, the student on 4 GPUs and the teacher on its own pool.
#
#   source $ATMOS_ROOT/env.sh
#   ARM=P FOLD=F1 SEED=42 GPUS=0,1,2,3,7 bash train/opd/run_opd.sh            # full run (36 h cap)
#   ARM=P FOLD=F1 SEED=42 GPUS=0,1,2,3,7 SMOKE=1 bash train/opd/run_opd.sh    # 2 steps
#   DRY=1 ... prints the command only
#
# ARM   P    privileged self-teacher (Qwen3-8B, teacher sees the given twin on outside rows), forward_kl_topk
#            (the pinned verl 5b79827b has no forward_kl_topk_tail; OPD-1 in docs/PREREG_OPD.md)
#       S    same teacher without privilege (data-opd/<fold>-self), forward_kl_topk
#       Pk3  privileged self-teacher, sampled-token k3
#       T    external teacher (TEACHER=<path>, TEACHER_TP=2), no privilege, forward_kl_topk
#       Pk2  privileged self-teacher, sampled-token k2 = (log-ratio)^2 / 2 with verl's per-token loss clamp switched off
#            (OPD-13: removes the clamp that zeroes Pk3's gradient on the most informative tokens)
#       Pshuf privileged self-teacher shown ANOTHER template's relation sentence (data-opd/<fold>-privshuf), forward_kl_topk
# The trainer runs from the patched verl worktree ($VERL_DIR, train/opd/patch_verl_privileged.py) with the shared venv; the
# worktree is put first on PYTHONPATH so Ray workers import the same tree. ATMOS_OPD_PRIVILEGED=1 makes a run without any
# teacher_prompt fail instead of silently degrading (P, Pk3 only).
set -euo pipefail
: "${ATMOS_ROOT:?source env.sh first}" "${ATMOS_REPO:?source env.sh first}"
: "${ARM:?P|S|Pk3|T|Pk2|Pshuf}" "${FOLD:?F1|F2}" "${GPUS:?comma-separated GPU ids: 4 for the student + teacher pool}"
SEED=${SEED:-42}; SMOKE=${SMOKE:-0}; STEPS=${STEPS:-120}; MAX_HOURS=${MAX_HOURS:-36}
VERL_DIR=${VERL_DIR:-$ATMOS_ROOT/scratch/verl-opd}
STUDENT=${STUDENT:-/path/to/shared/Qwen3-8B}
CLAMP=()
case $ARM in
    P)   DATA=$ATMOS_ROOT/data-opd/$FOLD-priv; TEACHER=${TEACHER:-$STUDENT}; TEACHER_TP=${TEACHER_TP:-1}; LOSS=forward_kl_topk; PRIV=1;;
    S)   DATA=$ATMOS_ROOT/data-opd/$FOLD-self; TEACHER=${TEACHER:-$STUDENT}; TEACHER_TP=${TEACHER_TP:-1}; LOSS=forward_kl_topk; PRIV=0;;
    Pk3) DATA=$ATMOS_ROOT/data-opd/$FOLD-priv; TEACHER=${TEACHER:-$STUDENT}; TEACHER_TP=${TEACHER_TP:-1}; LOSS=k3; PRIV=1;;
    T)   DATA=$ATMOS_ROOT/data-opd/$FOLD-self; : "${TEACHER:?T needs TEACHER=<path>}"; TEACHER_TP=${TEACHER_TP:-2}; LOSS=forward_kl_topk; PRIV=0;;
    Pk2) DATA=$ATMOS_ROOT/data-opd/$FOLD-priv; TEACHER=${TEACHER:-$STUDENT}; TEACHER_TP=${TEACHER_TP:-1}; LOSS=k2; PRIV=1
         CLAMP=(distillation.distillation_loss.loss_max_clamp=null);;
    Pshuf) DATA=$ATMOS_ROOT/data-opd/$FOLD-privshuf; TEACHER=${TEACHER:-$STUDENT}; TEACHER_TP=${TEACHER_TP:-1}; LOSS=forward_kl_topk; PRIV=1;;
    *) echo "unknown ARM=$ARM"; exit 1;;
esac
IFS=, read -ra G <<< "$GPUS"; NG=${#G[@]}; TEACHER_GPUS=$TEACHER_TP; STUDENT_GPUS=$((NG - TEACHER_GPUS))
[ "$STUDENT_GPUS" = 4 ] || { echo "need exactly 4 student GPUs (got $STUDENT_GPUS of $NG with teacher TP $TEACHER_TP)"; exit 1; }
EXP=${EXP:-$([ "$SMOKE" = 1 ] && echo smoke-opd-qwen3-8b-$FOLD-$ARM || echo opd-qwen3-8b-$FOLD-$ARM-s$SEED)}
grep -q "ATMOSCODER privileged" "$VERL_DIR/verl/experimental/agent_loop/agent_loop.py" || { echo "$VERL_DIR is not patched"; exit 1; }
test -f "$DATA/train.parquet" || { echo "no $DATA/train.parquet"; exit 1; }
gen_batch=$([ "$SMOKE" = 1 ] && echo 16 || echo 128)
# PPO_MAX_TOKENS (OPD-3): the top-k loss keeps full-vocabulary logits per micro-batch; at 24,576 tokens per GPU the backward
# pass ran out of memory on the 4-GPU student (70 GiB in use, +12.5 GiB requested). 8,192 cuts the micro-batch 3x; the
# mini-batch, optimiser steps and token-mean aggregation are unchanged.
PPO_MAX_TOKENS=${PPO_MAX_TOKENS:-8192}
OVER=(algorithm.filter_groups.enable=False data.gen_batch_size=$gen_batch trainer.n_gpus_per_node=$STUDENT_GPUS
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKENS
      distillation.enabled=True distillation.n_gpus_per_node=$TEACHER_GPUS distillation.nnodes=1
      distillation.teacher_models.teacher_model.model_path="$TEACHER"
      distillation.teacher_models.teacher_model.inference.name=vllm
      distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP
      distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.8
      distillation.teacher_models.teacher_model.inference.max_model_len=6144
      distillation.distillation_loss.loss_mode=$LOSS distillation.distillation_loss.topk=32
      distillation.distillation_loss.use_task_rewards=True distillation.distillation_loss.use_policy_gradient=False
      "${CLAMP[@]}")
ENV=(CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}" VERL_DIR="$VERL_DIR" ATMOS_OPD_PRIVILEGED=$PRIV ATMOS_OPD_DIAG="$ATMOS_ROOT/logs/opd/diag-$EXP.jsonl"
     SMOKE=$SMOKE SEED=$SEED MODEL_PATH="$STUDENT" DATA="$DATA" REWARD=regime EXP="$EXP" TOTAL_STEPS=$STEPS TEST_FREQ=20 SAVE_FREQ=5 KEEP_CKPT=1
     VAL_FILES="['$DATA/train_probe.parquet']" VAL_TEMP=0.3 CKPT_ROOT=${CKPT_ROOT:-/path/to/local/opd/checkpoints})
echo "[$(date +%F' '%T)] OPD $EXP: arm $ARM fold $FOLD seed $SEED student $STUDENT_GPUS GPUs teacher $TEACHER (TP $TEACHER_TP) loss $LOSS priv $PRIV data $DATA"
[ "${DRY:-0}" = 1 ] && { echo "env ${ENV[*]}"; echo "overrides ${OVER[*]}"; exit 0; }
mkdir -p "${CKPT_ROOT:-/path/to/local/opd/checkpoints}" "$ATMOS_ROOT/logs/opd"
HERE=$(cd "$(dirname "$0")" && pwd)   # the dapo script of this checkout (VERL_DIR-aware), not the shared $ATMOS_REPO one
env "${ENV[@]}" timeout --signal=TERM --kill-after=10m ${MAX_HOURS}h bash "$HERE/../dapo_qwen3_8b.sh" "${OVER[@]}"
echo "[$(date +%F' '%T)] OPD $EXP exited $?"
