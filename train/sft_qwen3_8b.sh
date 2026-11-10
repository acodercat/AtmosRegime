#!/usr/bin/env bash
# Supervised fine-tuning on the same 190 training parents the RL runs used, as the control
# for "would any training on this data produce the same split?".
#
#   source /path/to/workdir/env.sh
#   SOURCE=gold bash train/sft_qwen3_8b.sh
#   SOURCE=raft bash train/sft_qwen3_8b.sh      # needs export_sft --source raft first
#
# Deliberately matched to the RL arm wherever a choice exists, because a difference that
# comes from a hyper-parameter is not evidence about the algorithm:
#   - the same model, the same frozen split, the same 1140 rows;
#   - the same prompt (prompts.py), and a fenced target so engine.extract_code accepts it;
#   - evaluation at the same point: the RL comparison is step 120 of 200, i.e. 60% of the
#     way through the data, so SFT stops at epoch 3 of 5 and is evaluated there.
#
# Tokenisation (PREREG_X amendment X-10): `atmoscoder_rl/sft_dataset.py` renders each (system, user, assistant) row once with
# thinking off and trains only on the tokens after the generation prompt -- exactly the prompt the RL rollout and the
# evaluation use. verl's default per-message tokenisation with `ignore_input_ids_mismatch=True`, used by every SFT run before
# 2026-10-13, trained on the concatenated per-turn pieces, in which Qwen3's assistant turn lacks the empty
# `<think>\n\n</think>\n\n` block the policy's prompt ends with (and Qwen3.5's template refuses single messages outright).
# The mismatch check now stays on, so any difference fails loudly.
#
# Where it cannot match, and why: SFT needs a larger learning rate than RL's 1e-6 (one pass
# of imitation moves far less than 120 steps of policy gradient), so 1e-5 is the standard
# choice for 8B SFT and is what a reviewer would expect. That difference is the price of the
# comparison existing at all, and it is stated in the results rather than hidden.
set -euo pipefail
: "${ATMOS_ROOT:?source env.sh first}"
: "${ATMOS_REPO:?source env.sh first}"

MODEL_PATH=${MODEL_PATH:-/path/to/shared/Qwen3-8B}
SOURCE=${SOURCE:-gold}
EXP=${EXP:-sft-$SOURCE}
EPOCHS=${EPOCHS:-5}
LR=${LR:-1e-5}
DATA=$ATMOS_ROOT/data/sft_$SOURCE.parquet
OUT=${CKPT_ROOT:-$ATMOS_ROOT/checkpoints}/$EXP   # CKPT_ROOT: node-local disk for the X chains (X-10)
export TENSORBOARD_DIR=$ATMOS_ROOT/logs/tensorboard/$EXP

[ -f "$DATA" ] || { echo "missing $DATA — run atmoscoder_rl.export_sft --out-dir $ATMOS_ROOT/data --source $SOURCE"; exit 1; }
mkdir -p "$OUT" "$TENSORBOARD_DIR" "$ATMOS_ROOT/logs"
cd "$ATMOS_ROOT/verl"

# RMPAD=False (PREREG_DOSE_MINISTRAL DM-5/DM-6): verl's padding-free packing gives Ministral 3 a wrong forward; default unchanged.
# torchrun, not plain python: the SFT trainer expects torch.distributed's env:// rendezvous
# (RANK, WORLD_SIZE) to be set by the launcher, unlike main_ppo which brings up Ray itself.
"$ATMOS_ROOT/verl/.venv/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.sft_trainer \
    data.train_files="$DATA" \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=4096 \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=8192 \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=False \
    data.enable_thinking_default=False \
    data.custom_cls.path="$ATMOS_REPO/atmoscoder_rl/sft_dataset.py" \
    data.custom_cls.name=WholeConversationSFTDataset \
    model.path="$MODEL_PATH" \
    model.use_remove_padding=${RMPAD:-True} \
    optim.lr=$LR \
    trainer.project_name=atmoscoder-rl \
    trainer.experiment_name="$EXP" \
    trainer.logger="['console','tensorboard']" \
    trainer.default_local_dir="$OUT" \
    trainer.total_epochs=$EPOCHS \
    trainer.seed=${SEED:-1} \
    trainer.save_freq=${SAVE_FREQ:-100} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    "$@" 2>&1 | tee -a "$ATMOS_ROOT/logs/$EXP.log"
