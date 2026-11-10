#!/usr/bin/env bash
# DAPO on Qwen3-8B over the frozen `train` split, validated on every held-out split.
#
#   source /path/to/workdir/env.sh
#   EXP=dapo-qwen3-8b-run1 bash train/dapo_qwen3_8b.sh          # real run
#   SMOKE=1 bash train/dapo_qwen3_8b.sh                          # 2-step pipeline check
#
# gen_batch_size oversamples the training batch because dynamic sampling throws groups
# away: the smoke run evicted most of them at n=4. More rollouts per prompt means more
# within-group variance and less eviction, so 256 -> 128 is a starting guess;
# training/filter_groups/evicted_samples says whether it needs raising.
#
# Algorithm: GRPO advantages + the two DAPO pieces this problem needs — dynamic sampling
# (`filter_groups`: drop prompt groups whose rollouts all score the same, which on this
# corpus is most of them) and clip-higher. No KL term, no entropy bonus (DAPO). Overlong
# shaping is left off: a solve() is not a long chain of thought.
#
# Thinking is OFF for the policy (`enable_thinking=false` in the chat template): responses
# stay short and the rollout budget goes to samples, not tokens. Flip THINKING=on to train
# the reasoning variant; expect ~4x the response length.
#
# max_prompt_length=2048: the longest held-out prompt (test_book) is 1,973 tokens under the
# Qwen3 template; anything smaller would let filter_overlong_prompts silently drop it.
#
# Every validation set is a separate parquet with its own `data_source`, so verl reports
# val-core_score/atmoscoder/<split> per split — the pre-training row is step 0
# (`val_before_train`), which is the like-for-like baseline for the RL curve.
set -euo pipefail
: "${ATMOS_ROOT:?source env.sh first}"
: "${ATMOS_REPO:?source env.sh first}"

MODEL_PATH=${MODEL_PATH:-/path/to/shared/Qwen3-8B}
# Sandboxes per reward-worker process. reward.num_workers below is 8, so the machine-wide
# ceiling is 8 x 8 = 64 concurrent sandboxes on 224 cores. Uncapped, the first run put
# 20,698 bwrap processes in uninterruptible sleep and wedged the box; see sandbox.py.
export ATMOSCODER_SANDBOX_CONCURRENCY=${ATMOSCODER_SANDBOX_CONCURRENCY:-8}
THINKING=${THINKING:-off}
SMOKE=${SMOKE:-0}
# OVERLONG=1 turns on DAPO's overlong reward shaping. Run 2 (no shaping) showed why it
# exists even for short code answers: the policy learned to reason inside `#` comments,
# responses grew 315 -> 892 tokens in 40 steps, and under greedy validation decoding they
# degenerated into repetition loops that hit the 3072 cap (14 -> 89 of 525 truncated), each
# an unclosed fence handed to the interpreter as a solve() with no return. Nothing in a
# 0/1 outcome reward pushes back on that; a linear penalty over the last OVERLONG_LEN
# tokens before the cap does. VAL_TEMP samples validation at the evaluation runner's 0.3
# instead of greedy, which is what exposes the loops.
# REWARD=random swaps in the information-free control reward (verl_reward_random.py).
# Everything else stays byte-identical, which is the point: the two arms differ only in
# what the gradient is told.
# REWARD=regime is the training pilot's reward (verl_reward_regime.py, docs/PILOT_DESIGN.md): the
# answer reward on unlabelled rows, plus the judgement reward on rows whose truth carries a label.
# Both pilot arms use it with their own DATA (export_verl --pilot-arm A|B); a DATA dir holding the
# held-out template files validates on them too. SAVE_FREQ / TEST_FREQ / KEEP_CKPT keep the pilot's
# disk use and validation time down; none of them changes what is trained.
# REWARD=think (docs/THINK_RL.md): the answer reward applied to the text after </think>, for a policy
# trained with THINKING=on; a response whose thinking never closes has no answer and scores 0.
# RESP_LEN overrides max_response_length (thinking-on responses run 7-8k tokens at the median).
REWARD=${REWARD:-real}
RESP_LEN=${RESP_LEN:-}
OVERLONG=${OVERLONG:-0}
OVERLONG_LEN=${OVERLONG_LEN:-1024}
VAL_TEMP=${VAL_TEMP:-0}
EXP=${EXP:-$([ "$SMOKE" = 1 ] && echo smoke || echo dapo-qwen3-8b)}
DATA=${DATA:-$ATMOS_ROOT/data}   # DATA=.../data-targeted for the pre-registered targeted run
OUT=${CKPT_ROOT:-$ATMOS_ROOT/checkpoints}/$EXP   # CKPT_ROOT: node-local disk for the X chains (X-10)
VAL_OUT=$ATMOS_ROOT/checkpoints/$EXP/val_generations   # always on the shared disk
export TENSORBOARD_DIR=$ATMOS_ROOT/logs/tensorboard/$EXP

if [ "$SMOKE" = 1 ]; then
    train_batch_size=16;  gen_batch_size=24;  rollout_n=4;  ppo_mini_batch_size=8
    total_training_steps=2; test_freq=1; save_freq=-1; max_response_length=1024
    val_files="['$DATA/test_id.parquet']"
else
    train_batch_size=128; gen_batch_size=256; rollout_n=8;  ppo_mini_batch_size=32
    total_training_steps=${TOTAL_STEPS:-200}; test_freq=${TEST_FREQ:-10}; save_freq=${SAVE_FREQ:-10}; max_response_length=${RESP_LEN:-3072}
    val_files="'$DATA/test_id.parquet','$DATA/test_core_only.parquet','$DATA/test_book.parquet','$DATA/probe_trap.parquet','$DATA/probe_trap_traps.parquet','$DATA/probe_scaffolding.parquet','$DATA/probe_cross_domain.parquet'"
    for f in probe_tpl_heldout probe_tpl_heldout_judge; do
        [ -f "$DATA/$f.parquet" ] && val_files="$val_files,'$DATA/$f.parquet'"
    done
    val_files="[$val_files]"
fi
# VAL_FILES (PREREG_X amendment X-5): an explicit validation list replaces the default one above.
[ -n "${VAL_FILES:-}" ] && val_files="$VAL_FILES"
case $REWARD in
    real) reward_file=verl_reward.py;; random) reward_file=verl_reward_random.py;;
    regime) reward_file=verl_reward_regime.py;; think) reward_file=verl_reward_think.py;;
    *) echo "unknown REWARD=$REWARD"; exit 1;;
esac
enable_thinking=$([ "$THINKING" = on ] && echo true || echo false)
# RMPAD=False (PREREG_DOSE_MINISTRAL DM-5/DM-6): verl's padding-free packing gives Ministral 3 a wrong training-side
# forward (rollout/actor pearson 0.57 at step 1); the default is unchanged for every other model.
# PPO_MAX_TOKENS (DM-7): per-GPU token budget of every micro-batch (log-prob passes inherit it). Without remove-padding a
# micro-batch is padded to its longest row, so Ministral's 131k-vocabulary logits ran out of memory at 24,576; 8,192 fits.
# Only memory changes: token-mean aggregation over the same mini-batch gives the same gradient.
EXTRA=()
if [ "$OVERLONG" = 1 ]; then
    EXTRA+=(reward.reward_manager.name=dapo
            +reward.reward_kwargs.overlong_buffer_cfg.enable=True
            +reward.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG_LEN
            +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
            +reward.reward_kwargs.overlong_buffer_cfg.log=True
            +reward.reward_kwargs.max_resp_len=$max_response_length)
fi
[ -n "${KEEP_CKPT:-}" ] && EXTRA+=(trainer.max_actor_ckpt_to_keep=$KEEP_CKPT)
# MAX_MODEL_LEN: cap the rollout engine's sequence length for models whose config advertises a huge context (Ministral 3:
# 262,144 -> vLLM wants 34 GiB of KV cache for one request and refuses to start). 5120 = prompt 2048 + response 3072.
[ -n "${MAX_MODEL_LEN:-}" ] && EXTRA+=(actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN)
# PREREG_X X-14: X-chain runs (CKPT_ROOT set) save every 5 steps, so a run the X-13 guard stops before step 20 still
# has a checkpoint. Chains started before X-14 pass SAVE_FREQ=20; this applies from their next run.
[ -n "${CKPT_ROOT:-}" ] && [ "$SMOKE" != 1 ] && save_freq=5
# SEED (docs/PREREG_REPLICATION.md): data shuffle, rollout sampling and actor data loading. Unset = the
# configuration every earlier run used (verl defaults: data.seed null, the others 42).
[ -n "${SEED:-}" ] && EXTRA+=(data.seed=$SEED actor_rollout_ref.rollout.seed=$SEED
                              actor_rollout_ref.actor.data_loader_seed=$SEED)
# PREREG_X X-12: models with gated-delta-net layers (Qwen3.5) hung vLLM 0.24's rollout engine three times
# ("RPC call to sample_tokens timed out") on the FlashInfer 0.6.12 GDN prefill kernel; use vLLM's Triton
# GDN prefill for them. GDN_PREFILL=flashinfer restores the old path. Models without GDN layers are unchanged.
if grep -q '"linear_attention"' "$MODEL_PATH/config.json" 2>/dev/null; then
    EXTRA+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=${GDN_PREFILL:-triton})
fi
if [ "$VAL_TEMP" != 0 ]; then
    EXTRA+=(actor_rollout_ref.rollout.val_kwargs.do_sample=True
            actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMP)
fi

mkdir -p "$OUT" "$VAL_OUT" "$TENSORBOARD_DIR" "$ATMOS_ROOT/logs"
# VERL_DIR (PREREG_OPD): run the trainer from another verl checkout (the patched worktree) with the same venv; the caller
# also puts it first on PYTHONPATH so Ray workers import the same tree. Unset = the shared checkout, as every earlier run.
cd "${VERL_DIR:-$ATMOS_ROOT/verl}"
"$ATMOS_ROOT/verl/.venv/bin/python" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=score \
    algorithm.filter_groups.max_num_gen_batches=10 \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$val_files" \
    data.train_batch_size=$train_batch_size \
    data.gen_batch_size=$gen_batch_size \
    data.max_prompt_length=2048 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    "+data.apply_chat_template_kwargs.enable_thinking=$enable_thinking" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=${RMPAD:-True} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKENS:-24576} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$rollout_n \
    reward.custom_reward_function.path="$ATMOS_REPO/atmoscoder_rl/$reward_file" \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers=8 \
    trainer.project_name=atmoscoder-rl \
    trainer.experiment_name="$EXP" \
    trainer.logger="['console','tensorboard']" \
    trainer.default_local_dir="$OUT" \
    trainer.validation_data_dir="$VAL_OUT" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=$test_freq \
    trainer.save_freq=$save_freq \
    trainer.total_training_steps=$total_training_steps \
    "${EXTRA[@]}" "$@" 2>&1 | tee -a "$ATMOS_ROOT/logs/$EXP.log"
