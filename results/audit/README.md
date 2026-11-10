# Rollout / training consistency audit

`rollout_consistency.json` — per archived verl RL log under `$ATMOS_ROOT/logs/*.log`: steps, min / median
`training/rollout_actor_probs_pearson_corr`, max / median `rollout_corr/training_ppl / rollout_ppl`, max / median `rollout_corr/kl`,
max `actor/pg_clipfrac`, median `actor/grad_norm`, and the DM-6 gate (pearson ≥ 0.97 and ppl ratio ≤ 1.10 on every step).

    python -m explore.audit_rollout_consistency --logs $ATMOS_ROOT/logs --out results/audit/rollout_consistency.json
    python -m explore.audit_rollout_consistency --steps <one log>        # per-step table

State 2026-10-31 (109 runs): every Qwen3, Qwen3.5, Llama and Granite run has `kl_max` < 0.001. The only runs above the DM-8 gate
(`rollout_corr/kl` ≤ 0.002) are three Ministral 3 smokes made before the fix (`smoke-ministral-rmpadTrue`, `-rmpadFalse`,
`smoke-x-ministral3-8b`); the logs of voided runs were deleted on 2026-10-31 (DM-11, OPD-15).
Cause and fix: `docs/PREREG_DOSE_MINISTRAL.md` DM-4 to DM-10 (verl padding-free packing, then vLLM's YaRN attention factor on the
HF-format text checkpoint).

Serving check for the evaluation-only models (D series, same vLLM 0.24.0; `experiments/d_*/run/*/server.log`): Ministral 3 8B / 14B
were served from the official directories, which vLLM reads in Mistral format (`Resolved architecture: PixtralForConditionalGeneration`;
the params.json `yarn.apply_scale: false` is honoured), so the DM-9 defect does not apply to them. Granite 4.2 8B / 30B resolve to
`GraniteForCausalLM`, the architecture whose RL runs pass this audit. Qwen3.5 uses default RoPE (partial factor 0.25), Qwen3-8B none;
Gemma 4 has no RL run to audit and no YaRN-type scaling in its config.
