#!/usr/bin/env bash
# Generic launcher for E1/E2/E3 and the patched E4. Designed against the public
# command surface of Thinking-Space/Rethinking-OPD. Must be GPU-validated on the
# exact upstream commit before expensive runs.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config.env> <experiment-id> [adv-estimator] [teacher-enable]" >&2
  exit 2
fi
load_config "$1"
EXP_ID="$2"
ADV_ESTIMATOR="${3:-grpo}"
TEACHER_ENABLE="${4:-false}"

case "$EXP_ID" in
  E1_grpo_base|E2_opd_base|S0_hybrid_upstream_sanity|E3_hybrid_static|E4_hybrid_dynamic|E5_hybrid_dynamic_queue) ;;
  *) echo "Unknown/unsafe experiment id: $EXP_ID" >&2; exit 2 ;;
esac

[[ -d "$UPSTREAM_ROOT" ]] || { echo "Missing upstream: $UPSTREAM_ROOT" >&2; exit 1; }
[[ -f "$TRAIN_DATASET" ]] || { echo "Missing train parquet: $TRAIN_DATASET" >&2; exit 1; }
[[ -d "$STUDENT_MODEL" ]] || { echo "Missing student: $STUDENT_MODEL" >&2; exit 1; }
if [[ "$TEACHER_ENABLE" == "true" ]]; then
  [[ -d "$TEACHER_MODEL" ]] || { echo "Missing teacher: $TEACHER_MODEL" >&2; exit 1; }
fi

if [[ "$EXP_ID" == "S0_hybrid_upstream_sanity" || "$EXP_ID" == "E3_hybrid_static" ]]; then
  marker="$UPSTREAM_ROOT/.frontier_static_patch_applied"
  [[ -f "$marker" ]] || {
    echo "Refusing $EXP_ID: shape-safe two-branch patch marker missing: $marker" >&2
    echo "Run scripts/apply_static_patch.sh only after inspecting/testing the pinned upstream." >&2
    exit 3
  }
fi

if [[ "$EXP_ID" == "E4_hybrid_dynamic" || "$EXP_ID" == "E5_hybrid_dynamic_queue" ]]; then
  marker="$UPSTREAM_ROOT/.frontier_dynamic_patch_applied"
  [[ -f "$marker" ]] || {
    echo "Refusing E4: dynamic upstream integration marker missing: $marker" >&2
    echo "Read patches/CODEX_INTEGRATION.md; patch/test upstream, save diff, then create marker." >&2
    exit 3
  }
fi

if [[ "$EXP_ID" == "E5_hybrid_dynamic_queue" ]]; then
  marker="$UPSTREAM_ROOT/.frontier_queue_patch_applied"
  [[ -f "$marker" ]] || {
    echo "Refusing E5: concrete queue integration marker missing: $marker" >&2
    exit 3
  }
fi
if [[ "${ROLLOUT_REPEAT_INTERLEAVE:-true}" == "false" ]]; then
  marker="$UPSTREAM_ROOT/.frontier_balanced_rollout_patch_applied"
  [[ -f "$marker" ]] || {
    echo "Refusing balanced rollout: verified patch marker missing: $marker" >&2
    echo "Run scripts/apply_balanced_rollout_patch.sh first." >&2
    exit 3
  }
fi

RUN_TAG="${EXP_ID}_${TRAIN_VARIANT:-data}_seed${SEED:-42}_$(date +%Y%m%d_%H%M%S)"
if [[ -n "${RUN_DIR_OVERRIDE:-}" ]]; then
  RUN_DIR="$(readlink -f "$RUN_DIR_OVERRIDE")"
  RUNS_ROOT_REAL="$(readlink -f "$RUNS_ROOT")"
  case "$RUN_DIR" in
    "$RUNS_ROOT_REAL"/"${EXP_ID}"_*) ;;
    *) echo "Unsafe/mismatched RUN_DIR_OVERRIDE for $EXP_ID: $RUN_DIR" >&2; exit 4 ;;
  esac
  [[ -d "$RUN_DIR" && -f "$RUN_DIR/config.env" ]] || {
    echo "Resume directory is missing its recorded config: $RUN_DIR" >&2
    exit 4
  }
  cmp -s "$CONFIG_FILE" "$RUN_DIR/config.env" || {
    echo "Refusing resume because the recorded config differs: $RUN_DIR/config.env" >&2
    exit 4
  }
  RUN_TAG="$(basename "$RUN_DIR")"
  printf '%s resume_requested\n' "$(date -Is)" >> "$RUN_DIR/resume_events.log"
else
  RUN_DIR="$RUNS_ROOT/$RUN_TAG"
  record_run_meta "$RUN_DIR"
fi

export PYTHONUNBUFFERED=1
export PROJECT_NAME="${PROJECT_NAME:-FrontierGRPOOPD}"
export PYTHONPATH="$UPSTREAM_ROOT/verl:$UPSTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Company cluster runs are deliberately offline. Missing assets must fail in
# preflight instead of triggering an implicit Hub/network download.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export WANDB_DISABLED=true
export DO_NOT_TRACK=1
export RAY_memory_usage_threshold=0.99
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED="${SEED:-42}"
export FRONTIER_ROLLOUT_REPEAT_INTERLEAVE="$(bool_cap "${ROLLOUT_REPEAT_INTERLEAVE:-true}")"
export FRONTIER_TEACHER_ENTROPY_CHUNK_SIZE="${FRONTIER_TEACHER_ENTROPY_CHUNK_SIZE:-4096}"
# vLLM sleep mode uses a CUDA memory pool that is incompatible with PyTorch's
# expandable_segments allocator. Respect an explicit allocator config, but do
# not enable expandable segments by default.
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF
fi

MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH ))
if (( MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH > MAX_MODEL_LEN )); then
  MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH ))
fi

TEACHER_PATH="$TEACHER_MODEL"
if [[ "$TEACHER_ENABLE" != "true" ]]; then TEACHER_PATH="$STUDENT_MODEL"; fi

# E1 never consumes top-k candidates. Avoid the expensive discarded full-vocab
# top-k/log-softmax path without changing rollout samples, rewards, or GRPO loss.
EFFECTIVE_LOG_PROB_TOP_K="$LOG_PROB_TOP_K"
if [[ "$EXP_ID" == "E1_grpo_base" && "$TEACHER_ENABLE" != "true" ]]; then
  EFFECTIVE_LOG_PROB_TOP_K=0
fi

VAL_FILES="${VAL_FILES_OVERRIDE:-['$UPSTREAM_ROOT/datasets/test_data/AIME25/test.parquet','$UPSTREAM_ROOT/datasets/test_data/AMC23/test.parquet','$UPSTREAM_ROOT/datasets/test_data/AIME24/test.parquet']}"
REWARD_PATH="$UPSTREAM_ROOT/verl/verl/utils/reward_score/ttrl_math/__init__.py"

# Static upstream hybrid already exposes grpo_outcome_weight. Dynamic E4 adds
# extra config keys consumed only by our patch.
EXTRA_ARGS=()
if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then
  EXTRA_ARGS+=(
    "+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
    "+actor_rollout_ref.ref.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
    "+reward_model.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
  )
fi
if [[ "$(bool_cap "${FRONTIER_GRAD_DIAG_ENABLE:-false}")" == "True" ]]; then
  marker="$UPSTREAM_ROOT/.frontier_grad_interaction_diagnostics_v1"
  [[ -f "$marker" ]] || {
    echo "Refusing gradient diagnostics: verified patch marker missing: $marker" >&2
    exit 3
  }
  EXTRA_ARGS+=(
    "+actor_rollout_ref.actor.frontier_grad_diagnostics.enable=True"
    "+actor_rollout_ref.actor.frontier_grad_diagnostics.interval=${FRONTIER_GRAD_DIAG_INTERVAL:-16}"
    "+actor_rollout_ref.actor.frontier_grad_diagnostics.first_step=${FRONTIER_GRAD_DIAG_FIRST_STEP:--1}"
    "+actor_rollout_ref.actor.frontier_grad_diagnostics.min_param_numel=${FRONTIER_GRAD_DIAG_MIN_PARAM_NUMEL:-100000}"
    "+actor_rollout_ref.actor.frontier_grad_diagnostics.max_param_numel=${FRONTIER_GRAD_DIAG_MAX_PARAM_NUMEL:-80000000}"
  )
fi
if [[ "$(bool_cap "${FRONTIER_FUSED_TOPK_ACTOR:-false}")" == "True" ]]; then
  EXTRA_ARGS+=(
    "actor_rollout_ref.model.use_fused_kernels=True"
    "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch"
    "actor_rollout_ref.actor.use_fused_kernels=True"
    "reward_model.model.use_fused_kernels=False"
  )
fi
if [[ "$EXP_ID" == "E3_hybrid_static" || "$EXP_ID" == "E4_hybrid_dynamic" || "$EXP_ID" == "E5_hybrid_dynamic_queue" ]]; then
  EXTRA_ARGS+=("+algorithm.frontier_static.normalize_branches=True")
fi
if [[ "$EXP_ID" == "E4_hybrid_dynamic" || "$EXP_ID" == "E5_hybrid_dynamic_queue" ]]; then
  EXTRA_ARGS+=(
    "+algorithm.frontier.enable=True"
    "+algorithm.frontier.grpo_gamma=${FRONTIER_GRPO_GAMMA:-1.0}"
    "+algorithm.frontier.grpo_min=${FRONTIER_GRPO_MIN:-0.0}"
    "+algorithm.frontier.grpo_max=${FRONTIER_GRPO_MAX:-1.0}"
    "+algorithm.frontier.opd_tau_c=${FRONTIER_OPD_TAU_C:-0.5}"
    "+algorithm.frontier.opd_kappa=${FRONTIER_OPD_KAPPA:-0.1}"
    "+algorithm.frontier.opd_min=${FRONTIER_OPD_MIN:-0.0}"
    "+algorithm.frontier.opd_max=${FRONTIER_OPD_MAX:-1.0}"
  )
fi
if [[ "$EXP_ID" == "E5_hybrid_dynamic_queue" ]]; then
  EXTRA_ARGS+=(
    "+algorithm.frontier_queue.enable=True"
    "data.dataloader_num_workers=0"
    "data.sampler.class_path=${QUEUE_SAMPLER_CLASS_PATH:-pkg://frontier.verl_queue_sampler}"
    "data.sampler.class_name=${QUEUE_SAMPLER_CLASS_NAME:-FrontierQueueSampler}"
    "+data.sampler.max_total_revisits=${QUEUE_MAX_TOTAL_REVISITS:-128}"
    "+data.sampler.state_path=$RUN_DIR/queue_state.json"
    "+data.sampler.events_path=$RUN_DIR/queue_events.jsonl"
  )
fi
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
  EXTRA_ARGS+=("trainer.total_training_steps=$TOTAL_TRAINING_STEPS")
fi
ACTOR_PARAM_OFFLOAD_H=$(bool_cap "${ACTOR_PARAM_OFFLOAD:-false}")
ACTOR_OPT_OFFLOAD_H=$(bool_cap "${ACTOR_OPTIMIZER_OFFLOAD:-false}")
ACTOR_FSDP_STRATEGY_H="${ACTOR_FSDP_STRATEGY:-fsdp}"
ACTOR_FSDP_OFFLOAD_POLICY_H=$(bool_cap "${ACTOR_FSDP_OFFLOAD_POLICY:-false}")
TEACHER_PARAM_OFFLOAD_H=$(bool_cap "${TEACHER_PARAM_OFFLOAD:-false}")
REWARD_FSDP_STRATEGY_H="${REWARD_FSDP_STRATEGY:-fsdp}"
REF_PARAM_OFFLOAD_H=$(bool_cap "${REF_PARAM_OFFLOAD:-true}")
GRAD_CKPT_H=$(bool_cap "${ENABLE_GRADIENT_CHECKPOINTING:-true}")
ACT_OFFLOAD_H=$(bool_cap "${ENABLE_ACTIVATION_OFFLOAD:-true}")
TEACHER_ENABLE_H=$(bool_cap "$TEACHER_ENABLE")
ENABLE_THINKING_H=$(bool_cap "${ENABLE_THINKING:-false}")
REMOVE_PADDING_H=$(bool_cap "${USE_REMOVE_PADDING:-true}")

cd "$UPSTREAM_ROOT"
if [[ "$(bool_cap "${FRONTIER_MANAGE_RAY:-true}")" == "True" ]]; then
  # Never stop a shared Ray cluster from the A30 profile.
  ray stop --force >/dev/null 2>&1 || true
  ray start --head >/dev/null
  sleep 3
fi

telemetry_pid=""
cleanup_telemetry() {
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap cleanup_telemetry EXIT INT TERM
if [[ "$(bool_cap "${ENABLE_GPU_TELEMETRY:-false}")" == "True" ]]; then
  bash "$PROJECT_ROOT/frontier_company/sample_gpu_telemetry.sh" \
    "$RUN_DIR/gpu_telemetry" "${GPU_TELEMETRY_INTERVAL:-1}" &
  telemetry_pid=$!
fi

CMD=(python3 -m verl.trainer.main_ppo
  "algorithm.adv_estimator=$ADV_ESTIMATOR"
  "algorithm.grpo_outcome_weight=${GRPO_OUTCOME_WEIGHT:-1.0}"
  "data.shuffle=False"
  "data.seed=${SEED:-42}"
  "data.train_files=$TRAIN_DATASET"
  "data.val_files=$VAL_FILES"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESP_LENGTH"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "+data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING_H"
  "actor_rollout_ref.model.path=$STUDENT_MODEL"
  "actor_rollout_ref.model.use_remove_padding=$REMOVE_PADDING_H"
  "actor_rollout_ref.model.enable_activation_offload=$ACT_OFFLOAD_H"
  "actor_rollout_ref.model.enable_gradient_checkpointing=$GRAD_CKPT_H"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS:-1}"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
  "actor_rollout_ref.actor.strategy=$ACTOR_FSDP_STRATEGY_H"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD_H"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPT_OFFLOAD_H"
  "actor_rollout_ref.actor.fsdp_config.offload_policy=$ACTOR_FSDP_OFFLOAD_POLICY_H"
  "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU"
  "actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD_H"
  "actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.temperature=$TEMPERATURE"
  "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-1.0}"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "+actor_rollout_ref.rollout.log_prob_top_k=$EFFECTIVE_LOG_PROB_TOP_K"
  "+actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY"
  "+actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE"
  "+actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.n=$N_RESPONSES"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "+actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH"
  "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N"
  "actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE"
  "actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P"
  "actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY"
  "actor_rollout_ref.rollout.calculate_log_probs=True"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "reward_model.enable=$TEACHER_ENABLE_H"
  "reward_model.strategy=$REWARD_FSDP_STRATEGY_H"
  "+reward_model.reward_kwargs.enable_format_reward=False"
  "reward_model.model.path=$TEACHER_PATH"
  "reward_model.model.input_tokenizer=null"
  "reward_model.model.use_remove_padding=$REMOVE_PADDING_H"
  "reward_model.model.fsdp_config.param_offload=$TEACHER_PARAM_OFFLOAD_H"
  "+reward_model.model.dtype=$MODEL_DTYPE"
  "reward_model.micro_batch_size_per_gpu=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "custom_reward_function.path=$REWARD_PATH"
  "custom_reward_function.name=reward_func"
  "trainer.val_before_train=False"
  "trainer.logger=['console']"
  "trainer.project_name=$PROJECT_NAME"
  "trainer.experiment_name=$RUN_TAG"
  "trainer.n_gpus_per_node=$N_GPUS"
  "trainer.nnodes=1"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2}"
  "trainer.resume_mode=auto"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.total_epochs=$TOTAL_EPOCHS"
  "trainer.default_local_dir=$RUN_DIR/checkpoint"
  "trainer.is_plot=False"
  "${EXTRA_ARGS[@]}"
)

printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
echo "RUN_DIR=$RUN_DIR"
echo "Estimator=$ADV_ESTIMATOR teacher_enable=$TEACHER_ENABLE"
echo "Effective log_prob_top_k=$EFFECTIVE_LOG_PROB_TOP_K"

set +e
if [[ -x /usr/bin/time ]]; then
  /usr/bin/time -v "${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"
  rc=${PIPESTATUS[0]}
else
  start_epoch=$(date +%s)
  "${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"
  rc=${PIPESTATUS[0]}
  end_epoch=$(date +%s)
  echo "launcher_wall_seconds=$((end_epoch - start_epoch))" | tee -a "$RUN_DIR/train.log"
fi
set -e
cleanup_telemetry
telemetry_pid=""
if [[ "$(bool_cap "${FRONTIER_MANAGE_RAY:-true}")" == "True" ]]; then
  ray stop --force >/dev/null 2>&1 || true
fi

echo "$rc" > "$RUN_DIR/exit_code.txt"
exit "$rc"
