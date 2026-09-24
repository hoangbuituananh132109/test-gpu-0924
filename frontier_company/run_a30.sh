#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
CONFIG_PATH="${1:-$SCRIPT_DIR/config_a30_4gpu_smoke.env}"
EXP_ID="${2:-E1_grpo_base}"
load_config "$CONFIG_PATH"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline WANDB_DISABLED=true DO_NOT_TRACK=1
export PYTHONPATH="$UPSTREAM_ROOT/verl:$UPSTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VAL_FILES_OVERRIDE="$VAL_DATASET"

if [[ "${ATTN_IMPLEMENTATION:-auto}" == auto ]]; then
  if python -c 'import flash_attn' >/dev/null 2>&1; then
    export ATTN_IMPLEMENTATION=flash_attention_2 USE_REMOVE_PADDING=true
  else
    export ATTN_IMPLEMENTATION=sdpa USE_REMOVE_PADDING=false
  fi
fi

bash "$SCRIPT_DIR/preflight_a30.sh" "$CONFIG_PATH"
echo "[$(date -Is)] A30 attention=$ATTN_IMPLEMENTATION remove_padding=$USE_REMOVE_PADDING"
case "$EXP_ID" in
  E1_grpo_base) exec bash "$SCRIPT_DIR/launch_verl.sh" "$CONFIG_PATH" "$EXP_ID" grpo false ;;
  E2_opd_base) exec bash "$SCRIPT_DIR/launch_verl.sh" "$CONFIG_PATH" "$EXP_ID" token_reward_direct true ;;
  S0_hybrid_upstream_sanity|E3_hybrid_static|E4_hybrid_dynamic|E5_hybrid_dynamic_queue)
    exec bash "$SCRIPT_DIR/launch_verl.sh" "$CONFIG_PATH" "$EXP_ID" token_reward_direct_plus_grpo true ;;
  *) echo "Unsupported experiment: $EXP_ID" >&2; exit 2 ;;
esac
