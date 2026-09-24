#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
CONFIG_PATH="${1:-$SCRIPT_DIR/config_a30_4gpu_smoke.env}"
EXP_ID="${2:-E1_grpo_base}"
load_config "$CONFIG_PATH"
PID_FILE="$RUNS_ROOT/control/${EXP_ID}.pid"
LOG_FILE="$RUNS_ROOT/control/${EXP_ID}.launcher.log"
if [[ -s "$PID_FILE" ]]; then
  run_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$run_pid" =~ ^[0-9]+$ ]] && kill -0 "$run_pid" 2>/dev/null; then
    echo "RUNNING experiment=$EXP_ID pid=$run_pid"
  else
    echo "NOT_RUNNING experiment=$EXP_ID recorded_pid=$run_pid"
  fi
else
  echo "NO_PID_FILE experiment=$EXP_ID"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
[[ -f "$LOG_FILE" ]] && tail -60 "$LOG_FILE"
