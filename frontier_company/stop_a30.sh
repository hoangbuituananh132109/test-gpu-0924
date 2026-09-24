#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
CONFIG_PATH="${1:-$SCRIPT_DIR/config_a30_4gpu_smoke.env}"
EXP_ID="${2:-E1_grpo_base}"
load_config "$CONFIG_PATH"
PID_FILE="$RUNS_ROOT/control/${EXP_ID}.pid"
[[ -s "$PID_FILE" ]] || { echo "No PID file: $PID_FILE" >&2; exit 1; }
run_pid="$(tr -d '[:space:]' < "$PID_FILE")"
[[ "$run_pid" =~ ^[0-9]+$ ]] || { echo "Invalid PID file" >&2; exit 1; }
kill -0 "$run_pid" 2>/dev/null || { echo "PID $run_pid is not active"; exit 0; }
cmdline="$(tr '\0' ' ' < "/proc/$run_pid/cmdline")"
if [[ "$cmdline" != *"frontier_company/run_a30.sh"* ]]; then
  echo "Refusing to stop unrelated PID $run_pid: $cmdline" >&2
  exit 3
fi
kill -TERM -- "-$run_pid"
echo "Sent TERM to verified A30 process group $run_pid ($EXP_ID)"
