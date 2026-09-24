#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
CONFIG_PATH="${1:-$SCRIPT_DIR/config_b200_8gpu.env}"
EXP_ID="${2:-E5_hybrid_dynamic_queue}"
load_config "$CONFIG_PATH"

CONTROL_DIR="$RUNS_ROOT/control"
mkdir -p "$CONTROL_DIR"
PID_FILE="$CONTROL_DIR/${EXP_ID}.pid"
LOG_FILE="$CONTROL_DIR/${EXP_ID}.launcher.log"

if [[ -s "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Refusing duplicate launch: PID $old_pid is active ($PID_FILE)" >&2
    exit 3
  fi
fi

setsid bash "$SCRIPT_DIR/run_b200.sh" "$CONFIG_PATH" "$EXP_ID" \
  >"$LOG_FILE" 2>&1 < /dev/null &
run_pid=$!
printf '%s\n' "$run_pid" > "$PID_FILE"
sleep 3
if ! kill -0 "$run_pid" 2>/dev/null; then
  echo "Launch exited early; inspect $LOG_FILE" >&2
  tail -80 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "STARTED experiment=$EXP_ID pid=$run_pid"
echo "PID_FILE=$PID_FILE"
echo "LOG_FILE=$LOG_FILE"
tail -20 "$LOG_FILE" || true
