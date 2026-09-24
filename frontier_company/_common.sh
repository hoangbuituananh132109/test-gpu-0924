#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

load_config() {
  local cfg="${1:-}"
  if [[ -z "$cfg" ]]; then
    echo "Usage: $0 <config.env>" >&2
    exit 2
  fi
  if [[ "$cfg" != /* ]]; then cfg="$PROJECT_ROOT/$cfg"; fi
  [[ -f "$cfg" ]] || { echo "Missing config: $cfg" >&2; exit 2; }
  set -a
  # shellcheck disable=SC1090
  source "$cfg"
  set +a
  export CONFIG_FILE="$cfg"

  export UPSTREAM_ROOT="${UPSTREAM_ROOT:-$PROJECT_ROOT}"
  export RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs}"
  mkdir -p "$RUNS_ROOT"

  # Resolve project-relative asset paths.
  for name in TRAIN_DATASET STUDENT_MODEL TEACHER_MODEL; do
    local val="${!name:-}"
    if [[ -n "$val" && "$val" != /* ]]; then
      printf -v "$name" '%s/%s' "$PROJECT_ROOT" "$val"
      export "$name"
    fi
  done
}

bool_cap() {
  case "${1,,}" in
    1|true|yes|on) echo True ;;
    0|false|no|off) echo False ;;
    *) echo "$1" ;;
  esac
}

record_run_meta() {
  local run_dir="$1"
  mkdir -p "$run_dir"
  cp "$CONFIG_FILE" "$run_dir/config.env"
  if [[ -d "$UPSTREAM_ROOT/.git" ]]; then
    git -C "$UPSTREAM_ROOT" rev-parse HEAD > "$run_dir/upstream_commit.txt" || true
  fi
  {
    date -Is
    uname -a
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    python --version 2>&1 || true
  } > "$run_dir/environment.txt"
}
