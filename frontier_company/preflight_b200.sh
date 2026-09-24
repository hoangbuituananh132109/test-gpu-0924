#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
CONFIG_PATH="${1:-$SCRIPT_DIR/config_b200_8gpu.env}"
load_config "$CONFIG_PATH"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export WANDB_DISABLED=true
export DO_NOT_TRACK=1
export PYTHONPATH="$UPSTREAM_ROOT/verl:$UPSTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

fail=0
for path in "$UPSTREAM_ROOT" "$TRAIN_DATASET" "$VAL_DATASET" "$STUDENT_MODEL" "$TEACHER_MODEL"; do
  if [[ -e "$path" ]]; then
    echo "OK: $path"
  else
    echo "MISSING: $path" >&2
    fail=1
  fi
done

command -v nvidia-smi >/dev/null 2>&1 || { echo "MISSING: nvidia-smi" >&2; fail=1; }
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  if (( ${#gpu_names[@]} < N_GPUS )); then
    echo "Need $N_GPUS GPUs, found ${#gpu_names[@]}" >&2
    fail=1
  fi
  for ((i=0; i<N_GPUS && i<${#gpu_names[@]}; i++)); do
    echo "GPU $i: ${gpu_names[$i]}"
    if [[ "${gpu_names[$i]}" != *B200* ]]; then
      echo "GPU $i is not B200" >&2
      fail=1
    fi
  done
  nvidia-smi topo -m || true
fi

mkdir -p "$RUNS_ROOT"
[[ -w "$RUNS_ROOT" ]] || { echo "RUNS_ROOT is not writable: $RUNS_ROOT" >&2; fail=1; }

if (( fail )); then
  exit 1
fi

python - "$TRAIN_DATASET" "$VAL_DATASET" "$STUDENT_MODEL" "$TEACHER_MODEL" "$EXPECTED_TRAIN_ROWS" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

train_path, val_path, student_path, teacher_path, expected_rows = sys.argv[1:]
required = ["torch", "transformers", "pyarrow", "vllm", "verl", "math_verify", "latex2sympy2_extended"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))

import pyarrow.parquet as pq
import torch
import transformers
import vllm
from transformers import AutoConfig, AutoTokenizer

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("cuda_devices", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print("cuda", index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))

required_columns = {"data_source", "prompt", "reward_model"}
for label, raw_path in (("train", train_path), ("val", val_path)):
    path = Path(raw_path)
    metadata = pq.read_metadata(path)
    columns = set(pq.read_schema(path).names)
    absent = sorted(required_columns - columns)
    if absent:
        raise SystemExit(f"{label} parquet missing columns: {absent}")
    print(label, "rows", metadata.num_rows, "columns", sorted(columns))
    if label == "train" and int(expected_rows) > 0 and metadata.num_rows != int(expected_rows):
        raise SystemExit(f"expected {expected_rows} train rows, found {metadata.num_rows}")

for label, raw_path in (("student", student_path), ("teacher", teacher_path)):
    path = Path(raw_path)
    for filename in ("config.json", "tokenizer_config.json"):
        if not (path / filename).is_file():
            raise SystemExit(f"{label} missing {filename}: {path}")
    if not list(path.glob("*.safetensors")):
        raise SystemExit(f"{label} has no safetensors weights: {path}")
    cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    print(label, cfg.model_type, getattr(cfg, "num_hidden_layers", None), len(tok))

print("B200_OFFLINE_PREFLIGHT_OK")
PY
