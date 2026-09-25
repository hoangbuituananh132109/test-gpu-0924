#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
load_config "${1:-$SCRIPT_DIR/config_a30_4gpu_smoke.env}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline WANDB_DISABLED=true DO_NOT_TRACK=1
export PYTHONPATH="$UPSTREAM_ROOT/verl:$UPSTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for path in "$UPSTREAM_ROOT/verl" "$TRAIN_DATASET" "$VAL_DATASET" "$STUDENT_MODEL" "$TEACHER_MODEL"; do
  [[ -e "$path" ]] || { echo "MISSING: $path" >&2; exit 1; }
  echo "OK: $path"
done
command -v nvidia-smi >/dev/null || { echo 'MISSING: nvidia-smi' >&2; exit 1; }
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#gpu_names[@]} == N_GPUS )) || { echo "Expected $N_GPUS visible GPUs, got ${#gpu_names[@]}" >&2; exit 1; }
for name in "${gpu_names[@]}"; do
  [[ "$name" == *A30* ]] || { echo "Expected A30, got $name" >&2; exit 1; }
done
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
nvidia-smi topo -m
mkdir -p "$RUNS_ROOT"
[[ -w "$RUNS_ROOT" ]] || { echo "Not writable: $RUNS_ROOT" >&2; exit 1; }

python - "$TRAIN_DATASET" "$VAL_DATASET" "$STUDENT_MODEL" "$TEACHER_MODEL" "$EXPECTED_TRAIN_ROWS" <<'PY'
import importlib.util
import sys
from pathlib import Path

train, val, student, teacher, expected = sys.argv[1:]
required = ('torch', 'transformers', 'pyarrow', 'vllm', 'verl', 'math_verify', 'latex2sympy2_extended')
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('Missing local packages: ' + ', '.join(missing))

import pyarrow.parquet as pq
import torch
import transformers
import vllm
from transformers import AutoConfig, AutoTokenizer
from frontier_company.check_pair_tokenizer import check_compatible

print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('transformers', transformers.__version__, 'vllm', vllm.__version__)
if torch.cuda.device_count() != 4:
    raise SystemExit(f'Expected 4 CUDA devices, got {torch.cuda.device_count()}')
required_columns = {'data_source', 'prompt', 'reward_model'}
for label, raw in [('train', train), ('val', val)]:
    meta = pq.read_metadata(raw)
    missing_columns = required_columns - set(pq.read_schema(raw).names)
    if missing_columns:
        raise SystemExit(f'{label} missing columns: {sorted(missing_columns)}')
    print(label, 'rows', meta.num_rows)
    if label == 'train' and int(expected) > 0 and meta.num_rows != int(expected):
        raise SystemExit(f'Expected {expected} train rows, got {meta.num_rows}')
tokenizers = {}
for label, raw in [('student', student), ('teacher', teacher)]:
    path = Path(raw)
    for file in ('config.json', 'tokenizer_config.json'):
        if not (path / file).is_file():
            raise SystemExit(f'{label} missing {file}: {path}')
    if not list(path.glob('*.safetensors')):
        raise SystemExit(f'{label} missing safetensors: {path}')
    cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    tokenizers[label] = tok
    print(label, cfg.model_type, 'vocab', len(tok))
check_compatible(tokenizers['student'], tokenizers['teacher'])
print('TOKENIZER_COMPATIBLE')
print('A30_OFFLINE_PREFLIGHT_OK')
PY
