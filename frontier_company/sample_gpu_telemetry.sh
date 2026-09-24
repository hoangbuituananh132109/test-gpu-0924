#!/usr/bin/env bash
# Low-overhead NVIDIA telemetry sampler for systems-only profiling.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output-prefix> [interval-seconds]" >&2
  exit 2
fi

OUT_PREFIX="$1"
INTERVAL="${2:-1}"
GPU_OUT="${OUT_PREFIX}_gpu.csv"
PROCESS_OUT="${OUT_PREFIX}_process.log"

mkdir -p "$(dirname "$OUT_PREFIX")"
printf '%s\n' \
  'epoch_s,index,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c,sm_clock_mhz,memory_clock_mhz,pstate' \
  > "$GPU_OUT"
printf '%s\n' '# epoch_s followed by one nvidia-smi pmon sample (-s um)' > "$PROCESS_OUT"

while :; do
  epoch_s="$(date +%s.%N)"
  while IFS= read -r row; do
    printf '%s,%s\n' "$epoch_s" "$row" >> "$GPU_OUT"
  done < <(
    nvidia-smi \
      --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem,pstate \
      --format=csv,noheader,nounits
  )
  printf '# %s\n' "$epoch_s" >> "$PROCESS_OUT"
  nvidia-smi pmon -c 1 -s um >> "$PROCESS_OUT" 2>&1 || true
  sleep "$INTERVAL"
done
