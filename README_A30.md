# Offline 4×A30 runner (code snapshot)

This private code-only snapshot is meant to be downloaded manually to
`/nlp/anhhbt/test-gpu-0924` on the company machine. It contains a patched
`verl` tree, `frontier` algorithms and a conservative 4×A30 profile. No
models, datasets, checkpoints or credentials are included. Do not use the
older B200 scripts on A30.

The 4×A30 profile starts with the previously used Qwen3-1.7B-Instruct student
and Qwen3-4B-Base-GRPO teacher. It does **not** imply that 4B→8B/14B/32B
fits 24 GB per GPU. Those are separate exploratory profiles after a memory
probe. See [PLAN_A30.md](frontier_company/PLAN_A30.md).

## Offline handoff

On a machine that has GitHub access, download the private branch as a ZIP.
Move the ZIP to the company machine by your approved transfer method and
extract it under `/nlp/anhhbt/test-gpu-0924`. The company machine must not
fetch the repo, models or packages from the public network.

Place verified local model directories and a DAPO **training** parquet under
`/shared-storage/models`. To create the fixed 1,024-prompt pilot, use only
an approved local DAPO train parquet:

```bash
cd /nlp/anhhbt/test-gpu-0924
python3 frontier_company/make_pilot_subset.py \
  /shared-storage/models/dapo_processed_train.parquet \
  /shared-storage/models/dapo_processed_pilot_1024.parquet --rows 1024 --seed 42
```

If the source filename differs, use its actual local path; never use
AIME/AMC/MATH500 evaluation data. Keep and log the output SHA-256. Do not
recreate the subset between comparison arms.

First inspect the existing runtime; do not install anything automatically:

```bash
cd /nlp/anhhbt/test-gpu-0924
bash frontier_company/preflight_a30.sh frontier_company/config_a30_4gpu_smoke.env
```

Preflight checks exactly four visible A30s, topology, local packages, local
model files/tokenizers and the fixed train parquet. It fails before creating
Ray workers. If the existing environment lacks `verl` import, first try:

```bash
PYTHONPATH="$PWD/verl:$PWD" python3 -c 'import verl.trainer.main_ppo, frontier'
```

Only if required and approved, install the already-local checkout without
dependency downloads: `python3 -m pip install -e ./verl --no-deps --no-build-isolation`.
Any missing dependencies must be supplied through the company's approved
offline channel; the runner itself keeps Hub/network access disabled.

## Two-step systems smoke, then meaningful profile

```bash
cd /nlp/anhhbt/test-gpu-0924
bash frontier_company/start_a30_background.sh frontier_company/config_a30_4gpu_smoke.env E1_grpo_base
bash frontier_company/status_a30.sh frontier_company/config_a30_4gpu_smoke.env E1_grpo_base
```

Smoke uses response ceiling 1,024 and two optimizer steps. It is only a
pipeline/OOM/logging check, never a paper result. Confirm the launcher PID,
log, exit code, phase timings and GPU memory. The long profile is 7,168
response tokens, G=4, batch=4, full BF16 with offload:

```bash
bash frontier_company/preflight_a30.sh frontier_company/config_a30_4gpu.env
bash frontier_company/start_a30_background.sh frontier_company/config_a30_4gpu.env E1_grpo_base
```

Other IDs: `E2_opd_base`, `E3_hybrid_static`, `E4_hybrid_dynamic`,
`E5_hybrid_dynamic_queue`. Run comparable arms with the same immutable config,
subset, model hashes and optimizer budget. Do not launch multiple arms on the
same four A30s without a resource-isolation design. The A30 launcher does not
stop/start the machine's shared Ray cluster (`FRONTIER_MANAGE_RAY=false`).

Stop only the verified process group for a named experiment:

```bash
bash frontier_company/stop_a30.sh frontier_company/config_a30_4gpu.env E1_grpo_base
```

This code has local static checks only; it has **not** been executed on the
company A30s. Reported research runs in the old project policy required
personally rented compute. Treat company-machine runs as engineering evidence
until resource attribution/publication permission is explicitly resolved.
