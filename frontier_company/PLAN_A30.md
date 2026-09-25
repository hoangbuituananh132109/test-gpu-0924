# 4×A30 decision card

1. **Environment audit, zero GPU training:** run `preflight_a30.sh`; record
   CUDA/driver, topology, free VRAM, RAM, disk, package versions, local model
   hashes and DAPO train hash. Avoid old environment installation if imports
   already work. Missing assets fail closed.
2. **Systems smoke:** 1,024 output tokens, G=4, batch=4, two steps on the new
   1.7B-Instruct→4B-Instruct pair. Require finite native GRPO/OPD losses, verifier
   outputs, correct 4-rank behavior, checkpoint/log integrity and no OOM.
   Smoke accuracy is not a scientific result.
3. **Signal census:** on the same fixed 1,024 train prompts, measure k/G
   histogram (f0, mixed, f1), verifier pass rate, response p50/p90/p99,
   cap-hit, teacher reachability/disagreement and OPD/GRPO gradient norms and
   cosine where both are active. No updates in this census. Freeze the G and
   response ceiling before comparisons; G=8 or 16k are separate profiles.
4. **Larger-pair feasibility:** separately run the 4B-Instruct→8B-Instruct
   two-step profile,
   first E1 then E2/E4 only if the teacher is available locally. It is a
   systems probe, not a comparable result. Inspect peak VRAM and OOM, then
   decide whether a 7,168-token profile is feasible. Do not infer 14B/32B
   feasibility from an 8B teacher run.
5. **Matched pilot:** if the census has useful mixed groups, run E1 GRPO,
   E2 OPD, E4 dynamic and E5 queue with the same 1,024 prompts, seed, G=4,
   7,168 ceiling, full BF16/offload, and optimizer budget. E3 static is the
   control for dynamic weighting. Keep native GRPO group advantages and native
   teacher distributional OPD; dynamic weights are outer scales. Never infer
   branch influence from a near-zero scalar policy loss alone—use gradient
   magnitudes/cosine and active-token/group counts.
6. **Scale decision:** only after matched eval on held-out AIME/AMC/MATH500
   and cost/throughput review, consider the 17K DAPO train set. Do not use
   benchmark rows for training, queue control, threshold fitting or teacher
   selection. Evaluate the untrained student and teacher as reference points.

The profile is intentionally conservative for 24 GB/GPU: TP=1 rollout,
vLLM fraction 0.30, microbatch=1, gradient checkpointing and parameter/
optimizer/activation offload. Measure rollout, teacher scoring, update,
weight sync/offload separately before increasing batch or vLLM fraction.
Change systems knobs for **all** comparison arms together. If the 7,168
ceiling has material cap hits, raise it for all arms; do not add a length
penalty or silently truncate reasoning.

The proposed 4B→8B pair has a separate 1,024-token feasibility config;
14B/32B do not. Their own offline model assets, full-weight memory preflight
and equal-budget baselines are prerequisites. Four A30s do not imply each pair
will fit or that multiple experiments can run concurrently.

The GitHub work branch is code-only. Its train/eval parquet files are in a
separate local offline ZIP with a hash manifest; models are supplied by the
company machine. The previous 4B-Base-GRPO teacher runs are not a matched
control for these Instruct-teacher experiments. Before a scientific pilot,
lock the primary held-out metric, optimizer-step budget and model hashes for
all compared E1–E5 arms; the two-step probes answer feasibility only.
