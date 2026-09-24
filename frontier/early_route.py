"""Pure helpers for E5 verifier-first teacher routing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


ORDINARY = 0
REVISIT = 1
PADDING = 2


def teacher_compute_mask(
    uids: Sequence[object],
    visit_phase: torch.Tensor,
    true_reward_score: torch.Tensor,
) -> torch.Tensor:
    """Select rows that still need OPD after the cheap rule verifier.

    Initial 0/G, revisit G/G, and masked padding never reach the teacher.
    All decisions are group-level and use the same nonzero-correctness rule as
    the native GRPO implementation.
    """

    row_count = int(true_reward_score.shape[0])
    if len(uids) != row_count or int(visit_phase.numel()) != row_count:
        raise RuntimeError("E5 early-route inputs do not share the same row count")
    phases = visit_phase.detach().reshape(-1).to("cpu", dtype=torch.long)
    if not torch.all((phases >= ORDINARY) & (phases <= PADDING)):
        raise RuntimeError("E5 visit phase must be ordinary=0, revisit=1, or padding=2")

    sequence_reward = true_reward_score.detach().reshape(row_count, -1).sum(dim=-1)
    mask = torch.ones(row_count, dtype=torch.bool, device=sequence_reward.device)
    uid_array = np.asarray([str(uid) for uid in uids], dtype=object)
    for uid in dict.fromkeys(uid_array.tolist()):
        rows_np = np.flatnonzero(uid_array == uid)
        rows_cpu = torch.as_tensor(rows_np, dtype=torch.long)
        group_phases = phases.index_select(0, rows_cpu)
        if not torch.all(group_phases == group_phases[0]):
            raise RuntimeError(f"E5 prompt group {uid!r} contains mixed visit phases")
        phase = int(group_phases[0].item())
        rows = rows_cpu.to(sequence_reward.device)
        group_size = int(rows.numel())
        correct = int(torch.count_nonzero(sequence_reward.index_select(0, rows)).item())
        skip_teacher = (
            phase == PADDING
            or (phase == ORDINARY and correct == 0)
            or (phase == REVISIT and correct == group_size)
        )
        if skip_teacher:
            mask.index_fill_(0, rows, False)
    return mask


def scatter_partial_tensors(
    batch_size: int,
    active_indices: torch.Tensor,
    partial_tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Scatter worker outputs back to full row order with inert placeholders."""

    full: dict[str, torch.Tensor] = {}
    for key, value in partial_tensors.items():
        fill = -1.0e9 if "log_probs" in key and key != "old_log_probs" else 0
        target = value.new_full((batch_size, *value.shape[1:]), fill)
        target.index_copy_(0, active_indices.to(value.device, dtype=torch.long), value)
        full[key] = target
    return full


def empty_only_student_outputs(
    responses: torch.Tensor,
    top_k: int,
) -> dict[str, torch.Tensor]:
    """Create shape-safe inert tensors when every row skips OPD."""

    batch_size, response_length = responses.shape
    device = responses.device
    floats_2d = torch.zeros((batch_size, response_length), dtype=torch.float32, device=device)
    floats_3d = torch.full(
        (batch_size, response_length, top_k), -1.0e9, dtype=torch.float32, device=device
    )
    return {
        "old_log_probs": floats_2d.clone(),
        "entropys": floats_2d.clone(),
        "student_top_k_ids": torch.zeros(
            (batch_size, response_length, top_k), dtype=torch.long, device=device
        ),
        "student_top_k_log_probs": floats_3d.clone(),
        "teacher_on_student_log_probs": floats_3d.clone(),
        "rm_scores": torch.zeros(
            (batch_size, response_length, top_k), dtype=torch.float32, device=device
        ),
    }
