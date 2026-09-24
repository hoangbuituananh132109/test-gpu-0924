"""Torch helpers for wiring the controller into verl without reimplementing GRPO/OPD.

These functions only compute outer group weights / scalar branch normalization.
They never construct native GRPO advantages or teacher OPD distributions.
"""
from __future__ import annotations
from typing import Iterable


def _torch():
    import torch
    return torch


def masked_rms(x, mask, eps: float = 1e-8):
    torch = _torch()
    m = mask
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    m = m.expand_as(x).bool()
    vals = x[m]
    if vals.numel() == 0:
        return torch.ones((), dtype=x.dtype, device=x.device)
    return torch.sqrt(vals.float().pow(2).mean() + eps).to(dtype=x.dtype).detach()


def scalar_rms_normalize(x, mask, eps: float = 1e-8):
    return x / masked_rms(x, mask, eps=eps).clamp_min(eps)


def group_mean_rows(values, index):
    """Broadcast each prompt group's row mean back to its member rows."""
    torch = _torch()
    if values.dim() != 1:
        raise ValueError("values must be a one-dimensional row tensor")
    keys = index.tolist() if hasattr(index, "tolist") else list(index)
    if len(keys) != values.shape[0]:
        raise ValueError("index length does not match values")
    groups = {}
    for row, key in enumerate(keys):
        groups.setdefault(key, []).append(row)
    result = torch.empty_like(values)
    for rows in groups.values():
        row_ids = torch.tensor(rows, device=values.device, dtype=torch.long)
        result[row_ids] = values[row_ids].mean()
    return result


def group_binary_stats(true_reward_score, index, response_mask=None):
    """Return row-wise group success rate p and correct count k.

    `true_reward_score` may be a response-level vector or token-level reward tensor;
    for token-level input it is summed over non-batch dimensions, matching the
    outcome-score convention used by verl's GRPO estimator.
    """
    torch = _torch()
    scores = true_reward_score
    if scores.dim() > 1:
        if response_mask is not None and scores.dim() == response_mask.dim():
            scores = (scores * response_mask).sum(dim=tuple(range(1, scores.dim())))
        else:
            scores = scores.sum(dim=tuple(range(1, scores.dim())))
    success = (scores > 0).to(torch.float32)
    # index can be numpy array / list / tensor / strings. Use a simple stable map;
    # batch sizes are small enough that clarity matters more than micro-optimization.
    keys = index.tolist() if hasattr(index, "tolist") else list(index)
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    p = torch.empty(len(keys), dtype=torch.float32, device=success.device)
    k = torch.empty(len(keys), dtype=torch.float32, device=success.device)
    gsize = torch.empty(len(keys), dtype=torch.float32, device=success.device)
    for ids in groups.values():
        ids_t = torch.tensor(ids, device=success.device, dtype=torch.long)
        kk = success[ids_t].sum()
        gg = float(len(ids))
        p[ids_t] = kk / gg
        k[ids_t] = kk
        gsize[ids_t] = gg
    return p, k, gsize


def grpo_outer_row_weights(true_reward_score, index, response_mask=None, gamma: float = 1.0,
                           min_weight: float = 0.0, max_weight: float = 1.0):
    """Row-wise outer GRPO weights, zero for all-wrong/all-correct groups."""
    torch = _torch()
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    p, k, gsize = group_binary_stats(true_reward_score, index, response_mask=response_mask)
    mixed = (k > 0) & (k < gsize)
    w = torch.pow(1.0 - p, gamma)
    w = torch.clamp(w, min=min_weight, max=max_weight)
    return torch.where(mixed, w, torch.zeros_like(w)), p, k, gsize


def opd_outer_group_weights(compatibility, disagreement, tau_c: float = 0.5, kappa: float = 0.1,
                            min_weight: float = 0.0, max_weight: float = 1.0):
    """Vectorized OPD outer weights for already-computed group/prompt C and D."""
    torch = _torch()
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    c = torch.as_tensor(compatibility)
    d = torch.as_tensor(disagreement, device=c.device, dtype=c.dtype)
    if torch.any((c < 0) | (c > 1)):
        raise ValueError("compatibility outside [0,1]")
    if torch.any(d < 0):
        raise ValueError("disagreement must be non-negative")
    w = d / (d + kappa)
    w = torch.clamp(w, min=min_weight, max=max_weight)
    return torch.where(c >= tau_c, w, torch.zeros_like(w))
