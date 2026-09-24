"""Global scalar branch normalization.

The purpose is to match GRPO and OPD branch *scale* without mean-centering or
otherwise altering the internal ordering/sign geometry of either signal.
"""
from __future__ import annotations
import math
from typing import Iterable, Sequence


def rms(values: Iterable[float], eps: float = 1e-8) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 1.0
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("non-finite value")
    return math.sqrt(sum(v * v for v in vals) / len(vals) + eps)


def rms_normalize(values: Sequence[float], eps: float = 1e-8) -> list[float]:
    s = rms(values, eps=eps)
    return [float(v) / s for v in values]
