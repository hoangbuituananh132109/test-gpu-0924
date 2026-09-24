"""Framework-independent primitives for the Frontier GRPO×OPD controller."""

from .weights import (
    ControllerConfig,
    ControllerDecision,
    compute_controller_decision,
    grpo_group_weight,
    opd_outer_weight,
)
from .queues import DeferredCurriculum, QueueConfig

__all__ = [
    "ControllerConfig",
    "ControllerDecision",
    "compute_controller_decision",
    "grpo_group_weight",
    "opd_outer_weight",
    "DeferredCurriculum",
    "QueueConfig",
]
