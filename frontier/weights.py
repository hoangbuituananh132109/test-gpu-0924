"""Controller weighting primitives.

These functions deliberately do NOT implement GRPO or OPD themselves. They compute
outer activation/scaling decisions so the native upstream GRPO advantages and OPD
teacher-distribution signal can remain intact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


def _clip(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError("clip lower bound must be <= upper bound")
    return min(max(float(x), float(lo)), float(hi))


def grpo_group_weight(
    correct_count: int,
    group_size: int,
    *,
    gamma: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    require_mixed_group: bool = True,
) -> float:
    """Return the *outer* GRPO group weight.

    Starter hypothesis: w=(1-p)^gamma, p=k/G. This prioritizes rare-success
    groups. It does not replace the native group-normalized GRPO advantages.

    If ``require_mixed_group`` is true, all-wrong/all-correct groups receive zero
    because ordinary binary-reward GRPO has no within-group outcome variance.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not 0 <= correct_count <= group_size:
        raise ValueError("correct_count must lie in [0, group_size]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    if require_mixed_group and correct_count in (0, group_size):
        return 0.0
    p = correct_count / group_size
    raw = (1.0 - p) ** gamma
    return _clip(raw, min_weight, max_weight)


def opd_outer_weight(
    compatibility: float,
    disagreement: float,
    *,
    tau_c: float = 0.5,
    kappa: float = 0.1,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> float:
    """Return the *outer* OPD weight.

    ``compatibility`` C is assumed to be normalized to [0,1]. ``disagreement`` D
    is non-negative and should refer to the teacher/student disagreement in the
    same reachable/support region whenever possible.

        w = I[C >= tau_c] * D/(D+kappa)

    Kappa prevents arbitrarily large raw disagreements from creating unbounded
    weights. The native OPD token-level distributional signal is not changed.
    """
    if not math.isfinite(compatibility) or not 0.0 <= compatibility <= 1.0:
        raise ValueError("compatibility must be finite and in [0,1]")
    if not math.isfinite(disagreement) or disagreement < 0:
        raise ValueError("disagreement must be finite and non-negative")
    if not 0 <= tau_c <= 1:
        raise ValueError("tau_c must be in [0,1]")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")

    if compatibility < tau_c:
        return 0.0
    raw = disagreement / (disagreement + kappa)
    return _clip(raw, min_weight, max_weight)


@dataclass(frozen=True)
class ControllerConfig:
    grpo_gamma: float = 1.0
    grpo_min_weight: float = 0.0
    grpo_max_weight: float = 1.0
    opd_tau_c: float = 0.5
    opd_kappa: float = 0.1
    opd_min_weight: float = 0.0
    opd_max_weight: float = 1.0


@dataclass(frozen=True)
class ControllerDecision:
    group_size: int
    correct_count: int
    success_rate: float
    grpo_active: bool
    grpo_weight: float
    opd_active: bool
    opd_weight: float
    compatibility: Optional[float]
    disagreement: Optional[float]
    outcome_state: str
    branch_state: str


def compute_controller_decision(
    correct_count: int,
    group_size: int,
    *,
    compatibility: Optional[float] = None,
    disagreement: Optional[float] = None,
    config: ControllerConfig = ControllerConfig(),
    static_hybrid: bool = False,
) -> ControllerDecision:
    """Compute a routing/outer-weight decision for one prompt group.

    ``static_hybrid=True`` is useful for E3: GRPO is active on mixed groups with
    weight 1; OPD is active whenever C/D values are supplied, also with weight 1.
    No dynamic competence scaling is applied.
    """
    if group_size <= 0 or not 0 <= correct_count <= group_size:
        raise ValueError("invalid correctness group")
    p = correct_count / group_size
    mixed = 0 < correct_count < group_size
    outcome_state = "hard" if correct_count == 0 else "easy" if correct_count == group_size else "frontier"

    if static_hybrid:
        w_r = 1.0 if mixed else 0.0
    else:
        w_r = grpo_group_weight(
            correct_count,
            group_size,
            gamma=config.grpo_gamma,
            min_weight=config.grpo_min_weight,
            max_weight=config.grpo_max_weight,
        )

    opd_available = compatibility is not None and disagreement is not None
    if static_hybrid:
        w_o = 1.0 if opd_available else 0.0
    elif opd_available:
        w_o = opd_outer_weight(
            float(compatibility),
            float(disagreement),
            tau_c=config.opd_tau_c,
            kappa=config.opd_kappa,
            min_weight=config.opd_min_weight,
            max_weight=config.opd_max_weight,
        )
    else:
        w_o = 0.0

    r_active = w_r > 0
    o_active = w_o > 0
    if r_active and o_active:
        branch = "both"
    elif r_active:
        branch = "grpo_only"
    elif o_active:
        branch = "opd_only"
    else:
        branch = "neither"

    return ControllerDecision(
        group_size=group_size,
        correct_count=correct_count,
        success_rate=p,
        grpo_active=r_active,
        grpo_weight=w_r,
        opd_active=o_active,
        opd_weight=w_o,
        compatibility=compatibility,
        disagreement=disagreement,
        outcome_state=outcome_state,
        branch_state=branch,
    )


class EMAScale:
    """Tiny diagnostic helper for tracking relative branch magnitudes.

    This is *not* enabled automatically in any scientific baseline. It can be
    used to log scale mismatch or as a separately declared ablation later.
    """

    def __init__(self, beta: float = 0.98, eps: float = 1e-8) -> None:
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0,1)")
        self.beta = beta
        self.eps = eps
        self.value: Optional[float] = None

    def update(self, magnitude: float) -> float:
        if not math.isfinite(magnitude) or magnitude < 0:
            raise ValueError("magnitude must be finite and non-negative")
        if self.value is None:
            self.value = float(magnitude)
        else:
            self.value = self.beta * self.value + (1 - self.beta) * float(magnitude)
        return self.value

    def normalized(self, magnitude: float) -> float:
        if self.value is None:
            return float(magnitude)
        return float(magnitude) / (self.value + self.eps)
