"""Concrete synchronous E5 sampler for the pinned verl trainer.

The sampler emits every ordinary prompt once.  Constant verifier groups are
queued by their GRPO state even when OPD can teach them immediately: 0/G is a
hard item and G/G is an easy item.  Revisited prompts flow through the normal
trainer again, so generation is fresh and uses the current policy.
"""

from __future__ import annotations

from collections import defaultdict, deque
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from frontier.queues import DeferredCurriculum, QueueConfig
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


def _plain(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else [value]


class FrontierQueueSampler(AbstractCurriculumSampler):
    """FIFO hard/easy revisit queue with bounded fresh-rollout attempts."""

    def __init__(self, data_source, data_config) -> None:
        self.data_source = data_source
        cfg = data_config.sampler
        self.max_total_revisits = int(cfg.get("max_total_revisits", 128))
        self.easy_revisit_fraction = float(cfg.get("easy_revisit_fraction", 0.25))
        if not 0.0 <= self.easy_revisit_fraction <= 1.0:
            raise ValueError("easy_revisit_fraction must be in [0, 1]")
        self.state_path = str(cfg.get("state_path", ""))
        self.events_path = str(cfg.get("events_path", ""))
        self.queue = DeferredCurriculum(
            QueueConfig(
                max_attempts=int(cfg.get("max_attempts", 2)),
                base_backoff_passes=1,
                max_backoff_passes=2,
            )
        )
        self.prompt_to_index = self._build_prompt_index()
        self.index_to_prompt = {item: prompt_id for prompt_id, item in self.prompt_to_index.items()}
        self.ordinary_cursor = 0
        self.phase = "ordinary"
        self.phase_pass = 0
        self.phase_items: list[str] = []
        self.phase_cursor = 0
        self.revisit_visits = 0
        self.hard_revisit_visits = 0
        self.easy_revisit_visits = 0
        self.ordinary_visits = 0
        self.optimizer_updates = 0
        self.rollout_tokens = 0
        self.teacher_tokens = 0
        self._inflight: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._persist("initialized")

    def _build_prompt_index(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        frame = getattr(self.data_source, "dataframe", None)
        for item in range(len(self.data_source)):
            row = frame[item] if frame is not None else self.data_source[item]
            extra = row.get("extra_info") or {}
            prompt_id = str(extra.get("index", item))
            if prompt_id in mapping:
                raise RuntimeError(f"E5 requires unique prompt IDs; duplicate {prompt_id!r}")
            mapping[prompt_id] = item
        return mapping

    def __len__(self) -> int:
        # Upper bound used only for progress/step planning.  The iterator ends
        # early when fewer prompts qualify; the trainer patch saves naturally.
        return len(self.data_source) + self.max_total_revisits

    def _enqueue_inflight(self, prompt_id: str, *, mode: str, pass_id: int) -> None:
        self._inflight[prompt_id].append({"mode": mode, "pass": pass_id})

    def _begin_revisit_pass(self, pass_id: int) -> None:
        hard: list[str] = []
        easy: list[str] = []
        for prompt_id in self.queue.due(pass_id):
            item = self.queue.get(prompt_id)
            (hard if item["kind"] == "hard" else easy).append(str(prompt_id))
        # Reserve a deterministic fraction of the remaining budget for easy
        # saturation checks so the much larger hard queue cannot starve them.
        remaining = max(0, self.max_total_revisits - self.revisit_visits)
        easy_slots = min(len(easy), int(round(remaining * self.easy_revisit_fraction)))
        hard_slots = min(len(hard), remaining - easy_slots)
        unused = remaining - hard_slots - easy_slots
        if unused:
            extra_hard = min(len(hard) - hard_slots, unused)
            hard_slots += extra_hard
            unused -= extra_hard
        if unused:
            easy_slots += min(len(easy) - easy_slots, unused)
        selected_hard = hard[:hard_slots]
        selected_easy = easy[:easy_slots]
        if (len(selected_hard) + len(selected_easy)) % 2:
            if len(selected_hard) > len(selected_easy):
                dropped = selected_hard.pop()
            else:
                dropped = selected_easy.pop()
            self._event(
                {
                    "event": "unpaired_revisit_deferred",
                    "prompt_id": dropped,
                    "pass": pass_id,
                    "reason": "train_batch_size_2_drop_last_guard",
                }
            )
        # Independent FIFO within each queue; hard first is fixed and logged.
        self.phase = f"revisit_{pass_id}"
        self.phase_pass = pass_id
        self.phase_items = selected_hard + selected_easy
        self._event(
            {
                "event": "revisit_pass_planned",
                "pass": pass_id,
                "due_hard": len(hard),
                "due_easy": len(easy),
                "selected_hard": len(selected_hard),
                "selected_easy": len(selected_easy),
                "remaining_budget_before_pass": remaining,
                "easy_revisit_fraction": self.easy_revisit_fraction,
            }
        )
        self.phase_cursor = 0
        self._persist("pass_boundary")

    def __iter__(self) -> Iterator[int]:
        while self.phase == "ordinary" and self.ordinary_cursor < len(self.data_source):
            item = self.ordinary_cursor
            self.ordinary_cursor += 1
            prompt_id = self.index_to_prompt[item]
            self.ordinary_visits += 1
            self._enqueue_inflight(prompt_id, mode="ordinary", pass_id=0)
            yield item

        if self.phase == "ordinary":
            self._begin_revisit_pass(1)

        while self.phase in ("revisit_1", "revisit_3"):
            while self.phase_cursor < len(self.phase_items):
                if self.revisit_visits >= self.max_total_revisits:
                    self.phase = "complete"
                    self._persist("revisit_budget_exhausted")
                    return
                prompt_id = self.phase_items[self.phase_cursor]
                self.phase_cursor += 1
                self.revisit_visits += 1
                queue_kind = self.queue.get(prompt_id)["kind"]
                if queue_kind == "hard":
                    self.hard_revisit_visits += 1
                else:
                    self.easy_revisit_visits += 1
                self._enqueue_inflight(prompt_id, mode="revisit", pass_id=self.phase_pass)
                yield self.prompt_to_index[prompt_id]
            if self.phase == "revisit_1":
                self._begin_revisit_pass(3)
            else:
                self.phase = "complete"

        self._persist("iterator_complete")

    @staticmethod
    def _batch_value(batch, key: str):
        if key in batch.batch:
            return batch.batch[key]
        if key in batch.non_tensor_batch:
            return batch.non_tensor_batch[key]
        raise KeyError(f"E5 sampler missing required batch field: {key}")

    def update(self, batch) -> None:
        prompt_ids = [str(value) for value in _plain(self._batch_value(batch, "index"))]
        neither = [float(value) for value in _plain(self._batch_value(batch, "frontier_metric__neither_active_frac"))]
        success = [float(value) for value in _plain(self._batch_value(batch, "frontier_metric__group_success_rate_mean"))]
        if not (len(prompt_ids) == len(neither) == len(success)):
            raise RuntimeError("E5 sampler diagnostic shapes do not match prompt IDs")

        # Preserve prompt-level controller evidence, not just the final queue
        # decision. This makes all four states (both, GRPO-only, OPD-only,
        # neither) and branch-scale diagnostics auditable after the run.
        diagnostic_names = (
            "grpo_active_frac",
            "opd_active_frac",
            "grpo_weight_mean",
            "opd_weight_mean",
            "compatibility_mean",
            "disagreement_mean",
            "grpo_native_rms",
            "opd_native_rms",
            "grpo_effective_rms",
            "opd_effective_rms",
            "grpo_token_l2_proxy",
            "opd_token_l2_proxy",
            "token_l2_proxy_ratio",
        )
        diagnostics: dict[str, list[float]] = {}
        for name in diagnostic_names:
            key = f"frontier_metric__{name}"
            try:
                diagnostics[name] = [float(value) for value in _plain(self._batch_value(batch, key))]
            except KeyError:
                continue

        grouped: dict[str, list[int]] = defaultdict(list)
        for row, prompt_id in enumerate(prompt_ids):
            grouped[prompt_id].append(row)
        for prompt_id, rows in grouped.items():
            if not self._inflight[prompt_id]:
                raise RuntimeError(f"E5 sampler has no inflight visit for {prompt_id}")
            visit = self._inflight[prompt_id].popleft()
            is_neither = float(np.mean([neither[row] for row in rows])) >= 0.5
            p = float(np.mean([success[row] for row in rows]))
            kind = "hard" if p <= 0.0 else "easy" if p >= 1.0 else None
            grpo_learnable = kind is None
            prompt_metrics = {
                name: float(np.mean([values[row] for row in rows]))
                for name, values in diagnostics.items()
            }
            grpo_active = prompt_metrics.get("grpo_active_frac", 0.0) >= 0.5
            opd_active = prompt_metrics.get("opd_active_frac", 0.0) >= 0.5
            if grpo_active and opd_active:
                branch_state = "both"
            elif grpo_active:
                branch_state = "grpo_only"
            elif opd_active:
                branch_state = "opd_only"
            else:
                branch_state = "neither"
            previous_kind = None
            if visit["mode"] == "ordinary":
                if kind is not None:
                    self.queue.defer(
                        prompt_id,
                        kind,
                        current_pass=0,
                        reason=f"ordinary:grpo_constant:{kind}:p={p:.6f}:branch={branch_state}",
                    )
            else:
                previous_kind = self.queue.get(prompt_id)["kind"]
                self.queue.record_revisit(
                    prompt_id,
                    became_active=grpo_learnable,
                    current_pass=int(visit["pass"]),
                    new_kind=kind,
                    note=(
                        f"revisit_pass={visit['pass']}:p={p:.6f}:"
                        f"grpo_learnable={int(grpo_learnable)}:branch={branch_state}"
                    ),
                )
            try:
                queue_entry = self.queue.get(prompt_id)
            except KeyError:
                queue_entry = None
            if visit["mode"] == "ordinary":
                transition = f"initial_{kind or 'partial'}"
            else:
                transition = f"{previous_kind}_to_{kind or 'partial'}"
            if visit["mode"] == "ordinary" and kind is not None:
                decision = f"learn_opd_defer_{kind}" if opd_active else f"defer_{kind}"
            elif visit["mode"] == "revisit" and grpo_learnable:
                decision = "recovered_grpo"
            elif visit["mode"] == "revisit" and kind is not None:
                decision = f"learn_opd_remain_{kind}" if opd_active else f"remain_{kind}"
            else:
                decision = "learn_now"
            self._event(
                {
                    "event": "visit_result",
                    "prompt_id": prompt_id,
                    "mode": visit["mode"],
                    "pass": visit["pass"],
                    "success_rate": p,
                    "neither_active": is_neither,
                    "grpo_learnable": grpo_learnable,
                    "grpo_state_consistent": grpo_learnable == grpo_active,
                    "queue_kind": kind,
                    "queued_after_visit": queue_entry is not None,
                    "queue_attempts": queue_entry["attempts"] if queue_entry else None,
                    "branch_state": branch_state,
                    "update_branches": branch_state,
                    "opd_update_now": opd_active,
                    "grpo_update_now": grpo_active,
                    "transition": transition,
                    "recovered_grpo": visit["mode"] == "revisit" and grpo_learnable,
                    "decision": decision,
                    "metrics": prompt_metrics,
                }
            )

        self.optimizer_updates += 1
        response_mask = self._batch_value(batch, "response_mask")
        tokens = int(response_mask.detach().sum().cpu().item())
        self.rollout_tokens += tokens
        self.teacher_tokens += tokens
        self._persist("batch_update")

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "queue": self.queue.state_dict(),
            "ordinary_cursor": self.ordinary_cursor,
            "phase": self.phase,
            "phase_pass": self.phase_pass,
            "phase_items": self.phase_items,
            "phase_cursor": self.phase_cursor,
            "revisit_visits": self.revisit_visits,
            "hard_revisit_visits": self.hard_revisit_visits,
            "easy_revisit_visits": self.easy_revisit_visits,
            "ordinary_visits": self.ordinary_visits,
            "optimizer_updates": self.optimizer_updates,
            "rollout_tokens": self.rollout_tokens,
            "teacher_tokens": self.teacher_tokens,
            "inflight": {key: list(value) for key, value in self._inflight.items() if value},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.queue = DeferredCurriculum.from_state_dict(state["queue"])
        for name in (
            "ordinary_cursor",
            "phase_pass",
            "phase_cursor",
            "revisit_visits",
            "hard_revisit_visits",
            "easy_revisit_visits",
            "ordinary_visits",
            "optimizer_updates",
            "rollout_tokens",
            "teacher_tokens",
        ):
            setattr(self, name, int(state.get(name, 0)))
        self.phase = str(state.get("phase", "ordinary"))
        self.phase_items = [str(item) for item in state.get("phase_items", [])]
        self._inflight = defaultdict(deque)
        for key, values in state.get("inflight", {}).items():
            self._inflight[str(key)].extend(values)
        self._persist("state_restored")

    def _payload(self, reason: str) -> dict[str, Any]:
        return {
            "reason": reason,
            "ordinary_prompt_visits": self.ordinary_visits,
            "queue_revisit_visits": self.revisit_visits,
            "hard_revisit_visits": self.hard_revisit_visits,
            "easy_revisit_visits": self.easy_revisit_visits,
            "optimizer_updates": self.optimizer_updates,
            "generated_rollout_tokens": self.rollout_tokens,
            "teacher_tokens": self.teacher_tokens,
            "phase": self.phase,
            "queue": self.queue.snapshot(),
            "state": self.state_dict(),
        }

    def _persist(self, reason: str) -> None:
        if not self.state_path:
            return
        target = Path(self.state_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self._payload(reason), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def _event(self, payload: dict[str, Any]) -> None:
        if not self.events_path:
            return
        target = Path(self.events_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
