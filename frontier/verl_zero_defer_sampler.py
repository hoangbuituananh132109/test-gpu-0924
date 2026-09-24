"""E5 all-zero defer curriculum for the pinned synchronous verl trainer.

Every prompt is visited once in the ordinary stream.  A prompt whose fresh
G-rollout verifier group is 0/G receives no training credit on that visit and
is queued.  After the ordinary pass, every queued prompt is revisited exactly
once with the then-current policy.  The loss patch decides which native E4
branches are usable on that revisit.

The sampler carries visit phase into the batch *before* generation.  The v4
trainer runs the rule verifier immediately after rollout so first-visit 0/G can
bypass student-top-k and teacher OPD forwards, not merely mask their loss.
"""

from __future__ import annotations

from collections import defaultdict, deque
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from verl.experimental.dataset.sampler import AbstractCurriculumSampler


ORDINARY = 0
REVISIT = 1
PADDING = 2


def _plain(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else [value]


class ZeroDeferCurriculumSampler(AbstractCurriculumSampler):
    """One ordinary pass followed by one fresh revisit of every initial 0/G."""

    def __init__(self, data_source, data_config) -> None:
        self.data_source = data_source
        cfg = data_config.sampler
        configured_budget = int(cfg.get("max_total_revisits", len(data_source)))
        if configured_budget < 1:
            configured_budget = len(data_source)
        self.max_total_revisits = configured_budget
        self.state_path = str(cfg.get("state_path", ""))
        self.events_path = str(cfg.get("events_path", ""))
        self.prompt_to_index = self._build_prompt_index()
        self.index_to_prompt = {item: prompt_id for prompt_id, item in self.prompt_to_index.items()}

        self.ordinary_cursor = 0
        self.phase = "ordinary"
        self.revisit_items: list[dict[str, Any]] = []
        self.revisit_cursor = 0
        self.hard_queue: list[str] = []
        self.hard_queue_set: set[str] = set()
        self._inflight: dict[str, deque[dict[str, Any]]] = defaultdict(deque)

        self.ordinary_visits = 0
        self.initial_zero_count = 0
        self.revisit_visits = 0
        self.padding_visits = 0
        self.optimizer_steps = 0
        self.prompt_training_credits = 0
        self.revisit_training_credits = 0
        self.recovered_to_mixed = 0
        self.recovered_to_all_correct = 0
        self.remained_all_wrong = 0
        self.generated_rollout_tokens = 0
        self.teacher_tokens = 0
        self.suppressed_first_visit_tokens = 0
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
        # Upper bound for the trainer scheduler.  The natural-end trainer patch
        # saves when fewer than all prompts enter the queue.  One masked padding
        # visit covers the possible odd queue size under drop_last=True.
        return len(self.data_source) + self.max_total_revisits + 1

    def _enqueue_inflight(self, prompt_id: str, mode: int) -> None:
        self._inflight[prompt_id].append({"mode": mode})

    def _padding_prompt(self, final_revisit_prompt: str) -> str:
        for prompt_id in self.prompt_to_index:
            if prompt_id != final_revisit_prompt:
                return prompt_id
        raise RuntimeError("E5 batch-size padding requires at least two unique prompts")

    def _plan_revisits(self) -> None:
        if len(self.hard_queue) > self.max_total_revisits:
            raise RuntimeError(
                "E5 all-zero queue budget would drop prompts: "
                f"queued={len(self.hard_queue)} budget={self.max_total_revisits}"
            )
        self.revisit_items = [
            {"prompt_id": prompt_id, "mode": REVISIT}
            for prompt_id in self.hard_queue
        ]
        if len(self.revisit_items) % 2:
            padding_id = self._padding_prompt(self.revisit_items[-1]["prompt_id"])
            self.revisit_items.append({"prompt_id": padding_id, "mode": PADDING})
            self._event(
                {
                    "event": "masked_padding_planned",
                    "prompt_id": padding_id,
                    "reason": "train_batch_size_2_drop_last_guard",
                }
            )
        self.phase = "revisit" if self.revisit_items else "complete"
        self.revisit_cursor = 0
        self._event(
            {
                "event": "revisit_pass_planned",
                "queued_all_wrong": len(self.hard_queue),
                "fresh_revisits": len(self.hard_queue),
                "masked_padding": len(self.revisit_items) - len(self.hard_queue),
            }
        )
        self._persist("ordinary_pass_complete")

    def __iter__(self) -> Iterator[int]:
        while self.phase == "ordinary" and self.ordinary_cursor < len(self.data_source):
            item = self.ordinary_cursor
            self.ordinary_cursor += 1
            prompt_id = self.index_to_prompt[item]
            self.ordinary_visits += 1
            self._enqueue_inflight(prompt_id, ORDINARY)
            yield item

        if self.phase == "ordinary":
            self._plan_revisits()

        while self.phase == "revisit" and self.revisit_cursor < len(self.revisit_items):
            visit = self.revisit_items[self.revisit_cursor]
            self.revisit_cursor += 1
            prompt_id = str(visit["prompt_id"])
            mode = int(visit["mode"])
            if mode == REVISIT:
                self.revisit_visits += 1
            else:
                self.padding_visits += 1
            self._enqueue_inflight(prompt_id, mode)
            yield self.prompt_to_index[prompt_id]

        if self.phase == "revisit":
            self.phase = "complete"
        self._persist("iterator_complete")

    @staticmethod
    def _batch_value(batch, key: str):
        if key in batch.batch:
            return batch.batch[key]
        if key in batch.non_tensor_batch:
            return batch.non_tensor_batch[key]
        raise KeyError(f"E5 sampler missing required batch field: {key}")

    def annotate_batch(self, batch):
        """Attach ordinary/revisit state before repeat and loss computation."""
        prompt_ids = [str(value) for value in _plain(self._batch_value(batch, "index"))]
        modes: list[int] = []
        for prompt_id in prompt_ids:
            if not self._inflight[prompt_id]:
                raise RuntimeError(f"E5 sampler has no inflight visit for {prompt_id}")
            modes.append(int(self._inflight[prompt_id][0]["mode"]))
        batch.batch["frontier_queue_visit_phase"] = torch.tensor(modes, dtype=torch.int64)
        return batch

    def update(self, batch) -> None:
        prompt_ids = [str(value) for value in _plain(self._batch_value(batch, "index"))]
        success = [
            float(value)
            for value in _plain(self._batch_value(batch, "frontier_metric__group_success_rate_mean"))
        ]
        credit = [
            float(value)
            for value in _plain(self._batch_value(batch, "frontier_metric__training_credit_spent_frac"))
        ]
        opd_eligible = [
            float(value)
            for value in _plain(self._batch_value(batch, "frontier_metric__opd_eligible_before_queue_frac"))
        ]
        try:
            teacher_scored = [
                float(value)
                for value in _plain(self._batch_value(batch, "frontier_metric__teacher_scored_frac"))
            ]
        except KeyError:
            teacher_scored = [1.0] * len(prompt_ids)
        response_mask = self._batch_value(batch, "response_mask")
        row_tokens = response_mask.detach().sum(dim=-1).cpu().tolist()
        if not (
            len(prompt_ids)
            == len(success)
            == len(credit)
            == len(opd_eligible)
            == len(teacher_scored)
            == len(row_tokens)
        ):
            raise RuntimeError("E5 sampler diagnostic shapes do not match prompt IDs")

        diagnostic_names = (
            "group_success_count",
            "group_size",
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
            "first_zero_deferred_frac",
            "revisit_all_correct_retired_frac",
            "teacher_scored_frac",
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
            mode = int(visit["mode"])
            p = float(np.mean([success[row] for row in rows]))
            spent_credit = float(np.mean([credit[row] for row in rows])) >= 0.5
            teacher_eligible = float(np.mean([opd_eligible[row] for row in rows])) >= 0.5
            teacher_was_scored = float(np.mean([teacher_scored[row] for row in rows])) >= 0.5
            token_count = int(sum(int(row_tokens[row]) for row in rows))
            prompt_metrics = {
                name: float(np.mean([values[row] for row in rows]))
                for name, values in diagnostics.items()
            }

            if mode == ORDINARY and p <= 0.0:
                if prompt_id in self.hard_queue_set:
                    raise RuntimeError(f"E5 attempted to queue {prompt_id} twice")
                self.hard_queue.append(prompt_id)
                self.hard_queue_set.add(prompt_id)
                self.initial_zero_count += 1
                if spent_credit:
                    raise RuntimeError("E5 invariant violated: first-visit 0/G spent training credit")
                self.suppressed_first_visit_tokens += token_count
                decision = "defer_initial_zero_no_update"
            elif mode == REVISIT:
                if p <= 0.0:
                    self.remained_all_wrong += 1
                    decision = "revisit_zero_opd_fallback" if spent_credit else "revisit_zero_no_usable_signal"
                elif p >= 1.0:
                    self.recovered_to_all_correct += 1
                    if spent_credit:
                        raise RuntimeError("E5 invariant violated: 0/G->G/G revisit spent training credit")
                    decision = "revisit_all_correct_retire"
                else:
                    self.recovered_to_mixed += 1
                    decision = "revisit_mixed_e4_update" if spent_credit else "revisit_mixed_no_usable_signal"
                if spent_credit:
                    self.revisit_training_credits += 1
            elif mode == PADDING:
                if spent_credit:
                    raise RuntimeError("E5 invariant violated: padding spent training credit")
                decision = "masked_padding"
            else:
                decision = "ordinary_e4_update" if spent_credit else "ordinary_no_usable_signal"

            if spent_credit:
                self.prompt_training_credits += 1
            self._event(
                {
                    "event": "visit_result",
                    "prompt_id": prompt_id,
                    "mode": {ORDINARY: "ordinary", REVISIT: "revisit", PADDING: "padding"}[mode],
                    "success_rate": p,
                    "opd_eligible_before_queue": teacher_eligible,
                    "teacher_scored": teacher_was_scored,
                    "training_credit_spent": spent_credit,
                    "decision": decision,
                    "response_tokens": token_count,
                    "metrics": prompt_metrics,
                }
            )

        self.optimizer_steps += 1
        tokens = int(response_mask.detach().sum().cpu().item())
        self.generated_rollout_tokens += tokens
        self.teacher_tokens += int(
            sum(int(row_tokens[row]) for row, scored in enumerate(teacher_scored) if scored >= 0.5)
        )
        self._persist("batch_update")

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 4,
            "ordinary_cursor": self.ordinary_cursor,
            "phase": self.phase,
            "revisit_items": self.revisit_items,
            "revisit_cursor": self.revisit_cursor,
            "hard_queue": self.hard_queue,
            "ordinary_visits": self.ordinary_visits,
            "initial_zero_count": self.initial_zero_count,
            "revisit_visits": self.revisit_visits,
            "padding_visits": self.padding_visits,
            "optimizer_steps": self.optimizer_steps,
            "prompt_training_credits": self.prompt_training_credits,
            "revisit_training_credits": self.revisit_training_credits,
            "recovered_to_mixed": self.recovered_to_mixed,
            "recovered_to_all_correct": self.recovered_to_all_correct,
            "remained_all_wrong": self.remained_all_wrong,
            "generated_rollout_tokens": self.generated_rollout_tokens,
            "teacher_tokens": self.teacher_tokens,
            "suppressed_first_visit_tokens": self.suppressed_first_visit_tokens,
            "inflight": {key: list(value) for key, value in self._inflight.items() if value},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != 4:
            raise RuntimeError("refusing to resume E5 curriculum from incompatible queue semantics")
        for name in (
            "ordinary_cursor",
            "revisit_cursor",
            "ordinary_visits",
            "initial_zero_count",
            "revisit_visits",
            "padding_visits",
            "optimizer_steps",
            "prompt_training_credits",
            "revisit_training_credits",
            "recovered_to_mixed",
            "recovered_to_all_correct",
            "remained_all_wrong",
            "generated_rollout_tokens",
            "teacher_tokens",
            "suppressed_first_visit_tokens",
        ):
            setattr(self, name, int(state.get(name, 0)))
        self.phase = str(state.get("phase", "ordinary"))
        self.revisit_items = list(state.get("revisit_items", []))
        self.hard_queue = [str(item) for item in state.get("hard_queue", [])]
        self.hard_queue_set = set(self.hard_queue)
        self._inflight = defaultdict(deque)
        for key, values in state.get("inflight", {}).items():
            self._inflight[str(key)].extend(values)
        self._persist("state_restored")

    def _payload(self, reason: str) -> dict[str, Any]:
        return {
            "reason": reason,
            "semantics": "all initial 0/G deferred; one fresh revisit; at most one training credit per prompt",
            "phase": self.phase,
            "ordinary_prompt_visits": self.ordinary_visits,
            "initial_zero_count": self.initial_zero_count,
            "initial_zero_fraction": self.initial_zero_count / max(1, self.ordinary_visits),
            "queue_revisit_visits": self.revisit_visits,
            "padding_visits": self.padding_visits,
            "optimizer_steps": self.optimizer_steps,
            "prompt_training_credits": self.prompt_training_credits,
            "revisit_training_credits": self.revisit_training_credits,
            "recovered_to_mixed": self.recovered_to_mixed,
            "recovered_to_all_correct": self.recovered_to_all_correct,
            "remained_all_wrong": self.remained_all_wrong,
            "generated_rollout_tokens": self.generated_rollout_tokens,
            "teacher_tokens": self.teacher_tokens,
            "suppressed_first_visit_tokens": self.suppressed_first_visit_tokens,
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
