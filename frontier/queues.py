"""Framework-independent deferred curriculum queue.

E5 uses this only after the dynamic hybrid (E4) is established. The queue stores
prompt identifiers, not cached rollouts, because revisits must be generated under
the current policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Literal

QueueKind = Literal["hard", "easy"]


@dataclass(frozen=True)
class QueueConfig:
    max_attempts: int = 4
    base_backoff_passes: int = 1
    max_backoff_passes: int = 8


@dataclass
class _Entry:
    prompt_id: Hashable
    kind: QueueKind
    attempts: int = 0
    next_pass: int = 0
    history: List[str] = field(default_factory=list)


class DeferredCurriculum:
    """Bounded easy/hard queue with exponential pass-level backoff."""

    def __init__(self, config: QueueConfig = QueueConfig()) -> None:
        if config.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if config.base_backoff_passes < 1:
            raise ValueError("base_backoff_passes must be >= 1")
        self.config = config
        self._entries: Dict[Hashable, _Entry] = {}

    def defer(self, prompt_id: Hashable, kind: QueueKind, current_pass: int, reason: str = "") -> None:
        if kind not in ("hard", "easy"):
            raise ValueError("kind must be hard or easy")
        entry = self._entries.get(prompt_id)
        if entry is None:
            entry = _Entry(prompt_id=prompt_id, kind=kind)
            self._entries[prompt_id] = entry
        else:
            entry.kind = kind
        if reason:
            entry.history.append(reason)
        delay = min(
            self.config.base_backoff_passes * (2 ** entry.attempts),
            self.config.max_backoff_passes,
        )
        entry.next_pass = current_pass + delay

    def due(self, current_pass: int) -> List[Hashable]:
        return [
            pid
            for pid, e in self._entries.items()
            if e.next_pass <= current_pass and e.attempts < self.config.max_attempts
        ]

    def record_revisit(
        self,
        prompt_id: Hashable,
        *,
        became_active: bool,
        current_pass: int,
        new_kind: QueueKind | None = None,
        note: str = "",
    ) -> None:
        entry = self._entries[prompt_id]
        entry.attempts += 1
        if note:
            entry.history.append(note)
        if became_active:
            del self._entries[prompt_id]
            return
        if entry.attempts >= self.config.max_attempts:
            entry.next_pass = 10**18
            return
        if new_kind is not None:
            entry.kind = new_kind
        delay = min(
            self.config.base_backoff_passes * (2 ** entry.attempts),
            self.config.max_backoff_passes,
        )
        entry.next_pass = current_pass + delay

    def snapshot(self) -> dict:
        hard = sum(e.kind == "hard" for e in self._entries.values())
        easy = sum(e.kind == "easy" for e in self._entries.values())
        exhausted = sum(e.attempts >= self.config.max_attempts for e in self._entries.values())
        return {"hard": hard, "easy": easy, "exhausted": exhausted, "total": len(self._entries)}

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-serializable queue state in stable FIFO order."""
        return {
            "config": {
                "max_attempts": self.config.max_attempts,
                "base_backoff_passes": self.config.base_backoff_passes,
                "max_backoff_passes": self.config.max_backoff_passes,
            },
            "entries": [
                {
                    "prompt_id": entry.prompt_id,
                    "kind": entry.kind,
                    "attempts": entry.attempts,
                    "next_pass": entry.next_pass,
                    "history": list(entry.history),
                }
                for entry in self._entries.values()
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DeferredCurriculum":
        """Restore state produced by :meth:`state_dict`."""
        config = QueueConfig(**state["config"])
        queue = cls(config)
        for item in state.get("entries", []):
            entry = _Entry(
                prompt_id=item["prompt_id"],
                kind=item["kind"],
                attempts=int(item.get("attempts", 0)),
                next_pass=int(item.get("next_pass", 0)),
                history=list(item.get("history", [])),
            )
            queue._entries[entry.prompt_id] = entry
        return queue

    def get(self, prompt_id: Hashable) -> dict:
        e = self._entries[prompt_id]
        return {
            "prompt_id": e.prompt_id,
            "kind": e.kind,
            "attempts": e.attempts,
            "next_pass": e.next_pass,
            "history": list(e.history),
        }
