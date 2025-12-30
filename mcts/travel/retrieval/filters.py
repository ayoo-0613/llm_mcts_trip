from __future__ import annotations

from typing import Any, Iterable, Set


class DedupFilter:
    def __init__(self, field: str, used_set: Iterable[Any]) -> None:
        self._field = field
        self._used_set: Set[Any] = set(used_set or [])

    def __call__(self, candidate: dict) -> bool:
        return candidate.get(self._field) not in self._used_set
