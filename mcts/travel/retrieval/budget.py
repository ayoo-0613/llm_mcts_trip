from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


@dataclass
class BudgetResult:
    candidates: list
    cap: Optional[float]
    info: Dict[str, Any]
    relaxed: bool


class BudgetAllocator:
    def __init__(
        self,
        parsed_get: Callable[..., Any],
        remaining_counts: Callable[[Any, Any], Dict[str, int]],
        budget_scales: Dict[str, float],
    ) -> None:
        self._parsed_get = parsed_get
        self._remaining_counts = remaining_counts
        self._budget_scales = budget_scales

    def caps(self, parsed: Any, state: Any, phase: str) -> Tuple[Optional[float], Dict[str, Any]]:
        budget = self._parsed_get(parsed, "budget", default=None)
        try:
            budget_f = float(budget)
        except Exception:
            return None, {}
        if budget_f <= 0:
            return None, {}

        remaining = budget_f - float(getattr(state, "cost", 0.0) or 0.0)
        if remaining <= 0:
            return None, {"budget_remaining": remaining}

        counts = self._remaining_counts(parsed, state)
        weights = {
            "flight": 0.35 * self._budget_scales.get("flight", 1.0) * max(1, counts["flight"]),
            "stay": 0.45 * self._budget_scales.get("stay", 1.0) * max(1, counts["stay_nights"] or counts["stay"]),
            "meal": 0.15 * self._budget_scales.get("meal", 1.0) * max(1, counts["meal"]),
        }
        weights = {k: v for k, v in weights.items() if counts.get(k, 0) > 0}
        total_weight = sum(weights.values()) if weights else 0.0
        if total_weight <= 0:
            return None, {"budget_remaining": remaining}

        phase_budget = remaining * (weights.get(phase, 0.0) / total_weight)
        info = {
            "budget_remaining": remaining,
            "phase_budget": phase_budget,
            "phase": phase,
            "counts": counts,
            "weights": weights,
        }

        if phase == "flight":
            per_slot = phase_budget / max(1, counts["flight"])
            people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
            return per_slot / float(people), info
        if phase == "stay":
            nights = max(1, counts["stay_nights"] or counts["stay"])
            return phase_budget / float(nights), info
        if phase == "meal":
            per_slot = phase_budget / max(1, counts["meal"])
            people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
            return per_slot / float(people), info
        if phase == "attraction":
            return None, info
        return None, info


class RelaxationController:
    def __init__(self, budget_scales: Dict[str, float]) -> None:
        self._budget_scales = budget_scales

    def bump(self, phase: str) -> bool:
        current = float(self._budget_scales.get(phase, 1.0) or 1.0)
        new = min(3.0, current * 1.25)
        if new <= current:
            return False
        delta = new - current
        self._budget_scales[phase] = new

        others = [p for p in self._budget_scales if p != phase]
        if not others:
            return True
        total_other = sum(float(self._budget_scales[p] or 0.0) for p in others)
        if total_other <= 0:
            return True
        min_scale = 0.4
        for p in others:
            val = float(self._budget_scales[p] or 0.0)
            reduction = delta * (val / total_other)
            self._budget_scales[p] = max(min_scale, val - reduction)
        return True


def filter_with_budget_relax(
    parsed: Any,
    state: Any,
    phase: str,
    candidates: Iterable[Any],
    *,
    allocator: BudgetAllocator,
    relaxer: RelaxationController,
    filter_fn: Callable[[Any, Optional[float]], bool],
    max_attempts: int = 3,
) -> BudgetResult:
    cap, info = allocator.caps(parsed, state, phase)
    relaxed = False
    base_list = list(candidates)
    if cap is None:
        return BudgetResult(base_list, cap, info, relaxed)

    kept = [cand for cand in base_list if filter_fn(cand, cap)]
    if kept:
        return BudgetResult(kept, cap, info, relaxed)

    relaxed = True
    for _ in range(max_attempts):
        if not relaxer.bump(phase):
            break
        cap, info = allocator.caps(parsed, state, phase)
        if cap is None:
            break
        kept = [cand for cand in base_list if filter_fn(cand, cap)]
        if kept:
            return BudgetResult(kept, cap, info, relaxed)

    return BudgetResult([], cap, info, relaxed)
