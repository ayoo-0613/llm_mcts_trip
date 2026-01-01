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
        cap_multipliers: Optional[Dict[str, float]] = None,
    ) -> None:
        self._parsed_get = parsed_get
        self._remaining_counts = remaining_counts
        self._budget_scales = budget_scales
        self._cap_multipliers = cap_multipliers or {}

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

        phase = str(phase or "").lower().strip()
        if phase in ("flight", "segment"):
            phase_key = "segment"
        elif phase in ("meal", "daily"):
            phase_key = "daily"
        elif phase in ("stay", "hotel"):
            phase_key = "stay"
        else:
            phase_key = phase

        counts = self._remaining_counts(parsed, state)
        segment_count = int(counts.get("flight", 0) or 0)
        stay_count = int(counts.get("stay_nights") or counts.get("stay", 0) or 0)
        daily_count = int(counts.get("meal", 0) or 0)
        segment_slots = max(1, segment_count)
        stay_slots = max(1, stay_count)
        daily_slots = max(1, daily_count)
        segment_scale = self._budget_scales.get("segment", self._budget_scales.get("flight", 1.0))
        daily_scale = self._budget_scales.get("daily", self._budget_scales.get("meal", 1.0))
        phase_counts = {
            "segment": segment_count,
            "stay": stay_count,
            "daily": daily_count,
        }
        weights = {
            "segment": 0.35 * segment_scale if segment_count > 0 else 0.0,
            "stay": 0.45 * self._budget_scales.get("stay", 1.0) if stay_count > 0 else 0.0,
            "daily": 0.15 * daily_scale if daily_count > 0 else 0.0,
        }
        weights = {k: v for k, v in weights.items() if v > 0}
        total_weight = sum(weights.values()) if weights else 0.0
        if total_weight <= 0:
            return None, {"budget_remaining": remaining}

        phase_budget = remaining * (weights.get(phase_key, 0.0) / total_weight)
        info = {
            "budget_remaining": remaining,
            "phase_budget": phase_budget,
            "phase": phase,
            "phase_key": phase_key,
            "counts": counts,
            "phase_counts": phase_counts,
            "weights": weights,
            "inactive_phases": [k for k, v in phase_counts.items() if v <= 0],
        }

        base_cap: Optional[float] = None
        if phase_key == "segment":
            per_slot = phase_budget / segment_slots
            people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
            base_cap = per_slot / float(people)
        elif phase_key == "stay":
            base_cap = phase_budget / float(stay_slots)
        elif phase_key == "daily":
            per_slot = phase_budget / daily_slots
            people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
            base_cap = per_slot / float(people)
        elif phase == "attraction":
            base_cap = None

        cap_multiplier = float(self._cap_multipliers.get(phase_key, 1.0) or 1.0)
        cap = base_cap * cap_multiplier if base_cap is not None else None
        info.update({"base_cap": base_cap, "cap_multiplier": cap_multiplier, "slot_cap": cap})
        return cap, info


class RelaxationController:
    def __init__(
        self, reserve_scales: Dict[str, float], cap_multipliers: Optional[Dict[str, float]] = None
    ) -> None:
        self._reserve_scales = reserve_scales
        self._cap_multipliers = cap_multipliers or {}
        self._min_reserve_scale = 0.4
        self._max_cap_multiplier = 3.0
        self._reserve_step = 0.85
        self._cap_step = 1.25

    @staticmethod
    def _phase_key(phase: str) -> str:
        p = str(phase or "").lower().strip()
        if p in ("flight", "segment"):
            return "segment"
        if p in ("hotel", "stay"):
            return "stay"
        if p in ("meal", "daily"):
            return "daily"
        return p

    def reserve_scale(self, phase: str) -> float:
        phase_key = self._phase_key(phase)
        return float(self._reserve_scales.get(phase_key, 1.0) or 1.0)

    def cap_multiplier(self, phase: str) -> float:
        phase_key = self._phase_key(phase)
        return float(self._cap_multipliers.get(phase_key, 1.0) or 1.0)

    def bump(self, phase: str) -> bool:
        phase_key = self._phase_key(phase)
        changed = False
        reserve_scale = float(self._reserve_scales.get(phase_key, 1.0) or 1.0)
        new_reserve = max(self._min_reserve_scale, reserve_scale * self._reserve_step)
        if new_reserve < reserve_scale:
            self._reserve_scales[phase_key] = new_reserve
            changed = True

        cap_multiplier = float(self._cap_multipliers.get(phase_key, 1.0) or 1.0)
        new_cap = min(self._max_cap_multiplier, cap_multiplier * self._cap_step)
        if new_cap > cap_multiplier:
            self._cap_multipliers[phase_key] = new_cap
            changed = True
        return changed


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
