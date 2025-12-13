from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from mcts.travel import filters


class RelaxationController:
    """Stepwise filter relaxation with a safe fallback."""

    def __init__(self, max_tries: int = 6, goal: Any = None):
        self.max_tries = max_tries
        self.goal = goal

    def relax_and_query(self, kb, slot, filt: Dict[str, Any], state: Any, cap: int) -> List[Any]:
        current = copy.deepcopy(filt) if isinstance(filt, dict) else {}
        for attempt in range(max(1, self.max_tries)):
            candidates = kb.query(slot, current, state, cap=cap)
            if candidates:
                return candidates
            current = self._relax_once(slot, current, attempt)

        # Final fallback: cheapest/closest options ignoring the failing filter.
        if hasattr(kb, "fallback_candidates"):
            try:
                return kb.fallback_candidates(slot, state, cap=cap)
            except Exception:
                pass
        # If no specialized fallback, use a minimal default filter.
        base = filters.default_filter(getattr(slot, "type", None), goal=self.goal, state=state, slot=slot)
        return kb.query(slot, base, state, cap=cap)

    def _relax_once(self, slot, filt: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        stype = getattr(slot, "type", None) or (slot.get("type") if isinstance(slot, dict) else None)
        stype = str(stype or "").lower()
        f = copy.deepcopy(filt)

        if stype == "flight":
            if attempt == 0:
                f.pop("max_price", None)
                f.pop("min_price", None)
            elif attempt == 1:
                f.pop("earliest_depart", None)
                f.pop("latest_depart", None)
            elif attempt == 2:
                f.pop("max_duration", None)
                f.pop("max_stops", None)
            elif attempt == 3:
                f.pop("date", None)
            else:
                f["avoid_ids"] = []
        elif stype == "hotel":
            if attempt == 0:
                if f.get("max_price") is not None:
                    f["max_price"] = f["max_price"] * 1.3
            elif attempt == 1:
                f.pop("room_type", None)
                f.pop("house_rules", None)
            elif attempt == 2:
                f.pop("min_review", None)
            elif attempt == 3:
                f.pop("min_occupancy", None)
            else:
                f["avoid_ids"] = []
        elif stype == "restaurant" or stype == "meal":
            if attempt == 0:
                f.pop("cuisines", None)
            elif attempt == 1:
                f.pop("max_cost", None)
            elif attempt == 2:
                f.pop("min_rating", None)
            else:
                f["avoid_ids"] = []
        elif stype == "attraction":
            if attempt == 0:
                f.pop("categories", None)
            elif attempt == 1:
                f.pop("max_distance_km", None)
            else:
                f["avoid_ids"] = []
        return f
