from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

from mcts.travel import filters

logger = logging.getLogger(__name__)


class LLMFilterGenerator:
    """LLM-only filter generator with strict JSON validation and caching."""

    def __init__(self, llm: Optional[Any] = None, cache: Optional[Dict] = None, enable: bool = True):
        self.llm = llm
        self.enable = enable
        self.cache: Dict[Tuple[str, str, str], Dict[str, Any]] = cache if cache is not None else {}
        self.last_info: Dict[str, Any] = {}

    def _make_cache_key(self, goal, state, slot) -> Tuple[str, str, str]:
        goal_sig = self._goal_signature(goal)
        state_sig = self._state_signature(state)
        slot_sig = self._slot_signature(slot)
        return goal_sig, state_sig, slot_sig

    @staticmethod
    def _goal_signature(goal) -> str:
        if goal is None:
            return ""
        parts = [
            getattr(goal, "origin", ""),
            getattr(goal, "destination", ""),
            str(getattr(goal, "start_date", "")),
            str(getattr(goal, "duration_days", "")),
            str(getattr(goal, "budget", "")),
            "|".join(getattr(goal, "must_visit_cities", []) or []),
            "|".join(getattr(goal, "priority_cities", []) or []),
        ]
        return "|".join(parts)

    @staticmethod
    def _state_signature(state) -> str:
        if state is None:
            return ""
        if hasattr(state, "signature"):
            try:
                return state.signature()
            except Exception:
                pass
        return repr(state)

    @staticmethod
    def _slot_signature(slot) -> str:
        if slot is None:
            return ""
        if hasattr(slot, "signature"):
            try:
                return slot.signature()
            except Exception:
                pass
        try:
            return json.dumps(slot.__dict__, sort_keys=True, default=str)
        except Exception:
            return repr(slot)

    def propose(self, goal, state, slot, user_query: str = "") -> Dict[str, Any]:
        """Return a validated filter dict for this slot."""
        filter_type = getattr(slot, "type", None) or (slot.get("type") if isinstance(slot, dict) else None)
        filter_type = str(filter_type or "").lower()
        if filter_type == "meal":
            filter_type = "restaurant"
        key = self._make_cache_key(goal, state, slot)
        if key in self.cache:
            self.last_info = {"cache_hit": True, "used_llm": False, "filter_type": filter_type}
            return self.cache[key]

        if not self.enable or self.llm is None or not filter_type:
            filt = filters.default_filter(filter_type, goal, state, slot)
            self.cache[key] = filt
            self.last_info = {"cache_hit": False, "used_llm": False, "filter_type": filter_type}
            return filt

        prompt = self._build_prompt(goal, state, slot, user_query)
        try:
            raw = self._call_llm(prompt)
            parsed = self._parse_json(raw)
        except Exception as exc:  # pragma: no cover - depends on runtime llm
            logger.warning("LLM filter generation failed (%s); using default filter", exc)
            parsed = {}
        filt = filters.validate_and_normalize(parsed, filter_type, goal=goal, state=state, slot=slot)
        self.cache[key] = filt
        self.last_info = {"cache_hit": False, "used_llm": True, "filter_type": filter_type}
        return filt

    def _call_llm(self, prompt: str) -> str:
        if callable(self.llm):
            return self.llm(prompt)
        if hasattr(self.llm, "generate"):
            return self.llm.generate(prompt)  # type: ignore[attr-defined]
        if hasattr(self.llm, "__call__"):
            return self.llm(prompt)
        raise RuntimeError("No callable LLM provided for LLMFilterGenerator")

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(raw[start : end + 1])
                except Exception:
                    return {}
        return {}

    def _build_prompt(self, goal, state, slot, user_query: str) -> str:
        goal_text = goal.as_text() if hasattr(goal, "as_text") else str(goal)
        slot_hint = self._slot_hint(slot)
        state_summary = self._state_summary(state)
        kb_fields = self._kb_field_hint(slot)
        return (
            "You generate a JSON filter for querying a travel knowledge base. "
            "Return ONLY JSON, no prose. Do not invent fields outside the schema.\n"
            f"User query: {user_query}\n"
            f"Goal: {goal_text}\n"
            f"Current partial plan: {state_summary}\n"
            f"Slot: {slot_hint}\n"
            f"Allowed fields: {kb_fields}\n"
            "Output a single JSON object matching the allowed fields."
        )

    @staticmethod
    def _state_summary(state) -> str:
        if state is None:
            return ""
        parts = []
        city_seq = getattr(state, "city_sequence", None)
        if city_seq:
            parts.append("cities=" + "->".join(city_seq))
        outbound = getattr(state, "outbound_flight", None)
        if outbound:
            parts.append(f"outbound={outbound.get('origin')}->{outbound.get('destination')}")
        stays = getattr(state, "city_stays", {})
        stay_txt = [f"{c}:{s.get('name') if s else 'missing'}" for c, s in (stays or {}).items()]
        if stay_txt:
            parts.append("stays=" + ";".join(stay_txt))
        meals = getattr(state, "meals", {})
        filled_meals = []
        for day, slots in (meals or {}).items():
            for slot, meal in slots.items():
                if meal:
                    filled_meals.append(f"d{day}:{slot}:{meal.get('name')}")
        if filled_meals:
            parts.append("meals=" + ",".join(filled_meals[:6]))
        return " | ".join(parts)

    @staticmethod
    def _slot_hint(slot) -> str:
        try:
            return json.dumps(slot.__dict__, ensure_ascii=False, default=str)
        except Exception:
            return str(slot)

    @staticmethod
    def _kb_field_hint(slot) -> str:
        stype = getattr(slot, "type", None) or (slot.get("type") if isinstance(slot, dict) else None)
        stype = str(stype or "").lower()
        if stype == "flight":
            return "origin, destination, date, max_price, min_price, earliest_depart, latest_depart, max_duration, max_stops, sort_by (price|duration|depart|arrive), avoid_ids"
        if stype == "hotel":
            return "city, max_price, min_price, min_review, room_type, house_rules, min_occupancy, sort_by (price|review), avoid_ids"
        if stype == "restaurant":
            return "city, cuisines, max_cost, min_rating, meal_type, sort_by (rating|cost), avoid_ids"
        if stype == "attraction":
            return "city, categories, max_distance_km, sort_by (rating|distance|name), avoid_ids"
        return ""
