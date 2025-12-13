from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


def _safe_json_load(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                return {}
    return {}


@dataclass
class PhasePlan:
    segment: Dict[str, Any] = field(default_factory=dict)
    stay: Dict[str, Any] = field(default_factory=dict)
    daily: Dict[str, Any] = field(default_factory=dict)


class PhasePlanGenerator:
    """Generate per-phase retrieval plans. LLM is called once per phase+goal."""

    def __init__(self, llm: Optional[Any] = None, enable: bool = True):
        self.llm = llm
        self.enable = enable
        self.cache: Dict[Tuple[str, str], PhasePlan] = {}
        self.last_info: Dict[str, Any] = {}

    def _goal_sig(self, goal) -> str:
        parts = [
            getattr(goal, "origin", "") or "",
            getattr(goal, "destination", "") or "",
            str(getattr(goal, "start_date", "") or ""),
            str(getattr(goal, "duration_days", "") or ""),
            str(getattr(goal, "budget", "") or ""),
            "|".join(getattr(goal, "preferences", []) or []),
            "|".join(getattr(goal, "must_visit_cities", []) or []),
        ]
        return "|".join(parts)

    def get_or_build(self, goal, phase_name: str, user_query: str = "") -> PhasePlan:
        key = (self._goal_sig(goal), phase_name)
        if key in self.cache:
            self.last_info = {"cache_hit": True, "used_llm": False, "phase": phase_name}
            return self.cache[key]

        plan = self._default_plan(goal)
        if self.enable and self.llm is not None:
            raw = self._call_llm(self._prompt(goal, phase_name, user_query))
            parsed = _safe_json_load(raw)
            plan = self._merge_and_validate(plan, parsed, phase_name, goal)
            self.last_info = {"cache_hit": False, "used_llm": True, "phase": phase_name}
        else:
            self.last_info = {"cache_hit": False, "used_llm": False, "phase": phase_name}

        self.cache[key] = plan
        return plan

    def _call_llm(self, prompt: str) -> str:
        if callable(self.llm):
            return self.llm(prompt)
        if hasattr(self.llm, "generate"):
            return self.llm.generate(prompt)  # type: ignore[attr-defined]
        if hasattr(self.llm, "__call__"):
            return self.llm(prompt)
        return ""

    def _default_plan(self, goal) -> PhasePlan:
        budget = getattr(goal, "budget", None)
        daily_meal_max = None
        if budget:
            try:
                daily_meal_max = max(
                    20.0,
                    float(budget) / max(1.0, float(getattr(goal, "duration_days", 3))) / 6.0,
                )
            except Exception:
                daily_meal_max = None
        return PhasePlan(
            segment={
                "sort_by": "price",
                "max_stops": 1,
                "time_window": None,
            },
            stay={
                "sort_by": "price",
                "min_review": 3.0,
                "room_type": [],
                "house_rules": [],
                "max_price_per_night": None,
            },
            daily={
                "meal": {
                    "sort_by": "rating",
                    "min_rating": None,
                    "max_cost": daily_meal_max,
                    "cuisines": [],
                },
                "attraction": {
                    "sort_by": "rating",
                    "categories": [],
                    "max_distance_km": None,
                },
            },
        )

    def _prompt(self, goal, phase_name: str, user_query: str) -> str:
        goal_text = goal.as_text() if hasattr(goal, "as_text") else str(goal)
        if phase_name == "SEGMENT":
            schema = (
                "Return JSON with keys: sort_by (price|duration|depart|arrive), "
                "max_stops (int or null), time_window (morning|afternoon|evening|null)."
            )
        elif phase_name == "STAY":
            schema = (
                "Return JSON with keys: sort_by (price|review), min_review (number|null), "
                "room_type (array), house_rules (array), max_price_per_night (number|null)."
            )
        else:
            schema = (
                "Return JSON with keys: "
                "meal:{cuisines:array, min_rating:number|null, max_cost:number|null, sort_by(rating|cost)}, "
                "attraction:{categories:array, max_distance_km:number|null, sort_by(rating|distance|name)}."
            )
        return (
            "You are generating a PHASE retrieval plan for a travel planning KB.\n"
            "Return ONLY JSON. Do not include city/origin/destination/date.\n"
            f"User query: {user_query}\n"
            f"Goal: {goal_text}\n"
            f"Phase: {phase_name}\n"
            f"Schema: {schema}\n"
        )

    def _merge_and_validate(self, base: PhasePlan, parsed: Dict[str, Any], phase_name: str, goal) -> PhasePlan:
        if phase_name == "SEGMENT":
            seg = dict(base.segment)
            for k in ("sort_by", "max_stops", "time_window"):
                if k in parsed:
                    seg[k] = parsed[k]
            base.segment = seg
        elif phase_name == "STAY":
            stay = dict(base.stay)
            for k in ("sort_by", "min_review", "room_type", "house_rules", "max_price_per_night"):
                if k in parsed:
                    stay[k] = parsed[k]
            base.stay = stay
        else:
            daily = dict(base.daily)
            if isinstance(parsed.get("meal"), dict):
                meal = dict(daily.get("meal", {}))
                for k in ("cuisines", "min_rating", "max_cost", "sort_by"):
                    if k in parsed["meal"]:
                        meal[k] = parsed["meal"][k]
                daily["meal"] = meal
            if isinstance(parsed.get("attraction"), dict):
                att = dict(daily.get("attraction", {}))
                for k in ("categories", "max_distance_km", "sort_by"):
                    if k in parsed["attraction"]:
                        att[k] = parsed["attraction"][k]
                daily["attraction"] = att
            base.daily = daily
        return base
