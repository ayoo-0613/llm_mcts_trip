from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from mcts.travel import filters

DEFAULT_BUDGET_RATIOS = {"flight": 0.35, "stay": 0.45, "meal": 0.15, "attraction": 0.05}
# Canonical budget keys TravelEnv expects; aliases map LLM outputs to these names.
BUDGET_CAP_KEY_ALIASES = {
    "flight_total": ("flight_total", "flight_cap", "flight_budget", "flight"),
    "stay_total": ("stay_total", "accommodation_total", "hotel_total", "stay_budget", "accommodation_budget"),
    "meal_total": ("meal_total", "food_total", "food_budget", "meal_budget", "meals_total"),
    "att_total": (
        "att_total",
        "attraction_total",
        "attractions_total",
        "entertainment_total",
        "activity_total",
        "activity_budget",
    ),
}
BUDGET_PER_UNIT_KEY_ALIASES = {
    "stay_per_night": ("stay_per_night", "accommodation_per_night", "hotel_per_night", "nightly_budget"),
    "meal_per_meal": ("meal_per_meal", "meal_per_slot", "food_per_meal", "per_meal_budget", "meal_budget_each"),
}


def call_llm(llm: Any, prompt: str) -> str:
    """Unified LLM call helper used by filter/phase planners."""
    if llm is None:
        raise RuntimeError("No callable LLM provided")
    if callable(llm):
        return llm(prompt)
    if hasattr(llm, "generate"):
        return llm.generate(prompt)  # type: ignore[attr-defined]
    if hasattr(llm, "__call__"):
        return llm(prompt)
    raise RuntimeError("LLM object is not callable")


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
class BudgetPlan:
    ratios: Dict[str, float] = field(default_factory=dict)
    caps: Dict[str, float] = field(default_factory=dict)
    per_unit: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhasePlan:
    budget_plan: BudgetPlan = field(default_factory=BudgetPlan)
    segment: Dict[str, Any] = field(default_factory=dict)
    stay: Dict[str, Any] = field(default_factory=dict)
    daily: Dict[str, Any] = field(default_factory=dict)


class PhasePlanGenerator:
    """Generate per-phase retrieval blueprints to guide KB retrieval in TravelEnv."""

    SCHEMA_VERSION = "budgetplan_v2"

    def __init__(self, llm_client: Optional[Any] = None, enable: bool = True, enable_budget_llm: bool = False):
        self.llm_client = llm_client
        self.enable = enable
        # If False (default), budget allocation falls back to defaults; LLM only shapes preferences.
        self.enable_budget_llm = enable_budget_llm
        self.cache: Dict[Tuple[str, str, str], PhasePlan] = {}
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
        key = (self._goal_sig(goal), phase_name, self.SCHEMA_VERSION)
        if key in self.cache:
            self.last_info = {"cache_hit": True, "used_llm": False, "phase": phase_name}
            return self.cache[key]

        plan = self._default_plan(goal)
        if self.enable and self.llm_client is not None:
            raw = self._call_llm(self._prompt(goal, phase_name, user_query))
            parsed = _safe_json_load(raw)
            plan = self._merge_and_validate(plan, parsed, phase_name, goal)
            self.last_info = {"cache_hit": False, "used_llm": True, "phase": phase_name}
        else:
            plan = self._merge_and_validate(plan, {}, phase_name, goal)
            self.last_info = {"cache_hit": False, "used_llm": False, "phase": phase_name}

        self.cache[key] = plan
        return plan

    def _call_llm(self, prompt: str) -> str:
        try:
            return call_llm(self.llm_client, prompt)
        except Exception:
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
            budget_plan=BudgetPlan(
                ratios=DEFAULT_BUDGET_RATIOS.copy(),
                caps={},
                per_unit={},
                notes={},
            ),
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
        budget_text = getattr(goal, "budget", None)
        days = getattr(goal, "duration_days", None)
        meals_per_day = 3
        try:
            meals_per_day = len(getattr(goal, "meal_slots", [])) or 3
        except Exception:
            meals_per_day = 3
        base_info = (
            f"Total budget: {budget_text}; Days: {days}; "
            f"City target: {getattr(goal, 'visiting_city_number', None)}; "
            f"Require flight: {getattr(goal, 'require_flight', None)}; "
            f"Return required: {getattr(goal, 'return_required', None)}; "
            f"Require accommodation: {getattr(goal, 'require_accommodation', None)}; "
            f"Meals per day: {meals_per_day}; "
            f"Attractions per day min/max: "
            f"{getattr(goal, 'attractions_per_day_min', None)}/"
            f"{getattr(goal, 'attractions_per_day_max', None)}"
        )
        if phase_name == "SEGMENT":
            schema = (
                "Return JSON with keys: "
                "budget_plan:{ratios:obj,caps:obj,per_unit:obj,notes:obj}, "
                "segment:{sort_by (price|duration|depart|arrive), "
                "max_stops (int or null), time_window (morning|afternoon|evening|null)}."
            )
        elif phase_name == "STAY":
            schema = (
                "Return JSON with keys: "
                "budget_plan:{ratios:obj,caps:obj,per_unit:obj,notes:obj}, "
                "stay:{sort_by (price|review), min_review (number|null), "
                "room_type (array), house_rules (array), max_price_per_night (number|null)}."
            )
        else:
            schema = (
                "Return JSON with keys: "
                "budget_plan:{ratios:obj,caps:obj,per_unit:obj,notes:obj}, "
                "meal:{cuisines:array, min_rating:number|null, max_cost:number|null, sort_by(rating|cost)}, "
                "attraction:{categories:array, max_distance_km:number|null, sort_by(rating|distance|name)}."
            )
        return (
            "You are generating a PHASE retrieval plan for a travel planning KB.\n"
            "Return ONLY JSON. Do not include city/origin/destination/date.\n"
            f"Context: {base_info}\n"
            f"User query: {user_query}\n"
            f"Goal: {goal_text}\n"
            f"Phase: {phase_name}\n"
            f"Schema: {schema}\n"
            "Rules: budget_plan.ratios should sum near 1.0; caps/per_unit must be non-negative numbers; "
            "provide actionable caps consistent with total budget.\n"
        )

    def _merge_segment_preferences(self, base_segment: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
        segment = dict(base_segment)
        if not isinstance(parsed, dict):
            return segment
        for key in ("sort_by", "max_stops", "time_window"):
            if key in parsed:
                segment[key] = parsed[key]
        return segment

    def _merge_stay_preferences(self, base_stay: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
        stay = dict(base_stay)
        if not isinstance(parsed, dict):
            return stay
        for key in ("sort_by", "min_review", "room_type", "house_rules", "max_price_per_night"):
            if key in parsed:
                stay[key] = parsed[key]
        return stay

    def _merge_daily_preferences(self, base_daily: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
        daily = dict(base_daily)
        if isinstance(parsed.get("meal"), dict):
            meal = dict(daily.get("meal", {}))
            for key in ("cuisines", "min_rating", "max_cost", "sort_by"):
                if key in parsed["meal"]:
                    meal[key] = parsed["meal"][key]
            daily["meal"] = meal
        if isinstance(parsed.get("attraction"), dict):
            attraction = dict(daily.get("attraction", {}))
            for key in ("categories", "max_distance_km", "sort_by"):
                if key in parsed["attraction"]:
                    attraction[key] = parsed["attraction"][key]
            daily["attraction"] = attraction
        return daily

    def _merge_and_validate(self, base: PhasePlan, parsed: Dict[str, Any], phase_name: str, goal) -> PhasePlan:
        # budget plan (optional LLM)
        if self.enable_budget_llm and isinstance(parsed.get("budget_plan"), dict):
            bp_raw = parsed.get("budget_plan", {}) or {}
        else:
            bp_raw = {}
        base.budget_plan = self._normalize_budget_plan(bp_raw, goal)

        if phase_name == "SEGMENT":
            base.segment = self._merge_segment_preferences(base.segment, parsed)
        elif phase_name == "STAY":
            base.stay = self._merge_stay_preferences(base.stay, parsed)
        else:
            base.daily = self._merge_daily_preferences(base.daily, parsed)
        return base

    def _canonicalize_keys(self, raw: Dict[str, Any], alias_map: Dict[str, Tuple[str, ...]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        alias_lookup = {alias: canonical for canonical, aliases in alias_map.items() for alias in aliases}
        canonicalized: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for key, value in (raw or {}).items():
            target = alias_lookup.get(key, None)
            if target:
                if target not in canonicalized:
                    canonicalized[target] = value
            else:
                extras[key] = value
        return canonicalized, extras

    def _coerce_positive_number_map(self, raw: Dict[str, Any]) -> Dict[str, float]:
        clean: Dict[str, float] = {}
        for key, value in (raw or {}).items():
            try:
                clean[str(key)] = max(0.0, float(value))
            except Exception:
                continue
        return clean

    def _normalize_ratios(self, ratios_in: Dict[str, Any]) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        if isinstance(ratios_in, dict):
            ratios = self._coerce_positive_number_map(ratios_in)
        if not ratios:
            ratios = DEFAULT_BUDGET_RATIOS.copy()
        total = sum(ratios.values())
        if total <= 0:
            ratios = DEFAULT_BUDGET_RATIOS.copy()
            total = sum(ratios.values())
        return {k: v / total for k, v in ratios.items()}

    def _normalize_budget_plan(self, bp_raw: Dict[str, Any], goal) -> BudgetPlan:
        total = getattr(goal, "budget", None)
        days = getattr(goal, "duration_days", 0) or 0
        meals_per_day = 3
        try:
            meals_per_day = len(getattr(goal, "meal_slots", [])) or 3
        except Exception:
            meals_per_day = 3

        bp = BudgetPlan()
        ratios_in = bp_raw.get("ratios") if isinstance(bp_raw, dict) else {}
        caps_in = bp_raw.get("caps") if isinstance(bp_raw, dict) else {}
        per_in = bp_raw.get("per_unit") if isinstance(bp_raw, dict) else {}
        notes_in = bp_raw.get("notes") if isinstance(bp_raw, dict) else {}

        ratios = self._normalize_ratios(ratios_in)
        bp.ratios = ratios

        caps_raw, caps_extra_input = self._canonicalize_keys(
            caps_in if isinstance(caps_in, dict) else {}, BUDGET_CAP_KEY_ALIASES
        )
        caps: Dict[str, float] = self._coerce_positive_number_map(caps_raw)

        def _cap_key(key: str, ratio_key: str) -> float:
            if key in caps:
                return caps[key]
            if total is None:
                return 0.0
            return max(0.0, float(total) * ratios.get(ratio_key, 0.0))

        caps["flight_total"] = _cap_key("flight_total", "flight")
        caps["stay_total"] = _cap_key("stay_total", "stay")
        caps["meal_total"] = _cap_key("meal_total", "meal")
        caps["att_total"] = _cap_key("att_total", "attraction")

        if total is not None:
            total_f = float(total)
            for k in list(caps.keys()):
                caps[k] = min(caps[k], total_f)
            caps_sum = sum(caps.values())
            if caps_sum > total_f and caps_sum > 0:
                scale = total_f / caps_sum
                for k in caps:
                    caps[k] *= scale
        bp.caps = caps

        per_raw, per_extra_input = self._canonicalize_keys(
            per_in if isinstance(per_in, dict) else {}, BUDGET_PER_UNIT_KEY_ALIASES
        )
        per_unit: Dict[str, float] = self._coerce_positive_number_map(per_raw)
        nights = max(1, days if isinstance(days, int) else 1)
        if "stay_per_night" not in per_unit:
            per_unit["stay_per_night"] = caps.get("stay_total", 0.0) / float(nights or 1)
        total_meals = max(1, days * meals_per_day) if days else meals_per_day
        if "meal_per_meal" not in per_unit:
            per_unit["meal_per_meal"] = caps.get("meal_total", 0.0) / float(total_meals)
        bp.per_unit = per_unit
        notes = dict(notes_in) if isinstance(notes_in, dict) else {}
        if caps_extra_input:
            notes["raw_extra_caps_input"] = caps_extra_input
        if per_extra_input:
            notes["raw_extra_per_unit_input"] = per_extra_input
        bp.notes = notes
        return self._clean_budget_plan(bp)

    def _clean_budget_plan(self, bp: BudgetPlan) -> BudgetPlan:
        clean_caps: Dict[str, float] = {}
        for key in ("flight_total", "stay_total", "meal_total", "att_total"):
            if key in bp.caps and bp.caps[key] is not None:
                try:
                    clean_caps[key] = float(bp.caps[key])
                except Exception:
                    continue
        clean_per: Dict[str, float] = {}
        for key in ("stay_per_night", "meal_per_meal"):
            if key in bp.per_unit and bp.per_unit[key] is not None:
                try:
                    clean_per[key] = float(bp.per_unit[key])
                except Exception:
                    continue
        extra_caps = {k: v for k, v in bp.caps.items() if k not in clean_caps} if bp.caps else {}
        extra_per = {k: v for k, v in bp.per_unit.items() if k not in clean_per} if bp.per_unit else {}
        bp.caps = clean_caps
        bp.per_unit = clean_per
        if extra_caps or extra_per:
            bp.notes = dict(bp.notes or {})
            bp.notes["raw_extra_caps"] = extra_caps
            bp.notes["raw_extra_per_unit"] = extra_per
        return bp


@dataclass
class FilterBuildResult:
    filt: Dict[str, Any]
    event: Dict[str, Any]
    plan: Optional[Any] = None
    uncapped_filter: Optional[Dict[str, Any]] = None


class FilterPolicy:
    def build_slot_filter(
        self,
        goal,
        state,
        slot,
        user_query: str = "",
    ) -> FilterBuildResult:
        raise NotImplementedError


class PhasePlanFilterPolicy(FilterPolicy):
    """Default policy: use PhasePlan (LLM) + budget clamps to build slot filters."""

    def __init__(
        self,
        phase_planner: Optional[PhasePlanGenerator] = None,
        cost_estimator: Optional[Callable[[Any], float]] = None,
        total_days: int = 0,
        meal_slots: Optional[List[str]] = None,
        debug: bool = False,
        enable_budget_caps: bool = False,
    ):
        self.phase_planner = phase_planner
        self.cost_estimator = cost_estimator or (lambda state: 0.0)
        self.total_days = total_days
        self.meal_slots = meal_slots or ["breakfast", "lunch", "dinner"]
        self.debug = debug
        self.enable_budget_caps = enable_budget_caps
        self.last_event: Dict[str, Any] = {}
        self._last_plan: Optional[PhasePlan] = None

    def build_slot_filter(self, goal, state, slot, user_query: str = "") -> FilterBuildResult:
        base = filters.default_filter(getattr(slot, "type", None), goal, state, slot)
        plan = None
        plan_info: Dict[str, Any] = {}
        phase_name = state.phase.name if hasattr(state, "phase") and state.phase is not None else ""
        if self.phase_planner is not None:
            plan = self.phase_planner.get_or_build(goal, phase_name, user_query=user_query)
            plan_info = getattr(self.phase_planner, "last_info", {}) or {}

        has_preferences = bool(getattr(goal, "preferences", []) or [])
        filt = self._merge_plan_preferences(base, plan, slot, allow_preferences=has_preferences)
        uncapped_filt = dict(filt)
        filt["avoid_ids"] = self._avoid_ids_for_slot(state, slot)
        if self.enable_budget_caps:
            filt = self._apply_phase_budget_caps(goal, state, slot, filt, plan)
            filt = self._budget_adjust(goal, state, slot, filt)

        ftype = "restaurant" if getattr(slot, "type", None) == "meal" else getattr(slot, "type", None)
        filt = filters.validate_and_normalize(filt, ftype, goal=goal, state=state, slot=slot)

        event: Dict[str, Any] = {
            "plan_cache_hit": plan_info.get("cache_hit", False),
            "plan_used_llm": plan_info.get("used_llm", False),
            "phase": phase_name,
        }
        try:
            spend = self._phase_spend_so_far(state)
            total_left = None
            if getattr(goal, "budget", None) is not None:
                total_left = max(0.0, float(goal.budget) - float(self.cost_estimator(state)))
            bp = getattr(plan, "budget_plan", None) if plan else None
            event["budget"] = {
                "global_left": total_left,
                "phase_spend": spend,
                "phase_caps": getattr(bp, "caps", None) if bp else None,
                "phase_per_unit": getattr(bp, "per_unit", None) if bp else None,
            }
        except Exception:
            if self.debug:
                raise
        self.last_event = event
        self._last_plan = plan
        return FilterBuildResult(filt=filt, event=event, plan=plan, uncapped_filter=uncapped_filt)

    def _merge_plan_preferences(
        self, base: Dict[str, Any], plan: Optional[PhasePlan], slot, allow_preferences: bool = True
    ) -> Dict[str, Any]:
        filt = dict(base)
        if plan is None:
            return filt
        if getattr(slot, "type", None) == "flight":
            seg_cfg = getattr(plan, "segment", {}) or {}
            filt["sort_by"] = seg_cfg.get("sort_by", filt.get("sort_by"))
            filt["max_stops"] = seg_cfg.get("max_stops", filt.get("max_stops"))
        elif getattr(slot, "type", None) == "hotel":
            stay_cfg = getattr(plan, "stay", {}) or {}
            filt["sort_by"] = stay_cfg.get("sort_by", filt.get("sort_by"))
            filt["min_review"] = stay_cfg.get("min_review", filt.get("min_review"))
            if stay_cfg.get("room_type") is not None:
                filt["room_type"] = stay_cfg["room_type"]
            if stay_cfg.get("house_rules") is not None:
                filt["house_rules"] = stay_cfg["house_rules"]
            if stay_cfg.get("max_price_per_night") is not None:
                filt["max_price"] = stay_cfg["max_price_per_night"]
        elif getattr(slot, "type", None) in ("meal", "restaurant"):
            daily_cfg = getattr(plan, "daily", {}) or {}
            meal_cfg = daily_cfg.get("meal", {}) if isinstance(daily_cfg, dict) else {}
            filt["sort_by"] = meal_cfg.get("sort_by", filt.get("sort_by"))
            if allow_preferences:
                filt["cuisines"] = meal_cfg.get("cuisines", filt.get("cuisines"))
            else:
                filt["cuisines"] = []
            filt["min_rating"] = meal_cfg.get("min_rating", filt.get("min_rating"))
            filt["max_cost"] = meal_cfg.get("max_cost", filt.get("max_cost"))
            if getattr(slot, "meal_type", None):
                filt["meal_type"] = slot.meal_type
        elif getattr(slot, "type", None) == "attraction":
            daily_cfg = getattr(plan, "daily", {}) or {}
            att_cfg = daily_cfg.get("attraction", {}) if isinstance(daily_cfg, dict) else {}
            filt["sort_by"] = att_cfg.get("sort_by", filt.get("sort_by"))
            filt["categories"] = att_cfg.get("categories", filt.get("categories"))
            filt["max_distance_km"] = att_cfg.get("max_distance_km", filt.get("max_distance_km"))
        return filt

    def relax_budget_for_slot(self, filt: Dict[str, Any], slot) -> Optional[Dict[str, Any]]:
        """Return a filter with budget caps removed for this slot, or None if nothing to change."""
        stype = getattr(slot, "type", None)
        relaxed = dict(filt)
        changed = False
        if stype in ("flight", "hotel"):
            if relaxed.get("max_price") is not None:
                relaxed["max_price"] = None
                changed = True
        elif stype in ("meal", "restaurant", "attraction"):
            if relaxed.get("max_cost") is not None:
                relaxed["max_cost"] = None
                changed = True
        return relaxed if changed else None

    @staticmethod
    def _restaurant_ids_all(state) -> set:
        ids = set()
        for day_map in getattr(state, "meals", {}).values():
            for meal in day_map.values():
                if meal and meal.get("id") is not None:
                    ids.add(meal["id"])
        return ids

    @staticmethod
    def _attraction_ids_all(state) -> set:
        ids = set()
        for day_map in getattr(state, "attractions", {}).values():
            for att in day_map.values():
                if att and att.get("id") is not None:
                    ids.add(att["id"])
        return ids

    def _avoid_ids_for_slot(self, state, slot) -> List[str]:
        stype = getattr(slot, "type", None)
        if stype in ("meal", "restaurant"):
            return list(self._restaurant_ids_all(state))
        if stype == "attraction":
            return list(self._attraction_ids_all(state))
        return []

    def _budget_adjust(self, goal, state, slot, filt: Dict[str, Any]) -> Dict[str, Any]:
        if getattr(goal, "budget", None) is None:
            return filt
        try:
            left = float(goal.budget) - float(self.cost_estimator(state))
        except Exception:
            left = 0.0
        left = max(0.0, left)
        updated = dict(filt)
        if getattr(slot, "type", None) == "hotel":
            cap = left * 0.5
            if cap > 0 and (updated.get("max_price") is None or updated["max_price"] > cap):
                updated["max_price"] = cap
        elif getattr(slot, "type", None) in ("meal", "restaurant"):
            meals_per_day = max(1, len(self.meal_slots) or 1)
            cap = left / max(1, (self.total_days * meals_per_day)) * 1.2 if self.total_days else left
            if cap > 0 and (updated.get("max_cost") is None or updated["max_cost"] > cap):
                updated["max_cost"] = cap
        return updated

    def _phase_spend_so_far(self, state) -> Dict[str, float]:
        spend = {"flight": 0.0, "stay": 0.0, "meal": 0.0, "attraction": 0.0}

        for seg in getattr(state, "segment_modes", {}).values():
            if isinstance(seg, dict) and seg.get("mode") == "flight":
                detail = seg.get("detail", {}) or {}
                try:
                    if detail.get("price") is not None:
                        spend["flight"] += float(detail["price"])
                    elif detail.get("cost") is not None:
                        spend["flight"] += float(detail["cost"])
                except Exception:
                    continue

        for stay in getattr(state, "city_stays", {}).values():
            if stay and stay.get("price") is not None:
                try:
                    spend["stay"] += float(stay["price"])
                except Exception:
                    continue

        for day_map in getattr(state, "meals", {}).values():
            for meal in day_map.values():
                if meal and meal.get("cost") is not None:
                    try:
                        spend["meal"] += float(meal["cost"])
                    except Exception:
                        continue

        for day_map in getattr(state, "attractions", {}).values():
            for att in day_map.values():
                if att and att.get("cost") is not None:
                    try:
                        spend["attraction"] += float(att["cost"])
                    except Exception:
                        continue
        return spend

    def _apply_phase_budget_caps(self, goal, state, slot, filt: Dict[str, Any], plan: Any) -> Dict[str, Any]:
        """Clamp filters with phase/global budget caps derived from PhasePlan."""
        if getattr(goal, "budget", None) is None or plan is None:
            return filt
        bp = getattr(plan, "budget_plan", None)
        if bp is None:
            return filt

        caps = getattr(bp, "caps", {}) or {}
        per_unit = getattr(bp, "per_unit", {}) or {}
        spend = self._phase_spend_so_far(state)
        try:
            total_left = max(0.0, float(goal.budget) - float(self.cost_estimator(state)))
        except Exception:
            total_left = 0.0

        out = dict(filt)
        stype = getattr(slot, "type", None)

        if stype == "flight":
            flight_left = max(0.0, float(caps.get("flight_total", total_left)) - spend["flight"])
            cap_val = self._safe_min_caps(out.get("max_price"), total_left, flight_left, floor=0.0)
            out["max_price"] = cap_val

        elif stype == "hotel":
            stay_left = max(0.0, float(caps.get("stay_total", total_left)) - spend["stay"])
            per_night = per_unit.get("stay_per_night", None)
            cap_val = self._safe_min_caps(out.get("max_price"), total_left, stay_left, per_night, floor=0.0)
            out["max_price"] = cap_val

        elif stype in ("meal", "restaurant"):
            meal_left = max(0.0, float(caps.get("meal_total", total_left)) - spend["meal"])
            per_meal = per_unit.get("meal_per_meal", None)
            cap_val = self._safe_min_caps(out.get("max_cost"), total_left, meal_left, per_meal, floor=0.0)
            out["max_cost"] = cap_val

        elif stype == "attraction":
            att_left = max(0.0, float(caps.get("att_total", total_left)) - spend["attraction"])
            cap_val = self._safe_min_caps(out.get("max_cost"), total_left, att_left, floor=0.0)
            out["max_cost"] = cap_val

        return out

    @staticmethod
    def _safe_min_caps(*vals, floor: float = 0.0) -> float:
        cleaned = []
        for v in vals:
            if v is None:
                continue
            try:
                cleaned.append(float(v))
            except Exception:
                continue
        if not cleaned:
            return floor
        return max(floor, min(cleaned))
