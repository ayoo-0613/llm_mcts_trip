from __future__ import annotations

import copy
import itertools
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from mcts.travel import filters
from mcts.travel.filtering import FilterPolicy, PhasePlanFilterPolicy, PhasePlanGenerator
from mcts.travel.knowledge_base import TravelKnowledgeBase, TripGoal
from mcts.travel.relaxation import RelaxationController


# ----------------------------
# Phase 定义：高层 → 交通 → 住宿 → 每日行程
# ----------------------------
class Phase(Enum):
    CITY = 1       # 选择城市序列
    SEGMENT = 2    # 选择城市间交通/segment mode
    STAY = 3       # 为每个城市选住宿
    DAILY = 4      # 填三餐+景点，直到可以 finish


# ----------------------------
# Reward & 槽位配置
# ----------------------------
DEFAULT_REWARD_CFG = {
    "step_bonus": 0.1,
    "budget_violation_penalty": -20.0,
    "missing_flight_penalty": -10.0,
    "missing_return_penalty": -10.0,
    "missing_stay_penalty": -8.0,
    "missing_city_penalty": -8.0,
    "missing_segment_penalty": -6.0,
    "missing_must_city_penalty": -6.0,
    "meal_missing_penalty": -4.0,
    "attraction_missing_penalty": -4.0,
    "duplicate_meal_penalty": -3.0,
    "duplicate_restaurant_across_days_penalty": -2.0,
    "duplicate_attraction_penalty": -3.0,
    "preference_bonus": 2.0,
    "poi_bonus": 1.0,
    "finish_success_bonus": 6.0,
    "finish_fail_penalty": -6.0,
}

MEAL_SLOTS = ["breakfast", "lunch", "dinner"]
ATTRACTION_SLOTS = ["spot"]  # single attraction slot per day


# ----------------------------
# 状态定义
# ----------------------------
@dataclass
class TravelState:
    outbound_flight: Optional[Dict] = None
    return_flight: Optional[Dict] = None
    accommodation: Optional[Dict] = None  # legacy single-city stay
    city_stays: Dict[str, Optional[Dict]] = field(default_factory=dict)  # city -> accommodation

    meals: Dict[int, Dict[str, Optional[Dict]]] = field(default_factory=dict)        # day -> slot -> restaurant
    attractions: Dict[int, Dict[str, Optional[Dict]]] = field(default_factory=dict)  # day -> slot -> attraction

    city_sequence: List[str] = field(default_factory=list)
    segment_modes: Dict[int, Dict] = field(default_factory=dict)  # segment idx -> {"mode": str, "detail": Dict}

    cost: float = 0.0
    preference_matches: int = 0
    violations: List[str] = field(default_factory=list)
    is_terminal: bool = False

    # 新增：当前所处阶段
    phase: Phase = Phase.CITY

    def clone(self) -> "TravelState":
        return TravelState(
            outbound_flight=copy.deepcopy(self.outbound_flight),
            return_flight=copy.deepcopy(self.return_flight),
            accommodation=copy.deepcopy(self.accommodation),
            city_stays=copy.deepcopy(self.city_stays),
            meals=copy.deepcopy(self.meals),
            attractions=copy.deepcopy(self.attractions),
            city_sequence=list(self.city_sequence),
            segment_modes=copy.deepcopy(self.segment_modes),
            cost=self.cost,
            preference_matches=self.preference_matches,
            violations=list(self.violations),
            is_terminal=self.is_terminal,
            phase=self.phase,
        )

    def signature(self) -> str:
        """Deterministic signature for caching."""
        parts: List[str] = [f"phase:{self.phase.name}"]
        parts.append("cities:" + "|".join(self.city_sequence))

        seg_parts = []
        for idx in sorted(self.segment_modes):
            seg = self.segment_modes[idx]
            mode = seg.get("mode") if isinstance(seg, dict) else None
            detail = seg.get("detail", {}) if isinstance(seg, dict) else {}
            seg_id = ""
            if isinstance(detail, dict):
                seg_id = str(detail.get("id") or detail.get("origin") or detail.get("destination") or "")
            seg_parts.append(f"{idx}:{mode}:{seg_id}")
        parts.append("segs:" + ";".join(seg_parts))

        stay_parts = []
        for city in sorted(self.city_stays):
            stay = self.city_stays[city]
            stay_id = stay.get("id") if stay else "na"
            stay_parts.append(f"{city}:{stay_id}")
        parts.append("stays:" + ";".join(stay_parts))

        meal_parts = []
        for day in sorted(self.meals):
            slot_map = self.meals[day]
            slot_txt = ",".join(
                f"{slot}:{slot_map[slot]['id'] if slot_map[slot] else 'na'}" for slot in sorted(slot_map)
            )
            meal_parts.append(f"d{day}:{slot_txt}")
        parts.append("meals:" + "|".join(meal_parts))

        att_parts = []
        for day in sorted(self.attractions):
            slot_map = self.attractions[day]
            slot_txt = ",".join(
                f"{slot}:{slot_map[slot]['id'] if slot_map[slot] else 'na'}" for slot in sorted(slot_map)
            )
            att_parts.append(f"d{day}:{slot_txt}")
        parts.append("atts:" + "|".join(att_parts))

        parts.append(f"cost:{self.cost:.2f}")
        return "||".join(parts)


@dataclass(frozen=True)
class Slot:
    type: str  # "flight"|"hotel"|"meal"|"attraction"|"finish"|"city"
    day: Optional[int] = None
    meal_type: Optional[str] = None
    seg: Optional[int] = None
    city: Optional[str] = None
    date: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None

    def signature(self) -> str:
        return (
            f"{self.type}|d{self.day}|slot:{self.meal_type}|seg:{self.seg}|"
            f"{self.city}|{self.date}|{self.origin}->{self.destination}"
        )


class TravelEnv:
    def __init__(self, knowledge_base: TravelKnowledgeBase, goal: TripGoal,
                 max_steps: int = 40, top_k: int = 5,
                 reward_cfg: Optional[Dict] = None, debug: bool = False,
                 candidate_cap: int = 80,
                 filter_policy: Optional[FilterPolicy] = None,
                 phase_planner: Optional[PhasePlanGenerator] = None,
                 relaxer: Optional[RelaxationController] = None,
                 relax_max_tries: int = 6,
                 user_query: str = "",
                 log_filter_usage: bool = False,
                 enable_budget_caps: bool = False):
        self.kb = knowledge_base
        self.goal = goal
        self.max_steps = max_steps
        self.top_k = top_k
        self.reward_cfg = reward_cfg.copy() if reward_cfg is not None else DEFAULT_REWARD_CFG.copy()
        self.debug = debug
        self.candidate_cap = candidate_cap
        self.user_query = user_query
        self.log_filter_usage = log_filter_usage

        self.total_days = goal.duration_days or 3
        self.meal_slots = MEAL_SLOTS
        self.attraction_slots = ATTRACTION_SLOTS
        self.phase_planner = phase_planner
        self.filter_policy: FilterPolicy = filter_policy or PhasePlanFilterPolicy(
            phase_planner=self.phase_planner,
            cost_estimator=self._estimate_cost,
            total_days=self.total_days,
            meal_slots=self.meal_slots,
            debug=self.debug,
            enable_budget_caps=enable_budget_caps,
        )

        self.base_state = self._empty_state()
        self.base_history: List[str] = []

        # 初始化城市/phase（单城直接跳过城市搜索）
        self._transport_cache: Dict[Tuple[str, str], bool] = {}
        self._ensure_destination_required()
        self._expand_state_locations()
        self._init_cities_and_phase(self.base_state)

        self.state = self.base_state.clone()
        self.history: List[str] = []
        self.steps = 0
        self.action_payloads: Dict[str, Tuple] = {}
        self.last_info: Dict = {}
        self.last_slot: Optional[Slot] = None
        self._transport_cache: Dict[Tuple[str, str], bool] = {}
        self._last_phase_plan = None

        self.relaxer = relaxer or RelaxationController(max_tries=relax_max_tries, goal=goal)
        self.candidate_cache: Dict[Tuple[str, Slot], Dict[str, Any]] = {}
        self.filter_cache: Dict[Tuple[str, Slot], Dict[str, Any]] = {}
        self.filter_events: List[Dict[str, Any]] = []
        self.executed_filter_events: List[Dict[str, Any]] = []
        self.filter_event_keys: set = set()
        self.executed_filter_event_keys: set = set()

    # ----------------------------
    # 初始化 & helper
    # ----------------------------
    def _empty_state(self) -> TravelState:
        meals = {day: {slot: None for slot in self.meal_slots}
                 for day in range(1, self.total_days + 1)}
        attractions = {day: {slot: None for slot in self.attraction_slots}
                       for day in range(1, self.total_days + 1)}
        return TravelState(meals=meals, attractions=attractions)

    def _ensure_destination_required(self) -> None:
        """Only build candidate cities when destination is a state; leave city untouched."""
        dest = self.goal.destination
        if not dest:
            return

        # Destination is a state → seed candidate_cities from that state.
        if self.kb.is_state(dest):
            dest_cities = self.kb.cities_in_state(dest)
            cap = max(self.top_k * 3, (self.goal.visiting_city_number or 1) * 10)
            self.goal.candidate_cities = dest_cities[:cap]
            return

        # Destination is a city: do not build candidate_cities. If multi-city requested, raise.
        city_target = self.goal.visiting_city_number or 1
        if city_target > 1:
            raise ValueError(
                f"Destination '{dest}' is a city but visiting_city_number={city_target}. "
                "No candidate_cities will be built automatically; please provide candidate_city list "
                "or change destination to a state."
            )

    def _has_transport_cached(self, src: str, dst: str, require_flight: bool = False) -> bool:
        key = (self.kb._normalize_city(src), self.kb._normalize_city(dst), require_flight)
        if hasattr(self, "_transport_cache") and key in self._transport_cache:
            return self._transport_cache[key]
        has = self.kb.has_any_transport(src, dst, require_flight=require_flight)
        if hasattr(self, "_transport_cache"):
            self._transport_cache[key] = has
        return has

    def _expand_state_locations(self) -> None:
        """Only expand when destination is a state; city destinations stay untouched."""
        if not self.goal.destination or not self.kb.is_state(self.goal.destination):
            return

        cities = self.kb.cities_in_state(self.goal.destination)
        cap = max(self.top_k * 3, (self.goal.visiting_city_number or 1) * 10)
        self.goal.candidate_cities = cities[:cap]


    def _init_cities_and_phase(self, state: TravelState) -> None:
        """
        根据 Goal 决定：
        - 是否需要 CITY 阶段（多城市/城市不确定）
        - 单城市场景直接设置 city_sequence 并从 SEGMENT 开始
        - fixed_city_order 场景直接用给定顺序并从 SEGMENT 开始
        """
        city_target = self.goal.visiting_city_number or 1
        dest_norm = self.kb._normalize_city(self.goal.destination) if self.goal.destination else None
        dest_is_city = bool(dest_norm and dest_norm in getattr(self.kb, "city_set_norm", {}))
        dest_is_state = bool(dest_norm and dest_norm in getattr(self.kb, "state_norm_map", {}))

        # 1) fixed_city_order：已经完全给定城市顺序 → 不需要 CITY 搜索
        if self.goal.fixed_city_order:
            state.city_sequence = list(self.goal.fixed_city_order)
            state.phase = Phase.SEGMENT
            return

        # 2) 单城市 & destination 是明确城市 → 直接锁定城市，跳过 CITY
        if city_target == 1 and self.goal.destination and dest_is_city and not dest_is_state:
            state.city_sequence = [self.goal.destination]
            state.phase = Phase.SEGMENT
            return

        # 3) 否则：需要 CITY 阶段由 MCTS+LLM 搜索城市集合
        state.phase = Phase.CITY

    def reset(self, goal: Optional[TripGoal] = None) -> Tuple[str, List[str]]:
        if goal is not None:
            self.goal = goal
            self.total_days = goal.duration_days or 3
            if isinstance(self.filter_policy, PhasePlanFilterPolicy):
                self.filter_policy.total_days = self.total_days
                self.filter_policy.meal_slots = self.meal_slots
            self.base_state = self._empty_state()
            self.base_history = []
            self._last_phase_plan = None
            self._ensure_destination_required()
            self._expand_state_locations()
            self._init_cities_and_phase(self.base_state)
            self.candidate_cache = {}
            self.filter_cache = {}
            self.filter_events = []
            self.executed_filter_events = []
            self.filter_event_keys = set()
            self.executed_filter_event_keys = set()
            if hasattr(self.relaxer, "goal"):
                self.relaxer.goal = goal

        self.state = self.base_state.clone()
        # 再次确保 fixed_city_order 下 city_sequence 正确
        if not self.state.city_sequence and self.goal.fixed_city_order:
            self.state.city_sequence = list(self.goal.fixed_city_order)

        self.state.cost = self._estimate_cost(self.state)
        self.history = list(self.base_history)
        self.steps = 0
        self._transport_cache = {}
        self._last_phase_plan = None
        if not getattr(self, "_preserve_filter_log", False):
            self.filter_events = []
            self.executed_filter_events = []
            self.filter_event_keys = set()
            self.executed_filter_event_keys = set()
        self._preserve_filter_log = False

        obs = self._observation(self.state)
        valid_actions = self.get_valid_actions()
        return obs, valid_actions

    def apply_action(self, action: str):
        """Apply action to the anchor state after planning."""
        prev_events = list(self.executed_filter_events)
        prev_keys = set(self.executed_filter_event_keys)
        self._preserve_filter_log = True
        self.reset()
        obs, reward, done, history, valid_actions = self.step(action)
        # 保留之前的过滤日志，便于输出完整的 filter_usage，仅记录真实执行路径
        self.executed_filter_events = prev_events + list(self.filter_events)
        self.filter_events = list(self.executed_filter_events)
        self.executed_filter_event_keys = prev_keys | set(self.filter_event_keys)
        self.filter_event_keys = set(self.executed_filter_event_keys)
        self.base_state = self.state.clone()
        self.base_history = list(self.history)
        return obs, reward, done, history, valid_actions

    def replay(self, history: List[str]) -> None:
        """
        Reset env and replay a sequence of actions to rebuild state.
        Used by MCTS simulations to align world state with history.
        """
        self.reset()
        for act in history:
            if act is None:
                break
            self.step(act)

    # ----------------------------
    # Cost / helper 统计
    # ----------------------------
    def _estimate_cost(self, state: TravelState) -> float:
        """Single-source cost estimator to avoid double counting."""
        cost = 0.0

        # Transport: only from segment_modes
        for seg in (state.segment_modes or {}).values():
            if not isinstance(seg, dict):
                continue
            detail = seg.get("detail", {}) or {}
            if not isinstance(detail, dict):
                continue
            if detail.get("price") is not None:
                cost += float(detail["price"])
            elif detail.get("cost") is not None:
                cost += float(detail["cost"])

        # Stay: prefer city_stays; fallback to legacy accommodation only if city_stays empty
        if state.city_stays:
            for stay in state.city_stays.values():
                if stay is not None and stay.get("price") is not None:
                    cost += float(stay["price"])
        else:
            if state.accommodation is not None and state.accommodation.get("price") is not None:
                cost += float(state.accommodation["price"])

        # Meals
        for day in (state.meals or {}).values():
            for meal in (day or {}).values():
                if meal is not None and meal.get("cost") is not None:
                    cost += float(meal["cost"])

        # Attractions (if priced)
        for day in (state.attractions or {}).values():
            for att in (day or {}).values():
                if att is not None and att.get("cost") is not None:
                    cost += float(att["cost"])

        return cost

    def _ensure_violation(self, state: TravelState, violation: str) -> None:
        if violation not in state.violations:
            state.violations.append(violation)

    def _matches_preference(self, restaurant: Dict) -> bool:
        meal_cuisines = []
        if hasattr(self.goal, "constraints"):
            meal_cuisines = (self.goal.constraints.get("meal", {}) or {}).get("cuisines", [])
        if not meal_cuisines:
            return False
        cuisines = str(restaurant.get("cuisines") or restaurant.get("Cuisines") or "").lower()
        return any(pref.lower() in cuisines for pref in meal_cuisines)

    def _failure_reasons(self, state: TravelState) -> Dict[str, List[str]]:
        reasons: Dict[str, List[str]] = {
            "missing_meals": [],
            "missing_attractions": [],
            "missing_stays": [],
            "missing_segments": [],
            "budget_over": [],
        }
        segments = self._segments(state)
        for idx, src, dst in segments:
            if idx not in state.segment_modes:
                reasons["missing_segments"].append(f"seg{idx}:{src}->{dst}")
        for city in state.city_sequence:
            if self.goal.require_accommodation and state.city_stays.get(city) is None:
                reasons["missing_stays"].append(city)
        for day in range(1, self.total_days + 1):
            for slot, meal in state.meals.get(day, {}).items():
                if meal is None:
                    reasons["missing_meals"].append(f"d{day}:{slot}")
            city = self._city_for_day(state, day)
            if self.kb.get_attractions(city):
                for slot, att in state.attractions.get(day, {}).items():
                    if att is None:
                        reasons["missing_attractions"].append(f"d{day}:{slot}")
        if self.goal.budget is not None and state.cost > self.goal.budget:
            reasons["budget_over"].append(f"{state.cost:.1f}>{self.goal.budget:.1f}")
        return {k: v for k, v in reasons.items() if v}

    def _count_attractions_day(self, state: TravelState, day: int) -> int:
        return sum(1 for a in state.attractions.get(day, {}).values() if a is not None)

    def _restaurant_ids_day(self, state: TravelState, day: int) -> set:
        return {m["id"] for m in state.meals.get(day, {}).values() if m}

    def _restaurant_ids_all(self, state: TravelState) -> set:
        ids = set()
        for day_map in state.meals.values():
            for m in day_map.values():
                if m:
                    ids.add(m["id"])
        return ids

    def _attraction_ids_all(self, state: TravelState) -> set:
        ids = set()
        for day_map in state.attractions.values():
            for a in day_map.values():
                if a:
                    ids.add(a["id"])
        return ids

    def _city_for_day(self, state: TravelState, day: int) -> Optional[str]:
        seq = state.city_sequence or self.goal.fixed_city_order or self.goal.must_visit_cities
        if not seq:
            return self.goal.destination
        idx = min(len(seq) - 1, int((day - 1) * len(seq) / max(1, self.total_days)))
        return seq[idx]

    def _segments(self, state: TravelState) -> List[Tuple[int, str, str]]:
        seq = state.city_sequence
        segments: List[Tuple[int, str, str]] = []
        if seq:
            segments.append((0, self.goal.origin, seq[0]))
            for i in range(1, len(seq)):
                segments.append((i, seq[i - 1], seq[i]))
            if self.goal.return_required:
                segments.append((len(seq), seq[-1], self.goal.origin))
        elif self.goal.destination:
            # fallback single-destination routing
            segments.append((0, self.goal.origin, self.goal.destination))
            if self.goal.return_required:
                segments.append((1, self.goal.destination, self.goal.origin))
        return segments

    def _allowed_transport_modes(self) -> List[str]:
        allowed = self.goal.transport_allowed_modes or ["flight", "taxi", "self-driving"]
        forbidden = set(m.lower() for m in (self.goal.transport_forbidden_modes or []))

        if hasattr(self.goal, "constraints"):
            tcons = self.goal.constraints.get("transport", {}) or {}
            if tcons.get("allow"):
                allowed = list(tcons["allow"])
            forbidden |= set(m.lower() for m in (tcons.get("forbid") or []))

        return [m for m in allowed if m not in forbidden]

    # ----------------------------
    # Phase 完成判定 & 推进
    # ----------------------------
    def _city_phase_done(self, state: TravelState) -> bool:
        city_target = self.goal.visiting_city_number or 1
        if len(state.city_sequence) < city_target:
            return False
        must_norm = {self.kb._normalize_city(c) for c in self.goal.must_visit_cities}
        if must_norm:
            seq_norm = {self.kb._normalize_city(c) for c in state.city_sequence}
            if not must_norm.issubset(seq_norm):
                return False
        return True

    def _segment_phase_done(self, state: TravelState) -> bool:
        segments = self._segments(state)
        for idx, _, _ in segments:
            if idx not in state.segment_modes:
                return False
        return True

    def _stay_phase_done(self, state: TravelState) -> bool:
        if not self.goal.require_accommodation:
            return True
        for city in state.city_sequence:
            if state.city_stays.get(city) is None:
                return False
        return True

    def _advance_phase_if_ready(self, state: TravelState) -> None:
        # 按顺序推进 Phase
        if state.phase == Phase.CITY and self._city_phase_done(state):
            state.phase = Phase.SEGMENT
        if state.phase == Phase.SEGMENT and self._segment_phase_done(state):
            state.phase = Phase.STAY
        if state.phase == Phase.STAY and self._stay_phase_done(state):
            state.phase = Phase.DAILY

    # ----------------------------
    # Observation（保持原始文本形式）
    # ----------------------------
    def _observation(self, state: TravelState) -> str:
        parts = [self.goal.as_text(), f"Phase: {state.phase.name}"]

        if state.city_sequence:
            parts.append(f"Cities selected: {' -> '.join(state.city_sequence)}")
        if state.segment_modes:
            seg_txt = []
            for idx, seg in state.segment_modes.items():
                detail = seg.get("detail", {})
                mode = seg.get("mode")
                if isinstance(detail, dict) and "origin" in detail and "destination" in detail:
                    seg_txt.append(f"seg{idx}:{detail['origin']}->{detail['destination']} via {mode}")
                else:
                    seg_txt.append(f"seg{idx}:{mode}")
            parts.append("Transport: " + "; ".join(seg_txt))
        for city, stay in state.city_stays.items():
            if stay:
                price_val = stay.get("price")
                price_txt = f"${price_val:.0f}" if price_val is not None else "$?"
                parts.append(f"Stay {city}: {stay.get('name')} {stay.get('room_type')} {price_txt}")

        for day in range(1, self.total_days + 1):
            day_meals = state.meals[day]
            meal_txt = ", ".join(f"{slot}:{meal['name']}" for slot, meal in day_meals.items() if meal)
            if meal_txt:
                parts.append(f"Day {day} meals: {meal_txt}")
            day_atts = state.attractions[day]
            att_txt = ", ".join(f"{slot}:{att['name']}" for slot, att in day_atts.items() if att)
            if att_txt:
                parts.append(f"Day {day} attractions: {att_txt}")

        pending = []
        city_target = self.goal.visiting_city_number or 1
        if len(state.city_sequence) < city_target:
            pending.append(f"cities missing {city_target - len(state.city_sequence)}")
        segments = self._segments(state)
        for idx, src, dst in segments:
            if idx not in state.segment_modes:
                pending.append(f"segment {idx} {src}->{dst} mode")
        for city in state.city_sequence:
            if state.city_stays.get(city) is None and self.goal.require_accommodation:
                pending.append(f"stay in {city}")
        for day in range(1, self.total_days + 1):
            meals_missing = [slot for slot, meal in state.meals[day].items() if meal is None]
            if meals_missing:
                pending.append(f"day{day} meals {len(meals_missing)} missing")
            att_count = self._count_attractions_day(state, day)
            if att_count < self.goal.attractions_per_day_min:
                pending.append(f"day{day} attractions missing {self.goal.attractions_per_day_min - att_count}")
        if pending:
            parts.append("Pending: " + ", ".join(pending))

        if self.goal.budget is not None:
            budget_left = self.goal.budget - self._estimate_cost(state)
            parts.append(f"Budget left estimate: {budget_left:.0f}")
        parts.append(f"Allowed transport: {', '.join(self._allowed_transport_modes())}")
        return " | ".join(parts)

    # ----------------------------
    # Slot & candidates
    # ----------------------------
    def _next_slot(self, state: TravelState) -> Optional[Slot]:
        """Decide the next slot to fill based on phase and pending requirements."""
        self._advance_phase_if_ready(state)
        city_target = self.goal.visiting_city_number or 1
        if state.phase == Phase.CITY and len(state.city_sequence) < city_target:
            return Slot(type="city")

        if state.phase == Phase.SEGMENT:
            for idx, src, dst in self._segments(state):
                if idx not in state.segment_modes:
                    return Slot(
                        type="flight",
                        seg=idx,
                        origin=src,
                        destination=dst,
                        date=self.goal.start_date,
                    )

        if state.phase == Phase.STAY and self.goal.require_accommodation:
            for city in state.city_sequence:
                if state.city_stays.get(city) is None:
                    return Slot(type="hotel", city=city)

        if state.phase == Phase.DAILY:
            for day in range(1, self.total_days + 1):
                missing_slots = [slot for slot, meal in state.meals[day].items() if meal is None]
                if missing_slots:
                    city = self._city_for_day(state, day)
                    return Slot(type="meal", day=day, meal_type=missing_slots[0], city=city)
            for day in range(1, self.total_days + 1):
                if self._count_attractions_day(state, day) < self.goal.attractions_per_day_min:
                    for slot_name, att in state.attractions[day].items():
                        if att is None:
                            city = self._city_for_day(state, day)
                            return Slot(type="attraction", day=day, meal_type=slot_name, city=city)

        if self._can_finish(state):
            return Slot(type="finish")
        return None

    def _actions_from_candidates(self, slot: Slot, candidates: List[Dict]) -> List[str]:
        """Convert KB candidates into action strings + payloads."""
        actions: List[str] = []
        self.action_payloads = {}
        if not candidates:
            return actions

        if slot.type == "flight":
            for f in candidates:
                seg_idx = slot.seg if slot.seg is not None else -1
                price_val = float(f.get("price", 0) or 0.0)
                action = (
                    f"move:seg{seg_idx}:flight:{f.get('id')} {f.get('origin')}->{f.get('destination')} "
                    f"${price_val:.0f} {f.get('depart', '?')}-{f.get('arrive', '?')}"
                )
                self.action_payloads[action] = ("segment_mode", seg_idx, "flight", f)
                actions.append(action)
        elif slot.type == "hotel":
            for stay in candidates:
                city = slot.city or stay.get("city")
                price_val = float(stay.get("price", 0) or 0.0)
                action = (
                    f"stay:{city}:{stay.get('id')} {stay.get('name')} "
                    f"{stay.get('room_type')} ${price_val:.0f}"
                )
                self.action_payloads[action] = ("stay_city", city, stay)
                actions.append(action)
        elif slot.type == "meal":
            for rest in candidates:
                cost_val = float(rest.get("cost", 0) or 0.0)
                action = (
                    f"eat:d{slot.day}:{slot.meal_type}:{rest.get('id')} "
                    f"{rest.get('name')} {rest.get('cuisines')} ${cost_val:.0f} rating {rest.get('rating', 0)}"
                )
                self.action_payloads[action] = ("meal", slot.day, slot.meal_type, rest)
                actions.append(action)
        elif slot.type == "attraction":
            for att in candidates:
                city = att.get("city") or slot.city
                action = f"visit:d{slot.day}:{slot.meal_type or 'spot'}:{att.get('id')} {att.get('name')} @ {city}"
                self.action_payloads[action] = ("attraction", slot.day, slot.meal_type or "spot", att)
                actions.append(action)
        elif slot.type == "finish":
            actions.append("finish")
        return actions

    @staticmethod
    def _assert_candidates_respect_caps(slot: Slot, filt: Dict[str, Any], candidates: List[Dict]) -> None:
        stype = getattr(slot, "type", None)
        if stype in ("flight", "hotel"):
            mp = filt.get("max_price")
            if mp is not None:
                try:
                    mp_f = float(mp)
                    bad = [
                        c for c in candidates if c.get("price") is not None and float(c["price"]) > mp_f + 1e-6
                    ]
                    if bad:
                        raise RuntimeError(
                            f"[CAP VIOLATION] {stype} max_price={mp_f} bad0_price={bad[0].get('price')} bad0={bad[0]}"
                        )
                except Exception:
                    pass
        if stype in ("meal", "restaurant", "attraction"):
            mc = filt.get("max_cost")
            if mc is not None:
                try:
                    mc_f = float(mc)
                    bad = [
                        c for c in candidates if c.get("cost") is not None and float(c["cost"]) > mc_f + 1e-6
                    ]
                    if bad:
                        raise RuntimeError(
                            f"[CAP VIOLATION] {stype} max_cost={mc_f} bad0_cost={bad[0].get('cost')} bad0={bad[0]}"
                        )
                except Exception:
                    pass

    @staticmethod
    def _slot_to_dict(slot: Slot) -> Dict[str, Any]:
        try:
            return asdict(slot)
        except Exception:
            return {
                "type": getattr(slot, "type", None),
                "day": getattr(slot, "day", None),
                "meal_type": getattr(slot, "meal_type", None),
                "seg": getattr(slot, "seg", None),
                "city": getattr(slot, "city", None),
                "date": getattr(slot, "date", None),
                "origin": getattr(slot, "origin", None),
                "destination": getattr(slot, "destination", None),
            }

    def _ground_fallback_actions(self, slot: Slot) -> List[str]:
        """Non-flight transport fallback for segments when flight candidates are empty."""
        actions: List[str] = []
        if not slot.origin or not slot.destination:
            return actions
        distance = self.kb.distance_km(slot.origin, slot.destination)
        if distance is None:
            return actions
        for mode in self._allowed_transport_modes():
            if mode == "flight":
                continue
            cost = distance
            seg_idx = slot.seg if slot.seg is not None else -1
            action = (
                f"move:seg{seg_idx}:{mode}:{slot.origin}->{slot.destination} "
                f"{distance:.0f}km cost {cost:.0f}"
            )
            payload_detail = {
                "origin": slot.origin,
                "destination": slot.destination,
                "distance": distance,
                "cost": cost,
                "fallback_nonflight": True,
            }
            self.action_payloads[action] = ("segment_mode", seg_idx, mode, payload_detail)
            actions.append(action)
        return actions

    def _compute_actions_for_slot(self, slot: Optional[Slot]) -> List[str]:
        """Core action builder that also tracks the current slot for callers."""
        self.last_slot = slot
        if isinstance(getattr(self, "last_info", None), dict):
            try:
                self.last_info["slot"] = self._slot_to_dict(slot) if slot else None
            except Exception:
                self.last_info["slot"] = None
        if slot is None:
            self.action_payloads = {}
            return []
        key = (self.state.signature(), slot)

        if key in self.candidate_cache:
            cached = self.candidate_cache[key]
            self.action_payloads = copy.deepcopy(cached.get("payloads", {}))
            if self.log_filter_usage:
                filt = self.filter_cache.get(key)
                event = cached.get("event", {})
                print(
                    f"[FILTER] slot={slot.type} cache_hit=True plan_used_llm={event.get('plan_used_llm', False)} "
                    f"filter={filt} event={event}"
                )
            if cached.get("event") and key not in self.filter_event_keys:
                self.filter_events.append(dict(cached["event"], cache_hit=True))
                self.filter_event_keys.add(key)
            return list(cached.get("actions", []))

        actions: List[str] = []
        candidates: List[Dict] = []
        relaxed = False
        filt: Dict[str, Any] = {}
        policy_event: Dict[str, Any] = {}

        if slot.type == "city":
            self.action_payloads = {}
            actions = self._build_city_actions(self.state, relaxed=False)
            if not actions:
                actions = self._build_city_actions(self.state, relaxed=True)
            self._last_phase_plan = None
        else:
            res = self.filter_policy.build_slot_filter(self.goal, self.state, slot, user_query=self.user_query)
            filt = res.filt if res is not None else filters.default_filter(slot.type, self.goal, self.state, slot)
            self._last_phase_plan = getattr(res, "plan", None) if res else None
            policy_event = getattr(res, "event", {}) if res else {}
            uncapped_filt = getattr(res, "uncapped_filter", None) if res else None
            self.filter_cache[key] = filt
            if self.log_filter_usage:
                print(
                    f"[FILTER] slot={slot.type} phase={self.state.phase.name} "
                    f"event={policy_event} filter={filt}"
                )
            candidates = self.kb.query(slot, filt, self.state, cap=self.candidate_cap)
            try:
                self._assert_candidates_respect_caps(slot, filt, candidates)
            except Exception as e:
                if self.debug:
                    raise
                else:
                    print(f"[WARN] {e}")
            relaxed = False
            if not candidates:
                # Soft-cap fallback: re-query without budget caps if provided
                if uncapped_filt and uncapped_filt != filt:
                    filt = uncapped_filt
                    candidates = self.kb.query(slot, filt, self.state, cap=self.candidate_cap)
                    policy_event["soft_cap_relaxed"] = True
                    if self.log_filter_usage:
                        print(f"[FILTER] slot={slot.type} soft-cap relaxed -> {len(candidates)} candidates")
                # Budget-only relaxation: drop price caps for this slot before invoking relaxer
                if not candidates and hasattr(self.filter_policy, "relax_budget_for_slot"):
                    relaxed_budget_filt = self.filter_policy.relax_budget_for_slot(filt, slot)
                    if relaxed_budget_filt:
                        filt = relaxed_budget_filt
                        candidates = self.kb.query(slot, filt, self.state, cap=self.candidate_cap)
                        policy_event["budget_relaxed"] = True
                        if self.log_filter_usage:
                            print(f"[FILTER] slot={slot.type} budget-relaxed -> {len(candidates)} candidates")
                if not candidates:
                    candidates = self.relaxer.relax_and_query(self.kb, slot, filt, self.state, cap=self.candidate_cap)
                    relaxed = True
                    if self.log_filter_usage:
                        print(f"[FILTER] slot={slot.type} relaxed -> {len(candidates)} candidates")
            actions = self._actions_from_candidates(slot, candidates)
            if not actions and slot.type == "flight":
                actions = self._ground_fallback_actions(slot)

            # update cache with the final filter actually used after any soft-cap relaxation
            self.filter_cache[key] = filt

        event = dict(policy_event or {})
        event.update(
            {
                "slot": self._slot_to_dict(slot),
                "relaxed": relaxed if slot.type != "city" else False,
                "filter": filt if slot.type != "city" else {},
                "candidates": len(candidates) if slot.type != "city" else len(actions),
                "actions": len(actions),
            }
        )
        if key not in self.filter_event_keys:
            self.filter_events.append(event)
            self.filter_event_keys.add(key)
        self.candidate_cache[key] = {
            "actions": actions,
            "payloads": copy.deepcopy(self.action_payloads),
            "event": event,
        }
        if self.debug:
            print(f"[DEBUG] Slot {slot.type} produced {len(actions)} actions (cache miss)")
        return actions

    def get_valid_actions_with_slot(self) -> Tuple[Optional[Slot], List[str], Dict[str, Tuple]]:
        """
        Return the next slot alongside its valid actions and payloads.

        This is useful for planners (e.g., MCTS) that need slot context to build
        structured prompts or priors. It shares the same caching logic as
        ``get_valid_actions``.
        """
        slot = self._next_slot(self.state)
        actions = self._compute_actions_for_slot(slot)
        return slot, actions, copy.deepcopy(self.action_payloads)

    def get_valid_actions(self) -> List[str]:
        _, actions, _ = self.get_valid_actions_with_slot()
        return actions

    # ----------------------------
    # Phase-aware action builder（严格→放宽）
    # ----------------------------
    def _build_valid_actions(self, state: TravelState) -> List[str]:
        """Backwards-compatible wrapper for legacy callers."""
        self.state = state
        return self.get_valid_actions()

    def _build_phase_actions(self, state: TravelState, relaxed: bool) -> List[str]:
        # Legacy path disabled; always use slot-based action builder.
        self.state = state
        return self.get_valid_actions()

    def _log_empty_actions(self, state: TravelState) -> None:
        """仅在 debug 下打印空动作原因，便于定位。"""
        if not self.debug:
            return
        phase = state.phase.name if isinstance(state.phase, Phase) else state.phase
        pending = []
        segments = self._segments(state)
        for idx, src, dst in segments:
            if idx not in state.segment_modes:
                pending.append(f"seg{idx}:{src}->{dst}")
        for day in range(1, self.total_days + 1):
            meals_missing = [slot for slot, meal in state.meals[day].items() if meal is None]
            att_missing = self.goal.attractions_per_day_min - self._count_attractions_day(state, day)
            pending.append(f"d{day}:meals_missing={len(meals_missing)},att_missing={max(att_missing,0)}")
        print(
            "[DEBUG] No valid actions | "
            f"phase={phase}, cities={state.city_sequence}, "
            f"candidates={self.goal.candidate_cities}, must={self.goal.must_visit_cities}, "
            f"priority={self.goal.priority_cities}, pending=({'; '.join(pending)})"
        )

    def _build_city_actions(self, state: TravelState, relaxed: bool = False) -> List[str]:
        actions: List[str] = []
        city_target = self.goal.visiting_city_number or 1
        origin = self.goal.origin

        # 已经满足城市数量 → 不需要更多动作
        if len(state.city_sequence) >= city_target:
            return actions

        # 组合模式：一次性决定全组城市
        candidates = self.goal.candidate_cities or self.kb.get_candidate_cities(
            destination_hint=self.goal.destination,
            must_visit=self.goal.must_visit_cities,
            priority=self.goal.priority_cities,
            top_k=max(self.top_k * 2, city_target),
        )
        for must in self.goal.must_visit_cities:
            if must not in candidates:
                candidates.insert(0, must)
        for pri in self.goal.priority_cities:
            if pri not in candidates:
                candidates.append(pri)

        # 不重复选择已在序列里的城市
        prefix = list(state.city_sequence)
        remaining_needed = city_target - len(prefix)
        pool = [c for c in candidates if c not in prefix]
        # 控制组合爆炸：限制搜索池大小
        pool = pool[: max(self.top_k * 3, remaining_needed)]

        combos: List[Tuple[List[str], float]] = []

        for combo in itertools.permutations(pool, remaining_needed):
            seq = prefix + list(combo)
            score = self._score_city_bundle(seq, origin)
            if score is None:
                continue
            combos.append((seq, score))
            if len(combos) > 500:  # 防止极端爆炸
                break

        # relaxed 模式：允许重复城市兜底
        if relaxed and not combos:
            for combo in itertools.permutations(pool + prefix, remaining_needed):
                seq = prefix + list(combo)
                score = self._score_city_bundle(seq, origin, allow_repeat=True)
                if score is None:
                    continue
                combos.append((seq, score))
                if len(combos) > 500:
                    break

        # 按分数排序取前 top_k
        combos = sorted(combos, key=lambda x: x[1], reverse=True)[: self.top_k]
        for seq, score in combos:
            action = f"choose_city_bundle:[{','.join(seq)}]"
            self.action_payloads[action] = ("choose_city_bundle", seq)
            actions.append(action)

        return actions

    def _score_city_bundle(self, seq: List[str], origin: str, allow_repeat: bool = False) -> Optional[float]:
        """对城市组合进行快速打分；返回 None 表示不可行。"""
        if not seq:
            return None
        # 去重检查
        if not allow_repeat and len(set(seq)) != len(seq):
            return None

        score = 0.0

        # origin -> 第一城 的可达性（如 require_flight 则强制航班）
        if self.goal.require_flight:
            if not self._has_transport_cached(origin, seq[0], require_flight=True):
                return None
            score += 1.0
        else:
            if self.kb.has_any_transport(origin, seq[0], require_flight=False):
                score += 0.5

        # 城市间连通性（允许距离矩阵）
        for i in range(len(seq) - 1):
            if not self.kb.has_any_transport(seq[i], seq[i + 1], require_flight=False):
                return None
            score += 0.4

        # 返程连通性（如果需要返程）
        if self.goal.return_required:
            if self.goal.require_flight:
                # 返程必须可飞，否则直接丢弃该组合
                if not self._has_transport_cached(seq[-1], origin, require_flight=True):
                    return None
                score += 0.5
            else:
                if self.kb.has_any_transport(seq[-1], origin, require_flight=False):
                    score += 0.2

        # 数据覆盖：住宿/餐饮/景点桶
        for city in seq:
            has_stay = bool(self.kb.get_accommodations(city, top_k=1))
            has_rest = bool(self.kb.get_restaurants(city, preferences=self.goal.preferences, top_k=1))
            has_att = bool(self.kb.get_attractions(city, top_k=1))
            score += 0.1 * has_stay + 0.1 * has_rest + 0.1 * has_att

        return score

    def _build_segment_actions(self, state: TravelState, relaxed: bool = False) -> List[str]:
        raise RuntimeError("Legacy _build_segment_actions called; use get_valid_actions() instead")


    def _build_stay_actions(self, state: TravelState, relaxed: bool = False) -> List[str]:
        raise RuntimeError("Legacy _build_stay_actions called; use get_valid_actions() instead")

    def _build_daily_actions(self, state: TravelState, relaxed: bool = False) -> List[str]:
        raise RuntimeError("Legacy _build_daily_actions called; use get_valid_actions() instead")

    # ----------------------------
    # Success / Finish 条件
    # ----------------------------
    def get_goal(self) -> str:
        return self.goal.as_text()

    def is_success(self, state: Optional[TravelState] = None) -> bool:
        state = state or self.state
        city_target = self.goal.visiting_city_number or 1
        if len(state.city_sequence) < city_target:
            return False
        must_norm = {self.kb._normalize_city(c) for c in self.goal.must_visit_cities}
        if must_norm:
            seq_norm = {self.kb._normalize_city(c) for c in state.city_sequence}
            if not must_norm.issubset(seq_norm):
                return False

        segments = self._segments(state)
        for idx, _, _ in segments:
            if idx not in state.segment_modes:
                return False

        if self.goal.require_flight:
            if segments:
                first_seg = state.segment_modes.get(0)
                if not first_seg or first_seg.get("mode") != "flight":
                    return False
                if self.goal.return_required:
                    last_idx = segments[-1][0]
                    last_seg = state.segment_modes.get(last_idx)
                    if not last_seg or last_seg.get("mode") != "flight":
                        return False

        if self.goal.require_accommodation:
            for city in state.city_sequence:
                if state.city_stays.get(city) is None:
                    return False
        for day in range(1, self.total_days + 1):
            if any(meal is None for meal in state.meals[day].values()):
                return False
            city = self._city_for_day(state, day)
            # 若该城市无景点数据，则跳过景点约束
            if self.kb.get_attractions(city):
                att_count = self._count_attractions_day(state, day)
                if att_count < self.goal.attractions_per_day_min:
                    return False
        if self.goal.budget is not None and state.cost > self.goal.budget:
            return False
        # 记录失败原因以便调试/输出
        self.last_failure_reasons = {}
        return True

    def _can_finish(self, state: Optional[TravelState] = None) -> bool:
        """结构性检查（不含预算），用于是否允许出现 finish 动作。"""
        state = state or self.state
        city_target = self.goal.visiting_city_number or 1
        if len(state.city_sequence) < city_target:
            return False
        must_norm = {self.kb._normalize_city(c) for c in self.goal.must_visit_cities}
        if must_norm:
            seq_norm = {self.kb._normalize_city(c) for c in state.city_sequence}
            if not must_norm.issubset(seq_norm):
                return False
        segments = self._segments(state)
        for idx, _, _ in segments:
            if idx not in state.segment_modes:
                return False
        if self.goal.require_accommodation:
            for city in state.city_sequence:
                if state.city_stays.get(city) is None:
                    return False
        for day in range(1, self.total_days + 1):
            if any(meal is None for meal in state.meals[day].values()):
                return False
            att_count = self._count_attractions_day(state, day)
            if att_count < self.goal.attractions_per_day_min:
                return False
        return True

    # ----------------------------
    # Step: 应用 action + Phase 推进 + reward
    # ----------------------------
    def step(self, action: str):
        self.steps += 1
        reward = self.reward_cfg.get("step_bonus", 0.0)
        self.last_info = {}

        # finish 逻辑：先判断能不能 finish
        if action == "finish":
            if not self._can_finish(self.state):
                # 不允许提前 finish：强制负奖励，继续规划
                reward = -1.0
                obs = self._observation(self.state)
                valid_actions = self.get_valid_actions()
                slot_dict = self._slot_to_dict(self.last_slot) if self.last_slot else None
                self.last_info = {
                    "violations": list(self.state.violations),
                    "cost": self.state.cost,
                    "slot": slot_dict,
                }
                return obs, reward, False, self.history, valid_actions

            # 可以 finish：终局打分
            self.history.append(action)
            reward += self._finish_and_score(self.state)
            obs = self._observation(self.state)
            slot_dict = self._slot_to_dict(self.last_slot) if self.last_slot else None
            self.last_info = {
                "violations": list(self.state.violations),
                "cost": self.state.cost,
                "slot": slot_dict,
            }
            return obs, reward, True, self.history, []

        # 非 finish 动作
        payload = self.action_payloads.get(action)
        if payload is None:
            # 非法动作，轻微惩罚，重新给 action 列表
            reward = -1.0
            obs = self._observation(self.state)
            valid_actions = self.get_valid_actions()
            slot_dict = self._slot_to_dict(self.last_slot) if self.last_slot else None
            self.last_info = {
                "violations": list(self.state.violations),
                "cost": self.state.cost,
                "slot": slot_dict,
            }
            return obs, reward, False, self.history, valid_actions

        kind = payload[0]
        if kind == "choose_city":
            _, city = payload
            if city not in self.state.city_sequence:
                self.state.city_sequence.append(city)
                # route 变化 → 清空已有 segment 决策
                self.state.segment_modes = {}
                self.state.outbound_flight = None
                self.state.return_flight = None

        elif kind == "segment_mode":
            _, seg_idx, mode, detail = payload
            self.state.segment_modes[seg_idx] = {"mode": mode, "detail": detail}
            segments = self._segments(self.state)
            last_idx = segments[-1][0] if segments else None
            if mode == "flight":
                if seg_idx == 0:
                    self.state.outbound_flight = detail
                if self.goal.return_required and last_idx is not None and seg_idx == last_idx:
                    self.state.return_flight = detail
            else:
                # 非航班兜底：如果要求航班但使用了其他交通，记录违规
                if self.goal.require_flight and (seg_idx == 0 or (self.goal.return_required and seg_idx == last_idx)):
                    self._ensure_violation(self.state, "missing_flight")

        elif kind == "stay_city":
            _, city, stay = payload
            if self.state.city_stays.get(city) is None:
                self.state.city_stays[city] = stay
            # legacy accommodation not used once city_stays is set
            self.state.accommodation = None

        elif kind == "choose_city_bundle":
            _, seq = payload
            self.state.city_sequence = list(seq)
            # route 变化 → 清空已有决策
            self.state.segment_modes = {}
            self.state.outbound_flight = None
            self.state.return_flight = None
            self.state.city_stays = {}
            # 清空每日餐饮/景点
            self.state.meals = {day: {slot: None for slot in self.meal_slots}
                                for day in range(1, self.total_days + 1)}
            self.state.attractions = {day: {slot: None for slot in self.attraction_slots}
                                      for day in range(1, self.total_days + 1)}
            self.state.preference_matches = 0
            self.state.violations = []
            self.state.is_terminal = False
            self.state.cost = 0.0

        elif kind == "meal":
            _, day, slot, rest = payload
            if self.state.meals[day][slot] is None:
                day_ids = self._restaurant_ids_day(self.state, day)
                if rest["id"] in day_ids:
                    reward += self.reward_cfg.get("duplicate_meal_penalty", 0.0)
                elif rest["id"] in self._restaurant_ids_all(self.state):
                    reward += self.reward_cfg.get("duplicate_restaurant_across_days_penalty", 0.0)
                else:
                    self.state.meals[day][slot] = rest
                    if self._matches_preference(rest):
                        self.state.preference_matches += 1
                        reward += self.reward_cfg.get("preference_bonus", 0.0)

        elif kind == "attraction":
            _, day, slot, attr = payload
            if self.state.attractions[day][slot] is None:
                if attr["id"] in self._attraction_ids_all(self.state):
                    reward += self.reward_cfg.get("duplicate_attraction_penalty", 0.0)
                elif self._count_attractions_day(self.state, day) < self.goal.attractions_per_day_max:
                    self.state.attractions[day][slot] = attr
                    reward += self.reward_cfg.get("poi_bonus", 0.0)

        # 更新 cost & 预算惩罚
        self.state.cost = self._estimate_cost(self.state)
        if self.goal.budget is not None and self.state.cost > self.goal.budget:
            self._ensure_violation(self.state, "budget")
            reward += self.reward_cfg.get("budget_violation_penalty", 0.0)

        self.history.append(action)

        # Phase 推进：在当前步骤之后，看看是否可以进入下一阶段
        self._advance_phase_if_ready(self.state)

        done = self.is_success(self.state) or self.steps >= self.max_steps
        if done:
            reward += self._finish_and_score(self.state)

        obs = self._observation(self.state)
        valid_actions = self.get_valid_actions() if not done else []
        failure_reasons = self._failure_reasons(self.state) if not self.is_success(self.state) else {}
        slot_dict = self._slot_to_dict(self.last_slot) if self.last_slot else None
        self.last_info = {
            "violations": list(self.state.violations),
            "cost": self.state.cost,
            "failure_reasons": failure_reasons,
            "slot": slot_dict,
        }
        return obs, reward, done, self.history, valid_actions

    def _finish_and_score(self, state: TravelState) -> float:
        state.is_terminal = True
        state.cost = self._estimate_cost(state)
        reward = 0.0

        city_target = self.goal.visiting_city_number or 1
        if len(state.city_sequence) < city_target:
            self._ensure_violation(state, "city_count")
            reward += (city_target - len(state.city_sequence)) * self.reward_cfg.get("missing_city_penalty", 0.0)

        must_norm = {self.kb._normalize_city(c) for c in self.goal.must_visit_cities}
        if must_norm:
            seq_norm = {self.kb._normalize_city(c) for c in state.city_sequence}
            missing_must = [c for c in must_norm if c not in seq_norm]
            if missing_must:
                self._ensure_violation(state, "must_city")
                reward += len(missing_must) * self.reward_cfg.get("missing_must_city_penalty", 0.0)

        segments = self._segments(state)
        for idx, src, dst in segments:
            if idx not in state.segment_modes:
                self._ensure_violation(state, f"segment{idx}")
                reward += self.reward_cfg.get("missing_segment_penalty", 0.0)

        if self.goal.require_flight:
            if segments:
                first_seg = state.segment_modes.get(0)
                if not first_seg or first_seg.get("mode") != "flight":
                    self._ensure_violation(state, "outbound")
                    reward += self.reward_cfg.get("missing_flight_penalty", 0.0)
                if self.goal.return_required:
                    last_idx = segments[-1][0]
                    last_seg = state.segment_modes.get(last_idx)
                    if not last_seg or last_seg.get("mode") != "flight":
                        self._ensure_violation(state, "return")
                        reward += self.reward_cfg.get("missing_return_penalty", 0.0)

        if self.goal.require_accommodation:
            for city in state.city_sequence:
                if state.city_stays.get(city) is None:
                    self._ensure_violation(state, f"stay_{city}")
                    reward += self.reward_cfg.get("missing_stay_penalty", 0.0)

        for day in range(1, self.total_days + 1):
            missing_meals = sum(1 for meal in state.meals[day].values() if meal is None)
            if missing_meals:
                self._ensure_violation(state, f"day{day}_meals")
                reward += missing_meals * self.reward_cfg.get("meal_missing_penalty", 0.0)
            att_count = self._count_attractions_day(state, day)
            if att_count < self.goal.attractions_per_day_min:
                missing_att = self.goal.attractions_per_day_min - att_count
                self._ensure_violation(state, f"day{day}_attractions")
                reward += missing_att * self.reward_cfg.get("attraction_missing_penalty", 0.0)
            # duplicate penalties
            day_ids = self._restaurant_ids_day(state, day)
            if len(day_ids) < len([m for m in state.meals[day].values() if m]):
                self._ensure_violation(state, f"day{day}_duplicate_meals")
                reward += self.reward_cfg.get("duplicate_meal_penalty", 0.0)

        # cross-day duplicate restaurants
        all_rest = self._restaurant_ids_all(state)
        total_meals = sum(1 for day_map in state.meals.values() for v in day_map.values() if v)
        if len(all_rest) < total_meals:
            self._ensure_violation(state, "duplicate_restaurants")
            reward += self.reward_cfg.get("duplicate_restaurant_across_days_penalty", 0.0)

        # global duplicate attraction penalty
        att_all = self._attraction_ids_all(state)
        total_atts = sum(1 for day_map in state.attractions.values() for v in day_map.values() if v)
        if len(att_all) < total_atts:
            self._ensure_violation(state, "duplicate_attractions")
            reward += self.reward_cfg.get("duplicate_attraction_penalty", 0.0)

        if self.goal.budget is not None and state.cost > self.goal.budget:
            self._ensure_violation(state, "budget")
            reward += self.reward_cfg.get("budget_violation_penalty", 0.0)

        reward += state.preference_matches * self.reward_cfg.get("preference_bonus", 0.0)

        if self.is_success(state):
            reward += self.reward_cfg.get("finish_success_bonus", 0.0)
        else:
            reward += self.reward_cfg.get("finish_fail_penalty", 0.0)
        return reward
