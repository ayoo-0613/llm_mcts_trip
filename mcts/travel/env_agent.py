from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from mcts.travel.env.knowledge_base import TravelKnowledgeBase
from mcts.travel.retrieval_agent import RetrievalAgent


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
    "step_bonus": 1.0,
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
    "preference_bonus": 10.0,
    "poi_bonus": 1.0,
    "finish_success_bonus": 6.0,
    "finish_fail_penalty": -6.0,
    "feasibility_weight": 5.0,
    "preference_weight": 1.0,
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

    # Optional: explicit day-to-city allocation (CITY phase output)
    day_to_city: Dict[int, str] = field(default_factory=dict)  # day (1-based) -> city

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
            day_to_city=copy.deepcopy(self.day_to_city),
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
        if self.day_to_city:
            day_parts = [f"{d}:{self.day_to_city[d]}" for d in sorted(self.day_to_city)]
            parts.append("day_city:" + ",".join(day_parts))
        else:
            parts.append("day_city:")

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
    type: str  # "segment"|"hotel"|"meal"|"attraction"|"finish"|"city" (legacy: "flight")
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
    def __init__(
        self,
        knowledge_base: TravelKnowledgeBase,
        max_steps: int = 40,
        top_k: int = 5,
        reward_cfg: Optional[Dict] = None,
        debug: bool = False,
        candidate_cap: int = 80,
        user_query: str = "",
        log_filter_usage: bool = False,
        retrieval_agent: Optional[RetrievalAgent] = None,
        goal_parsed: Optional[Dict[str, Any]] = None,
    ):
        self.kb = knowledge_base
        self.goal_parsed = goal_parsed or {}
        self.goal = self.goal_parsed
        self.max_steps = max_steps
        self.top_k = top_k
        self.reward_cfg = reward_cfg.copy() if reward_cfg is not None else DEFAULT_REWARD_CFG.copy()
        self.debug = debug
        self.candidate_cap = candidate_cap
        self.user_query = user_query
        self.log_filter_usage = log_filter_usage

        self.total_days = self._parsed_int("duration_days", "days", default=3)
        self.meal_slots = MEAL_SLOTS
        self.attraction_slots = ATTRACTION_SLOTS

        self.base_state = self._empty_state()
        self.base_history: List[str] = []

        # 初始化城市/phase（单城直接跳过城市搜索）
        self._ensure_destination_required()
        self._expand_state_locations()
        self._init_cities_and_phase(self.base_state)

        self.state = self.base_state.clone()
        self.history: List[str] = []
        self.steps = 0
        self.action_payloads: Dict[str, Tuple] = {}
        self.last_info: Dict = {}
        self.last_slot: Optional[Slot] = None
        self._last_phase_plan = None

        self.retrieval_agent = retrieval_agent or RetrievalAgent(
            kb=self.kb,
            top_k=self.top_k,
            candidate_cap=self.candidate_cap,
            debug=self.debug,
            log_filter_usage=self.log_filter_usage,
        )
        self.candidate_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.filter_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.filter_events: List[Dict[str, Any]] = []
        self.executed_filter_events: List[Dict[str, Any]] = []
        self.filter_event_keys: set = set()
        self.executed_filter_event_keys: set = set()

    # ----------------------------
    # 初始化 & helper
    # ----------------------------
    def _parsed_get(self, *keys: str, default: Any = None) -> Any:
        parsed = self.goal_parsed or {}
        for key in keys:
            if key in parsed:
                val = parsed.get(key)
                if val is not None:
                    return val
        return default

    def _parsed_list(self, *keys: str, default: Optional[List[Any]] = None) -> List[Any]:
        val = self._parsed_get(*keys, default=None)
        if val is None:
            return list(default or [])
        if isinstance(val, list):
            return list(val)
        return [val]

    def _parsed_int(self, *keys: str, default: Optional[int] = None) -> Optional[int]:
        val = self._parsed_get(*keys, default=None)
        if val is None:
            return default
        try:
            return int(val)
        except Exception:
            return default

    def _parsed_float(self, *keys: str, default: Optional[float] = None) -> Optional[float]:
        val = self._parsed_get(*keys, default=None)
        if val is None:
            return default
        try:
            return float(val)
        except Exception:
            return default

    def _parsed_bool(self, *keys: str, default: bool = False) -> bool:
        val = self._parsed_get(*keys, default=None)
        if val is None:
            return default
        return bool(val)

    def _parsed_start_date(self) -> Optional[str]:
        val = self._parsed_get("start_date", "date", "dates", default=None)
        if isinstance(val, list):
            return val[0] if val else None
        return val

    def _parsed_constraints(self) -> Dict[str, Any]:
        cons = self.goal_parsed.get("constraints")
        return cons if isinstance(cons, dict) else {}

    def _goal_text(self) -> str:
        origin = self._parsed_get("origin", "org", default=None)
        destination = self._parsed_get("destination", "dest", default=None)
        start_date = self._parsed_start_date()
        days = self._parsed_int("duration_days", "days", default=None)
        budget = self._parsed_float("budget", default=None)
        people = self._parsed_int("people_number", default=1) or 1
        visit_num = self._parsed_int("visiting_city_number", default=1) or 1
        prefs = self._parsed_list("preferences", default=[])
        must_cities = self._parsed_list("must_visit_cities", default=[])
        priority_cities = self._parsed_list("priority_cities", default=[])
        budget_text = f"${budget:.0f}" if budget is not None else "unspecified"
        days_text = f"{days} day(s)" if days is not None else "unspecified"
        parts = [
            f"Trip from {origin} to {destination}",
            f"start date: {start_date or 'unspecified'}",
            f"duration: {days_text}",
            f"budget: {budget_text}",
            f"people: {people}",
            f"city count target: {visit_num}",
            f"preferences: {', '.join(prefs) if prefs else 'None'}",
            f"must cities: {', '.join(must_cities) if must_cities else 'None'}",
            f"priority cities: {', '.join(priority_cities) if priority_cities else 'None'}",
        ]
        return "; ".join(parts)

    def _parsed_transport_forbid(self) -> set:
        return {
            str(m).lower()
            for m in self._parsed_list(
                "transport_forbid", "transport_forbidden", "transport_forbidden_modes", default=[]
            )
            if m
        }

    def _require_flight(self) -> bool:
        val = self._parsed_get("require_flight", default=None)
        if val is not None:
            return bool(val)
        return "flight" not in self._parsed_transport_forbid()

    def _return_required(self) -> bool:
        val = self._parsed_get("return_required", default=None)
        if val is not None:
            return bool(val)
        return True

    def _budget_remaining(self, state: TravelState) -> Optional[float]:
        budget = self._parsed_float("budget", default=None)
        if budget is None:
            return None
        try:
            budget_f = float(budget)
        except Exception:
            return None
        return budget_f - self._estimate_cost(state)

    def _planned_accommodation_days_for_city(self, state: TravelState, city: str) -> int:
        if not city or self.total_days <= 1:
            return 0
        target = self.kb._normalize_city(city)
        count = 0
        for day in range(1, self.total_days):  # exclude last day
            c = self._city_for_day(state, day)
            if c and self.kb._normalize_city(c) == target:
                count += 1
        return count

    def _estimate_action_cost(self, payload: Optional[Tuple], state: TravelState) -> Optional[float]:
        if not payload:
            return None
        kind = payload[0]
        people = self._parsed_int("people_number", default=1) or 1
        if kind == "segment_mode":
            mode = payload[2] if len(payload) > 2 else None
            detail = payload[3] if len(payload) > 3 else None
            if not isinstance(detail, dict):
                return None
            cost = detail.get("price")
            if cost is None:
                cost = detail.get("cost")
            if cost is None:
                return None
            try:
                base = float(cost)
            except Exception:
                return None
            if mode == "flight":
                return base * max(1, people)
            if mode == "taxi":
                return base * max(1, math.ceil(people / 4.0))
            if mode == "self-driving":
                return base * max(1, math.ceil(people / 5.0))
            return base
        if kind == "stay_city":
            detail = payload[2] if len(payload) > 2 else None
            if not isinstance(detail, dict):
                return None
            price = detail.get("price")
            if price is None:
                return None
            try:
                price_f = float(price)
            except Exception:
                return None
            city = payload[1] if len(payload) > 1 else detail.get("city")
            nights = self._planned_accommodation_days_for_city(state, str(city)) if city else 0
            if nights <= 0:
                return 0.0
            occ = detail.get("occupancy")
            try:
                occ_i = int(occ) if occ is not None else None
            except Exception:
                occ_i = None
            occ_i = max(1, occ_i) if occ_i else None
            rooms = math.ceil(people / float(occ_i or people))
            return price_f * rooms * nights
        if kind == "meal":
            detail = payload[3] if len(payload) > 3 else None
            if not isinstance(detail, dict):
                return None
            cost = detail.get("cost")
            if cost is None:
                return None
            try:
                return float(cost) * max(1, people)
            except Exception:
                return None
        if kind == "attraction":
            return 0.0
        return 0.0

    def _action_feasible(self, payload: Optional[Tuple], state: TravelState) -> bool:
        if not payload:
            return True
        kind = payload[0]
        if kind == "meal":
            detail = payload[3] if len(payload) > 3 else None
            if isinstance(detail, dict):
                rid = detail.get("id")
                if rid is not None:
                    if rid in self._restaurant_ids_all(state):
                        return False
        return True

    def _filter_actions_by_constraints(
        self,
        actions: List[str],
        payloads: Dict[str, Tuple],
        state: TravelState,
    ) -> Tuple[List[str], Dict[str, Tuple]]:
        if not actions or not payloads:
            return actions, payloads
        kept_actions: List[str] = []
        kept_payloads: Dict[str, Tuple] = {}
        for action in actions:
            payload = payloads.get(action)
            if self._action_feasible(payload, state):
                kept_actions.append(action)
                kept_payloads[action] = payload
        if not kept_actions:
            return actions, payloads
        return kept_actions, kept_payloads

    def _feasibility_scores(self, state: TravelState) -> Tuple[float, float]:
        budget_score = 0.0
        remaining = self._budget_remaining(state)
        budget = self._parsed_float("budget", default=None)
        if remaining is not None and budget:
            try:
                budget_score = max(-1.0, min(1.0, float(remaining) / float(budget)))
            except Exception:
                budget_score = 0.0
        total_meals = sum(1 for day_map in state.meals.values() for v in day_map.values() if v)
        pref_score = 0.0
        if total_meals > 0:
            pref_score = float(state.preference_matches) / float(total_meals)
        return budget_score, pref_score

    def _empty_state(self) -> TravelState:
        meals = {day: {slot: None for slot in self.meal_slots}
                 for day in range(1, self.total_days + 1)}
        attractions = {day: {slot: None for slot in self.attraction_slots}
                       for day in range(1, self.total_days + 1)}
        return TravelState(meals=meals, attractions=attractions)

    def _ensure_destination_required(self) -> None:
        """Only build candidate cities when destination is a state; leave city untouched."""
        dest = self._parsed_get("destination", "dest", default=None)
        if not dest:
            return
        days = self._parsed_int("duration_days", "days", default=None)
        dest_norm = self.kb._normalize_city(dest)
        dest_is_city = bool(dest_norm and dest_norm in getattr(self.kb, "city_set_norm", {}))
        prefer_city = bool(dest_is_city and days == 3)
        if prefer_city:
            return

        # Destination is a state → seed candidate_cities from that state.
        if self.kb.is_state(dest) and days and days > 3:
            dest_cities = self.kb.cities_in_state(dest)
            city_target = self._parsed_int("visiting_city_number", default=1) or 1
            cap = max(self.top_k * 3, city_target * 10)
            self.goal_parsed["candidate_cities"] = dest_cities[:cap]
            return

        # Destination is a city: do not build candidate_cities. If multi-city requested, raise.
        city_target = self._parsed_int("visiting_city_number", default=1) or 1
        if city_target > 1:
            raise ValueError(
                f"Destination '{dest}' is a city but visiting_city_number={city_target}. "
                "No candidate_cities will be built automatically; please provide candidate_city list "
                "or change destination to a state."
            )

    def _expand_state_locations(self) -> None:
        """Only expand when destination is a state; city destinations stay untouched."""
        dest = self._parsed_get("destination", "dest", default=None)
        if not dest or not self.kb.is_state(dest):
            return
        days = self._parsed_int("duration_days", "days", default=None)
        dest_norm = self.kb._normalize_city(dest)
        dest_is_city = bool(dest_norm and dest_norm in getattr(self.kb, "city_set_norm", {}))
        if dest_is_city and days == 3:
            return
        if not days or days <= 3:
            return

        cities = self.kb.cities_in_state(dest)
        city_target = self._parsed_int("visiting_city_number", default=1) or 1
        cap = max(self.top_k * 3, city_target * 10)
        self.goal_parsed["candidate_cities"] = cities[:cap]


    def _init_cities_and_phase(self, state: TravelState) -> None:
        """
        根据解析后的目标数据决定：
        - 是否需要 CITY 阶段（多城市/城市不确定）
        - 单城市场景直接设置 city_sequence 并从 SEGMENT 开始
        - fixed_city_order 场景直接用给定顺序并从 SEGMENT 开始
        """
        city_target = self._parsed_int("visiting_city_number", default=1) or 1
        destination = self._parsed_get("destination", "dest", default=None)
        dest_norm = self.kb._normalize_city(destination) if destination else None
        dest_is_city = bool(dest_norm and dest_norm in getattr(self.kb, "city_set_norm", {}))
        dest_is_state = bool(dest_norm and dest_norm in getattr(self.kb, "state_norm_map", {}))
        days = self._parsed_int("duration_days", "days", default=None)
        if dest_is_city and days == 3:
            dest_is_state = False
        fixed_order = self._parsed_list("fixed_city_order", default=[])

        # 1) fixed_city_order：已经完全给定城市顺序 → 不需要 CITY 搜索
        if fixed_order:
            state.city_sequence = list(fixed_order)
            state.phase = Phase.SEGMENT
            return

        # 2) 单城市 & destination 是明确城市 → 直接锁定城市，跳过 CITY
        if city_target == 1 and destination and dest_is_city and not dest_is_state:
            state.city_sequence = [destination]
            state.phase = Phase.SEGMENT
            return

        # 3) 否则：需要 CITY 阶段由 MCTS+LLM 搜索城市集合
        state.phase = Phase.CITY

    def reset(self, goal_parsed: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
        if goal_parsed is not None:
            self.goal_parsed = goal_parsed or {}
            self.goal = self.goal_parsed
            self.total_days = self._parsed_int("duration_days", "days", default=3)
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

        self.state = self.base_state.clone()
        # 再次确保 fixed_city_order 下 city_sequence 正确
        if not self.state.city_sequence:
            fixed_order = self._parsed_list("fixed_city_order", default=[])
            if fixed_order:
                self.state.city_sequence = list(fixed_order)

        self.state.cost = self._estimate_cost(self.state)
        self.history = list(self.base_history)
        self.steps = 0
        try:
            self.retrieval_agent.reset()
        except Exception:
            pass
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
        """
        Cost estimator aligned with TravelPlanner evaluation logic.

        Key points (match evaluation/hard_constraint.py):
        - flight: price * people_number
        - taxi/self-driving: base_cost * ceil(people_number / k) where k is 4/5
        - accommodation: price * ceil(people_number / max_occupancy) * nights
          where nights ~= number of (non-last) days spent in that city.
        - meals: per-meal cost * people_number
        - attractions: ignored (evaluation does not count attraction costs)
        """
        cost = 0.0
        people = self._parsed_int("people_number", default=1) or 1
        people = max(1, people)

        # Transport: from segment_modes
        for seg in (state.segment_modes or {}).values():
            if not isinstance(seg, dict):
                continue
            mode = str(seg.get("mode") or "").lower()
            detail = seg.get("detail", {}) or {}
            if not isinstance(detail, dict):
                continue
            base = detail.get("price")
            if base is None:
                base = detail.get("cost")
            if base is None:
                continue
            try:
                base_f = float(base)
            except Exception:
                continue
            if mode == "flight":
                cost += base_f * people
            elif mode == "taxi":
                cost += base_f * math.ceil(people / 4.0)
            elif mode == "self-driving":
                cost += base_f * math.ceil(people / 5.0)
            else:
                cost += base_f

        # Stay: prefer city_stays; fallback to legacy accommodation only if city_stays empty.
        #
        # We model "nights" as the number of non-last days allocated to the city.
        # This aligns with commonsense_constraint.is_not_absent which allows
        # accommodation to be absent on the last day, and our submission formatter
        # which emits accommodation="-" on the last day.
        nights_by_city: Dict[str, int] = {}
        for day in range(1, max(1, self.total_days)):  # exclude last day
            city = self._city_for_day(state, day)
            if not city:
                continue
            nights_by_city[city] = nights_by_city.get(city, 0) + 1

        if state.city_stays:
            for city, stay in state.city_stays.items():
                if not stay or stay.get("price") is None:
                    continue
                nights = int(nights_by_city.get(city, 0) or 0)
                if nights <= 0:
                    continue
                try:
                    price = float(stay["price"])
                except Exception:
                    continue
                occ = stay.get("occupancy")
                try:
                    occ_i = int(occ) if occ is not None and not math.isnan(float(occ)) else None
                except Exception:
                    occ_i = None
                occ_i = max(1, occ_i) if occ_i else None
                rooms = math.ceil(people / float(occ_i or people))
                cost += price * rooms * nights
        else:
            # Legacy single accommodation (rare)
            stay = state.accommodation
            if stay is not None and stay.get("price") is not None:
                try:
                    price = float(stay["price"])
                except Exception:
                    price = 0.0
                occ = stay.get("occupancy")
                try:
                    occ_i = int(occ) if occ is not None else None
                except Exception:
                    occ_i = None
                occ_i = max(1, occ_i) if occ_i else None
                rooms = math.ceil(people / float(occ_i or people))
                nights = max(0, self.total_days - 1)
                cost += price * rooms * nights

        # Meals (per person)
        for day in (state.meals or {}).values():
            for meal in (day or {}).values():
                if meal is not None and meal.get("cost") is not None:
                    try:
                        cost += float(meal["cost"]) * people
                    except Exception:
                        continue

        return cost

    def _ensure_violation(self, state: TravelState, violation: str) -> None:
        if violation not in state.violations:
            state.violations.append(violation)

    def _matches_preference(self, restaurant: Dict) -> bool:
        meal_cuisines: List[str] = []
        cons = self._parsed_constraints()
        if cons:
            meal_cuisines = (cons.get("meal", {}) or {}).get("cuisines", [])
        if not meal_cuisines:
            meal_cuisines = self._parsed_list("preferences", default=[])
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
            require_accommodation = self._parsed_bool("require_accommodation", default=True)
            if require_accommodation and state.city_stays.get(city) is None:
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
        budget = self._parsed_float("budget", default=None)
        if budget is not None and state.cost > budget:
            reasons["budget_over"].append(f"{state.cost:.1f}>{budget:.1f}")
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
        if state.day_to_city and day in state.day_to_city:
            return state.day_to_city.get(day)
        fixed_order = self._parsed_list("fixed_city_order", default=[])
        must_cities = self._parsed_list("must_visit_cities", default=[])
        seq = state.city_sequence or fixed_order or must_cities
        if not seq:
            return self._parsed_get("destination", "dest", default=None)
        idx = min(len(seq) - 1, int((day - 1) * len(seq) / max(1, self.total_days)))
        return seq[idx]

    def _segments(self, state: TravelState) -> List[Tuple[int, str, str]]:
        seq = state.city_sequence
        segments: List[Tuple[int, str, str]] = []
        origin = self._parsed_get("origin", "org", default=None)
        destination = self._parsed_get("destination", "dest", default=None)
        return_required = self._return_required()
        if seq:
            if origin:
                segments.append((0, origin, seq[0]))
            for i in range(1, len(seq)):
                segments.append((i, seq[i - 1], seq[i]))
            if return_required and origin:
                segments.append((len(seq), seq[-1], origin))
        elif destination and origin:
            # fallback single-destination routing
            segments.append((0, origin, destination))
            if return_required:
                segments.append((1, destination, origin))
        return segments

    def _allowed_transport_modes(self) -> List[str]:
        allowed = self._parsed_list("transport_allow", "transport_allowed_modes", default=[])
        if not allowed:
            allowed = ["flight", "taxi", "self-driving"]
        allowed = [str(m).lower() for m in allowed if m]
        forbidden = set(
            str(m).lower()
            for m in self._parsed_list("transport_forbid", "transport_forbidden", "transport_forbidden_modes", default=[])
            if m
        )

        cons = self._parsed_constraints()
        if cons:
            tcons = cons.get("transport", {}) or {}
            if tcons.get("allow"):
                allowed = [str(m).lower() for m in (tcons.get("allow") or []) if m]
            forbidden |= set(str(m).lower() for m in (tcons.get("forbid") or []) if m)

        return [m for m in allowed if m not in forbidden]

    # ----------------------------
    # Phase 完成判定 & 推进
    # ----------------------------
    def _city_phase_done(self, state: TravelState) -> bool:
        city_target = self._parsed_int("visiting_city_number", default=1) or 1
        if len(state.city_sequence) < city_target:
            return False
        must_norm = {self.kb._normalize_city(c) for c in self._parsed_list("must_visit_cities", default=[])}
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
        require_accommodation = self._parsed_bool("require_accommodation", default=True)
        if not require_accommodation:
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
        parts = [self._goal_text(), f"Phase: {state.phase.name}"]

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
        city_target = self._parsed_int("visiting_city_number", default=1) or 1
        if len(state.city_sequence) < city_target:
            pending.append(f"cities missing {city_target - len(state.city_sequence)}")
        segments = self._segments(state)
        for idx, src, dst in segments:
            if idx not in state.segment_modes:
                pending.append(f"segment {idx} {src}->{dst} mode")
        for city in state.city_sequence:
            require_accommodation = self._parsed_bool("require_accommodation", default=True)
            if state.city_stays.get(city) is None and require_accommodation:
                pending.append(f"stay in {city}")
        for day in range(1, self.total_days + 1):
            meals_missing = [slot for slot, meal in state.meals[day].items() if meal is None]
            if meals_missing:
                pending.append(f"day{day} meals {len(meals_missing)} missing")
            att_count = self._count_attractions_day(state, day)
            att_min = self._parsed_int("attractions_per_day_min", "attractions_min", default=1) or 1
            if att_count < att_min:
                pending.append(f"day{day} attractions missing {att_min - att_count}")
        if pending:
            parts.append("Pending: " + ", ".join(pending))

        budget = self._parsed_float("budget", default=None)
        if budget is not None:
            budget_left = budget - self._estimate_cost(state)
            parts.append(f"Budget left estimate: {budget_left:.0f}")
        parts.append(f"Allowed transport: {', '.join(self._allowed_transport_modes())}")
        return " | ".join(parts)

    # ----------------------------
    # Slot & candidates
    # ----------------------------
    def _next_slot(self, state: TravelState) -> Optional[Slot]:
        """Decide the next slot to fill based on phase and pending requirements."""
        self._advance_phase_if_ready(state)
        city_target = self._parsed_int("visiting_city_number", default=1) or 1
        if state.phase == Phase.CITY and len(state.city_sequence) < city_target:
            return Slot(type="city")

        if state.phase == Phase.SEGMENT:
            for idx, src, dst in self._segments(state):
                if idx not in state.segment_modes:
                    return Slot(
                        type="segment",
                        seg=idx,
                        origin=src,
                        destination=dst,
                        date=self._parsed_start_date(),
                    )

        require_accommodation = self._parsed_bool("require_accommodation", default=True)
        if state.phase == Phase.STAY and require_accommodation:
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
                att_min = self._parsed_int("attractions_per_day_min", "attractions_min", default=1) or 1
                if self._count_attractions_day(state, day) < att_min:
                    for slot_name, att in state.attractions[day].items():
                        if att is None:
                            city = self._city_for_day(state, day)
                            return Slot(type="attraction", day=day, meal_type=slot_name, city=city)

        if self._can_finish(state):
            return Slot(type="finish")
        return None

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

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        try:
            return int(val)
        except Exception:
            return default

    def _build_dead_end_meta(
        self,
        slot: Optional[Slot],
        actions: List[str],
        event: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if slot is None or actions:
            return None

        filter_usage = event.get("filter_usage", {}) if isinstance(event, dict) else {}
        kb_count = event.get("kb_count")
        if kb_count is None:
            kb_count = filter_usage.get("base_count")
        after_nonbudget = event.get("after_nonbudget_count")
        if after_nonbudget is None:
            after_nonbudget = filter_usage.get("base_count")
        after_budget = filter_usage.get("hard_kept")
        bundle_infeasible = bool(event.get("bundle_infeasible"))

        kb_v = self._safe_int(kb_count, 0)
        nonbudget_v = self._safe_int(after_nonbudget, 0)
        budget_v = self._safe_int(after_budget, 0)
        topk = len(actions)

        failure_code = filter_usage.get("failure_code") if isinstance(filter_usage, dict) else None
        if bundle_infeasible:
            reason = "constraint_mismatch"
        elif failure_code == "kb_empty":
            reason = "scarcity"
        elif failure_code == "hard_pruned_empty":
            reason = "budget"
        elif kb_v == 0:
            reason = "scarcity"
        elif nonbudget_v == 0:
            reason = "constraint_mismatch"
        elif budget_v == 0:
            reason = "budget"
        else:
            reason = "duplication"

        city_bundle = list(self.state.city_sequence or self._parsed_get("fixed_city_order", default=[]) or [])
        if not city_bundle:
            dest = self._parsed_get("destination", "dest", default=None)
            if dest:
                city_bundle = [str(dest)]

        slot_id = slot.signature() if hasattr(slot, "signature") else str(slot)
        city = getattr(slot, "city", None) or getattr(slot, "destination", None) or getattr(slot, "origin", None)
        budget_context = {
            "remaining": filter_usage.get("budget_remaining"),
            "slot_cap": filter_usage.get("slot_cap") or event.get("budget_cap"),
            "reserve": filter_usage.get("reserve"),
            "min_cost_nonbudget": filter_usage.get("min_cost_nonbudget"),
            "bundle_infeasible": bundle_infeasible,
        }

        dominant = event.get("dominant_nonbudget_filter")
        if bundle_infeasible and not dominant:
            dominant = "bundle_infeasible"

        return {
            "phase": getattr(slot, "type", None),
            "slot_id": slot_id,
            "city": city,
            "city_bundle": city_bundle,
            "reason": reason,
            "counts": {
                "kb": kb_v,
                "after_nonbudget": nonbudget_v,
                "after_budget": budget_v,
                "budget_removed": max(0, nonbudget_v - budget_v),
                "nonbudget_removed": max(0, kb_v - nonbudget_v),
                "topk": topk,
            },
            "dominant_nonbudget_filter": dominant,
            "budget_context": budget_context,
        }

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
        budget_rev = 0
        try:
            budget_rev = int(self.retrieval_agent.get_budget_revision())
        except Exception:
            budget_rev = 0

        key = (self.state.signature(), slot, budget_rev)

        if key in self.candidate_cache:
            cached = self.candidate_cache[key]
            self.action_payloads = copy.deepcopy(cached.get("payloads", {}))
            actions = list(cached.get("actions", []))
            if self.log_filter_usage:
                filt = self.filter_cache.get(key)
                event = cached.get("event", {})
                print(
                    f"[FILTER] slot={slot.type} cache_hit=True "
                    f"plan_used_llm={event.get('plan_used_llm', False)} filter={filt} event={event}"
                )
            if cached.get("event") and key not in self.filter_event_keys:
                self.filter_events.append(dict(cached["event"], cache_hit=True))
                self.filter_event_keys.add(key)
            return actions

        result = self.retrieval_agent.compute(
            self.goal_parsed,
            self.state,
            slot,
            user_query=self.user_query,
        )
        actions = list(result.actions or [])
        self.action_payloads = copy.deepcopy(result.payloads or {})
        candidates = list(result.candidates or [])
        relaxed = bool(result.relaxed)
        filt = dict(result.filt or {})
        policy_event = dict(result.policy_event or {})
        self._last_phase_plan = getattr(result, "plan", None)
        if slot.type != "city":
            self.filter_cache[key] = filt
        actions, self.action_payloads = self._filter_actions_by_constraints(
            actions, self.action_payloads, self.state
        )

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
        dead_end = self._build_dead_end_meta(slot, actions, event)
        if dead_end is not None:
            event["dead_end_meta"] = dead_end
        if key not in self.filter_event_keys:
            self.filter_events.append(event)
            self.filter_event_keys.add(key)
        self.candidate_cache[key] = {
            "actions": actions,
            "payloads": copy.deepcopy(self.action_payloads),
            "event": event,
        }
        if self.debug:
            print(f"[DEBUG] Slot {slot.type} produced {len(actions)} actions")
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
        return self._goal_text()

    def is_success(self, state: Optional[TravelState] = None) -> bool:
        state = state or self.state
        city_target = self._parsed_int("visiting_city_number", default=1) or 1
        if len(state.city_sequence) < city_target:
            return False
        must_norm = {self.kb._normalize_city(c) for c in self._parsed_list("must_visit_cities", default=[])}
        if must_norm:
            seq_norm = {self.kb._normalize_city(c) for c in state.city_sequence}
            if not must_norm.issubset(seq_norm):
                return False

        segments = self._segments(state)
        for idx, _, _ in segments:
            if idx not in state.segment_modes:
                return False

        require_flight = self._require_flight()
        return_required = self._return_required()
        if require_flight:
            if segments:
                first_seg = state.segment_modes.get(0)
                if not first_seg or first_seg.get("mode") != "flight":
                    return False
                if return_required:
                    last_idx = segments[-1][0]
                    last_seg = state.segment_modes.get(last_idx)
                    if not last_seg or last_seg.get("mode") != "flight":
                        return False

        require_accommodation = self._parsed_bool("require_accommodation", default=True)
        if require_accommodation:
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
                att_min = self._parsed_int("attractions_per_day_min", "attractions_min", default=1) or 1
                if att_count < att_min:
                    return False
        budget = self._parsed_float("budget", default=None)
        if budget is not None and state.cost > budget:
            return False
        # 记录失败原因以便调试/输出
        self.last_failure_reasons = {}
        return True

    def _can_finish(self, state: Optional[TravelState] = None) -> bool:
        """结构性检查（不含预算），用于是否允许出现 finish 动作。"""
        state = state or self.state
        city_target = self._parsed_int("visiting_city_number", default=1) or 1
        if len(state.city_sequence) < city_target:
            return False
        must_norm = {self.kb._normalize_city(c) for c in self._parsed_list("must_visit_cities", default=[])}
        if must_norm:
            seq_norm = {self.kb._normalize_city(c) for c in state.city_sequence}
            if not must_norm.issubset(seq_norm):
                return False
        segments = self._segments(state)
        for idx, _, _ in segments:
            if idx not in state.segment_modes:
                return False
        require_accommodation = self._parsed_bool("require_accommodation", default=True)
        if require_accommodation:
            for city in state.city_sequence:
                if state.city_stays.get(city) is None:
                    return False
        for day in range(1, self.total_days + 1):
            if any(meal is None for meal in state.meals[day].values()):
                return False
            att_count = self._count_attractions_day(state, day)
            att_min = self._parsed_int("attractions_per_day_min", "attractions_min", default=1) or 1
            if att_count < att_min:
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
                self.state.day_to_city = {}
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
                return_required = self._return_required()
                if return_required and last_idx is not None and seg_idx == last_idx:
                    self.state.return_flight = detail
            else:
                # 非航班兜底：如果要求航班但使用了其他交通，记录违规
                require_flight = self._require_flight()
                return_required = self._return_required()
                if require_flight and (seg_idx == 0 or (return_required and seg_idx == last_idx)):
                    self._ensure_violation(self.state, "missing_flight")

        elif kind == "stay_city":
            _, city, stay = payload
            if self.state.city_stays.get(city) is None:
                self.state.city_stays[city] = stay
            # legacy accommodation not used once city_stays is set
            self.state.accommodation = None

        elif kind == "choose_city_bundle":
            seq = payload[1] if len(payload) >= 2 else []
            day_splits = payload[2] if len(payload) >= 3 else None
            self.state.city_sequence = list(seq or [])
            self.state.day_to_city = {}

            # Apply explicit day allocation if present; otherwise default to even split + tail absorbs remainder.
            mapping: Dict[int, str] = {}
            if isinstance(day_splits, list):
                for item in day_splits:
                    if not isinstance(item, dict):
                        continue
                    try:
                        c = str(item.get("city"))
                        start = int(item.get("start_day"))
                        end = int(item.get("end_day"))
                    except Exception:
                        continue
                    if not c or start <= 0 or end < start:
                        continue
                    for d in range(start, end + 1):
                        mapping[d] = c
            if not mapping and self.state.city_sequence and self.total_days > 0:
                n = len(self.state.city_sequence)
                base = self.total_days // n
                rem = self.total_days % n
                spans = [base] * n
                if rem:
                    spans[-1] += rem
                day = 1
                for c, span in zip(self.state.city_sequence, spans):
                    for _ in range(max(0, int(span))):
                        mapping[day] = c
                        day += 1
            self.state.day_to_city = mapping

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
                else:
                    att_max = self._parsed_int("attractions_per_day_max", "attractions_max", default=1) or 1
                    if self._count_attractions_day(self.state, day) < att_max:
                        self.state.attractions[day][slot] = attr
                        reward += self.reward_cfg.get("poi_bonus", 0.0)

        # 更新 cost & 预算惩罚
        self.state.cost = self._estimate_cost(self.state)
        budget = self._parsed_float("budget", default=None)
        if budget is not None and self.state.cost > budget:
            self._ensure_violation(self.state, "budget")
            reward += self.reward_cfg.get("budget_violation_penalty", 0.0)

        budget_score, pref_score = self._feasibility_scores(self.state)
        reward += self.reward_cfg.get("feasibility_weight", 0.0) * budget_score
        reward += self.reward_cfg.get("preference_weight", 0.0) * pref_score

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

        city_target = self._parsed_int("visiting_city_number", default=1) or 1
        if len(state.city_sequence) < city_target:
            self._ensure_violation(state, "city_count")
            reward += (city_target - len(state.city_sequence)) * self.reward_cfg.get("missing_city_penalty", 0.0)

        must_norm = {self.kb._normalize_city(c) for c in self._parsed_list("must_visit_cities", default=[])}
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

        require_flight = self._require_flight()
        return_required = self._return_required()
        if require_flight:
            if segments:
                first_seg = state.segment_modes.get(0)
                if not first_seg or first_seg.get("mode") != "flight":
                    self._ensure_violation(state, "outbound")
                    reward += self.reward_cfg.get("missing_flight_penalty", 0.0)
                if return_required:
                    last_idx = segments[-1][0]
                    last_seg = state.segment_modes.get(last_idx)
                    if not last_seg or last_seg.get("mode") != "flight":
                        self._ensure_violation(state, "return")
                        reward += self.reward_cfg.get("missing_return_penalty", 0.0)

        require_accommodation = self._parsed_bool("require_accommodation", default=True)
        if require_accommodation:
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
            att_min = self._parsed_int("attractions_per_day_min", "attractions_min", default=1) or 1
            if att_count < att_min:
                missing_att = att_min - att_count
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

        budget = self._parsed_float("budget", default=None)
        if budget is not None and state.cost > budget:
            self._ensure_violation(state, "budget")
            reward += self.reward_cfg.get("budget_violation_penalty", 0.0)

        budget_score, pref_score = self._feasibility_scores(state)
        reward += self.reward_cfg.get("feasibility_weight", 0.0) * budget_score
        reward += self.reward_cfg.get("preference_weight", 0.0) * pref_score

        reward += state.preference_matches * self.reward_cfg.get("preference_bonus", 0.0)

        if self.is_success(state):
            reward += self.reward_cfg.get("finish_success_bonus", 0.0)
        else:
            reward += self.reward_cfg.get("finish_fail_penalty", 0.0)
        return reward
