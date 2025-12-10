from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from mcts.travel.knowledge_base import TravelKnowledgeBase, TripGoal


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
    "preference_bonus": 1.5,
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


class TravelEnv:
    def __init__(self, knowledge_base: TravelKnowledgeBase, goal: TripGoal,
                 max_steps: int = 40, top_k: int = 5,
                 reward_cfg: Optional[Dict] = None):
        self.kb = knowledge_base
        self.goal = goal
        self.max_steps = max_steps
        self.top_k = top_k
        self.reward_cfg = reward_cfg.copy() if reward_cfg is not None else DEFAULT_REWARD_CFG.copy()

        self.total_days = goal.duration_days or 3
        self.meal_slots = MEAL_SLOTS
        self.attraction_slots = ATTRACTION_SLOTS

        self.base_state = self._empty_state()
        self.base_history: List[str] = []

        # 初始化城市/phase（单城直接跳过城市搜索）
        self._ensure_destination_required()
        self._init_cities_and_phase(self.base_state)

        self.state = self.base_state.clone()
        self.history: List[str] = []
        self.steps = 0
        self.action_payloads: Dict[str, Tuple] = {}
        self.last_info: Dict = {}

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
        """保证 destination 至少在 must_visit & candidate_cities 里（多城场景用）"""
        dest = self.goal.destination
        if not dest:
            return
        dest_norm = self.kb._normalize_city(dest)
        must_norms = {self.kb._normalize_city(c) for c in self.goal.must_visit_cities}
        if dest_norm not in must_norms and not self.goal.fixed_city_order:
            self.goal.must_visit_cities.append(dest)
        if dest not in self.goal.candidate_cities:
            self.goal.candidate_cities.append(dest)

    def _init_cities_and_phase(self, state: TravelState) -> None:
        """
        根据 Goal 决定：
        - 是否需要 CITY 阶段（多城市/城市不确定）
        - 单城市场景直接设置 city_sequence 并从 SEGMENT 开始
        - fixed_city_order 场景直接用给定顺序并从 SEGMENT 开始
        """
        city_target = self.goal.visiting_city_number or 1

        # 1) fixed_city_order：已经完全给定城市顺序 → 不需要 CITY 搜索
        if self.goal.fixed_city_order:
            state.city_sequence = list(self.goal.fixed_city_order)
            state.phase = Phase.SEGMENT
            return

        # 2) 单城市 & destination 明确 → 直接锁定城市，跳过 CITY
        if city_target == 1 and self.goal.destination:
            state.city_sequence = [self.goal.destination]
            state.phase = Phase.SEGMENT
            return

        # 3) 否则：需要 CITY 阶段由 MCTS+LLM 搜索城市集合
        state.phase = Phase.CITY

    def reset(self, goal: Optional[TripGoal] = None) -> Tuple[str, List[str]]:
        if goal is not None:
            self.goal = goal
            self.total_days = goal.duration_days or 3
            self.base_state = self._empty_state()
            self.base_history = []
            self._ensure_destination_required()
            self._init_cities_and_phase(self.base_state)

        self.state = self.base_state.clone()
        # 再次确保 fixed_city_order 下 city_sequence 正确
        if not self.state.city_sequence and self.goal.fixed_city_order:
            self.state.city_sequence = list(self.goal.fixed_city_order)

        self.state.cost = self._estimate_cost(self.state)
        self.history = list(self.base_history)
        self.steps = 0

        obs = self._observation(self.state)
        valid_actions = self._build_valid_actions(self.state)
        return obs, valid_actions

    def apply_action(self, action: str):
        """Apply action to the anchor state after planning."""
        self.reset()
        obs, reward, done, history, valid_actions = self.step(action)
        self.base_state = self.state.clone()
        self.base_history = list(self.history)
        return obs, reward, done, history, valid_actions

    # ----------------------------
    # Cost / helper 统计
    # ----------------------------
    def _estimate_cost(self, state: TravelState) -> float:
        cost = 0.0
        if state.outbound_flight is not None:
            cost += state.outbound_flight["price"]
        if state.return_flight is not None:
            cost += state.return_flight["price"]
        if state.accommodation is not None:
            cost += state.accommodation["price"]
        for stay in state.city_stays.values():
            if stay is not None:
                cost += stay["price"]
        for mode in state.segment_modes.values():
            detail = mode.get("detail", {}) if isinstance(mode, dict) else {}
            if isinstance(detail, dict):
                if "price" in detail:
                    cost += float(detail["price"])
                elif "cost" in detail:
                    cost += float(detail["cost"])
        for day in state.meals.values():
            for meal in day.values():
                if meal is not None:
                    cost += meal["cost"]
        return cost

    def _ensure_violation(self, state: TravelState, violation: str) -> None:
        if violation not in state.violations:
            state.violations.append(violation)

    def _matches_preference(self, restaurant: Dict) -> bool:
        if not self.goal.preferences:
            return False
        cuisines = str(restaurant.get("cuisines") or restaurant.get("Cuisines") or "").lower()
        return any(pref.lower() in cuisines for pref in self.goal.preferences)

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
                parts.append(f"Stay {city}: {stay['name']} {stay['room_type']} ${stay['price']:.0f}")

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
    # Phase-aware action builder（严格→放宽）
    # ----------------------------
    def _build_valid_actions(self, state: TravelState) -> List[str]:
        """
        先按严格约束构造动作；若无动作，则放宽约束再构造一轮。
        不再在“无动作时”强行注入 finish。
        """
        # 严格模式
        self.action_payloads = {}
        actions = self._build_phase_actions(state, relaxed=False)
        if actions:
            return actions

        # 放宽模式（允许重复/超预算等，由 reward 惩罚）
        self.action_payloads = {}
        actions = self._build_phase_actions(state, relaxed=True)
        return actions  # 可能为空，由 MCTS 上层自己处理

    def _build_phase_actions(self, state: TravelState, relaxed: bool) -> List[str]:
        if state.phase == Phase.CITY:
            return self._build_city_actions(state, relaxed=relaxed)
        elif state.phase == Phase.SEGMENT:
            return self._build_segment_actions(state, relaxed=relaxed)
        elif state.phase == Phase.STAY:
            return self._build_stay_actions(state, relaxed=relaxed)
        elif state.phase == Phase.DAILY:
            return self._build_daily_actions(state, relaxed=relaxed)
        else:
            return []

    def _build_city_actions(self, state: TravelState, relaxed: bool = False) -> List[str]:
        actions: List[str] = []
        city_target = self.goal.visiting_city_number or 1

        # 城市还不够 → 搜索候选城市
        if len(state.city_sequence) < city_target:
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

            for city in candidates:
                if city not in state.city_sequence:
                    action = f"choose_city:{city}"
                    self.action_payloads[action] = ("choose_city", city)
                    actions.append(action)

            # relaxed 模式下仍然为空时，可以允许重复城市（极端兜底）
            if relaxed and not actions and candidates:
                for city in candidates:
                    action = f"choose_city:{city}"
                    self.action_payloads[action] = ("choose_city", city)
                    actions.append(action)

        return actions

    def _build_segment_actions(self, state: TravelState, relaxed: bool = False) -> List[str]:
        actions: List[str] = []
        segments = self._segments(state)
        if not segments:
            return actions

        last_idx = segments[-1][0]

        for idx, src, dst in segments:
            if idx in state.segment_modes:
                continue

            # --- 关键逻辑：去程/返程在 require_flight=True 时只能用 flight ---
            if self.goal.require_flight and (idx == 0 or idx == last_idx):
                mode_candidates = ["flight"]
            else:
                mode_candidates = self._allowed_transport_modes()

            for mode in mode_candidates:
                if mode == "flight":
                    # 这里不再用 max_price，避免因为预算筛空
                    flights = self.kb.get_flights(
                        src,
                        dst,
                        top_k=self.top_k,
                        # 不用 max_price=self.goal.budget
                    )
                    for f in flights:
                        action = (
                            f"move:seg{idx}:flight:{f['id']} {src}->{dst} "
                            f"${f['price']:.0f} {f['depart']}-{f['arrive']}"
                        )
                        self.action_payloads[action] = ("segment_mode", idx, "flight", f)
                        actions.append(action)
                else:
                    dist_km = self.kb.distance_km(src, dst)
                    if dist_km is None:
                        continue
                    cost = dist_km
                    action = f"move:seg{idx}:{mode}:{src}->{dst} {dist_km:.0f}km cost {cost:.0f}"
                    payload_detail = {"origin": src, "destination": dst, "distance": dist_km, "cost": cost}
                    self.action_payloads[action] = ("segment_mode", idx, mode, payload_detail)
                    actions.append(action)

        return actions


    def _build_stay_actions(self, state: TravelState, relaxed: bool = False) -> List[str]:
        actions: List[str] = []
        if not self.goal.require_accommodation:
            return actions

        for city in state.city_sequence:
            if state.city_stays.get(city) is None:
                stays = self.kb.get_accommodations(
                    city,
                    top_k=self.top_k if not relaxed else self.top_k * 3,
                    max_price=None if relaxed else self.goal.budget,
                )
                for s in stays:
                    action = f"stay:{city}:{s['id']} {s['name']} {s['room_type']} ${s['price']:.0f}"
                    self.action_payloads[action] = ("stay_city", city, s)
                    actions.append(action)
        return actions

    def _build_daily_actions(self, state: TravelState, relaxed: bool = False) -> List[str]:
        actions: List[str] = []

        # ---------- 餐厅 ----------
        for day in range(1, self.total_days + 1):
            city = self._city_for_day(state, day)
            restaurants = self.kb.get_restaurants(
                city,
                preferences=self.goal.preferences,
                top_k=self.top_k if not relaxed else self.top_k * 3,
            )
            used_ids = self._restaurant_ids_day(state, day)
            used_global = self._restaurant_ids_all(state)

            for slot in self.meal_slots:
                if state.meals[day][slot] is None:
                    for r in restaurants:
                        if not relaxed:
                            # 严格模式：禁止当天重复 / 跨天重复
                            if r["id"] in used_ids or r["id"] in used_global:
                                continue
                        # relaxed 模式：允许重复，交给 reward 去惩罚
                        action = (
                            f"eat:d{day}:{slot}:{r['id']} "
                            f"{r['name']} {r['cuisines']} ${r['cost']:.0f} rating {r['rating']}"
                        )
                        self.action_payloads[action] = ("meal", day, slot, r)
                        actions.append(action)

        # ---------- 景点 ----------
        for day in range(1, self.total_days + 1):
            city = self._city_for_day(state, day)
            current_att = self._count_attractions_day(state, day)
            if current_att >= self.goal.attractions_per_day_max:
                continue

            attractions = self.kb.get_attractions(
                city,
                top_k=self.top_k if not relaxed else self.top_k * 3,
            )
            used_att = self._attraction_ids_all(state)

            for slot in self.attraction_slots:
                if state.attractions[day][slot] is None:
                    for a in attractions:
                        if not relaxed:
                            if a["id"] in used_att:
                                continue
                        # relaxed 模式：允许重复景点，由 reward 惩罚
                        action = f"visit:d{day}:{slot}:{a['id']} {a['name']} @ {a['city']}"
                        self.action_payloads[action] = ("attraction", day, slot, a)
                        actions.append(action)

        # 只在严格模式下且结构完整时提供 finish
        if not relaxed and self._can_finish(state):
            actions.append("finish")
        return actions

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
            att_count = self._count_attractions_day(state, day)
            if att_count < self.goal.attractions_per_day_min:
                return False
        if self.goal.budget is not None and state.cost > self.goal.budget:
            return False
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
                valid_actions = self._build_valid_actions(self.state)
                self.last_info = {"violations": list(self.state.violations), "cost": self.state.cost}
                return obs, reward, False, self.history, valid_actions

            # 可以 finish：终局打分
            self.history.append(action)
            reward += self._finish_and_score(self.state)
            obs = self._observation(self.state)
            self.last_info = {"violations": list(self.state.violations), "cost": self.state.cost}
            return obs, reward, True, self.history, []

        # 非 finish 动作
        payload = self.action_payloads.get(action)
        if payload is None:
            # 非法动作，轻微惩罚，重新给 action 列表
            reward = -1.0
            obs = self._observation(self.state)
            valid_actions = self._build_valid_actions(self.state)
            self.last_info = {"violations": list(self.state.violations), "cost": self.state.cost}
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

        elif kind == "stay_city":
            _, city, stay = payload
            if self.state.city_stays.get(city) is None:
                self.state.city_stays[city] = stay

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
        valid_actions = self._build_valid_actions(self.state) if not done else []
        self.last_info = {"violations": list(self.state.violations), "cost": self.state.cost}
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
