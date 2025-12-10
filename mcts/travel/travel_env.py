from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from mcts.travel.knowledge_base import TravelKnowledgeBase, TripGoal

# Shaping constants for constraint-aware planning (per-step and terminal checks)
DEFAULT_REWARD_CFG = {
    "step_bonus": 0.1,
    "budget_violation_penalty": -20.0,
    "missing_flight_penalty": -20.0,
    "missing_return_penalty": -20.0,
    "missing_stay_penalty": -20.0,
    "missing_city_penalty": -20.0,
    "missing_segment_penalty": -20.0,
    "missing_must_city_penalty": -20.0,
    "meal_missing_penalty": -20.0,
    "attraction_missing_penalty": -4.0,
    "duplicate_meal_penalty": -5.0,
    "duplicate_restaurant_across_days_penalty": -2.0,
    "duplicate_attraction_penalty": -3.0,
    "invalid_action_penalty": -10.0,
    "preference_bonus": 1.5,
    "poi_bonus": 1.0,
    "finish_success_bonus": 6.0,
    "finish_fail_penalty": -6.0,
}

MEAL_SLOTS = ["breakfast", "lunch", "dinner"]
ATTRACTION_SLOTS = ["spot"]  # single attraction slot per day


@dataclass
class TravelState:
    phase: str = "city"  # city -> transport_mode -> accommodation -> restaurant -> attraction -> done
    city_seq: List[str] = field(default_factory=list)
    departure_dates: List[Optional[int]] = field(default_factory=list)  # day index per leg
    transport_modes: List[Optional[str]] = field(default_factory=list)  # mode per leg
    hotel_index: Dict[str, Optional[Dict]] = field(default_factory=dict)  # city -> accommodation dict
    restaurants: Dict[tuple, Optional[Dict]] = field(default_factory=dict)  # (day, slot)->restaurant
    attractions: Dict[int, Optional[Dict]] = field(default_factory=dict)  # day->attraction
    total_cost: float = 0.0
    cuisine_covered: set = field(default_factory=set)
    violated: bool = False
    preference_matches: int = 0
    violations: List[str] = field(default_factory=list)
    is_terminal: bool = False

    def clone(self) -> "TravelState":
        return TravelState(
            phase=self.phase,
            city_seq=list(self.city_seq),
            departure_dates=list(self.departure_dates),
            transport_modes=list(self.transport_modes),
            hotel_index=copy.deepcopy(self.hotel_index),
            restaurants=copy.deepcopy(self.restaurants),
            attractions=copy.deepcopy(self.attractions),
            total_cost=self.total_cost,
            cuisine_covered=set(self.cuisine_covered),
            violated=self.violated,
            preference_matches=self.preference_matches,
            violations=list(self.violations),
            is_terminal=self.is_terminal,
        )


class ActionType(Enum):
    ADD_CITY = 1
    SET_DEPARTURE_DATE = 2
    SET_TRANSPORT_MODE = 3
    CHOOSE_HOTEL = 4
    CHOOSE_RESTAURANT = 5
    CHOOSE_ATTRACTION = 6


@dataclass
class Action:
    type: ActionType
    payload: dict


class TravelEnv:
    def __init__(self, knowledge_base: TravelKnowledgeBase, goal: TripGoal,
                 max_steps: int = 40, top_k: int =30,
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
        self.state = self.base_state.clone()
        self.history: List[str] = []
        self.steps = 0
        self.action_payloads: Dict[str, Tuple] = {}
        self.last_info: Dict = {}
        self._ensure_destination_required()

    def _empty_state(self, phase: str = "city") -> TravelState:
        restaurants = {(day, slot): None for day in range(1, self.total_days + 1) for slot in self.meal_slots}
        attractions = {day: None for day in range(1, self.total_days + 1)}
        return TravelState(
            phase=phase,
            city_seq=[],
            departure_dates=[],
            transport_modes=[],
            hotel_index={},
            restaurants=restaurants,
            attractions=attractions,
            total_cost=0.0,
            cuisine_covered=set(),
            violated=False,
            preference_matches=0,
            violations=[],
            is_terminal=False,
        )

    def _ensure_destination_required(self) -> None:
        dest = self.goal.destination
        if not dest:
            return
        dest_norm = self.kb._normalize_city(dest)
        must_norms = {self.kb._normalize_city(c) for c in self.goal.must_visit_cities}
        if dest_norm not in must_norms and not self.goal.fixed_city_order:
            self.goal.must_visit_cities.append(dest)
        if dest not in self.goal.candidate_cities:
            self.goal.candidate_cities.append(dest)

    def reset(self, goal: Optional[TripGoal] = None) -> Tuple[str, List[str]]:
        init_phase = "city"
        init_city_seq: List[str] = []
        if goal is not None:
            self.goal = goal
            self.total_days = goal.duration_days or 3
            if (self.goal.visiting_city_number or 1) == 1 and self.goal.destination:
                init_city_seq = [self.goal.destination]
                init_phase = "transport_date"
        else:
            if (self.goal.visiting_city_number or 1) == 1 and self.goal.destination:
                init_city_seq = [self.goal.destination]
                init_phase = "transport_date"
        # If no new goal is provided, restore from current anchor
        if goal is None:
            self.state = self.base_state.clone()
            self.history = list(self.base_history)
            self.steps = len(self.history)
            obs = self._observation(self.state)
            valid_actions = self._build_valid_actions(self.state)
            return obs, valid_actions

        self.base_state = self._empty_state(phase=init_phase)
        self.base_state.city_seq = init_city_seq
        if self.goal.fixed_city_order:
            self.base_state.city_seq = list(self.goal.fixed_city_order)
            self.base_state.phase = "transport_date"
        # initialize legs if city_seq known
        num_legs = (len(self.base_state.city_seq) + 1) if self.base_state.city_seq else 0
        if num_legs > 0:
            self.base_state.departure_dates = [None] * num_legs
            self.base_state.transport_modes = [None] * num_legs
        self.base_history = []
        self._ensure_destination_required()
        self.state = self.base_state.clone()
        if not self.state.city_seq and self.goal.fixed_city_order:
            self.state.city_seq = list(self.goal.fixed_city_order)
        self.state.total_cost = self._estimate_cost(self.state)
        self.history = list(self.base_history)
        self.steps = 0
        obs = self._observation(self.state)
        valid_actions = self._build_valid_actions(self.state)
        return obs, valid_actions

    def apply_action(self, action: str):
        """Apply action to the anchor state after planning."""
        # Start from anchored base state/history, apply action, then update anchor
        self.state = self.base_state.clone()
        self.history = list(self.base_history)
        # rebuild action payloads for current state to avoid stale mappings
        current_valid = self._build_valid_actions(self.state)
        if action not in current_valid:
            # invalid request for current anchor, mark violation and finish
            self._ensure_violation(self.state, "invalid_action")
            obs = self._observation(self.state)
            self.last_info = {"violations": list(self.state.violations), "cost": self.state.total_cost}
            return obs, self.reward_cfg.get("invalid_action_penalty", -10.0), True, self.history, []
        obs, reward, done, history, valid_actions = self.step(action)
        self.base_state = self.state.clone()
        self.base_history = list(self.history)
        return obs, reward, done, history, valid_actions

    # --- Action (Enum) based API: skeleton for MCTS integration ---
    def valid_actions(self, state: Optional[TravelState] = None, goal: Optional[TripGoal] = None) -> List[Action]:
        """Return structured actions for the given state/goal; uses existing builder for now."""
        self.goal = goal or self.goal
        state = state or self.state
        # Reuse string-based builder then map to typed actions
        str_actions = self._build_valid_actions(state)
        actions: List[Action] = []
        for a in str_actions:
            payload = self.action_payloads.get(a)
            if payload is None:
                continue
            kind = payload[0]
            if kind == "choose_city":
                _, city = payload
                actions.append(Action(ActionType.ADD_CITY, {"city": city}))
            elif kind == "set_departure_date":
                _, seg_idx, day_idx = payload
                actions.append(Action(ActionType.SET_DEPARTURE_DATE, {"segment": seg_idx, "day": day_idx}))
            elif kind == "segment_mode":
                _, seg_idx, mode, detail = payload
                actions.append(
                    Action(ActionType.SET_TRANSPORT_MODE, {"segment": seg_idx, "mode": mode, "detail": detail})
                )
            elif kind == "stay_city":
                _, city, stay = payload
                actions.append(Action(ActionType.CHOOSE_HOTEL, {"city": city, "stay": stay}))
            elif kind == "meal":
                _, day, slot, rest = payload
                actions.append(
                    Action(ActionType.CHOOSE_RESTAURANT, {"day": day, "slot": slot, "restaurant": rest})
                )
            elif kind == "attraction":
                _, day, slot, attr = payload
                actions.append(Action(ActionType.CHOOSE_ATTRACTION, {"day": day, "slot": slot, "attraction": attr}))
        return actions

    def step_action(self, state: TravelState, action: Action, goal: Optional[TripGoal] = None) -> TravelState:
        """Pure functional transition using structured Action."""
        self.goal = goal or self.goal
        new_state = state.clone()
        if action.type == ActionType.ADD_CITY:
            city = action.payload["city"]
            if city not in new_state.city_seq:
                new_state.city_seq.append(city)
                num_legs = len(new_state.city_seq) + 1
                new_state.departure_dates = [None] * num_legs
                new_state.transport_modes = [None] * num_legs
                if len(new_state.city_seq) >= (self.goal.visiting_city_number or 1):
                    new_state.phase = "transport_date"
        elif action.type == ActionType.SET_DEPARTURE_DATE:
            seg_idx = action.payload["segment"]
            day_idx = action.payload["day"]
            if seg_idx >= len(new_state.departure_dates):
                new_state.departure_dates += [None] * (seg_idx + 1 - len(new_state.departure_dates))
            new_state.departure_dates[seg_idx] = day_idx
            if all(d is not None for d in new_state.departure_dates):
                new_state.phase = "transport_mode"
        elif action.type == ActionType.SET_TRANSPORT_MODE:
            seg_idx = action.payload["segment"]
            mode = action.payload["mode"]
            detail = action.payload.get("detail", {})
            if seg_idx >= len(new_state.transport_modes):
                new_state.transport_modes += [None] * (seg_idx + 1 - len(new_state.transport_modes))
            new_state.transport_modes[seg_idx] = {"mode": mode, "detail": detail}
            if all(m is not None for m in new_state.transport_modes):
                new_state.phase = "accommodation" if self.goal.require_accommodation else "restaurant"
        elif action.type == ActionType.CHOOSE_HOTEL:
            city = action.payload["city"]
            stay = action.payload["stay"]
            new_state.hotel_index[city] = stay
            if all(new_state.hotel_index.get(c) is not None for c in new_state.city_seq):
                new_state.phase = "restaurant"
        elif action.type == ActionType.CHOOSE_RESTAURANT:
            day = action.payload["day"]
            slot = action.payload["slot"]
            rest = action.payload["restaurant"]
            new_state.restaurants[(day, slot)] = rest
            cuisines = str(rest.get("cuisines") or rest.get("Cuisines") or "")
            for c in cuisines.split(","):
                if c.strip():
                    new_state.cuisine_covered.add(c.strip())
            all_filled = all(
                new_state.restaurants[(d, s)] is not None
                for d in range(1, self.total_days + 1)
                for s in self.meal_slots
            )
            if all_filled:
                new_state.phase = "attraction"
        elif action.type == ActionType.CHOOSE_ATTRACTION:
            day = action.payload["day"]
            attr = action.payload["attraction"]
            new_state.attractions[day] = attr
            if all(new_state.attractions.get(d) is not None for d in range(1, self.total_days + 1)):
                new_state.phase = "done"

        # budget / global constraint check (lightweight)
        new_state.total_cost = self._estimate_cost(new_state)
        if self.goal.budget is not None and new_state.total_cost > self.goal.budget:
            new_state.violated = True
        self._check_global_constraints(new_state)
        return new_state

    def _estimate_cost(self, state: TravelState) -> float:
        cost = 0.0
        for mode in state.transport_modes:
            if isinstance(mode, dict):
                detail = mode.get("detail", {})
                if "price" in detail:
                    cost += float(detail["price"])
                elif "cost" in detail:
                    cost += float(detail["cost"])
        for stay in state.hotel_index.values():
            if stay:
                cost += float(stay.get("price", 0.0))
        for meal in state.restaurants.values():
            if meal:
                cost += float(meal.get("cost", 0.0))
        return cost

    def _check_global_constraints(self, state: TravelState) -> None:
        if self.goal.budget is not None and state.total_cost > self.goal.budget:
            state.violated = True

    def _ensure_violation(self, state: TravelState, violation: str) -> None:
        if violation not in state.violations:
            state.violations.append(violation)

    def _matches_preference(self, restaurant: Dict) -> bool:
        if not self.goal.preferences:
            return False
        cuisines = str(restaurant.get("cuisines") or restaurant.get("Cuisines") or "").lower()
        return any(pref.lower() in cuisines for pref in self.goal.preferences)

    def _count_attractions_day(self, state: TravelState, day: int) -> int:
        return 1 if state.attractions.get(day) else 0

    def _restaurant_ids_day(self, state: TravelState, day: int) -> set:
        return {
            meal["id"]
            for (d, _), meal in state.restaurants.items()
            if d == day and meal is not None
        }

    def _restaurant_ids_all(self, state: TravelState) -> set:
        return {meal["id"] for meal in state.restaurants.values() if meal}

    def _attraction_ids_all(self, state: TravelState) -> set:
        return {att["id"] for att in state.attractions.values() if att}

    def _city_for_day(self, state: TravelState, day: int) -> Optional[str]:
        seq = state.city_seq or self.goal.fixed_city_order or self.goal.must_visit_cities
        if not seq:
            return self.goal.destination
        idx = min(len(seq) - 1, int((day - 1) * len(seq) / max(1, self.total_days)))
        return seq[idx]

    def _city_stay_days(self, state: TravelState, city: str) -> int:
        """Estimate how many days the traveler stays in a city based on sequence and total days."""
        seq = state.city_seq or self.goal.fixed_city_order or self.goal.must_visit_cities
        if not seq:
            return self.total_days
        if city not in seq:
            return 0
        pos = seq.index(city)
        city_count = len(seq)
        # rough allocation: divide days equally among cities
        base = self.total_days // city_count
        extra = 1 if pos < (self.total_days % city_count) else 0
        return max(1, base + extra)

    def _segments(self, state: TravelState) -> List[Tuple[int, str, str]]:
        seq = state.city_seq
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

    def _segment_date(self, seg_idx: int) -> Optional[str]:
        if not self.goal.start_date:
            return None
        try:
            start_dt = datetime.fromisoformat(self.goal.start_date)
        except Exception:
            return None
        try:
            target_dt = start_dt + timedelta(days=seg_idx)
            return target_dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    def _allowed_transport_modes(self) -> List[str]:
        allowed = self.goal.transport_allowed_modes or ["flight", "taxi", "self-driving"]
        forbidden = set(m.lower() for m in (self.goal.transport_forbidden_modes or []))
        return [m for m in allowed if m not in forbidden]

    def _observation(self, state: TravelState) -> str:
        parts = [self.goal.as_text()]
        if state.city_seq:
            parts.append(f"Cities selected: {' -> '.join(state.city_seq)}")
        if state.departure_dates:
            parts.append(f"Departure days: {state.departure_dates}")
        if state.transport_modes:
            seg_txt = []
            for idx, tm in enumerate(state.transport_modes):
                if tm:
                    detail = tm.get("detail", {})
                    mode = tm.get("mode")
                    if isinstance(detail, dict) and "origin" in detail and "destination" in detail:
                        seg_txt.append(f"seg{idx}:{detail['origin']}->{detail['destination']} via {mode}")
                    else:
                        seg_txt.append(f"seg{idx}:{mode}")
            if seg_txt:
                parts.append("Transport: " + "; ".join(seg_txt))
        for city, stay in state.hotel_index.items():
            if stay:
                parts.append(f"Stay {city}: {stay['name']} {stay['room_type']} ${stay['price']:.0f}")

        for day in range(1, self.total_days + 1):
            meals = [
                f"{slot}:{meal['name']}"
                for (d, slot), meal in state.restaurants.items()
                if d == day and meal
            ]
            if meals:
                parts.append(f"Day {day} meals: {', '.join(meals)}")
            att = state.attractions.get(day)
            if att:
                parts.append(f"Day {day} attraction: {att['name']}")

        pending = []
        city_target = self.goal.visiting_city_number or 1
        if len(state.city_seq) < city_target:
            pending.append(f"cities missing {city_target - len(state.city_seq)}")
        segments = self._segments(state)
        for idx, src, dst in segments:
            if idx >= len(state.transport_modes) or state.transport_modes[idx] is None:
                pending.append(f"segment {idx} {src}->{dst} mode")
        for city in state.city_seq:
            if self.goal.require_accommodation and state.hotel_index.get(city) is None:
                pending.append(f"stay in {city}")

        for day in range(1, self.total_days + 1):
            missing_meals = [slot for slot in self.meal_slots if state.restaurants[(day, slot)] is None]
            if missing_meals:
                pending.append(f"day{day} meals {len(missing_meals)} missing")
            if not state.attractions.get(day):
                pending.append(f"day{day} attraction missing")
        if pending:
            parts.append("Pending: " + ", ".join(pending))

        if self.goal.budget is not None:
            budget_left = self.goal.budget - self._estimate_cost(state)
            parts.append(f"Budget left estimate: {budget_left:.0f}")
        parts.append(f"Allowed transport: {', '.join(self._allowed_transport_modes())}")
        return " | ".join(parts)

    def render_itinerary(self, state: Optional[TravelState] = None) -> str:
        state = state or self.state
        lines = [self.goal.as_text()]
        if state.city_seq:
            lines.append("Cities: " + " -> ".join(state.city_seq))
        if state.transport_modes:
            seg_lines = []
            for idx, tm in enumerate(state.transport_modes):
                if tm:
                    detail = tm.get("detail", {})
                    mode = tm.get("mode")
                    ori = detail.get("origin", "?")
                    dst = detail.get("destination", "?")
                    seg_lines.append(f"seg{idx}:{ori}->{dst} via {mode}")
            if seg_lines:
                lines.append("Transport: " + "; ".join(seg_lines))
        for city, stay in state.hotel_index.items():
            if stay:
                lines.append(f"Stay in {city}: {stay.get('name')} ({stay.get('room_type')}) ${stay.get('price')}")
        for day in range(1, self.total_days + 1):
            meals = [
                f"{slot}:{meal['name']}"
                for (d, slot), meal in state.restaurants.items()
                if d == day and meal
            ]
            if meals:
                lines.append(f"Day {day} meals: {', '.join(meals)}")
            att = state.attractions.get(day)
            if att:
                lines.append(f"Day {day} attraction: {att['name']} ({att.get('city')})")
        lines.append(f"Cost: {state.total_cost}")
        return "\n".join(lines)

    def evaluate_leaf(self, state: Optional[TravelState] = None) -> float:
        state = state or self.state
        if getattr(state, "violated", False) or state.phase != "done":
            return -1e9
        # hard cuisine coverage if required
        if getattr(self.goal, "required_cuisines", None):
            required = {c.lower() for c in self.goal.required_cuisines}
            covered = {c.lower() for c in state.cuisine_covered}
            if not required.issubset(covered):
                return -5e8
        # LLM score (if available)
        llm_score = 0.0
        if hasattr(self, "llm_policy") and self.llm_policy is not None:
            itinerary_text = self.render_itinerary(state)
            llm_score = self.llm_policy.score_plan(self.goal.as_text(), self.history, itinerary_text)
        budget_slack = max(0.0, (self.goal.budget or 0) - state.total_cost) if self.goal.budget else 0.0
        if getattr(self.goal, "required_cuisines", None):
            required = {c.lower() for c in self.goal.required_cuisines}
            covered = {c.lower() for c in state.cuisine_covered}
            cuisine_ratio = len(required.intersection(covered)) / len(required)
        else:
            cuisine_ratio = 1.0
        return 1.0 * llm_score + 0.001 * budget_slack + 0.5 * cuisine_ratio

    # Phased valid-actions override to enforce staged search
    def _build_valid_actions(self, state: TravelState) -> List[str]:  # type: ignore[override]
        actions: List[str] = []
        self.action_payloads = {}

        if getattr(state, "is_terminal", False) or getattr(state, "violated", False):
            return ["finish"]

        city_target = self.goal.visiting_city_number or 1

        if state.phase == "city" and len(state.city_seq) >= city_target:
            state.phase = "transport_date"
            return self._build_valid_actions(state)

        if state.phase == "accommodation" and not self.goal.require_accommodation:
            state.phase = "restaurant"
            return self._build_valid_actions(state)

        if state.phase == "city":
            candidates = self.goal.candidate_cities or self.kb.get_candidate_cities(
                destination_hint=self.goal.destination,
                must_visit=self.goal.must_visit_cities,
                priority=self.goal.priority_cities,
                top_k=max(self.top_k * 2, city_target),
            )
            origin_norm = self.kb._normalize_city(self.goal.origin)
            for must in self.goal.must_visit_cities:
                if must not in candidates:
                    candidates.insert(0, must)
            for pri in self.goal.priority_cities:
                if pri not in candidates:
                    candidates.append(pri)
            for city in candidates:
                # Allow choosing the origin if origin == destination; otherwise skip adding the origin city again.
                if self.kb._normalize_city(city) == origin_norm and self.goal.origin != self.goal.destination:
                    continue
                if city not in state.city_seq:
                    action = f"choose_city:{city}"
                    self.action_payloads[action] = ("choose_city", city)
                    actions.append(action)
            # Fallback: if no candidates, still allow selecting the destination once to avoid immediate finish.
            if not actions and self.goal.destination and self.goal.destination not in state.city_seq:
                city = self.goal.destination
                action = f"choose_city:{city}"
                self.action_payloads[action] = ("choose_city", city)
                actions.append(action)
            if not actions:
                actions.append("finish")
            return actions

        if state.phase == "transport_date":
            segments = self._segments(state)
            num_legs = len(segments)
            if not state.departure_dates or len(state.departure_dates) < num_legs:
                state.departure_dates = [None] * num_legs
            for idx, _, _ in segments:
                if state.departure_dates[idx] is not None:
                    continue
                if idx == 0:
                    days = [0]
                elif idx == num_legs - 1:
                    days = [self.total_days - 1]
                else:
                    days = list(range(1, self.total_days - 1))
                    # avoid clashes
                    used = set(d for d in state.departure_dates if d is not None)
                    days = [d for d in days if d not in used]
                for d in days:
                    action = f"set_date:seg{idx}:day{d}"
                    self.action_payloads[action] = ("set_departure_date", idx, d)
                    actions.append(action)
            if not actions:
                actions.append("finish")
            return actions

        if state.phase == "restaurant":
            all_meals_filled = all(
                state.restaurants[(d, s)] is not None
                for d in range(1, self.total_days + 1)
                for s in self.meal_slots
            )
            if all_meals_filled:
                state.phase = "attraction"
                return self._build_valid_actions(state)

        if state.phase == "attraction":
            all_atts = all(state.attractions.get(d) is not None for d in range(1, self.total_days + 1))
            if all_atts:
                state.phase = "done"
                return ["finish"]

        if state.phase == "transport_mode":
            allowed_modes = self._allowed_transport_modes()
            # optional rule: if any flight/taxi chosen, avoid self-driving mixes
            if any(m and m.get("mode") in ["flight", "taxi"] for m in state.transport_modes):
                allowed_modes = [m for m in allowed_modes if m != "self-driving"]
            segments = self._segments(state)
            if not state.transport_modes or len(state.transport_modes) < len(segments):
                state.transport_modes = [None] * len(segments)
            for idx, src, dst in segments:
                if idx < len(state.transport_modes) and state.transport_modes[idx] is not None:
                    continue
                seg_date = None
                if state.departure_dates and idx < len(state.departure_dates) and state.departure_dates[idx] is not None:
                    if self.goal.start_date:
                        try:
                            base = datetime.fromisoformat(self.goal.start_date)
                            seg_date = (base + timedelta(days=state.departure_dates[idx])).strftime("%Y-%m-%d")
                        except Exception:
                            seg_date = None
                if seg_date is None:
                    seg_date = self._segment_date(idx)
                for mode in allowed_modes:
                    if mode == "flight":
                        flights = self.kb.get_flights(
                            src, dst, top_k=self.top_k, max_price=self.goal.budget, date_str=seg_date
                        )
                        if not flights and seg_date:
                            flights = self.kb.get_flights(src, dst, top_k=self.top_k, max_price=self.goal.budget)
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
            if not actions:
                actions.append("finish")
            return actions

        if state.phase == "accommodation":
            pending_cities = [c for c in state.city_seq if state.hotel_index.get(c) is None]
            if pending_cities:
                city = pending_cities[0]  # fill by order
                nights = self._city_stay_days(state, city)
                stays = self.kb.get_accommodations(
                    city,
                    top_k=self.top_k,
                    max_price=self.goal.budget,
                    room_type=self.goal.room_type,
                    house_rule=self.goal.house_rule,
                    min_nights=nights,
                )
                for s in stays:
                    rooms = max(1, self.goal.people_number or 1)
                    est_cost = float(s.get("price", 0.0)) * max(1, nights) * rooms
                    projected = self._estimate_cost(state) + est_cost
                    if self.goal.budget is not None and projected > self.goal.budget:
                        continue
                    action = f"stay:{city}:{s['id']} {s['name']} {s['room_type']} ${s['price']:.0f}"
                    self.action_payloads[action] = ("stay_city", city, s, nights)
                    actions.append(action)
            if not actions:
                actions.append("finish")
            return actions

        if state.phase == "restaurant":
            for day in range(1, self.total_days + 1):
                city = self._city_for_day(state, day)
                restaurants = self.kb.get_restaurants(
                    city,
                    preferences=self.goal.preferences,
                    top_k=self.top_k,
                )
                if not restaurants:
                    restaurants = [{
                        "id": f"placeholder-{city}-{day}",
                        "name": f"Any meal in {city}",
                        "cuisines": "any",
                        "cost": 0.0,
                        "rating": 0.0,
                        "city": city,
                    }]
                # if required cuisines exist, prefer ones covering uncovered cuisines
                if getattr(self.goal, "required_cuisines", None):
                    uncovered = {c.lower() for c in self.goal.required_cuisines} - {
                        c.lower() for c in state.cuisine_covered
                    }
                    priority = []
                    fallback = []
                    for r in restaurants:
                        cuisines = str(r.get("cuisines") or "").lower()
                        if any(c in cuisines for c in uncovered):
                            priority.append(r)
                        else:
                            fallback.append(r)
                    restaurants = priority + fallback
                # block duplicates within the same city across all days
                used_ids = {
                    rest["id"]
                    for (d, _), rest in state.restaurants.items()
                    if rest is not None and (rest.get("city") or rest.get("City")) == city
                }
                for slot in self.meal_slots:
                    if state.restaurants[(day, slot)] is None:
                        for r in restaurants:
                            if r["id"] in used_ids:
                                continue
                            action = f"eat:d{day}:{slot}:{r['id']} {r['name']} {r['cuisines']} ${r['cost']:.0f} rating {r['rating']}"
                            self.action_payloads[action] = ("meal", day, slot, r)
                            actions.append(action)
            if not actions:
                actions.append("finish")
            return actions

        if state.phase == "attraction":
            for day in range(1, self.total_days + 1):
                city = self._city_for_day(state, day)
                if state.attractions.get(day):
                    continue
                attractions = self.kb.get_attractions(city, top_k=self.top_k)
                used_att = {
                    att["id"]
                    for att_day, att in state.attractions.items()
                    if att is not None and att.get("city") == city
                }
                for a in attractions:
                    if a["id"] in used_att:
                        continue
                    action = f"visit:d{day}:spot:{a['id']} {a['name']} @ {a['city']}"
                    self.action_payloads[action] = ("attraction", day, "spot", a)
                    actions.append(action)
            if not actions:
                actions.append("finish")
            return actions

        return ["finish"]

    def get_goal(self) -> str:
        return self.goal.as_text()

    def is_success(self, state: Optional[TravelState] = None) -> bool:
        state = state or self.state
        city_target = self.goal.visiting_city_number or 1
        if len(state.city_seq) < city_target:
            return False
        must_norm = {self.kb._normalize_city(c) for c in self.goal.must_visit_cities}
        if must_norm:
            seq_norm = {self.kb._normalize_city(c) for c in state.city_seq}
            if not must_norm.issubset(seq_norm):
                return False

        segments = self._segments(state)
        if len(state.transport_modes) < len(segments):
            return False
        for idx in range(len(segments)):
            if idx >= len(state.transport_modes) or state.transport_modes[idx] is None:
                return False
        if state.departure_dates and len(state.departure_dates) >= len(segments):
            if any(d is None for d in state.departure_dates[: len(segments)]):
                return False

        if self.goal.require_accommodation:
            for city in state.city_seq:
                if state.hotel_index.get(city) is None:
                    return False
        for day in range(1, self.total_days + 1):
            if any(state.restaurants[(day, slot)] is None for slot in self.meal_slots):
                return False
            if state.attractions.get(day) is None:
                return False
        if self.goal.budget is not None and state.total_cost > self.goal.budget:
            return False
        # cuisine coverage if required_cuisines set
        if getattr(self.goal, "required_cuisines", None):
            req = {c.lower() for c in self.goal.required_cuisines}
            covered = {c.lower() for c in state.cuisine_covered}
            if not req.issubset(covered):
                return False
        return True

    def _can_finish(self, state: Optional[TravelState] = None) -> bool:
        state = state or self.state
        city_target = self.goal.visiting_city_number or 1
        if len(state.city_seq) < city_target:
            return False
        must_norm = {self.kb._normalize_city(c) for c in self.goal.must_visit_cities}
        if must_norm:
            seq_norm = {self.kb._normalize_city(c) for c in state.city_seq}
            if not must_norm.issubset(seq_norm):
                return False
        segments = self._segments(state)
        if len(state.transport_modes) < len(segments):
            return False
        if state.departure_dates and len(state.departure_dates) >= len(segments):
            if any(d is None for d in state.departure_dates[: len(segments)]):
                return False
        if self.goal.require_accommodation:
            for city in state.city_seq:
                if state.hotel_index.get(city) is None:
                    return False
        for day in range(1, self.total_days + 1):
            if any(state.restaurants[(day, slot)] is None for slot in self.meal_slots):
                return False
            if state.attractions.get(day) is None:
                return False
        return True

    def step(self, action: str):
        self.steps += 1
        reward = self.reward_cfg.get("step_bonus", 0.0)
        self.last_info = {}
        state_changed = False

        if action == "finish" and not self._can_finish(self.state):
            # Treat impossible finish as terminal failure with scoring so search does not loop.
            reward += self._finish_and_score(self.state)
            obs = self._observation(self.state)
            self.last_info = {"violations": list(self.state.violations), "cost": self.state.total_cost}
            return obs, reward, True, self.history, []

        if action == "finish":
            self.history.append(action)
            reward += self._finish_and_score(self.state)
            obs = self._observation(self.state)
            self.last_info = {"violations": list(self.state.violations), "cost": self.state.total_cost}
            return obs, reward, True, self.history, []

        payload = self.action_payloads.get(action)
        if payload is None:
            # Stale/invalid action: mark violation, score and terminate to avoid loops.
            self._ensure_violation(self.state, "invalid_action")
            reward += self.reward_cfg.get("invalid_action_penalty", -10.0)
            reward += self._finish_and_score(self.state)
            obs = self._observation(self.state)
            self.last_info = {"violations": list(self.state.violations), "cost": self.state.total_cost}
            return obs, reward, True, self.history, []

        kind = payload[0]
        if kind == "choose_city":
            _, city = payload
            if city not in self.state.city_seq:
                self.state.city_seq.append(city)
                num_legs = len(self.state.city_seq) + 1
                self.state.departure_dates = [None] * num_legs
                self.state.transport_modes = [None] * num_legs
                # phase progression
                if len(self.state.city_seq) >= (self.goal.visiting_city_number or 1):
                    self.state.phase = "transport_date"
                state_changed = True
        elif kind == "set_departure_date":
            _, seg_idx, day_idx = payload
            if seg_idx >= len(self.state.departure_dates):
                self.state.departure_dates += [None] * (seg_idx + 1 - len(self.state.departure_dates))
            if self.state.departure_dates[seg_idx] is None:
                self.state.departure_dates[seg_idx] = day_idx
                if all(d is not None for d in self.state.departure_dates):
                    self.state.phase = "transport_mode"
                state_changed = True
        elif kind == "segment_mode":
            _, seg_idx, mode, detail = payload
            if seg_idx >= len(self.state.transport_modes):
                self.state.transport_modes += [None] * (seg_idx + 1 - len(self.state.transport_modes))
            self.state.transport_modes[seg_idx] = {"mode": mode, "detail": detail}
            if all(m is not None for m in self.state.transport_modes):
                self.state.phase = "accommodation" if self.goal.require_accommodation else "restaurant"
            state_changed = True
        elif kind == "stay_city":
            _, city, stay, nights = payload
            if self.state.hotel_index.get(city) is None:
                cost = float(stay.get("price", 0.0)) * max(1, nights) * max(1, self.goal.people_number or 1)
                stay_with_cost = dict(stay)
                stay_with_cost["price"] = cost
                self.state.hotel_index[city] = stay_with_cost
                if all(self.state.hotel_index.get(c) is not None for c in self.state.city_seq):
                    self.state.phase = "restaurant"
                state_changed = True
        elif kind == "meal":
            _, day, slot, rest = payload
            key = (day, slot)
            if key in self.state.restaurants and self.state.restaurants[key] is None:
                day_ids = self._restaurant_ids_day(self.state, day)
                all_ids = self._restaurant_ids_all(self.state)
                if rest["id"] in day_ids:
                    reward += self.reward_cfg.get("duplicate_meal_penalty", 0.0)
                elif rest["id"] in all_ids:
                    reward += self.reward_cfg.get("duplicate_restaurant_across_days_penalty", 0.0)
                # Always fill the slot to ensure progress
                self.state.restaurants[key] = rest
                cuisines = str(rest.get("cuisines") or rest.get("Cuisines") or "")
                for c in cuisines.split(","):
                    if c.strip():
                        self.state.cuisine_covered.add(c.strip())
                state_changed = True
                if self._matches_preference(rest):
                    self.state.preference_matches += 1
                    reward += self.reward_cfg.get("preference_bonus", 0.0)
                # advance phase if all meals filled
                all_filled = all(
                    self.state.restaurants[(d, s)] is not None
                    for d in range(1, self.total_days + 1)
                    for s in self.meal_slots
                )
                if all_filled:
                    self.state.phase = "attraction"
            else:
                # invalid repeated fill on occupied slot
                self._ensure_violation(self.state, "duplicate_slot_meal")
                reward += self.reward_cfg.get("invalid_action_penalty", -10.0)
                reward += self._finish_and_score(self.state)
                obs = self._observation(self.state)
                self.last_info = {"violations": list(self.state.violations), "cost": self.state.total_cost}
                return obs, reward, True, self.history, []
        elif kind == "attraction":
            _, day, slot, attr = payload
            if self.state.attractions.get(day) is None:
                if attr["id"] in self._attraction_ids_all(self.state):
                    reward += self.reward_cfg.get("duplicate_attraction_penalty", 0.0)
                else:
                    self.state.attractions[day] = attr
                    state_changed = True
                    reward += self.reward_cfg.get("poi_bonus", 0.0)
                    if all(self.state.attractions.get(d) is not None for d in range(1, self.total_days + 1)):
                        self.state.phase = "done"
            else:
                self._ensure_violation(self.state, "duplicate_slot_attraction")
                reward += self.reward_cfg.get("invalid_action_penalty", -10.0)
                reward += self._finish_and_score(self.state)
                obs = self._observation(self.state)
                self.last_info = {"violations": list(self.state.violations), "cost": self.state.total_cost}
                return obs, reward, True, self.history, []

        if not state_changed and action != "finish":
            # No state mutation occurred; terminate with penalty to avoid loops.
            self._ensure_violation(self.state, "no_op_action")
            reward += self.reward_cfg.get("invalid_action_penalty", -10.0)
            reward += self._finish_and_score(self.state)
            obs = self._observation(self.state)
            self.last_info = {"violations": list(self.state.violations), "cost": self.state.total_cost}
            return obs, reward, True, self.history, []

        self.state.total_cost = self._estimate_cost(self.state)
        if self.goal.budget is not None and self.state.total_cost > self.goal.budget:
            self._ensure_violation(self.state, "budget")
            reward += self.reward_cfg.get("budget_violation_penalty", 0.0)
            self.state.violated = True
        self._check_global_constraints(self.state)

        self.history.append(action)

        done = self.is_success(self.state) or self.steps >= self.max_steps
        if done:
            reward += self._finish_and_score(self.state)

        obs = self._observation(self.state)
        valid_actions = self._build_valid_actions(self.state) if not done else []
        self.last_info = {"violations": list(self.state.violations), "cost": self.state.total_cost}
        return obs, reward, done, self.history, valid_actions

    def _finish_and_score(self, state: TravelState) -> float:
        state.is_terminal = True
        state.total_cost = self._estimate_cost(state)
        reward = 0.0

        city_target = self.goal.visiting_city_number or 1
        if len(state.city_seq) < city_target:
            self._ensure_violation(state, "city_count")
            reward += (city_target - len(state.city_seq)) * self.reward_cfg.get("missing_city_penalty", 0.0)

        must_norm = {self.kb._normalize_city(c) for c in self.goal.must_visit_cities}
        if must_norm:
            seq_norm = {self.kb._normalize_city(c) for c in state.city_seq}
            missing_must = [c for c in must_norm if c not in seq_norm]
            if missing_must:
                self._ensure_violation(state, "must_city")
                reward += len(missing_must) * self.reward_cfg.get("missing_must_city_penalty", 0.0)

        segments = self._segments(state)
        for idx, src, dst in segments:
            if idx >= len(state.transport_modes) or state.transport_modes[idx] is None:
                self._ensure_violation(state, f"segment{idx}")
                reward += self.reward_cfg.get("missing_segment_penalty", 0.0)

        if self.goal.require_flight and segments:
            first_seg = state.transport_modes[0] if state.transport_modes else None
            if not first_seg or first_seg.get("mode") != "flight":
                self._ensure_violation(state, "outbound")
                reward += self.reward_cfg.get("missing_flight_penalty", 0.0)
            if self.goal.return_required:
                last_idx = segments[-1][0]
                last_seg = state.transport_modes[last_idx] if last_idx < len(state.transport_modes) else None
                if not last_seg or last_seg.get("mode") != "flight":
                    self._ensure_violation(state, "return")
                    reward += self.reward_cfg.get("missing_return_penalty", 0.0)

        if self.goal.require_accommodation:
            for city in state.city_seq:
                if state.hotel_index.get(city) is None:
                    self._ensure_violation(state, f"stay_{city}")
                    reward += self.reward_cfg.get("missing_stay_penalty", 0.0)

        for day in range(1, self.total_days + 1):
            filled_meals = [
                state.restaurants[(day, slot)]
                for slot in self.meal_slots
                if state.restaurants[(day, slot)] is not None
            ]
            missing_meals = len(self.meal_slots) - len(filled_meals)
            if missing_meals:
                self._ensure_violation(state, f"day{day}_meals")
                reward += missing_meals * self.reward_cfg.get("meal_missing_penalty", 0.0)
            att_count = self._count_attractions_day(state, day)
            expected_atts = len(self.attraction_slots)
            if att_count < expected_atts:
                missing_att = expected_atts - att_count
                self._ensure_violation(state, f"day{day}_attractions")
                reward += missing_att * self.reward_cfg.get("attraction_missing_penalty", 0.0)
            # duplicate penalties
            day_ids = self._restaurant_ids_day(state, day)
            if len(day_ids) < len(filled_meals):
                self._ensure_violation(state, f"day{day}_duplicate_meals")
                reward += self.reward_cfg.get("duplicate_meal_penalty", 0.0)
        # cross-day duplicate restaurants
        all_rest = self._restaurant_ids_all(state)
        total_meals = sum(1 for v in state.restaurants.values() if v is not None)
        if len(all_rest) < total_meals:
            self._ensure_violation(state, "duplicate_restaurants")
            reward += self.reward_cfg.get("duplicate_restaurant_across_days_penalty", 0.0)

        # global duplicate attraction penalty
        att_all = self._attraction_ids_all(state)
        total_atts = sum(1 for v in state.attractions.values() if v is not None)
        if len(att_all) < total_atts:
            self._ensure_violation(state, "duplicate_attractions")
            reward += self.reward_cfg.get("duplicate_attraction_penalty", 0.0)

        if self.goal.budget is not None and state.total_cost > self.goal.budget:
            self._ensure_violation(state, "budget")
            reward += self.reward_cfg.get("budget_violation_penalty", 0.0)

        reward += state.preference_matches * self.reward_cfg.get("preference_bonus", 0.0)

        if self.is_success(state):
            reward += self.reward_cfg.get("finish_success_bonus", 0.0)
        else:
            reward += self.reward_cfg.get("finish_fail_penalty", 0.0)
        return reward
