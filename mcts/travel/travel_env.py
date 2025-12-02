from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mcts.travel.knowledge_base import TravelKnowledgeBase, TripGoal

# Shaping constants for constraint-aware planning (per-step and terminal checks)
DEFAULT_REWARD_CFG = {
    "step_bonus": 0.1,
    "budget_violation_penalty": -20.0,
    "missing_flight_penalty": -10.0,
    "missing_return_penalty": -10.0,
    "missing_stay_penalty": -8.0,
    "meal_missing_penalty": -4.0,
    "attraction_missing_penalty": -4.0,
    "preference_bonus": 1.5,
    "poi_bonus": 1.0,
    "finish_success_bonus": 6.0,
    "finish_fail_penalty": -6.0,
}

MEAL_SLOTS = ["breakfast", "lunch", "dinner"]
ATTRACTION_SLOTS = ["morning", "afternoon", "evening", "night"]


@dataclass
class TravelState:
    outbound_flight: Optional[Dict] = None
    return_flight: Optional[Dict] = None
    accommodation: Optional[Dict] = None
    meals: Dict[int, Dict[str, Optional[Dict]]] = field(default_factory=dict)  # day -> slot -> restaurant
    attractions: Dict[int, Dict[str, Optional[Dict]]] = field(default_factory=dict)  # day -> slot -> attraction
    cost: float = 0.0
    preference_matches: int = 0
    violations: List[str] = field(default_factory=list)
    is_terminal: bool = False

    def clone(self) -> "TravelState":
        return TravelState(
            outbound_flight=copy.deepcopy(self.outbound_flight),
            return_flight=copy.deepcopy(self.return_flight),
            accommodation=copy.deepcopy(self.accommodation),
            meals=copy.deepcopy(self.meals),
            attractions=copy.deepcopy(self.attractions),
            cost=self.cost,
            preference_matches=self.preference_matches,
            violations=list(self.violations),
            is_terminal=self.is_terminal,
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
        self.state = self.base_state.clone()
        self.history: List[str] = []
        self.steps = 0
        self.action_payloads: Dict[str, Tuple] = {}
        self.last_info: Dict = {}

    def _empty_state(self) -> TravelState:
        meals = {day: {slot: None for slot in self.meal_slots} for day in range(1, self.total_days + 1)}
        attractions = {day: {slot: None for slot in self.attraction_slots} for day in range(1, self.total_days + 1)}
        return TravelState(meals=meals, attractions=attractions)

    def reset(self, goal: Optional[TripGoal] = None) -> Tuple[str, List[str]]:
        if goal is not None:
            self.goal = goal
            self.total_days = goal.duration_days or 3
            self.base_state = self._empty_state()
            self.base_history = []
        self.state = self.base_state.clone()
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

    def _estimate_cost(self, state: TravelState) -> float:
        cost = 0.0
        if state.outbound_flight is not None:
            cost += state.outbound_flight["price"]
        if state.return_flight is not None:
            cost += state.return_flight["price"]
        if state.accommodation is not None:
            cost += state.accommodation["price"]
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

    def _observation(self, state: TravelState) -> str:
        parts = [self.goal.as_text()]
        if state.outbound_flight:
            f = state.outbound_flight
            parts.append(
                f"Outbound: {f['id']} {f['origin']}->{f['destination']} {f['depart']}-{f['arrive']} ${f['price']:.0f}"
            )
        if state.return_flight:
            f = state.return_flight
            parts.append(
                f"Return: {f['id']} {f['origin']}->{f['destination']} {f['depart']}-{f['arrive']} ${f['price']:.0f}"
            )
        if state.accommodation:
            s = state.accommodation
            parts.append(
                f"Stay: {s['name']} in {s['city']} ({s['room_type']}) ${s['price']:.0f}"
            )

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
        if self.goal.require_flight and state.outbound_flight is None:
            pending.append("outbound flight")
        if self.goal.require_flight and self.goal.return_required and state.return_flight is None:
            pending.append("return flight")
        if self.goal.require_accommodation and state.accommodation is None:
            pending.append("accommodation")

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
        dist = self.kb.distance_between(self.goal.origin, self.goal.destination)
        if dist:
            parts.append(f"Ground distance {dist}")
        return " | ".join(parts)

    def _build_valid_actions(self, state: TravelState) -> List[str]:
        actions: List[str] = []
        self.action_payloads = {}

        if self.goal.require_flight and state.outbound_flight is None:
            flights = self.kb.get_flights(
                self.goal.origin, self.goal.destination,
                top_k=self.top_k,
                max_price=self.goal.budget,
            )
            for f in flights:
                action = f"flight_out:{f['id']} {f['origin']}->{f['destination']} ${f['price']:.0f} {f['depart']}-{f['arrive']}"
                self.action_payloads[action] = ("outbound", f)
                actions.append(action)

        if self.goal.require_flight and self.goal.return_required and state.return_flight is None:
            flights = self.kb.get_flights(
                self.goal.destination, self.goal.origin,
                top_k=self.top_k,
                max_price=self.goal.budget,
            )
            for f in flights:
                action = f"flight_back:{f['id']} {f['origin']}->{f['destination']} ${f['price']:.0f} {f['depart']}-{f['arrive']}"
                self.action_payloads[action] = ("return", f)
                actions.append(action)

        if self.goal.require_accommodation and state.accommodation is None:
            stays = self.kb.get_accommodations(
                self.goal.destination,
                top_k=self.top_k,
                max_price=self.goal.budget,
            )
            for s in stays:
                action = f"stay:{s['id']} {s['name']} {s['room_type']} ${s['price']:.0f}"
                self.action_payloads[action] = ("accommodation", s)
                actions.append(action)

        restaurants = self.kb.get_restaurants(
            self.goal.destination,
            preferences=self.goal.preferences,
            top_k=self.top_k,
        )
        for day in range(1, self.total_days + 1):
            for slot in self.meal_slots:
                if state.meals[day][slot] is None:
                    for r in restaurants:
                        action = f"eat:d{day}:{slot}:{r['id']} {r['name']} {r['cuisines']} ${r['cost']:.0f} rating {r['rating']}"
                        self.action_payloads[action] = ("meal", day, slot, r)
                        actions.append(action)

        attractions = self.kb.get_attractions(self.goal.destination, top_k=self.top_k)
        for day in range(1, self.total_days + 1):
            current_att = self._count_attractions_day(state, day)
            if current_att >= self.goal.attractions_per_day_max:
                continue
            for slot in self.attraction_slots:
                if state.attractions[day][slot] is None:
                    for a in attractions:
                        action = f"visit:d{day}:{slot}:{a['id']} {a['name']} @ {a['city']}"
                        self.action_payloads[action] = ("attraction", day, slot, a)
                        actions.append(action)

        actions.append("finish")
        return actions

    def get_goal(self) -> str:
        return self.goal.as_text()

    def is_success(self, state: Optional[TravelState] = None) -> bool:
        state = state or self.state
        if self.goal.require_flight and state.outbound_flight is None:
            return False
        if self.goal.require_flight and self.goal.return_required and state.return_flight is None:
            return False
        if self.goal.require_accommodation and state.accommodation is None:
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

    def step(self, action: str):
        self.steps += 1
        reward = self.reward_cfg.get("step_bonus", 0.0)
        self.last_info = {}

        if action == "finish":
            self.history.append(action)
            reward += self._finish_and_score(self.state)
            obs = self._observation(self.state)
            self.last_info = {"violations": list(self.state.violations), "cost": self.state.cost}
            return obs, reward, True, self.history, []

        payload = self.action_payloads.get(action)
        if payload is None:
            reward = -1.0
            done = False
            obs = self._observation(self.state)
            valid_actions = self._build_valid_actions(self.state)
            self.last_info = {"violations": list(self.state.violations), "cost": self.state.cost}
            return obs, reward, done, self.history, valid_actions

        kind = payload[0]
        if kind == "outbound" and self.state.outbound_flight is None:
            _, flight = payload
            self.state.outbound_flight = flight
        elif kind == "return" and self.state.return_flight is None:
            _, flight = payload
            self.state.return_flight = flight
        elif kind == "accommodation" and self.state.accommodation is None:
            _, stay = payload
            self.state.accommodation = stay
        elif kind == "meal":
            _, day, slot, rest = payload
            if self.state.meals[day][slot] is None:
                self.state.meals[day][slot] = rest
                if self._matches_preference(rest):
                    self.state.preference_matches += 1
                    reward += self.reward_cfg.get("preference_bonus", 0.0)
        elif kind == "attraction":
            _, day, slot, attr = payload
            if self.state.attractions[day][slot] is None:
                if self._count_attractions_day(self.state, day) < self.goal.attractions_per_day_max:
                    self.state.attractions[day][slot] = attr
                    reward += self.reward_cfg.get("poi_bonus", 0.0)

        self.state.cost = self._estimate_cost(self.state)
        if self.goal.budget is not None and self.state.cost > self.goal.budget:
            self._ensure_violation(self.state, "budget")
            reward += self.reward_cfg.get("budget_violation_penalty", 0.0)

        self.history.append(action)

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

        if self.goal.require_flight and state.outbound_flight is None:
            self._ensure_violation(state, "outbound")
            reward += self.reward_cfg.get("missing_flight_penalty", 0.0)
        if self.goal.require_flight and self.goal.return_required and state.return_flight is None:
            self._ensure_violation(state, "return")
            reward += self.reward_cfg.get("missing_return_penalty", 0.0)
        if self.goal.require_accommodation and state.accommodation is None:
            self._ensure_violation(state, "accommodation")
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

        if self.goal.budget is not None and state.cost > self.goal.budget:
            self._ensure_violation(state, "budget")
            reward += self.reward_cfg.get("budget_violation_penalty", 0.0)

        reward += state.preference_matches * self.reward_cfg.get("preference_bonus", 0.0)

        if self.is_success(state):
            reward += self.reward_cfg.get("finish_success_bonus", 0.0)
        else:
            reward += self.reward_cfg.get("finish_fail_penalty", 0.0)
        return reward
