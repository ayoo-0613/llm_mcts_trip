from __future__ import annotations

import math
from typing import Any, Dict

from mcts.travel.retrieval import DedupFilter
from mcts.travel.retrieval.agent_impl import RetrievalAgent as _RetrievalAgentImpl
from mcts.travel.retrieval.agent_types import SlotActionResult


class RetrievalAgent(_RetrievalAgentImpl):
    # ----------------------------
    # Main slot entrypoint
    # ----------------------------
    def compute(
        self,
        parsed: Any,
        state: Any,
        slot: Any,
        *,
        user_query: str = "",
        batch_idx: int = 0,
    ) -> SlotActionResult:
        stype = getattr(slot, "type", None)

        if stype == "city":
            actions, payloads, event = self.build_city_actions(parsed, state)
            return SlotActionResult(
                actions=actions,
                payloads=payloads,
                candidates=[],
                relaxed=False,
                filt={},
                policy_event=event or {},
                plan=None,
                uncapped_filter=None,
            )

        self._ensure_city_pools(parsed)

        # Transport consistency guardrail (for evaluation compatibility):
        # - If any segment already used self-driving, force all remaining segments to self-driving.
        # - If any segment already used flight or taxi, never introduce self-driving.
        if stype == "flight":
            modes = []
            for seg in (getattr(state, "segment_modes", None) or {}).values():
                if isinstance(seg, dict) and seg.get("mode"):
                    modes.append(str(seg["mode"]).lower())
            has_self = "self-driving" in modes
            has_flight = "flight" in modes
            has_taxi = "taxi" in modes
            if has_self:
                actions, payloads = self._ground_fallback_actions(parsed, slot, force_mode="self-driving")
                return SlotActionResult(
                    actions=actions,
                    payloads=payloads,
                    candidates=[],
                    relaxed=False,
                    filt={},
                    policy_event={"forced_transport_mode": "self-driving"},
                    plan=None,
                    uncapped_filter=None,
                )
            require_flight = bool(self._parsed_get(parsed, "require_flight", default=False))
            origin = getattr(slot, "origin", None)
            destination = getattr(slot, "destination", None)
            people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
            allowed_modes = self._allowed_transport_modes(parsed)
            nonflight_modes = [m for m in allowed_modes if m != "flight"]
            if has_flight or has_taxi:
                nonflight_modes = [m for m in nonflight_modes if m != "self-driving"]
            distance = self.kb.distance_km(origin, destination)

            flights, event = self._flight_candidates(parsed, state, slot)
            min_flight_total = None
            if flights:
                try:
                    min_price = min(float(f.get("price") or 0.0) for f in flights if f.get("price") is not None)
                except ValueError:
                    min_price = None
                min_flight_total = min_price * float(people) if min_price is not None else None

            ground_costs: Dict[str, float] = {}
            if distance is not None:
                if "taxi" in nonflight_modes:
                    ground_costs["taxi"] = distance * math.ceil(people / 4.0)
                if "self-driving" in nonflight_modes:
                    ground_costs["self-driving"] = distance * math.ceil(people / 5.0)
            min_ground_mode = None
            min_ground_cost = None
            if ground_costs:
                min_ground_mode = min(ground_costs, key=ground_costs.get)
                min_ground_cost = ground_costs[min_ground_mode]

            chosen_mode = None
            if require_flight or not self._flight_allowed(parsed):
                chosen_mode = "flight" if flights else None
            else:
                if min_flight_total is None:
                    chosen_mode = min_ground_mode
                elif min_ground_cost is None:
                    chosen_mode = "flight"
                else:
                    chosen_mode = "flight" if min_flight_total <= min_ground_cost else min_ground_mode

            if chosen_mode == "flight":
                candidates = self._select_valid_batch(flights, self.top_k, batch_idx=batch_idx)
                action_objs = self._action_factory.build(slot, candidates)
                actions, payloads = self._action_factory.to_actions_payloads(action_objs)
                if not actions:
                    return SlotActionResult(
                        actions=actions,
                        payloads=payloads,
                        candidates=candidates,
                        relaxed=bool(event.get("budget_relaxed")),
                        filt={"budget_cap": event.get("budget_cap")},
                        policy_event=dict(event, flight_candidates_empty=True),
                        plan=None,
                        uncapped_filter=None,
                    )
                event = dict(event or {})
                event.update(
                    {
                        "chosen_mode": "flight",
                        "min_flight_total": min_flight_total,
                        "min_ground_cost": min_ground_cost,
                    }
                )
                return SlotActionResult(
                    actions=actions,
                    payloads=payloads,
                    candidates=candidates,
                    relaxed=bool(event.get("budget_relaxed")),
                    filt={"budget_cap": event.get("budget_cap")},
                    policy_event=event,
                    plan=None,
                    uncapped_filter=None,
                )

            if chosen_mode in ("taxi", "self-driving"):
                actions, payloads = self._ground_fallback_actions(parsed, slot, force_mode=chosen_mode)
                return SlotActionResult(
                    actions=actions,
                    payloads=payloads,
                    candidates=[],
                    relaxed=False,
                    filt={},
                    policy_event={
                        "chosen_mode": chosen_mode,
                        "min_flight_total": min_flight_total,
                        "min_ground_cost": min_ground_cost,
                    },
                    plan=None,
                    uncapped_filter=None,
                )

            return SlotActionResult(
                actions=[],
                payloads={},
                candidates=[],
                relaxed=bool(event.get("budget_relaxed")),
                filt={"budget_cap": event.get("budget_cap")},
                policy_event=dict(event, flight_candidates_empty=True),
                plan=None,
                uncapped_filter=None,
            )

        if stype == "hotel":
            candidates, event = self._hotel_candidates(parsed, state, slot)
            candidates = self._select_valid_batch(candidates, self.top_k, batch_idx=batch_idx)
            action_objs = self._action_factory.build(slot, candidates)
            actions, payloads = self._action_factory.to_actions_payloads(action_objs)
            return SlotActionResult(
                actions=actions,
                payloads=payloads,
                candidates=candidates,
                relaxed=bool(event.get("budget_relaxed")),
                filt={"budget_cap": event.get("budget_cap")},
                policy_event=event,
                plan=None,
                uncapped_filter=None,
            )

        if stype in ("meal", "restaurant"):
            candidates, event = self._meal_candidates(parsed, state, slot)
            used_ids = self._used_restaurant_ids(state)
            candidates = self._select_valid_batch(
                candidates,
                self.top_k,
                predicate=DedupFilter("id", used_ids),
                batch_idx=batch_idx,
            )
            action_objs = self._action_factory.build(slot, candidates)
            actions, payloads = self._action_factory.to_actions_payloads(action_objs)
            return SlotActionResult(
                actions=actions,
                payloads=payloads,
                candidates=candidates,
                relaxed=bool(event.get("budget_relaxed")),
                filt={"budget_cap": event.get("budget_cap")},
                policy_event=event,
                plan=None,
                uncapped_filter=None,
            )

        if stype == "attraction":
            candidates, event = self._attraction_candidates(parsed, state, slot)
            used_ids = self._used_attraction_ids(state)
            candidates = self._select_valid_batch(
                candidates,
                self.top_k,
                predicate=DedupFilter("id", used_ids),
                batch_idx=batch_idx,
            )
            action_objs = self._action_factory.build(slot, candidates)
            actions, payloads = self._action_factory.to_actions_payloads(action_objs)
            return SlotActionResult(
                actions=actions,
                payloads=payloads,
                candidates=candidates,
                relaxed=False,
                filt={},
                policy_event=event or {},
                plan=None,
                uncapped_filter=None,
            )

        if stype == "finish":
            action_objs = self._action_factory.build(slot, [])
            actions, payloads = self._action_factory.to_actions_payloads(action_objs)
            return SlotActionResult(
                actions=actions,
                payloads=payloads,
                candidates=[],
                relaxed=False,
                filt={},
                policy_event={},
                plan=None,
                uncapped_filter=None,
            )

        return SlotActionResult(
            actions=[],
            payloads={},
            candidates=[],
            relaxed=False,
            filt={},
            policy_event={},
            plan=None,
            uncapped_filter=None,
        )

