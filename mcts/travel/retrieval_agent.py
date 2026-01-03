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

        # Transport consistency guardrail for segment slots (evaluation compatibility):
        # - If any segment already used self-driving, only allow self-driving.
        # - If any segment already used flight or taxi, disallow self-driving.
        if stype in ("segment", "flight"):
            modes = []
            for seg in (getattr(state, "segment_modes", None) or {}).values():
                if isinstance(seg, dict) and seg.get("mode"):
                    modes.append(str(seg["mode"]).lower())
            if "self-driving" in modes:
                guardrail_modes = {"self-driving"}
            elif "flight" in modes or "taxi" in modes:
                guardrail_modes = {"flight", "taxi"}
            else:
                guardrail_modes = {"flight", "taxi", "self-driving"}

            query_modes = {m.lower() for m in (self._allowed_transport_modes(parsed) or [])}
            allowed_modes = {m for m in guardrail_modes if m in query_modes}
            require_flight = bool(self._parsed_get(parsed, "require_flight", default=False))
            if require_flight:
                allowed_modes &= {"flight"}
            dominant_nonbudget = None
            if not allowed_modes:
                dominant_nonbudget = "transport_mode_forbidden"

            origin = getattr(slot, "origin", None)
            destination = getattr(slot, "destination", None)
            people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))

            segment_pool = []
            flight_pool = []
            ground_pool = []
            flight_event = {}
            if "flight" in allowed_modes:
                flight_pool, flight_event = self._flight_pool(slot)
                for flight in flight_pool:
                    cand = dict(flight)
                    cand["mode"] = "flight"
                    segment_pool.append(cand)

            distance = None
            if origin and destination:
                distance = self.kb.distance_km(origin, destination)
            if distance is not None:
                base_cost = float(distance)
                if "taxi" in allowed_modes:
                    cand = {
                        "id": f"taxi:{origin}->{destination}",
                        "mode": "taxi",
                        "origin": origin,
                        "destination": destination,
                        "distance": distance,
                        "cost": base_cost,
                        "price": base_cost,
                        "fallback_nonflight": True,
                    }
                    segment_pool.append(cand)
                    ground_pool.append(cand)
                if "self-driving" in allowed_modes:
                    cand = {
                        "id": f"self-driving:{origin}->{destination}",
                        "mode": "self-driving",
                        "origin": origin,
                        "destination": destination,
                        "distance": distance,
                        "cost": base_cost,
                        "price": base_cost,
                        "fallback_nonflight": True,
                    }
                    segment_pool.append(cand)
                    ground_pool.append(cand)

            def _hard_cost(cand: Dict[str, Any]) -> float | None:
                mode = str(cand.get("mode") or "flight").lower()
                if mode == "flight":
                    price = cand.get("price")
                    if price is None or self._is_nan_value(price):
                        return None
                    return float(price) * float(people)
                base = cand.get("cost")
                if base is None:
                    base = cand.get("price")
                if base is None or self._is_nan_value(base):
                    return None
                base_f = float(base)
                if mode == "taxi":
                    return base_f * math.ceil(float(people) / 4.0)
                if mode == "self-driving":
                    return base_f * math.ceil(float(people) / 5.0)
                return None

            def _soft_cost(cand: Dict[str, Any]) -> float | None:
                total = _hard_cost(cand)
                if total is None:
                    return None
                return float(total) / float(people)

            candidates, budget_event = self._filter_candidates_with_budget(
                parsed,
                state,
                "segment",
                segment_pool,
                hard_cost_fn=_hard_cost,
                soft_cost_fn=_soft_cost,
            )

            def _sort_key(cand: Dict[str, Any]) -> tuple:
                soft_penalty = float(cand.get("soft_penalty") or 0.0)
                soft_cost = _soft_cost(cand)
                soft_cost_val = float(soft_cost) if soft_cost is not None else float("inf")
                duration = self._duration_minutes(cand.get("duration")) or float("inf")
                return (soft_penalty, soft_cost_val, duration)

            candidates.sort(key=_sort_key)
            candidates = self._select_valid_fill(candidates, self.top_k)

            actions: list[str] = []
            payloads: Dict[str, tuple] = {}
            seg_idx = getattr(slot, "seg", None)
            seg_val = seg_idx if seg_idx is not None else -1
            for cand in candidates:
                mode = str(cand.get("mode") or "flight").lower()
                if mode == "flight":
                    price_val = float(cand.get("price", 0) or 0.0)
                    text = (
                        f"move:seg{seg_val}:flight:{cand.get('id')} {cand.get('origin')}->{cand.get('destination')} "
                        f"${price_val:.0f} {cand.get('depart', '?')}-{cand.get('arrive', '?')}"
                    )
                    payloads[text] = ("segment_mode", seg_val, "flight", cand)
                    actions.append(text)
                    continue
                if mode in ("taxi", "self-driving"):
                    cand_origin = cand.get("origin") or origin
                    cand_destination = cand.get("destination") or destination
                    cand_distance = cand.get("distance")
                    if cand_distance is None and cand_origin and cand_destination:
                        cand_distance = self.kb.distance_km(cand_origin, cand_destination)
                    if cand_distance is None:
                        continue
                    cand_cost = cand.get("cost")
                    if cand_cost is None:
                        cand_cost = cand_distance
                    text = (
                        f"move:seg{seg_val}:{mode}:{cand_origin}->{cand_destination} "
                        f"{float(cand_distance):.0f}km cost {float(cand_cost):.0f}"
                    )
                    detail = dict(cand)
                    detail.setdefault("origin", cand_origin)
                    detail.setdefault("destination", cand_destination)
                    detail.setdefault("distance", cand_distance)
                    detail.setdefault("cost", cand_cost)
                    payloads[text] = ("segment_mode", seg_val, mode, detail)
                    actions.append(text)

            event = dict(budget_event or {})
            event.update(flight_event or {})
            event.update(
                {
                    "guardrail_modes": sorted(guardrail_modes),
                    "allowed_modes": sorted(allowed_modes),
                    "segment_pool": len(segment_pool),
                    "segment_pool_flights": len(flight_pool),
                    "segment_pool_ground": len(ground_pool),
                    "kb_count": len(segment_pool),
                    "after_nonbudget_count": len(segment_pool),
                    "dominant_nonbudget_filter": dominant_nonbudget,
                }
            )
            if not actions:
                event["segment_candidates_empty"] = True

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

        if stype == "hotel":
            candidates, event = self._hotel_candidates(parsed, state, slot)
            candidates = self._select_valid_fill(candidates, self.top_k)
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
            candidates = self._select_valid_fill(
                candidates,
                self.top_k,
                predicate=DedupFilter("id", used_ids),
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
            candidates = self._select_valid_fill(
                candidates,
                self.top_k,
                predicate=DedupFilter("id", used_ids),
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
