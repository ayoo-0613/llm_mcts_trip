from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mcts.travel import filters


@dataclass
class SlotActionResult:
    actions: List[str]
    payloads: Dict[str, Tuple]
    candidates: List[Dict[str, Any]]
    relaxed: bool
    filt: Dict[str, Any]
    policy_event: Dict[str, Any]
    plan: Optional[Any] = None
    uncapped_filter: Optional[Dict[str, Any]] = None


class RetrievalAgent:
    """
    Retrieval + filtering agent for TravelEnv.

    It is responsible for:
    - building slot filters (via FilterPolicy)
    - querying the KB
    - applying relaxation when candidates are empty
    - converting candidates into action strings + payloads (for EnvAgent to execute)
    - generating CITY-phase actions (city bundles)
    """

    def __init__(
        self,
        kb: Any,
        filter_policy: Any,
        relaxer: Any,
        *,
        top_k: int = 5,
        candidate_cap: int = 80,
        debug: bool = False,
        log_filter_usage: bool = False,
    ):
        self.kb = kb
        self.filter_policy = filter_policy
        self.relaxer = relaxer
        self.top_k = top_k
        self.candidate_cap = candidate_cap
        self.debug = debug
        self.log_filter_usage = log_filter_usage
        self._transport_cache: Dict[Tuple[str, str, bool], bool] = {}

    def reset(self) -> None:
        self._transport_cache = {}

    # ----------------------------
    # Shared helpers
    # ----------------------------
    def _infer_dest_kind(self, goal: Any) -> str:
        """
        Infer whether goal.destination is a state or a city using KB's background sets.

        Contract:
        - if destination is in KB state set -> "state"
        - else if destination exists -> "city"
        - else -> "unknown"
        """
        dest = getattr(goal, "destination", None)
        if not dest:
            return "unknown"
        try:
            if hasattr(self.kb, "is_state") and self.kb.is_state(dest):
                return "state"
        except Exception:
            pass
        return "city"

    @staticmethod
    def _flight_allowed(goal: Any) -> bool:
        """
        Determine if flight is allowed by goal's allow/forbid signals.

        flight_allowed=True means: not explicitly forbidden and not excluded by allow-list.
        """
        allow = getattr(goal, "transport_allowed_modes", None)
        forbid = set(m.lower() for m in (getattr(goal, "transport_forbidden_modes", None) or []))
        if allow is not None:
            try:
                allow_set = {str(m).lower() for m in allow}
            except Exception:
                allow_set = set()
            if allow_set and "flight" not in allow_set:
                return False
        if "flight" in forbid:
            return False

        cons = getattr(goal, "constraints", None) or {}
        tcons = cons.get("transport", {}) if isinstance(cons, dict) else {}
        if isinstance(tcons, dict):
            if tcons.get("allow"):
                try:
                    allow_set = {str(m).lower() for m in (tcons.get("allow") or [])}
                except Exception:
                    allow_set = set()
                if allow_set and "flight" not in allow_set:
                    return False
            if tcons.get("forbid"):
                forbid2 = {str(m).lower() for m in (tcons.get("forbid") or [])}
                if "flight" in forbid2:
                    return False
        return True

    def _flight_exists(self, src: str, dst: str) -> bool:
        try:
            return bool(self._has_transport_cached(src, dst, require_flight=True))
        except Exception:
            return False

    @staticmethod
    def _default_city_target_from_days(days: Optional[int]) -> int:
        if not days:
            return 1
        if days >= 7:
            return 3
        if days >= 5:
            return 2
        return 1

    @staticmethod
    def _compute_day_splits(total_days: int, seq: List[str]) -> List[Dict[str, Any]]:
        """
        Allocate contiguous day ranges to cities, using even split + tail absorbs remainder.
        Example: 7 days, 3 cities -> 2,2,3
        """
        if total_days <= 0 or not seq:
            return []
        n = len(seq)
        base = total_days // n
        rem = total_days % n
        splits: List[int] = [base] * n
        if rem:
            splits[-1] += rem
        out: List[Dict[str, Any]] = []
        day = 1
        for city, span in zip(seq, splits):
            start = day
            end = day + max(0, int(span)) - 1
            out.append({"city": city, "start_day": start, "end_day": end})
            day = end + 1
        return out

    def _has_transport_cached(self, src: str, dst: str, require_flight: bool = False) -> bool:
        key = (self.kb._normalize_city(src), self.kb._normalize_city(dst), require_flight)
        if key in self._transport_cache:
            return self._transport_cache[key]
        has = self.kb.has_any_transport(src, dst, require_flight=require_flight)
        self._transport_cache[key] = has
        return has

    @staticmethod
    def _allowed_transport_modes(goal: Any) -> List[str]:
        allowed = getattr(goal, "transport_allowed_modes", None) or ["flight", "taxi", "self-driving"]
        forbidden = set(m.lower() for m in (getattr(goal, "transport_forbidden_modes", None) or []))

        cons = getattr(goal, "constraints", None) or {}
        tcons = cons.get("transport", {}) if isinstance(cons, dict) else {}
        if isinstance(tcons, dict) and tcons.get("allow"):
            allowed = list(tcons["allow"])
        if isinstance(tcons, dict) and tcons.get("forbid"):
            forbidden |= set(m.lower() for m in (tcons.get("forbid") or []))

        return [m for m in allowed if m not in forbidden]

    def _planned_city_for_day(self, goal: Any, state: Any, day: int) -> Optional[str]:
        mapping = getattr(state, "day_to_city", None) or {}
        if isinstance(mapping, dict) and mapping and day in mapping:
            try:
                return str(mapping.get(day))
            except Exception:
                return mapping.get(day)
        seq = getattr(state, "city_sequence", None) or getattr(goal, "fixed_city_order", None) or getattr(goal, "must_visit_cities", None)
        seq = list(seq or [])
        if not seq:
            return getattr(goal, "destination", None)
        total_days = int(getattr(goal, "duration_days", 0) or 0)
        total_days = max(1, total_days)
        idx = min(len(seq) - 1, int((day - 1) * len(seq) / max(1, total_days)))
        return seq[idx]

    def _planned_accommodation_days_for_city(self, goal: Any, state: Any, city: str) -> int:
        """
        Number of days we expect to emit accommodation for this city in submission output.

        We follow the evaluation convention that accommodation may be absent on the last day,
        so we count days 1..(D-1) and map each day to its assigned city.
        """
        total_days = int(getattr(goal, "duration_days", 0) or 0)
        if total_days <= 1:
            return 0
        target = self.kb._normalize_city(city)
        count = 0
        for d in range(1, total_days):  # exclude last day
            c = self._planned_city_for_day(goal, state, d)
            if c and self.kb._normalize_city(c) == target:
                count += 1
        return count

    # ----------------------------
    # Action builders (slot -> actions/payloads)
    # ----------------------------
    @staticmethod
    def _actions_from_candidates(slot: Any, candidates: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Tuple]]:
        actions: List[str] = []
        payloads: Dict[str, Tuple] = {}
        if not candidates:
            return actions, payloads

        stype = getattr(slot, "type", None)
        if stype == "flight":
            seg_idx = getattr(slot, "seg", None)
            for f in candidates:
                seg_val = seg_idx if seg_idx is not None else -1
                price_val = float(f.get("price", 0) or 0.0)
                action = (
                    f"move:seg{seg_val}:flight:{f.get('id')} {f.get('origin')}->{f.get('destination')} "
                    f"${price_val:.0f} {f.get('depart', '?')}-{f.get('arrive', '?')}"
                )
                payloads[action] = ("segment_mode", seg_val, "flight", f)
                actions.append(action)
        elif stype == "hotel":
            slot_city = getattr(slot, "city", None)
            for stay in candidates:
                city = slot_city or stay.get("city")
                price_val = float(stay.get("price", 0) or 0.0)
                action = (
                    f"stay:{city}:{stay.get('id')} {stay.get('name')} "
                    f"{stay.get('room_type')} ${price_val:.0f}"
                )
                payloads[action] = ("stay_city", city, stay)
                actions.append(action)
        elif stype == "meal":
            day = getattr(slot, "day", None)
            meal_type = getattr(slot, "meal_type", None)
            for rest in candidates:
                cost_val = float(rest.get("cost", 0) or 0.0)
                action = (
                    f"eat:d{day}:{meal_type}:{rest.get('id')} "
                    f"{rest.get('name')} {rest.get('cuisines')} ${cost_val:.0f} rating {rest.get('rating', 0)}"
                )
                payloads[action] = ("meal", day, meal_type, rest)
                actions.append(action)
        elif stype == "attraction":
            day = getattr(slot, "day", None)
            slot_name = getattr(slot, "meal_type", None) or "spot"
            slot_city = getattr(slot, "city", None)
            for att in candidates:
                city = att.get("city") or slot_city
                action = f"visit:d{day}:{slot_name}:{att.get('id')} {att.get('name')} @ {city}"
                payloads[action] = ("attraction", day, slot_name, att)
                actions.append(action)
        elif stype == "finish":
            actions.append("finish")
        return actions, payloads

    @staticmethod
    def _assert_candidates_respect_caps(slot: Any, filt: Dict[str, Any], candidates: List[Dict[str, Any]]) -> None:
        stype = getattr(slot, "type", None)
        if stype in ("flight", "hotel"):
            mp = filt.get("max_price")
            if mp is not None:
                try:
                    mp_f = float(mp)
                    bad = [c for c in candidates if c.get("price") is not None and float(c["price"]) > mp_f + 1e-6]
                    if bad:
                        raise ValueError(
                            f"[CAP VIOLATION] {stype} max_price={mp_f} bad0_price={bad[0].get('price')} bad0={bad[0]}"
                        )
                except Exception:
                    pass
        if stype in ("meal", "restaurant", "attraction"):
            mc = filt.get("max_cost")
            if mc is not None:
                try:
                    mc_f = float(mc)
                    bad = [c for c in candidates if c.get("cost") is not None and float(c["cost"]) > mc_f + 1e-6]
                    if bad:
                        raise ValueError(
                            f"[CAP VIOLATION] {stype} max_cost={mc_f} bad0_cost={bad[0].get('cost')} bad0={bad[0]}"
                        )
                except Exception:
                    pass

    def _ground_fallback_actions(
        self, goal: Any, slot: Any, *, force_mode: Optional[str] = None
    ) -> Tuple[List[str], Dict[str, Tuple]]:
        """
        Non-flight transport fallback for segments when flight candidates are empty.

        NOTE: evaluation/commonsense_constraint.py rejects conflicting transport mixes:
          - Self-driving + Flight (anywhere in the plan)
          - Taxi + Self-driving (anywhere in the plan)
        So we emit at most ONE non-flight mode here (deterministic), and callers can
        force a specific mode to keep the whole itinerary consistent.
        """
        actions: List[str] = []
        payloads: Dict[str, Tuple] = {}
        origin = getattr(slot, "origin", None)
        destination = getattr(slot, "destination", None)
        if not origin or not destination:
            return actions, payloads
        distance = self.kb.distance_km(origin, destination)
        if distance is None:
            return actions, payloads
        seg_idx = getattr(slot, "seg", None)
        seg_val = seg_idx if seg_idx is not None else -1
        nonflight_modes = [m for m in self._allowed_transport_modes(goal) if m != "flight"]
        if not nonflight_modes:
            return actions, payloads

        if force_mode is not None:
            fm = str(force_mode).lower()
            if fm not in nonflight_modes:
                return actions, payloads
            mode = fm
        else:
            # Default preference: taxi first, then self-driving (stable across segments).
            mode = "taxi" if "taxi" in nonflight_modes else nonflight_modes[0]
        cost = distance
        action = f"move:seg{seg_val}:{mode}:{origin}->{destination} {distance:.0f}km cost {cost:.0f}"
        payload_detail = {
            "origin": origin,
            "destination": destination,
            "distance": distance,
            "cost": cost,
            "fallback_nonflight": True,
        }
        payloads[action] = ("segment_mode", seg_val, mode, payload_detail)
        actions.append(action)
        return actions, payloads

    # ----------------------------
    # CITY phase (bundle selection)
    # ----------------------------
    def _score_city_bundle(
        self,
        goal: Any,
        seq: List[str],
        origin: str,
        *,
        allow_repeat: bool = False,
        flight_only: bool = False,
        flight_when_available: bool = False,
    ) -> Optional[float]:
        if not seq:
            return None
        if not allow_repeat and len(set(seq)) != len(seq):
            return None

        score = 0.0
        require_flight = bool(getattr(goal, "require_flight", False))
        return_required = bool(getattr(goal, "return_required", True))
        preferences = getattr(goal, "preferences", []) or []

        def _edge_ok(src: str, dst: str) -> Tuple[bool, float]:
            if flight_only:
                if not self._flight_exists(src, dst):
                    return False, 0.0
                return True, 1.0
            if flight_when_available:
                if self._flight_exists(src, dst):
                    return True, 0.8
                if self.kb.has_any_transport(src, dst, require_flight=False):
                    return True, 0.2
                return False, 0.0
            if require_flight:
                if not self._flight_exists(src, dst):
                    return False, 0.0
                return True, 1.0
            if self.kb.has_any_transport(src, dst, require_flight=False):
                return True, 0.5
            return False, 0.0

        ok, bonus = _edge_ok(origin, seq[0])
        if not ok:
            return None
        score += bonus

        for i in range(len(seq) - 1):
            ok, _bonus = _edge_ok(seq[i], seq[i + 1])
            if not ok:
                return None
            score += 0.4 + (0.1 * _bonus)

        if return_required:
            ok, bonus = _edge_ok(seq[-1], origin)
            if not ok:
                return None
            score += 0.2 + (0.1 * bonus)

        for city in seq:
            has_stay = bool(self.kb.get_accommodations(city, top_k=1))
            has_rest = bool(self.kb.get_restaurants(city, preferences=preferences, top_k=1))
            has_att = bool(self.kb.get_attractions(city, top_k=1))
            score += 0.1 * has_stay + 0.1 * has_rest + 0.1 * has_att

        return score

    def build_city_actions(self, goal: Any, state: Any) -> Tuple[List[str], Dict[str, Tuple], Dict[str, Any]]:
        actions: List[str] = []
        payloads: Dict[str, Tuple] = {}
        event: Dict[str, Any] = {}

        dest_kind = self._infer_dest_kind(goal)
        total_days = int(getattr(goal, "duration_days", 0) or 0)
        flight_allowed = self._flight_allowed(goal)
        event.update(
            {
                "dest_kind": dest_kind,
                "flight_allowed": flight_allowed,
                "duration_days": total_days,
            }
        )

        dest = getattr(goal, "destination", None)
        origin = getattr(goal, "origin", None)
        if not origin:
            return actions, payloads, event

        existing_prefix = list(getattr(state, "city_sequence", []) or [])

        # Branch A: destination is not a state -> treat as single-city destination bundle.
        if dest_kind != "state":
            if dest:
                seq = [str(dest)]
                day_splits = self._compute_day_splits(total_days or len(seq), seq)
                action = f"choose_city_bundle:[{','.join(seq)}]"
                payloads[action] = ("choose_city_bundle", seq, day_splits)
                actions.append(action)
                event.update({"strategy": "direct_city_bundle", "actions": 1})
            else:
                event.update({"strategy": "direct_city_bundle", "actions": 0})
            return actions, payloads, event

        # Branch B: destination is a state -> choose n cities from state city pool.
        city_target = getattr(goal, "visiting_city_number", None) or self._default_city_target_from_days(total_days)
        if city_target <= 0:
            city_target = self._default_city_target_from_days(total_days)

        prefix = existing_prefix
        remaining_needed = max(0, int(city_target) - len(prefix))
        if remaining_needed == 0:
            event.update({"strategy": "state_bundle", "city_target": city_target, "remaining_needed": 0, "actions": 0})
            return actions, payloads, event

        state_cities = []
        try:
            if hasattr(self.kb, "cities_in_state") and dest:
                state_cities = list(self.kb.cities_in_state(dest) or [])
        except Exception:
            state_cities = []

        candidates = list(state_cities)
        injected_out_of_state: List[str] = []
        for must in (getattr(goal, "must_visit_cities", None) or []):
            if must not in candidates:
                injected_out_of_state.append(must)
                candidates.insert(0, must)
        for pri in (getattr(goal, "priority_cities", None) or []):
            if pri not in candidates:
                injected_out_of_state.append(pri)
                candidates.append(pri)

        pool = [c for c in candidates if c not in prefix]
        pool_before = len(pool)
        pool = pool[: max(self.top_k * 3, remaining_needed)]
        pool_after = len(pool)

        flight_only = flight_allowed
        combos_generated = 0
        combos: List[Tuple[List[str], float]] = []

        def _gen_combos(
            pool_list: List[str],
            *,
            allow_repeat: bool,
            flight_only_flag: bool,
            flight_when_available_flag: bool,
        ) -> List[Tuple[List[str], float]]:
            out: List[Tuple[List[str], float]] = []
            nonlocal combos_generated
            if remaining_needed <= 0:
                return out

            # NOTE: permutations() never repeats elements. For relaxed mode we want to allow
            # repeats so that we can construct sequences like [Denver, X, Denver] when only
            # a subset of state cities are flight-connected to origin for outbound/return.
            if allow_repeat:
                iterator = itertools.product(pool_list, repeat=remaining_needed)
            else:
                iterator = itertools.permutations(pool_list, remaining_needed)

            for combo in iterator:
                combos_generated += 1
                seq = prefix + list(combo)
                score = self._score_city_bundle(
                    goal,
                    seq,
                    origin,
                    allow_repeat=allow_repeat,
                    flight_only=flight_only_flag,
                    flight_when_available=flight_when_available_flag,
                )
                if score is None:
                    continue
                out.append((seq, score))
                if len(out) > 500:
                    break
                # Hard cap generation to avoid huge cartesian products.
                if combos_generated > 5000:
                    break
            return out

        combos = _gen_combos(pool, allow_repeat=False, flight_only_flag=flight_only, flight_when_available_flag=False)

        fallback_used = False
        fallback_mode = None
        relaxed_used = False
        # If strict flight-only yields nothing, fall back to "flight when available" (only use any_transport when no direct flight exists).
        if flight_only and not combos:
            fallback_used = True
            fallback_mode = "flight_when_available"
            combos = _gen_combos(pool, allow_repeat=False, flight_only_flag=False, flight_when_available_flag=True)

        if not combos:
            relaxed_used = True
            combos = _gen_combos(
                pool + prefix,
                allow_repeat=True,
                flight_only_flag=flight_only and not fallback_used,
                flight_when_available_flag=fallback_used,
            )
            if (flight_only and not fallback_used) and not combos:
                fallback_used = True
                fallback_mode = "flight_when_available"
                combos = _gen_combos(pool + prefix, allow_repeat=True, flight_only_flag=False, flight_when_available_flag=True)

        combos = sorted(combos, key=lambda x: x[1], reverse=True)[: self.top_k]
        for seq, _score in combos:
            day_splits = self._compute_day_splits(total_days or len(seq), seq)
            action = f"choose_city_bundle:[{','.join(seq)}]"
            payloads[action] = ("choose_city_bundle", seq, day_splits)
            actions.append(action)

        event.update(
            {
                "strategy": "state_bundle",
                "city_target": city_target,
                "remaining_needed": remaining_needed,
                "connectivity_mode": "flight_only" if (flight_only and not fallback_used) else (fallback_mode or "any_transport"),
                "flight_only": flight_only,
                "fallback_mode": fallback_mode,
                "pool_size_before_prefilter": pool_before,
                "pool_size_after_prefilter": pool_after,
                "combos_generated": combos_generated,
                "combos_kept_topk": len(actions),
                "relaxed_used": relaxed_used,
                "fallback_used": fallback_used,
                "injected_out_of_state": injected_out_of_state,
            }
        )
        return actions, payloads, event

    # ----------------------------
    # Main slot entrypoint
    # ----------------------------
    def compute(self, goal: Any, state: Any, slot: Any, *, user_query: str = "") -> SlotActionResult:
        stype = getattr(slot, "type", None)

        if stype == "city":
            actions, payloads, event = self.build_city_actions(goal, state)
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
                actions, payloads = self._ground_fallback_actions(goal, slot, force_mode="self-driving")
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
            if has_flight or has_taxi:
                # self-driving is disallowed once flight/taxi appears in the itinerary
                pass
            if not self._flight_allowed(goal):
                # If flight is forbidden by constraints, never emit flight candidates.
                # Keep the itinerary in a single non-flight mode to satisfy evaluator constraints.
                force = "taxi" if "taxi" in self._allowed_transport_modes(goal) else "self-driving"
                actions, payloads = self._ground_fallback_actions(goal, slot, force_mode=force)
                return SlotActionResult(
                    actions=actions,
                    payloads=payloads,
                    candidates=[],
                    relaxed=False,
                    filt={},
                    policy_event={"forced_transport_mode": force, "flight_forbidden": True},
                    plan=None,
                    uncapped_filter=None,
                )

        res = None
        if self.filter_policy is not None:
            res = self.filter_policy.build_slot_filter(goal, state, slot, user_query=user_query)
        filt = res.filt if res is not None else filters.default_filter(stype, goal, state, slot)
        policy_event = getattr(res, "event", {}) if res else {}
        plan = getattr(res, "plan", None) if res else None
        uncapped_filt = getattr(res, "uncapped_filter", None) if res else None

        # Enforce evaluation-compatible minimum nights for hotels:
        # the selected accommodation must satisfy its "minimum nights" requirement
        # given the planned number of (non-last) days in that city.
        if stype == "hotel":
            city = getattr(slot, "city", None) or filt.get("city")
            if city:
                stay_days = self._planned_accommodation_days_for_city(goal, state, str(city))
                if stay_days > 0:
                    filt = dict(filt or {})
                    filt["max_minimum_nights"] = int(stay_days)

        candidates: List[Dict[str, Any]] = []
        relaxed = False
        try:
            if self.log_filter_usage:
                phase_name = getattr(getattr(state, "phase", None), "name", getattr(state, "phase", None))
                print(f"[FILTER] slot={stype} phase={phase_name} event={policy_event} filter={filt}")
            candidates = self.kb.query(slot, filt, state, cap=self.candidate_cap)
            try:
                self._assert_candidates_respect_caps(slot, filt, candidates)
            except Exception as e:
                if self.debug:
                    raise
                print(f"[WARN] {e}")
        except Exception:
            if self.debug:
                raise
            candidates = []

        if not candidates:
            if uncapped_filt and uncapped_filt != filt:
                filt = uncapped_filt
                candidates = self.kb.query(slot, filt, state, cap=self.candidate_cap)
                policy_event = dict(policy_event or {})
                policy_event["soft_cap_relaxed"] = True
                if self.log_filter_usage:
                    print(f"[FILTER] slot={stype} soft-cap relaxed -> {len(candidates)} candidates")

            if not candidates and hasattr(self.filter_policy, "relax_budget_for_slot"):
                relaxed_budget_filt = self.filter_policy.relax_budget_for_slot(filt, slot)
                if relaxed_budget_filt:
                    filt = relaxed_budget_filt
                    candidates = self.kb.query(slot, filt, state, cap=self.candidate_cap)
                    policy_event = dict(policy_event or {})
                    policy_event["budget_relaxed"] = True
                    if self.log_filter_usage:
                        print(f"[FILTER] slot={stype} budget-relaxed -> {len(candidates)} candidates")

            if not candidates:
                candidates = self.relaxer.relax_and_query(self.kb, slot, filt, state, cap=self.candidate_cap)
                relaxed = True
                if self.log_filter_usage:
                    print(f"[FILTER] slot={stype} relaxed -> {len(candidates)} candidates")

        # Re-apply hard constraints even if the relaxer fell back to "unfiltered" candidates.
        if stype == "hotel":
            cap_n = filt.get("max_minimum_nights")
            if cap_n is not None:
                try:
                    cap_i = int(cap_n)
                    candidates = [
                        s
                        for s in (candidates or [])
                        if s.get("minimum_nights") is None or int(s.get("minimum_nights") or 0) <= cap_i
                    ]
                except Exception:
                    pass

        actions, payloads = self._actions_from_candidates(slot, candidates)
        if not actions and stype == "flight":
            # If flight already exists in other segments, prefer taxi fallback to avoid (Flight + Self-driving).
            modes = []
            for seg in (getattr(state, "segment_modes", None) or {}).values():
                if isinstance(seg, dict) and seg.get("mode"):
                    modes.append(str(seg["mode"]).lower())
            has_flight = "flight" in modes
            has_taxi = "taxi" in modes
            has_self = "self-driving" in modes
            if has_flight or has_taxi:
                actions, payloads = self._ground_fallback_actions(goal, slot, force_mode="taxi")
            elif has_self:
                actions, payloads = self._ground_fallback_actions(goal, slot, force_mode="self-driving")
            else:
                actions, payloads = self._ground_fallback_actions(goal, slot)
            if has_flight or has_taxi:
                actions = [a for a in actions if ":taxi:" in a]
                payloads = {a: payloads[a] for a in actions} if actions else {}
            elif has_self:
                actions = [a for a in actions if ":self-driving:" in a]
                payloads = {a: payloads[a] for a in actions} if actions else {}

        return SlotActionResult(
            actions=actions,
            payloads=payloads,
            candidates=candidates,
            relaxed=relaxed,
            filt=filt,
            policy_event=policy_event or {},
            plan=plan,
            uncapped_filter=uncapped_filt,
        )
