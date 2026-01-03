from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from mcts.travel.retrieval.city_bundle_v2 import CityBundleBuilderV2
from mcts.travel.retrieval.city_profile import CityProfileBuilder
from mcts.travel.retrieval import (
    ActionFactory,
    BudgetAllocator,
    ConstraintNormalizer,
    RelaxationController,
)
from mcts.travel.semantic.bundle_reranker import SemanticBundleReranker


class RetrievalAgent:
    """
    Retrieval + filtering agent for TravelEnv.

    It is responsible for:
    - coarse filtering to build a per-query candidate pool (hard constraints)
    - fine ranking per slot using local context and dynamic budget caps
    - converting candidates into action strings + payloads (for EnvAgent to execute)
    - generating CITY-phase actions (city bundles)
    """

    def __init__(
        self,
        kb: Any,
        *,
        top_k: int = 5,
        candidate_cap: int = 200,
        debug: bool = False,
        log_filter_usage: bool = False,
    ):
        self.kb = kb
        self.top_k = top_k
        self.candidate_cap = candidate_cap
        self.debug = debug
        self.log_filter_usage = log_filter_usage
        self._transport_cache: Dict[Tuple[str, str, bool], bool] = {}
        self._transport_cost_cache: Dict[Tuple[str, str, int, Tuple[str, ...]], Optional[float]] = {}
        self._parsed_sig: Optional[str] = None
        self._city_pool: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._city_cuisine_coverage: Dict[str, set] = {}
        self._city_min_stay_price: Dict[str, Optional[float]] = {}
        self._city_min_meal_cost: Dict[str, Optional[float]] = {}
        self._global_min_stay_price: Optional[float] = None
        self._global_min_meal_cost: Optional[float] = None
        self._budget_revision: int = 0
        self._budget_scales = {
            "segment": 1.0,
            "flight": 1.0,
            "stay": 1.0,
            "daily": 1.0,
            "meal": 1.0,
            "attraction": 1.0,
        }
        self._reserve_scales = {"segment": 1.0, "stay": 1.0, "daily": 1.0}
        self._cap_multipliers = {"segment": 1.0, "stay": 1.0, "daily": 1.0}
        self._constraint_normalizer = ConstraintNormalizer()
        self._budget_allocator = BudgetAllocator(
            self._parsed_get,
            self._remaining_counts,
            self._budget_scales,
            self._cap_multipliers,
        )
        self._relaxer = RelaxationController(self._reserve_scales, self._cap_multipliers)
        self._action_factory = ActionFactory()
        self.failure_memory = None
        self._city_profile_builder = CityProfileBuilder(self.kb)
        self._bundle_builder = CityBundleBuilderV2(self.kb, profile_builder=self._city_profile_builder)
        self._bundle_reranker = SemanticBundleReranker(max_keep=self.top_k)

    def reset(self) -> None:
        self._transport_cache = {}
        self._transport_cost_cache = {}
        self._parsed_sig = None
        self._city_pool = {}
        self._city_cuisine_coverage = {}
        self._city_min_stay_price = {}
        self._city_min_meal_cost = {}
        self._global_min_stay_price = None
        self._global_min_meal_cost = None
        self._budget_revision = 0
        self._budget_scales.clear()
        self._budget_scales.update(
            {
                "segment": 1.0,
                "flight": 1.0,
                "stay": 1.0,
                "daily": 1.0,
                "meal": 1.0,
                "attraction": 1.0,
            }
        )
        self._reserve_scales.clear()
        self._reserve_scales.update({"segment": 1.0, "stay": 1.0, "daily": 1.0})
        self._cap_multipliers.clear()
        self._cap_multipliers.update({"segment": 1.0, "stay": 1.0, "daily": 1.0})

    def set_failure_memory(self, failure_memory: Any) -> None:
        self.failure_memory = failure_memory

    def get_budget_scales(self) -> Dict[str, float]:
        return dict(self._budget_scales)

    def set_budget_scales(self, scales: Dict[str, float]) -> None:
        if not isinstance(scales, dict):
            return
        self._budget_scales.clear()
        segment_val = scales.get("segment", scales.get("flight", 1.0))
        daily_val = scales.get("daily", scales.get("meal", 1.0))
        base = {
            "segment": segment_val,
            "flight": segment_val,
            "stay": scales.get("stay", 1.0),
            "daily": daily_val,
            "meal": daily_val,
            "attraction": scales.get("attraction", 1.0),
        }
        for key, val in base.items():
            try:
                self._budget_scales[key] = float(val)
            except Exception:
                self._budget_scales[key] = 1.0

    def get_budget_revision(self) -> int:
        return int(self._budget_revision)

    def rebalance_for_phase(self, phase: str, *, attempts: int = 2) -> bool:
        phase = str(phase or "").lower().strip()
        if not phase:
            return False
        changed = False
        for _ in range(max(1, int(attempts))):
            try:
                if self._relaxer.bump(phase):
                    changed = True
            except Exception:
                continue
        if changed:
            self._budget_revision += 1
        return changed

    # ----------------------------
    # Shared helpers
    # ----------------------------
    @staticmethod
    def _parsed_get(parsed: Any, *keys: str, default: Any = None) -> Any:
        if not isinstance(parsed, dict):
            return default
        for key in keys:
            if key in parsed and parsed[key] is not None:
                return parsed[key]
        return default

    @staticmethod
    def _is_nan_value(value: Any) -> bool:
        if value is None:
            return False
        try:
            return math.isnan(float(value))
        except Exception:
            return False

    @staticmethod
    def _duration_days(parsed: Any) -> Optional[int]:
        val = RetrievalAgent._parsed_get(parsed, "duration_days", "days", default=None)
        if val is None:
            return None
        try:
            return int(val)
        except Exception:
            return None

    def _prefer_city_dest(self, parsed: Any) -> bool:
        dest = self._parsed_get(parsed, "destination", "dest", default=None)
        if not dest:
            return False
        days = self._duration_days(parsed)
        if days != 3:
            return False
        dest_norm = self.kb._normalize_city(dest)
        return dest_norm in getattr(self.kb, "city_set_norm", {})

    def _infer_dest_kind(self, parsed: Any) -> str:
        """
        Infer whether parsed.destination is a state or a city using KB's background sets.

        Contract:
        - if destination is in KB state set -> "state"
        - else if destination exists -> "city"
        - else -> "unknown"
        """
        if self._prefer_city_dest(parsed):
            return "city"
        dest = self._parsed_get(parsed, "destination", "dest", default=None)
        if not dest:
            return "unknown"
        try:
            if hasattr(self.kb, "is_state") and self.kb.is_state(dest):
                return "state"
        except Exception:
            pass
        return "city"

    @staticmethod
    def _select_valid_batch(
        candidates: List[Dict[str, Any]],
        k: int,
        *,
        predicate: Optional[Any] = None,
        batch_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        if not candidates or k <= 0:
            return []
        start_idx = max(0, int(batch_idx)) * k
        for start in range(start_idx, len(candidates), k):
            batch = candidates[start : start + k]
            if predicate is None:
                return batch
            kept: List[Dict[str, Any]] = []
            for cand in batch:
                if predicate(cand):
                    kept.append(cand)
            if kept:
                return kept
        return []

    @staticmethod
    def _select_valid_fill(
        candidates: List[Dict[str, Any]],
        k: int,
        *,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        start_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        if not candidates or k <= 0:
            return []
        out: List[Dict[str, Any]] = []
        i = max(0, int(start_idx))
        while i < len(candidates) and len(out) < k:
            cand = candidates[i]
            i += 1
            if predicate is not None and not predicate(cand):
                continue
            out.append(cand)
        return out

    @staticmethod
    def _used_restaurant_ids(state: Any) -> set:
        used = set()
        for day_map in (getattr(state, "meals", None) or {}).values():
            for meal in (day_map or {}).values():
                if isinstance(meal, dict):
                    rid = meal.get("id")
                    if rid is not None:
                        used.add(rid)
        return used

    @staticmethod
    def _used_attraction_ids(state: Any) -> set:
        used = set()
        for day_map in (getattr(state, "attractions", None) or {}).values():
            for att in (day_map or {}).values():
                if isinstance(att, dict):
                    aid = att.get("id")
                    if aid is not None:
                        used.add(aid)
        return used

    @staticmethod
    def _flight_allowed(parsed: Any) -> bool:
        """
        Determine if flight is allowed by parsed allow/forbid signals.

        flight_allowed=True means: not explicitly forbidden and not excluded by allow-list.
        """
        allow = RetrievalAgent._parsed_get(parsed, "transport_allowed_modes", "transport_allow", default=None)
        forbid = set(
            m.lower()
            for m in (RetrievalAgent._parsed_get(parsed, "transport_forbidden_modes", "transport_forbid", "transport_forbidden", default=[]) or [])
        )
        if allow is not None:
            try:
                allow_set = {str(m).lower() for m in allow}
            except Exception:
                allow_set = set()
            if allow_set and "flight" not in allow_set:
                return False
        if "flight" in forbid:
            return False

        cons = RetrievalAgent._parsed_get(parsed, "constraints", default=None) or {}
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
        transportation = RetrievalAgent._parsed_get(parsed, "transportation", default=None)
        if transportation:
            items = transportation if isinstance(transportation, list) else [transportation]
            for raw in items:
                if "no flight" in str(raw).lower():
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
    def _room_type_matches(room_type: Optional[str], allowed: List[str]) -> bool:
        if not allowed:
            return True
        rt = str(room_type or "").strip().lower()
        if not rt:
            return False
        normalized: List[str] = []
        for raw in allowed:
            t = str(raw).strip().lower()
            if not t:
                continue
            if t in ("entire room", "entire home", "entire home/apt"):
                normalized.append("entire home/apt")
            elif t == "not shared room":
                normalized.extend(["private room", "entire home/apt"])
            else:
                normalized.append(t)
        return rt in set(normalized)

    @staticmethod
    def _violates_house_rule(rule: str, stay: Dict[str, Any]) -> bool:
        if not rule:
            return False
        rule = rule.strip().lower()
        token_map = {
            "smoking": "no_smoking",
            "parties": "no_parties",
            "children under 10": "no_children_under_10",
            "visitors": "no_visitors",
            "pets": "no_pets",
        }
        token = token_map.get(rule)
        tokens = stay.get("house_rules_tokens")
        if token:
            if isinstance(tokens, set) and token in tokens:
                return True
            if isinstance(tokens, (list, tuple)) and token in set(tokens):
                return True
        raw = stay.get("house_rules")
        if isinstance(raw, str):
            raw_lc = raw.lower()
            if rule == "smoking" and "no smoking" in raw_lc:
                return True
            if rule == "parties" and "no parties" in raw_lc:
                return True
            if rule == "children under 10" and "no children under 10" in raw_lc:
                return True
            if rule == "visitors" and "no visitors" in raw_lc:
                return True
            if rule == "pets" and "no pets" in raw_lc:
                return True
        return False

    def _constraints_from_parsed(self, parsed: Any) -> Dict[str, Any]:
        return self._constraint_normalizer.to_constraints(parsed).as_dict()

    def _parsed_signature(self, parsed: Any) -> str:
        spec = self._constraint_normalizer.to_spec(parsed)
        return spec.signature()

    def _candidate_cities(self, parsed: Any) -> List[str]:
        candidate_cities = list(self._parsed_get(parsed, "candidate_cities", default=[]) or [])
        if candidate_cities:
            return candidate_cities
        dest = self._parsed_get(parsed, "destination", "dest", default=None)
        if dest and self._prefer_city_dest(parsed):
            return [str(dest)]
        if dest and hasattr(self.kb, "cities_in_state") and self.kb.is_state(dest):
            return list(self.kb.cities_in_state(dest) or [])
        if dest:
            return [str(dest)]
        return []

    def _coarse_stays(self, city: str, cons: Dict[str, Any]) -> List[Dict[str, Any]]:
        city_norm = self.kb._normalize_city(city)
        stays = list(getattr(self.kb, "_accommodation_buckets", {}).get(city_norm, []))
        room_types = cons.get("room_types") or []
        house_rules = cons.get("house_rules") or []
        min_occ = cons.get("min_occupancy")
        out: List[Dict[str, Any]] = []
        for stay in stays:
            price = stay.get("price")
            if price is None or self._is_nan_value(price):
                continue
            if min_occ is not None:
                occ = stay.get("occupancy")
                try:
                    if occ is not None and float(occ) < float(min_occ):
                        continue
                except Exception:
                    pass
            if room_types and not self._room_type_matches(stay.get("room_type"), room_types):
                continue
            if house_rules and any(self._violates_house_rule(rule, stay) for rule in house_rules):
                continue
            out.append(stay)
            if len(out) >= self.candidate_cap:
                break
        return out

    def _coarse_restaurants(self, city: str, cons: Dict[str, Any]) -> List[Dict[str, Any]]:
        city_norm = self.kb._normalize_city(city)
        restaurants = list(getattr(self.kb, "_restaurant_buckets", {}).get(city_norm, []))
        cuisines = [c.lower() for c in (cons.get("cuisines") or []) if c]
        if not cuisines:
            out = [r for r in restaurants if r.get("cost") is not None and not self._is_nan_value(r.get("cost"))]
            return out[: self.candidate_cap]

        hits: List[Dict[str, Any]] = []
        others: List[Dict[str, Any]] = []
        for rest in restaurants:
            cost = rest.get("cost")
            if cost is None or self._is_nan_value(cost):
                continue
            text = str(rest.get("cuisines_lc") or rest.get("cuisines") or "").lower()
            if any(c in text for c in cuisines):
                hits.append(rest)
            else:
                others.append(rest)

        out = hits[: self.candidate_cap]
        if len(out) < self.candidate_cap:
            out.extend(others[: self.candidate_cap - len(out)])
        return out

    def _coarse_attractions(self, city: str) -> List[Dict[str, Any]]:
        city_norm = self.kb._normalize_city(city)
        attractions = list(getattr(self.kb, "_attraction_buckets", {}).get(city_norm, []))
        return attractions[: self.candidate_cap]

    def _ensure_city_pools(self, parsed: Any) -> None:
        sig = self._parsed_signature(parsed)
        if sig == self._parsed_sig:
            return
        self._parsed_sig = sig
        self._city_pool = {}
        self._city_cuisine_coverage = {}
        self._city_min_stay_price = {}
        self._city_min_meal_cost = {}
        self._global_min_stay_price = None
        self._global_min_meal_cost = None
        self._budget_scales.clear()
        self._budget_scales.update(
            {
                "segment": 1.0,
                "flight": 1.0,
                "stay": 1.0,
                "daily": 1.0,
                "meal": 1.0,
                "attraction": 1.0,
            }
        )
        self._reserve_scales.clear()
        self._reserve_scales.update({"segment": 1.0, "stay": 1.0, "daily": 1.0})
        self._cap_multipliers.clear()
        self._cap_multipliers.update({"segment": 1.0, "stay": 1.0, "daily": 1.0})

        cons = self._constraints_from_parsed(parsed)
        cuisines = [c.lower() for c in (cons.get("cuisines") or []) if c]
        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
        base_cities = list(self._candidate_cities(parsed))
        extra_cities = list(self._parsed_get(parsed, "must_visit_cities", default=[]) or [])
        extra_cities += list(self._parsed_get(parsed, "priority_cities", default=[]) or [])
        extra_cities += list(self._parsed_get(parsed, "fixed_city_order", default=[]) or [])
        all_cities: List[str] = []
        seen = set()
        for city in base_cities + extra_cities:
            if city not in seen:
                seen.add(city)
                all_cities.append(city)
        for city in all_cities:
            city_norm = self.kb._normalize_city(city)
            stays = self._coarse_stays(city, cons)
            rests = self._coarse_restaurants(city, cons)
            atts = self._coarse_attractions(city)
            self._city_pool[city_norm] = {
                "hotel": stays,
                "meal": rests,
                "attraction": atts,
            }
            if cuisines:
                covered = set()
                for rest in rests:
                    text = str(rest.get("cuisines_lc") or rest.get("cuisines") or "").lower()
                    for c in cuisines:
                        if c in text:
                            covered.add(c)
                self._city_cuisine_coverage[city_norm] = covered
            else:
                self._city_cuisine_coverage[city_norm] = set()

            min_stay = None
            for stay in stays:
                price = stay.get("price")
                if price is None:
                    continue
                occ = stay.get("occupancy")
                try:
                    occ_i = int(occ) if occ is not None else None
                except Exception:
                    occ_i = None
                occ_i = max(1, occ_i) if occ_i else None
                rooms = math.ceil(float(people) / float(occ_i or people))
                try:
                    total_per_night = float(price) * float(rooms)
                except Exception:
                    continue
                if min_stay is None or total_per_night < min_stay:
                    min_stay = total_per_night
            self._city_min_stay_price[city_norm] = min_stay

            min_meal = None
            for rest in rests:
                cost = rest.get("cost")
                if cost is None:
                    continue
                try:
                    cost_f = float(cost)
                except Exception:
                    continue
                if min_meal is None or cost_f < min_meal:
                    min_meal = cost_f
            self._city_min_meal_cost[city_norm] = min_meal

        stay_vals = [v for v in self._city_min_stay_price.values() if v is not None]
        meal_vals = [v for v in self._city_min_meal_cost.values() if v is not None]
        self._global_min_stay_price = min(stay_vals) if stay_vals else None
        self._global_min_meal_cost = min(meal_vals) if meal_vals else None

    def _rank_candidate_cities(self, parsed: Any, cities: List[str], origin: str) -> List[str]:
        require_accommodation = bool(self._parsed_get(parsed, "require_accommodation", default=True))
        must = set(self._parsed_get(parsed, "must_visit_cities", default=[]) or [])
        priority = set(self._parsed_get(parsed, "priority_cities", default=[]) or [])

        scored: List[Tuple[str, float]] = []
        for city in cities:
            city_norm = self.kb._normalize_city(city)
            pool = self._city_pool.get(city_norm, {})
            has_stay = bool(pool.get("hotel"))
            if require_accommodation and not has_stay and city not in must:
                continue

            score = 0.0
            if has_stay:
                score += 1.0

            if self._flight_allowed(parsed):
                if self._flight_exists(origin, city):
                    score += 0.8
                elif self.kb.has_any_transport(origin, city, require_flight=False):
                    score += 0.2
                else:
                    score -= 0.5
            else:
                if self.kb.has_any_transport(origin, city, require_flight=False):
                    score += 0.2

            if city in priority:
                score += 0.5
            if city in must:
                score += 1.0
            scored.append((city, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

    def _min_transport_cost(self, parsed: Any, src: str, dst: str, people: int) -> Optional[float]:
        if not src or not dst:
            return None
        people = max(1, int(people or 1))
        allowed = tuple(sorted(self._allowed_transport_modes(parsed)))
        key = (self.kb._normalize_city(src), self.kb._normalize_city(dst), people, allowed)
        if key in self._transport_cost_cache:
            return self._transport_cost_cache[key]

        costs: List[float] = []
        if self._flight_allowed(parsed):
            flights = list(getattr(self.kb, "_flight_buckets", {}).get((key[0], key[1]), []))
            if flights:
                try:
                    min_price = min(
                        float(f.get("price") or 0.0)
                        for f in flights
                        if f.get("price") is not None and not self._is_nan_value(f.get("price"))
                    )
                except ValueError:
                    min_price = None
                if min_price is not None:
                    costs.append(min_price * float(people))

        distance = self.kb.distance_km(src, dst)
        if distance is not None:
            if "taxi" in allowed:
                costs.append(float(distance) * math.ceil(float(people) / 4.0))
            if "self-driving" in allowed:
                costs.append(float(distance) * math.ceil(float(people) / 5.0))

        out = min(costs) if costs else None
        self._transport_cost_cache[key] = out
        return out

    @staticmethod
    def _day_to_city_from_splits(day_splits: List[Dict[str, Any]], total_days: int) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
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
        # Fill gaps conservatively by carrying last city forward.
        if total_days and mapping:
            last = None
            for d in range(1, total_days + 1):
                if d in mapping:
                    last = mapping[d]
                elif last:
                    mapping[d] = last
        return mapping

    def _min_bundle_cost(
        self,
        parsed: Any,
        seq: List[str],
        day_splits: List[Dict[str, Any]],
        origin: str,
    ) -> Optional[float]:
        budget = self._parsed_get(parsed, "budget", default=None)
        try:
            budget_f = float(budget)
        except Exception:
            budget_f = None

        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
        total_days = int(self._parsed_get(parsed, "duration_days", "days", default=0) or 0)
        total_days = max(1, total_days)

        # Transport lower bound.
        transport_cost = 0.0
        return_required = bool(self._parsed_get(parsed, "return_required", default=True))
        legs: List[Tuple[str, str]] = []
        if seq:
            if origin:
                legs.append((origin, seq[0]))
            for i in range(1, len(seq)):
                legs.append((seq[i - 1], seq[i]))
            if return_required and origin:
                legs.append((seq[-1], origin))
        for src, dst in legs:
            cost = self._min_transport_cost(parsed, src, dst, people)
            if cost is None:
                return None
            transport_cost += cost

        # Stay lower bounds.
        day_to_city = self._day_to_city_from_splits(day_splits, total_days)
        stay_cost = 0.0
        require_accommodation = bool(self._parsed_get(parsed, "require_accommodation", default=True))

        for day in range(1, total_days + 1):
            city = day_to_city.get(day) or (seq[-1] if seq else None)
            if not city:
                continue
            city_norm = self.kb._normalize_city(city)

            # Accommodation for all days except the last.
            if require_accommodation and day != total_days:
                min_stay = self._city_min_stay_price.get(city_norm)
                if min_stay is None:
                    return None
                stay_cost += float(min_stay)

        total = transport_cost + stay_cost
        if budget_f is not None and total > budget_f:
            return None
        return total

    def _clone_state_for_bundle(
        self,
        state: Any,
        seq: List[str],
        day_splits: List[Dict[str, Any]],
        total_days: int,
    ) -> Any:
        if hasattr(state, "clone"):
            temp_state = state.clone()
        else:
            temp_state = copy.deepcopy(state)
        temp_state.city_sequence = list(seq or [])
        total_days = max(1, int(total_days or 0))
        day_to_city = self._day_to_city_from_splits(day_splits, total_days)
        temp_state.day_to_city = day_to_city
        return temp_state

    def _bundle_meets_caps(
        self,
        parsed: Any,
        state: Any,
        seq: List[str],
        day_splits: List[Dict[str, Any]],
        origin: str,
    ) -> bool:
        total_days = int(self._parsed_get(parsed, "duration_days", "days", default=0) or 0)
        total_days = max(1, total_days)
        temp_state = self._clone_state_for_bundle(state, seq, day_splits, total_days)
        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))

        segment_cap, _ = self._budget_allocator.caps(parsed, temp_state, "segment")
        stay_cap, _ = self._budget_allocator.caps(parsed, temp_state, "stay")
        # Segment cap check (use min transport cost vs per-person cap).
        if segment_cap is not None:
            return_required = bool(self._parsed_get(parsed, "return_required", default=True))
            legs: List[Tuple[str, str]] = []
            if seq:
                if origin:
                    legs.append((origin, seq[0]))
                for i in range(1, len(seq)):
                    legs.append((seq[i - 1], seq[i]))
                if return_required and origin:
                    legs.append((seq[-1], origin))
            for src, dst in legs:
                cost_total = self._min_transport_cost(parsed, src, dst, people)
                if cost_total is None:
                    return False
                if cost_total > float(segment_cap) * float(people):
                    return False

        # Stay cap check (per-night total for the party).
        if stay_cap is not None and bool(self._parsed_get(parsed, "require_accommodation", default=True)):
            for city in seq:
                nights = self._planned_accommodation_days_for_city(parsed, temp_state, city)
                if nights <= 0:
                    continue
                city_norm = self.kb._normalize_city(city)
                min_stay = self._city_min_stay_price.get(city_norm)
                if min_stay is None:
                    return False
                if float(min_stay) > float(stay_cap):
                    return False

        return True

    def _min_remaining_stay_meal_cost(self, parsed: Any, state: Any) -> Optional[float]:
        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
        total_days = int(self._parsed_get(parsed, "duration_days", "days", default=0) or 0)
        total_days = max(1, total_days)
        require_accommodation = bool(self._parsed_get(parsed, "require_accommodation", default=True))

        total = 0.0
        if require_accommodation:
            for city in (getattr(state, "city_sequence", None) or []):
                if (getattr(state, "city_stays", None) or {}).get(city) is not None:
                    continue
                nights = self._planned_accommodation_days_for_city(parsed, state, city)
                if nights <= 0:
                    continue
                city_norm = self.kb._normalize_city(city)
                min_stay = self._city_min_stay_price.get(city_norm)
                if min_stay is None:
                    return None
                total += float(min_stay) * float(nights)

        meals = getattr(state, "meals", None) or {}
        for day in range(1, total_days + 1):
            day_map = meals.get(day) or {}
            for meal in day_map.values():
                if meal is not None:
                    continue
                city = self._planned_city_for_day(parsed, state, day)
                if not city:
                    return None
                city_norm = self.kb._normalize_city(city)
                min_meal = self._city_min_meal_cost.get(city_norm)
                if min_meal is None:
                    return None
                total += float(min_meal) * float(people)

        return total

    def _apply_constraints_to_candidates(self, stype: str, candidates: List[Dict[str, Any]], parsed: Any) -> List[Dict[str, Any]]:
        cons = self._constraints_from_parsed(parsed)
        if not candidates:
            return candidates
        if stype == "hotel":
            room_type = list(cons.get("room_types") or [])
            house_rules = list(cons.get("house_rules") or [])
            filtered: List[Dict[str, Any]] = []
            for stay in candidates:
                if room_type and not self._room_type_matches(stay.get("room_type"), room_type):
                    continue
                if house_rules and any(self._violates_house_rule(rule, stay) for rule in house_rules):
                    continue
                filtered.append(stay)
            return filtered
        if stype in ("meal", "restaurant"):
            return candidates
        return candidates

    @staticmethod
    def _allowed_transport_modes(parsed: Any) -> List[str]:
        allowed = RetrievalAgent._parsed_get(parsed, "transport_allowed_modes", "transport_allow", default=None) or ["flight", "taxi", "self-driving"]
        forbidden = set(
            m.lower()
            for m in (RetrievalAgent._parsed_get(parsed, "transport_forbidden_modes", "transport_forbid", "transport_forbidden", default=[]) or [])
        )
        cons = RetrievalAgent._parsed_get(parsed, "constraints", default=None) or {}
        tcons = cons.get("transport", {}) if isinstance(cons, dict) else {}
        if isinstance(tcons, dict) and tcons.get("allow"):
            allowed = list(tcons["allow"])
        if isinstance(tcons, dict) and tcons.get("forbid"):
            forbidden |= set(m.lower() for m in (tcons.get("forbid") or []))

        transportation = RetrievalAgent._parsed_get(parsed, "transportation", default=None)
        if transportation:
            items = transportation if isinstance(transportation, list) else [transportation]
            for raw in items:
                low = str(raw).strip().lower()
                if "no flight" in low:
                    forbidden.add("flight")
                if "no self-driving" in low:
                    forbidden.add("self-driving")
                if "no taxi" in low:
                    forbidden.add("taxi")

        return [m for m in allowed if m and m not in forbidden]

    def _planned_city_for_day(self, parsed: Any, state: Any, day: int) -> Optional[str]:
        mapping = getattr(state, "day_to_city", None) or {}
        if isinstance(mapping, dict) and mapping and day in mapping:
            try:
                return str(mapping.get(day))
            except Exception:
                return mapping.get(day)
        seq = (
            getattr(state, "city_sequence", None)
            or RetrievalAgent._parsed_get(parsed, "fixed_city_order", default=None)
            or RetrievalAgent._parsed_get(parsed, "must_visit_cities", default=None)
        )
        seq = list(seq or [])
        if not seq:
            return RetrievalAgent._parsed_get(parsed, "destination", "dest", default=None)
        total_days = int(RetrievalAgent._parsed_get(parsed, "duration_days", "days", default=0) or 0)
        total_days = max(1, total_days)
        idx = min(len(seq) - 1, int((day - 1) * len(seq) / max(1, total_days)))
        return seq[idx]

    def _meal_allowed_cities(
        self,
        parsed: Any,
        state: Any,
        day: Optional[int],
        meal_type: Optional[str],
        city: Optional[str],
    ) -> Tuple[List[str], Dict[str, Any]]:
        allowed: List[str] = []
        event: Dict[str, Any] = {"day": day, "meal_type": meal_type}

        mapped_city = None
        mapping = getattr(state, "day_to_city", None)
        if isinstance(mapping, dict) and day is not None and day in mapping:
            try:
                mapped_city = str(mapping.get(day))
            except Exception:
                mapped_city = mapping.get(day)
        if mapped_city:
            allowed.append(mapped_city)
            event["source"] = "day_to_city"

        planned_city = None
        if not allowed and day is not None:
            planned_city = self._planned_city_for_day(parsed, state, int(day))
            if planned_city:
                allowed.append(planned_city)
                event["source"] = "planned_day_city"

        if city:
            allowed.append(str(city))
            event.setdefault("source", "slot_city")

        if not allowed:
            seq = list(getattr(state, "city_sequence", None) or [])
            if seq:
                allowed.extend(seq)
                event["source"] = "city_sequence"
            else:
                dest = self._parsed_get(parsed, "destination", "dest", default=None)
                if dest:
                    allowed.append(str(dest))
                    event["source"] = "destination"

        deduped: List[str] = []
        seen: set = set()
        for item in allowed:
            if not item:
                continue
            key = self.kb._normalize_city(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        if "source" not in event:
            event["source"] = "none"
        return deduped, event

    def _planned_accommodation_days_for_city(self, parsed: Any, state: Any, city: str) -> int:
        """
        Number of days we expect to emit accommodation for this city in submission output.

        We follow the evaluation convention that accommodation may be absent on the last day,
        so we count days 1..(D-1) and map each day to its assigned city.
        """
        total_days = int(self._parsed_get(parsed, "duration_days", "days", default=0) or 0)
        if total_days <= 1:
            return 0
        target = self.kb._normalize_city(city)
        count = 0
        for d in range(1, total_days):  # exclude last day
            c = self._planned_city_for_day(parsed, state, d)
            if c and self.kb._normalize_city(c) == target:
                count += 1
        return count

    def _segments_for_state(self, parsed: Any, state: Any) -> List[Tuple[int, str, str]]:
        seq = list(getattr(state, "city_sequence", []) or [])
        origin = self._parsed_get(parsed, "origin", "org", default=None)
        destination = self._parsed_get(parsed, "destination", "dest", default=None)
        return_required = bool(self._parsed_get(parsed, "return_required", default=True))
        segments: List[Tuple[int, str, str]] = []
        if seq:
            if origin:
                segments.append((0, origin, seq[0]))
            for i in range(1, len(seq)):
                segments.append((i, seq[i - 1], seq[i]))
            if return_required and origin:
                segments.append((len(seq), seq[-1], origin))
        elif destination and origin:
            segments.append((0, origin, destination))
            if return_required:
                segments.append((1, destination, origin))
        return segments

    def _remaining_counts(self, parsed: Any, state: Any) -> Dict[str, int]:
        segments = self._segments_for_state(parsed, state)
        remaining_segments = sum(1 for idx, _, _ in segments if idx not in (getattr(state, "segment_modes", None) or {}))

        remaining_stays = 0
        remaining_nights = 0
        require_accommodation = bool(self._parsed_get(parsed, "require_accommodation", default=True))
        if require_accommodation:
            for city in (getattr(state, "city_sequence", None) or []):
                if (getattr(state, "city_stays", None) or {}).get(city) is None:
                    remaining_stays += 1
                    remaining_nights += self._planned_accommodation_days_for_city(parsed, state, city)

        remaining_meals = 0
        meals = getattr(state, "meals", None) or {}
        for day_map in meals.values():
            for meal in (day_map or {}).values():
                if meal is None:
                    remaining_meals += 1

        remaining_attractions = 0
        attractions = getattr(state, "attractions", None) or {}
        for day_map in attractions.values():
            for att in (day_map or {}).values():
                if att is None:
                    remaining_attractions += 1

        return {
            "segment": remaining_segments,
            "flight": remaining_segments,
            "stay": remaining_stays,
            "stay_nights": remaining_nights,
            "meal": remaining_meals,
            "attraction": remaining_attractions,
        }

    def _budget_remaining(self, parsed: Any, state: Any) -> Optional[float]:
        budget = self._parsed_get(parsed, "budget", default=None)
        try:
            budget_f = float(budget)
        except Exception:
            return None
        if budget_f <= 0:
            return None
        spent = float(getattr(state, "cost", 0.0) or 0.0)
        return budget_f - spent

    def _phase_ledger(self, parsed: Any, state: Any) -> Dict[str, float]:
        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
        segment_spent = 0.0
        for seg in (getattr(state, "segment_modes", None) or {}).values():
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
                segment_spent += base_f * people
            elif mode == "taxi":
                segment_spent += base_f * math.ceil(people / 4.0)
            elif mode == "self-driving":
                segment_spent += base_f * math.ceil(people / 5.0)
            else:
                segment_spent += base_f

        stay_spent = 0.0
        total_days = int(self._parsed_get(parsed, "duration_days", "days", default=0) or 0)
        total_days = max(1, total_days)
        nights_by_city: Dict[str, int] = {}
        for day in range(1, total_days):
            city = self._planned_city_for_day(parsed, state, day)
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
                    occ_i = int(occ) if occ is not None else None
                except Exception:
                    occ_i = None
                occ_i = max(1, occ_i) if occ_i else None
                rooms = math.ceil(people / float(occ_i or people))
                stay_spent += price * rooms * nights
        else:
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
                nights = max(0, total_days - 1)
                stay_spent += price * rooms * nights

        daily_spent = 0.0
        for day in (state.meals or {}).values():
            for meal in (day or {}).values():
                if meal is not None and meal.get("cost") is not None:
                    try:
                        daily_spent += float(meal["cost"]) * people
                    except Exception:
                        continue

        total_spent = segment_spent + stay_spent + daily_spent
        return {
            "segment": segment_spent,
            "stay": stay_spent,
            "daily": daily_spent,
            "total": total_spent,
        }

    # ----------------------------
    # Action builders (slot -> actions/payloads)
    # ----------------------------
    @staticmethod
    def _assert_candidates_respect_caps(slot: Any, filt: Dict[str, Any], candidates: List[Dict[str, Any]]) -> None:
        stype = getattr(slot, "type", None)
        if stype in ("segment", "flight", "hotel"):
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
        self, parsed: Any, slot: Any, *, force_mode: Optional[str] = None
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
        nonflight_modes = [m for m in self._allowed_transport_modes(parsed) if m != "flight"]
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

    @staticmethod
    def _duration_minutes(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().lower()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            pass
        hours = 0
        minutes = 0
        m = re.search(r"(\d+)\s*hour", s)
        if m:
            hours = int(m.group(1))
        m = re.search(r"(\d+)\s*min", s)
        if m:
            minutes = int(m.group(1))
        m = re.search(r"(\d+)\s*h", s)
        if m:
            hours = max(hours, int(m.group(1)))
        m = re.search(r"(\d+)\s*m", s)
        if m:
            minutes = max(minutes, int(m.group(1)))
        if hours or minutes:
            return float(hours * 60 + minutes)
        return None

    @staticmethod
    def _cuisine_hits(text: str, cuisines: List[str]) -> List[str]:
        hits = []
        for c in cuisines:
            if c and c in text:
                hits.append(c)
        return hits

    def _missing_cuisines(self, parsed: Any, state: Any) -> List[str]:
        cons = self._constraints_from_parsed(parsed)
        required = {c.lower() for c in (cons.get("cuisines") or []) if c}
        if not required:
            return []
        used = set()
        origin = self._parsed_get(parsed, "origin", "org", default=None)
        origin_norm = self.kb._normalize_city(origin) if origin else None
        for day_map in (getattr(state, "meals", None) or {}).values():
            for meal in (day_map or {}).values():
                if not meal:
                    continue
                city = meal.get("city")
                if origin_norm and city and self.kb._normalize_city(city) == origin_norm:
                    continue
                text = str(meal.get("cuisines") or "").lower()
                for c in required:
                    if c in text:
                        used.add(c)
        missing = sorted(required - used)
        return missing

    def _flight_pool(self, slot: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        origin = getattr(slot, "origin", None)
        destination = getattr(slot, "destination", None)
        if not origin or not destination:
            return [], {}
        orig_norm = self.kb._normalize_city(origin)
        dest_norm = self.kb._normalize_city(destination)
        all_flights = [
            f
            for f in (getattr(self.kb, "_flight_buckets", {}).get((orig_norm, dest_norm), []) or [])
            if f.get("price") is not None and not self._is_nan_value(f.get("price"))
        ]
        min_price = None
        try:
            min_price = min(float(f.get("price") or 0.0) for f in all_flights if f.get("price") is not None)
        except ValueError:
            min_price = None
        event = {"min_price": min_price, "total_flights": len(all_flights)}
        return list(all_flights), event

    def _flight_candidates(self, parsed: Any, state: Any, slot: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        all_flights, pool_event = self._flight_pool(slot)
        if not all_flights and not pool_event:
            return [], {}
        kb_count = len(all_flights)
        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))

        def _hard_cost(flight: Dict[str, Any]) -> Optional[float]:
            price = flight.get("price")
            if price is None or self._is_nan_value(price):
                return None
            return float(price) * float(people)

        def _soft_cost(flight: Dict[str, Any]) -> Optional[float]:
            price = flight.get("price")
            if price is None or self._is_nan_value(price):
                return None
            return float(price)

        flights, budget_event = self._filter_candidates_with_budget(
            parsed,
            state,
            "segment",
            all_flights,
            hard_cost_fn=_hard_cost,
            soft_cost_fn=_soft_cost,
        )

        flights.sort(
            key=lambda f: (
                float(f.get("soft_penalty") or 0.0),
                float(f.get("price") or 0.0),
                self._duration_minutes(f.get("duration")) or float("inf"),
            )
        )
        event = dict(budget_event or {})
        event.update(pool_event or {})
        event.update({"kb_count": kb_count, "after_nonbudget_count": kb_count})
        return flights, event

    def _hotel_candidates(self, parsed: Any, state: Any, slot: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        city = getattr(slot, "city", None)
        if not city:
            return [], {}
        city_norm = self.kb._normalize_city(city)
        all_stays_raw = list(self._city_pool.get(city_norm, {}).get("hotel", []) or [])
        kb_count = len(all_stays_raw)
        all_stays = self._apply_constraints_to_candidates("hotel", all_stays_raw, parsed)
        after_constraint = len(all_stays)

        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
        nights = self._planned_accommodation_days_for_city(parsed, state, str(city))
        nights = max(1, nights)

        def _meets_min_nights(stay: Dict[str, Any]) -> bool:
            min_nights = stay.get("minimum_nights")
            if min_nights is None:
                return True
            try:
                return int(min_nights) <= nights
            except Exception:
                return True
        all_stays = [s for s in all_stays if _meets_min_nights(s)]
        after_nonbudget = len(all_stays)
        stays = list(all_stays)

        def _total_per_night(stay: Dict[str, Any]) -> Optional[float]:
            price = stay.get("price")
            if price is None or self._is_nan_value(price):
                return None
            occ = stay.get("occupancy")
            try:
                occ_i = int(occ) if occ is not None else None
            except Exception:
                occ_i = None
            occ_i = max(1, occ_i) if occ_i else None
            rooms = math.ceil(people / float(occ_i or people))
            try:
                return float(price) * rooms
            except Exception:
                return None

        min_stay = self._city_min_stay_price.get(city_norm)
        if min_stay is None:
            min_stay = self._global_min_stay_price
        reserve_exempt = float(min_stay) * float(nights) if min_stay is not None else 0.0

        def _hard_cost(stay: Dict[str, Any]) -> Optional[float]:
            total_per_night = _total_per_night(stay)
            if total_per_night is None:
                return None
            return total_per_night * float(nights)

        def _soft_cost(stay: Dict[str, Any]) -> Optional[float]:
            return _total_per_night(stay)

        stays, budget_event = self._filter_candidates_with_budget(
            parsed,
            state,
            "stay",
            all_stays,
            hard_cost_fn=_hard_cost,
            soft_cost_fn=_soft_cost,
            reserve_exempt=reserve_exempt,
        )

        stays.sort(
            key=lambda s: (
                float(s.get("soft_penalty") or 0.0),
                float(s.get("price") or 0.0),
                -(float(s.get("review") or 0.0)),
            )
        )
        event = dict(budget_event or {})
        dominant = None
        if kb_count > 0 and after_nonbudget == 0:
            if after_constraint == 0:
                cons = self._constraints_from_parsed(parsed)
                room_type = list(cons.get("room_types") or [])
                house_rules = list(cons.get("house_rules") or [])
                if room_type or house_rules:
                    dominant = "stay_constraints"
                else:
                    dominant = "stay_unavailable"
            else:
                dominant = "min_nights"
        event.update(
            {
                "planned_nights": nights,
                "kb_count": kb_count,
                "after_nonbudget_count": after_nonbudget,
                "dominant_nonbudget_filter": dominant,
            }
        )
        return stays, event

    def _meal_candidates(self, parsed: Any, state: Any, slot: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        city = getattr(slot, "city", None)
        day = getattr(slot, "day", None)
        meal_type = getattr(slot, "meal_type", None)
        allowed_cities, city_event = self._meal_allowed_cities(parsed, state, day, meal_type, city)
        pool: List[Dict[str, Any]] = []
        if allowed_cities:
            seen_norms = set()
            for c in allowed_cities:
                if not c:
                    continue
                city_norm = self.kb._normalize_city(c)
                if not city_norm or city_norm in seen_norms:
                    continue
                seen_norms.add(city_norm)
                pool.extend(self._city_pool.get(city_norm, {}).get("meal", []) or [])
        else:
            if not city:
                return [], {}
            city_norm = self.kb._normalize_city(city)
            pool = list(self._city_pool.get(city_norm, {}).get("meal", []) or [])
        kb_count = len(pool)

        cons = self._constraints_from_parsed(parsed)
        cuisines = [c.lower() for c in (cons.get("cuisines") or []) if c]
        missing = self._missing_cuisines(parsed, state)
        counts = self._remaining_counts(parsed, state)
        focus_missing = bool(missing and counts.get("meal", 0) <= len(missing))

        def _matches_missing(rest: Dict[str, Any]) -> bool:
            if not missing:
                return True
            text = str(rest.get("cuisines_lc") or rest.get("cuisines") or "").lower()
            return any(c in text for c in missing)

        base_candidates = [r for r in pool if _matches_missing(r)] if focus_missing else list(pool)
        base_candidates = self._apply_constraints_to_candidates("meal", base_candidates, parsed)
        after_nonbudget = len(base_candidates)

        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
        day = getattr(slot, "day", None)
        reserve_exempt = 0.0
        if day is not None:
            city_for_day = self._planned_city_for_day(parsed, state, int(day)) or city
            city_norm_for_day = self.kb._normalize_city(city_for_day) if city_for_day else None
            min_meal = self._city_min_meal_cost.get(city_norm_for_day) if city_norm_for_day else None
            if min_meal is None:
                min_meal = self._global_min_meal_cost
            if min_meal is not None:
                reserve_exempt = float(min_meal) * float(people)

        def _hard_cost(rest: Dict[str, Any]) -> Optional[float]:
            cost = rest.get("cost")
            if cost is None or self._is_nan_value(cost):
                return None
            return float(cost) * float(people)

        def _soft_cost(rest: Dict[str, Any]) -> Optional[float]:
            cost = rest.get("cost")
            if cost is None or self._is_nan_value(cost):
                return None
            return float(cost)

        candidates, budget_event = self._filter_candidates_with_budget(
            parsed,
            state,
            "daily",
            base_candidates,
            hard_cost_fn=_hard_cost,
            soft_cost_fn=_soft_cost,
            reserve_exempt=reserve_exempt,
        )

        def _rank_key(rest: Dict[str, Any]):
            text = str(rest.get("cuisines_lc") or rest.get("cuisines") or "").lower()
            match_missing = bool(missing and any(c in text for c in missing))
            return (
                0 if match_missing else 1,
                float(rest.get("soft_penalty") or 0.0),
                -(float(rest.get("rating") or 0.0)),
                float(rest.get("cost") or 0.0),
            )

        candidates.sort(key=_rank_key)
        event = dict(budget_event or {})
        dominant = None
        if kb_count > 0 and after_nonbudget == 0:
            if focus_missing and missing:
                dominant = "missing_cuisine"
            else:
                dominant = "meal_constraints"
        event.update(
            {
                "missing_cuisines": list(missing),
                "focus_missing": focus_missing,
                "cuisine_constraints": cuisines,
                "kb_count": kb_count,
                "after_nonbudget_count": after_nonbudget,
                "dominant_nonbudget_filter": dominant,
            }
        )
        return candidates, event

    def _attraction_candidates(self, parsed: Any, state: Any, slot: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        city = getattr(slot, "city", None)
        if not city:
            return [], {}
        city_norm = self.kb._normalize_city(city)
        candidates = list(self._city_pool.get(city_norm, {}).get("attraction", []) or [])
        kb_count = len(candidates)

        def _hard_cost(_: Dict[str, Any]) -> Optional[float]:
            return 0.0

        def _soft_cost(_: Dict[str, Any]) -> Optional[float]:
            return None

        candidates, budget_event = self._filter_candidates_with_budget(
            parsed,
            state,
            "daily",
            candidates,
            hard_cost_fn=_hard_cost,
            soft_cost_fn=_soft_cost,
            apply_hard_gate=False,
        )
        event = dict(budget_event or {})
        event.update({"kb_count": kb_count, "after_nonbudget_count": kb_count})
        return candidates, event

    # ----------------------------
    # CITY phase (bundle selection)
    # ----------------------------
    def _score_city_bundle(
        self,
        parsed: Any,
        state: Any,
        seq: List[str],
        origin: str,
        *,
        allow_repeat: bool = False,
        ) -> Optional[float]:
        """
        City-bundle scoring v2:
        - 不做 transport 可达性检查（不调用 kb.has_any_transport / _flight_exists）
        - 不做 flight/ground 权重
        - 以 “segment/stay/daily cap 可行性” 作为主要排序依据
        - 次级依据：基于 transport + city-level min stay / min meal 的 lower-bound 总成本（越低越好）

        返回：score（越大越好），或 None（无效，如重复城市且不允许）
        """

        if not seq:
            return None
        if not allow_repeat and len(set(seq)) != len(seq):
            return None

        # 确保 city pools / city_min_* 已就绪
        self._ensure_city_pools(parsed)

        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
        total_days = int(self._parsed_get(parsed, "duration_days", "days", default=0) or 0)
        total_days = max(1, total_days)

        # day splits: 你现有实现返回 [{"city", "start_day", "end_day"}, ...]
        day_splits = self._compute_day_splits(total_days, seq)

        # clone state 并注入 day_to_city + city_sequence（复用你现有 bundle cap 流程）
        temp_state = self._clone_state_for_bundle(state, seq, day_splits, total_days)

        # ---- 1) 取 stay / daily caps（复用 BudgetAllocator 的口径）----
        # 你当前 stay/meal 候选阶段都是走 _filter_candidates_with_budget(phase="stay"/"daily")
        # 这里我们直接取 caps 来做 city-bundle feasibility ranking
        segment_cap, _ = self._budget_allocator.caps(parsed, temp_state, "segment")
        stay_cap, _ = self._budget_allocator.caps(parsed, temp_state, "stay")
        daily_cap, _ = self._budget_allocator.caps(parsed, temp_state, "daily")

        # ---- 1.5) segment(transport) cap + lower-bound cost ----
        # cap 判定口径与 _bundle_meets_caps 对齐：
        # - segment_cap 是“人均每段 cap”
        # - 对比 cost_total(整段 party 总成本) <= segment_cap * people
        seg_viol = 0
        transport_lb_cost = 0.0  # lower bound transport cost (party-level)
        return_required = bool(self._parsed_get(parsed, "return_required", default=True))
        legs: List[Tuple[str, str]] = []
        if seq:
            if origin:
                legs.append((origin, seq[0]))
            for i in range(1, len(seq)):
                legs.append((seq[i - 1], seq[i]))
            if return_required and origin:
                legs.append((seq[-1], origin))
        for src, dst in legs:
            cost_total = self._min_transport_cost(parsed, src, dst, people)
            if cost_total is None:
                seg_viol += 1
                continue
            transport_lb_cost += float(cost_total)
            if segment_cap is not None and float(cost_total) > float(segment_cap) * float(people):
                seg_viol += 1

        # ---- 2) stay cap 可行性检查（与你 _bundle_meets_caps 的 stay 部分同口径）----
        require_accommodation = bool(self._parsed_get(parsed, "require_accommodation", default=True))

        stay_viol = 0
        stay_lb_cost = 0.0  # lower bound total stay cost (party-level)
        if require_accommodation and stay_cap is not None:
            for city in seq:
                nights = self._planned_accommodation_days_for_city(parsed, temp_state, city)
                if nights <= 0:
                    continue
                city_norm = self.kb._normalize_city(city)
                min_stay = self._city_min_stay_price.get(city_norm)
                if min_stay is None:
                    # 没有 stay 下界 -> 视为不可行（会强烈降分）
                    stay_viol += 1
                    continue
                # cap 判定：min_stay(每晚 party 总价) <= stay_cap
                if float(min_stay) > float(stay_cap):
                    stay_viol += 1
                stay_lb_cost += float(min_stay) * float(nights)
        elif require_accommodation:
            # 没有 stay_cap 时仍可用于成本 tie-break
            for city in seq:
                nights = self._planned_accommodation_days_for_city(parsed, temp_state, city)
                if nights <= 0:
                    continue
                city_norm = self.kb._normalize_city(city)
                min_stay = self._city_min_stay_price.get(city_norm)
                if min_stay is None:
                    stay_viol += 1
                    continue
                stay_lb_cost += float(min_stay) * float(nights)

        # ---- 3) daily cap（meal）可行性检查：对比每城 min_meal(人均每餐) 与 daily_cap ----
        # 你的 meal hard_cost 是 cost*people，soft_cost 是 cost（人均）；caps 通常对应 soft_cost 的量纲。
        # 因此这里采用：min_meal_per_person <= daily_cap 作为 cap 可行性
        meal_viol = 0
        meal_lb_cost = 0.0  # lower bound total meal cost (party-level)
        if daily_cap is not None:
            # 估计每城市的“餐次数”下界：用 day_to_city 分配后，按 3 meals/day
            day_to_city = getattr(temp_state, "day_to_city", None) or self._day_to_city_from_splits(day_splits, total_days)
            meals_per_day = 3
            meals_by_city: Dict[str, int] = {}
            for d in range(1, total_days + 1):
                c = day_to_city.get(d)
                if not c:
                    continue
                meals_by_city[c] = meals_by_city.get(c, 0) + meals_per_day

            for city, meal_cnt in meals_by_city.items():
                city_norm = self.kb._normalize_city(city)
                min_meal = self._city_min_meal_cost.get(city_norm)
                if min_meal is None:
                    meal_viol += 1
                    continue
                if float(min_meal) > float(daily_cap):
                    meal_viol += 1
                meal_lb_cost += float(min_meal) * float(people) * float(meal_cnt)
        else:
            # 没有 daily_cap 时仍给 tie-break 成本
            day_to_city = getattr(temp_state, "day_to_city", None) or self._day_to_city_from_splits(day_splits, total_days)
            meals_per_day = 3
            meals_by_city: Dict[str, int] = {}
            for d in range(1, total_days + 1):
                c = day_to_city.get(d)
                if not c:
                    continue
                meals_by_city[c] = meals_by_city.get(c, 0) + meals_per_day

            for city, meal_cnt in meals_by_city.items():
                city_norm = self.kb._normalize_city(city)
                min_meal = self._city_min_meal_cost.get(city_norm)
                if min_meal is None:
                    meal_viol += 1
                    continue
                meal_lb_cost += float(min_meal) * float(people) * float(meal_cnt)

        # ---- 4) 组装 score：优先满足 cap，其次更便宜 ----
        # cap 优先：viol 越少越好；再按 lower-bound total cost 越低越好
        total_viol = stay_viol + meal_viol + seg_viol
        lb_total = stay_lb_cost + meal_lb_cost + transport_lb_cost

        # 常数仅用于把“cap 可行性”放到首要排序
        CAP_OK_BONUS = 1e6
        VIOL_PENALTY = 1e5

        cap_ok = (total_viol == 0)
        score = (-lb_total) + (CAP_OK_BONUS if cap_ok else 0.0) - (VIOL_PENALTY * float(total_viol))

        return float(score)

    def build_city_actions(self, parsed: Any, state: Any) -> Tuple[List[str], Dict[str, Tuple], Dict[str, Any]]:
        actions: List[str] = []
        payloads: Dict[str, Tuple] = {}
        event: Dict[str, Any] = {}

        dest_kind = self._infer_dest_kind(parsed)
        total_days = int(self._parsed_get(parsed, "duration_days", "days", default=0) or 0)
        flight_allowed = self._flight_allowed(parsed)
        event.update(
            {
                "dest_kind": dest_kind,
                "flight_allowed": flight_allowed,
                "duration_days": total_days,
            }
        )

        dest = self._parsed_get(parsed, "destination", "dest", default=None)
        origin = self._parsed_get(parsed, "origin", "org", default=None)
        if not origin:
            return actions, payloads, event
        self._ensure_city_pools(parsed)

        existing_prefix = list(getattr(state, "city_sequence", []) or [])

        # Branch A: destination is not a state -> treat as single-city destination bundle.
        if dest_kind != "state":
            if dest:
                seq = [str(dest)]
                day_splits = self._compute_day_splits(total_days or len(seq), seq)
                bundle_cost = self._min_bundle_cost(parsed, seq, day_splits, origin)
                action = f"choose_city_bundle:[{','.join(seq)}]"
                payloads[action] = ("choose_city_bundle", seq, day_splits)
                actions.append(action)
                event.update(
                    {
                        "strategy": "direct_city_bundle",
                        "actions": 1,
                        "bundle_feasible": bundle_cost is not None,
                    }
                )
            else:
                event.update({"strategy": "direct_city_bundle", "actions": 0})
            return actions, payloads, event

        # Branch B: destination is a state -> choose n cities from state city pool.
        city_target = self._parsed_get(parsed, "visiting_city_number", default=None) or self._default_city_target_from_days(total_days)
        if city_target <= 0:
            city_target = self._default_city_target_from_days(total_days)

        prefix = existing_prefix
        remaining_needed = max(0, int(city_target) - len(prefix))
        if remaining_needed == 0:
            event.update({"strategy": "state_bundle", "city_target": city_target, "remaining_needed": 0, "actions": 0})
            return actions, payloads, event

        def _dedupe_city_list(items: List[str]) -> List[str]:
            out: List[str] = []
            seen: set = set()
            for city in items:
                if not city:
                    continue
                key = self.kb._normalize_city(city)
                if key in seen:
                    continue
                seen.add(key)
                out.append(city)
            return out

        state_cities = list(self._parsed_get(parsed, "candidate_cities", default=[]) or [])
        # Even if candidate_cities is provided, augment with KB state cities to avoid
        # hard-feasibility over-pruning on a narrow candidate list.
        try:
            if hasattr(self.kb, "cities_in_state") and dest:
                kb_state_cities = list(self.kb.cities_in_state(dest) or [])
                if kb_state_cities:
                    state_cities.extend(kb_state_cities)
        except Exception:
            pass
        state_cities = _dedupe_city_list(state_cities)

        candidates = list(state_cities)
        injected_out_of_state: List[str] = []
        for must in (self._parsed_get(parsed, "must_visit_cities", default=[]) or []):
            if must not in candidates:
                injected_out_of_state.append(must)
                candidates.insert(0, must)
        for pri in (self._parsed_get(parsed, "priority_cities", default=[]) or []):
            if pri not in candidates:
                injected_out_of_state.append(pri)
                candidates.append(pri)
        candidates = _dedupe_city_list(candidates)

        origin_norm = self.kb._normalize_city(origin) if origin else None
        block_origin_mid = bool(origin_norm and total_days in (5, 7))
        if block_origin_mid:
            candidates = [c for c in candidates if self.kb._normalize_city(c) != origin_norm]

        prefix_norm = {self.kb._normalize_city(c) for c in prefix if c}
        pool = [c for c in candidates if self.kb._normalize_city(c) not in prefix_norm]
        pool_before = len(pool)
        pool = self._rank_candidate_cities(parsed, pool, origin)
        pool_limit = max(self.top_k * 4, remaining_needed * 4)
        if total_days in (5, 7) and remaining_needed > 1:
            pool_limit = max(pool_limit, remaining_needed * 6)
        pool = pool[:pool_limit]
        pool_after = len(pool)

        combos_generated = 0
        budget_pruned = 0
        cap_pruned = 0
        budget_relaxed = False
        fallback_used = False
        relaxed_used = False
        flight_only = flight_allowed
        fallback_mode = None

        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
        constraints = self._constraints_from_parsed(parsed)
        return_required = bool(self._parsed_get(parsed, "return_required", default=True))
        require_accommodation = bool(self._parsed_get(parsed, "require_accommodation", default=True))

        self._bundle_reranker.max_keep = self.top_k

        def _build_bundles(allow_repeat: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            return self._bundle_builder.build(
                parsed=parsed,
                origin=origin,
                candidates=pool,
                city_target=city_target,
                total_days=total_days,
                people=people,
                constraints=constraints,
                prefix=prefix,
                allow_repeat=allow_repeat,
                failure_memory=self.failure_memory,
                return_required=return_required,
                require_accommodation=require_accommodation,
                min_transport_cost_fn=lambda src, dst: self._min_transport_cost(parsed, src, dst, people),
                day_splits_fn=self._compute_day_splits,
            )

        bundles, build_event = _build_bundles(False)
        if not bundles:
            fallback_used = True
            relaxed_used = True
            bundles, fallback_event = _build_bundles(True)
            if fallback_event:
                for key, val in fallback_event.items():
                    if isinstance(val, int):
                        build_event[key] = int(build_event.get(key, 0) or 0) + int(val)

        combos_generated = int(build_event.get("combos_generated", 0) or 0)
        hotel_pruned = int(build_event.get("hotel_pruned", 0) or 0)
        attraction_pruned = int(build_event.get("attraction_pruned", 0) or 0)
        meal_pruned = int(build_event.get("meal_pruned", 0) or 0)
        return_pruned = int(build_event.get("return_pruned", 0) or 0)
        excluded_bundles = int(build_event.get("excluded_bundles", 0) or 0)

        budget_val = self._parsed_get(parsed, "budget", default=None)
        try:
            budget_f = float(budget_val)
        except Exception:
            budget_f = None

        cap_ok: List[Dict[str, Any]] = []
        filtered: List[Dict[str, Any]] = []
        cap_relaxed = False
        for entry in bundles:
            seq = entry.get("bundle") or []
            day_splits = entry.get("day_splits") or self._compute_day_splits(total_days or len(seq), seq)
            if not self._bundle_meets_caps(parsed, state, seq, day_splits, origin):
                cap_pruned += 1
                continue
            cap_ok.append(entry)
            min_cost = self._min_bundle_cost(parsed, seq, day_splits, origin)
            if min_cost is None:
                budget_pruned += 1
                continue
            entry.setdefault("features", {})["min_bundle_cost"] = min_cost
            if budget_f is not None and min_cost > budget_f:
                budget_pruned += 1
                continue
            filtered.append(entry)

        # If caps prune everything, fall back to hard-feasible bundles so MCTS can
        # attempt downstream relaxation/search instead of returning zero CITY actions.
        if not cap_ok and bundles:
            cap_relaxed = True
            cap_ok = list(bundles)
            filtered = []
            for entry in cap_ok:
                seq = entry.get("bundle") or []
                day_splits = entry.get("day_splits") or self._compute_day_splits(total_days or len(seq), seq)
                min_cost = self._min_bundle_cost(parsed, seq, day_splits, origin)
                if min_cost is None:
                    budget_pruned += 1
                    continue
                entry.setdefault("features", {})["min_bundle_cost"] = min_cost
                if budget_f is not None and min_cost > budget_f:
                    budget_pruned += 1
                    continue
                filtered.append(entry)

        if not filtered and cap_ok:
            budget_relaxed = True
            filtered = cap_ok

        reranked = self._bundle_reranker.rerank(
            constraints=constraints,
            failure_memory=self.failure_memory,
            bundles=filtered,
        )
        bundle_lookup = {tuple(entry.get("bundle") or []): entry for entry in filtered}
        selected: List[Dict[str, Any]] = []
        for item in reranked:
            seq = item.get("bundle") or []
            entry = bundle_lookup.get(tuple(seq))
            if entry:
                selected.append(entry)

        seen_seq: set = set()
        for entry in selected:
            seq = entry.get("bundle") or []
            seq_key = tuple(self.kb._normalize_city(c) if c else "" for c in seq)
            if seq_key in seen_seq:
                continue
            seen_seq.add(seq_key)
            day_splits = entry.get("day_splits") or self._compute_day_splits(total_days or len(seq), seq)
            action = f"choose_city_bundle:[{','.join(seq)}]"
            transport_lb = entry.get("features", {}).get("transport_lb_cost")
            if transport_lb is None:
                transport_lb = 0.0
                legs: List[Tuple[str, str]] = []
                if seq:
                    if origin:
                        legs.append((origin, seq[0]))
                    for i in range(1, len(seq)):
                        legs.append((seq[i - 1], seq[i]))
                    if return_required and origin:
                        legs.append((seq[-1], origin))
                for src, dst in legs:
                    c = self._min_transport_cost(parsed, src, dst, people)
                    if c is None:
                        transport_lb = float("inf")
                        break
                    transport_lb += float(c)

            payloads[action] = (
                "choose_city_bundle",
                seq,
                day_splits,
                {"transport_lb": transport_lb, "transport_lb_per_person": (transport_lb / float(people)) if people else transport_lb},
            )
            actions.append(action)

        event.update(
            {
                "strategy": "state_bundle_v2",
                "city_target": city_target,
                "remaining_needed": remaining_needed,
                "connectivity_mode": "flight_only" if (flight_only and not fallback_used) else (fallback_mode or "any_transport"),
                "flight_only": flight_only,
                "fallback_mode": fallback_mode,
                "pool_size_before_prefilter": pool_before,
                "pool_size_after_prefilter": pool_after,
                "combos_generated": combos_generated,
                "combos_kept_topk": len(actions),
                "budget_pruned": budget_pruned,
                "cap_pruned": cap_pruned,
                "budget_relaxed": budget_relaxed,
                "relaxed_used": relaxed_used,
                "fallback_used": fallback_used,
                "hotel_pruned": hotel_pruned,
                "attraction_pruned": attraction_pruned,
                "meal_pruned": meal_pruned,
                "return_pruned": return_pruned,
                "excluded_bundles": excluded_bundles,
                "reranked_used": True,
                "cap_relaxed": cap_relaxed,
                "injected_out_of_state": injected_out_of_state,
                "origin_filtered": block_origin_mid,
            }
        )
        return actions, payloads, event

    def _phase_key(self, phase: str) -> str:
        p = str(phase or "").lower().strip()
        if p in ("flight", "segment"):
            return "segment"
        if p in ("hotel", "stay"):
            return "stay"
        if p in ("meal", "daily"):
            return "daily"
        return p
    def _hard_gate_pass(
        self,
        *,
        budget_remaining: float,
        cand_cost: float,
        reserve: float,
    ) -> tuple[bool, str]:
        if cand_cost > budget_remaining:
            return False, "over_budget"
        if (budget_remaining - cand_cost) < reserve:
            return False, "violate_reserve"
        return True, "ok"

    def _estimate_min_future_cost(self, parsed: Any, state: Any) -> tuple[float, dict]:
        people = max(1, int(self._parsed_get(parsed, "people_number", default=None) or 1))
        total_days = int(self._parsed_get(parsed, "duration_days", "days", default=0) or 0)
        total_days = max(1, total_days)
        require_accommodation = bool(self._parsed_get(parsed, "require_accommodation", default=True))

        stay_total = 0.0
        meal_total = 0.0
        missing_stay_nights = 0
        missing_meals = 0
        stay_global = self._global_min_stay_price
        meal_global = self._global_min_meal_cost

        if require_accommodation:
            for city in (getattr(state, "city_sequence", None) or []):
                if (getattr(state, "city_stays", None) or {}).get(city) is not None:
                    continue
                nights = self._planned_accommodation_days_for_city(parsed, state, city)
                if nights <= 0:
                    continue
                city_norm = self.kb._normalize_city(city)
                min_stay = self._city_min_stay_price.get(city_norm)
                if min_stay is None:
                    missing_stay_nights += nights
                    continue
                stay_total += float(min_stay) * float(nights)

        meals = getattr(state, "meals", None) or {}
        for day in range(1, total_days + 1):
            day_map = meals.get(day) or {}
            for meal in (day_map or {}).values():
                if meal is not None:
                    continue
                city = self._planned_city_for_day(parsed, state, day)
                city_norm = self.kb._normalize_city(city) if city else None
                min_meal = self._city_min_meal_cost.get(city_norm) if city_norm else None
                if min_meal is None:
                    missing_meals += 1
                    continue
                meal_total += float(min_meal) * float(people)

        bundle_infeasible = bool(missing_stay_nights > 0 or missing_meals > 0)
        reserve = stay_total + meal_total
        detail = {
            "stay_total": stay_total,
            "meal_total": meal_total,
            "missing_stay_nights": missing_stay_nights,
            "missing_meals": missing_meals,
            "global_min_stay": stay_global,
            "global_min_meal": meal_global,
            "bundle_infeasible": bundle_infeasible,
        }
        return reserve, detail

    def _filter_candidates_with_budget(
        self,
        parsed: Any,
        state: Any,
        phase_key: str,
        candidates: List[Dict[str, Any]],
        *,
        hard_cost_fn: Callable[[Dict[str, Any]], Optional[float]],
        soft_cost_fn: Callable[[Dict[str, Any]], Optional[float]],
        reserve_exempt: float = 0.0,
        max_attempts: int = 3,
        apply_hard_gate: bool = True,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        base_count = len(candidates)
        budget_remaining = self._budget_remaining(parsed, state)
        phase_key = self._phase_key(phase_key)
        min_cost_nonbudget: Optional[float] = None
        for cand in candidates:
            soft_cost = soft_cost_fn(cand)
            if soft_cost is None or self._is_nan_value(soft_cost):
                continue
            try:
                cost_f = float(soft_cost)
            except Exception:
                continue
            if min_cost_nonbudget is None or cost_f < min_cost_nonbudget:
                min_cost_nonbudget = cost_f

        reserve_raw, reserve_detail = self._estimate_min_future_cost(parsed, state)
        bundle_infeasible = bool(reserve_detail.get("bundle_infeasible")) if isinstance(reserve_detail, dict) else False
        if bundle_infeasible and budget_remaining is not None:
            reserve_raw = float(budget_remaining) + float(reserve_exempt or 0.0) + 1.0
        reserve_raw = max(0.0, float(reserve_raw) - float(reserve_exempt or 0.0))
        reserve_scale = float(self._relaxer.reserve_scale(phase_key))
        reserve = reserve_raw * reserve_scale

        hard_reject = {"over_budget": 0, "violate_reserve": 0, "missing_cost": 0}
        hard_kept: List[Dict[str, Any]] = []
        relaxed = False
        relax_steps: List[Dict[str, Any]] = []

        def _apply_gate(reserve_val: float) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
            kept: List[Dict[str, Any]] = []
            reject = {"over_budget": 0, "violate_reserve": 0, "missing_cost": 0}
            for cand in candidates:
                hard_cost = hard_cost_fn(cand)
                if hard_cost is None or self._is_nan_value(hard_cost):
                    reject["missing_cost"] += 1
                    continue
                ok, reason = self._hard_gate_pass(
                    budget_remaining=float(budget_remaining),
                    cand_cost=float(hard_cost),
                    reserve=float(reserve_val),
                )
                if ok:
                    kept.append(cand)
                else:
                    reject[reason] = reject.get(reason, 0) + 1
            return kept, reject

        if not apply_hard_gate or budget_remaining is None:
            hard_kept = list(candidates)
        else:
            hard_kept, hard_reject = _apply_gate(reserve)
            if not hard_kept and base_count > 0:
                for attempt in range(max_attempts):
                    if not self._relaxer.bump(phase_key):
                        break
                    relaxed = True
                    reserve_scale = float(self._relaxer.reserve_scale(phase_key))
                    reserve = reserve_raw * reserve_scale
                    hard_kept, hard_reject = _apply_gate(reserve)
                    relax_steps.append(
                        {
                            "attempt": attempt + 1,
                            "reserve_scale": reserve_scale,
                            "cap_multiplier": float(self._relaxer.cap_multiplier(phase_key)),
                            "reserve": reserve,
                            "kept": len(hard_kept),
                        }
                    )
                    if hard_kept:
                        break
                if relaxed:
                    self._budget_revision += 1

        slot_cap, cap_info = self._budget_allocator.caps(parsed, state, phase_key)
        cap_multiplier = cap_info.get("cap_multiplier") if isinstance(cap_info, dict) else None

        penalties: List[float] = []
        ratios: List[float] = []
        enriched: List[Dict[str, Any]] = []
        for cand in hard_kept:
            soft_cost = soft_cost_fn(cand)
            penalty = 0.0
            ratio = None
            if soft_cost is not None and not self._is_nan_value(soft_cost):
                penalty, pinfo = self._soft_penalty(
                    cand_cost=float(soft_cost),
                    slot_cap=slot_cap,
                    phase_key=phase_key,
                )
                ratio = pinfo.get("r") if isinstance(pinfo, dict) else None
            cand_copy = dict(cand)
            cand_copy["soft_penalty"] = float(penalty)
            if ratio is not None:
                cand_copy["soft_ratio"] = float(ratio)
            enriched.append(cand_copy)
            penalties.append(float(penalty))
            if ratio is not None:
                ratios.append(float(ratio))

        soft_stats: Dict[str, Any] = {
            "count": len(enriched),
            "min": min(penalties) if penalties else 0.0,
            "max": max(penalties) if penalties else 0.0,
            "avg": (sum(penalties) / len(penalties)) if penalties else 0.0,
            "over_zero": sum(1 for p in penalties if p > 0.0),
        }
        if ratios:
            soft_stats.update(
                {
                    "ratio_min": min(ratios),
                    "ratio_max": max(ratios),
                    "ratio_avg": sum(ratios) / len(ratios),
                }
            )

        failure_code = None
        if base_count == 0:
            failure_code = "kb_empty"
        elif apply_hard_gate and budget_remaining is not None and not hard_kept:
            failure_code = "hard_pruned_empty"

        filter_usage = {
            "phase_key": phase_key,
            "base_count": base_count,
            "hard_kept": len(hard_kept),
            "hard_reject": hard_reject,
            "budget_remaining": budget_remaining,
            "reserve_raw": reserve_raw,
            "reserve_exempt": reserve_exempt,
            "reserve_scale": reserve_scale,
            "reserve": reserve,
            "slot_cap": slot_cap,
            "slot_cap_base": cap_info.get("base_cap") if isinstance(cap_info, dict) else None,
            "cap_multiplier": cap_multiplier,
            "min_cost_nonbudget": min_cost_nonbudget,
            "bundle_infeasible": bundle_infeasible,
            "soft_penalty": soft_stats,
            "relax_steps": relax_steps,
            "failure_code": failure_code,
            "phase_ledger": self._phase_ledger(parsed, state),
        }

        event = {
            "budget_cap": slot_cap,
            "budget_relaxed": relaxed,
            "filter_usage": filter_usage,
        }
        event.update(reserve_detail or {})
        return enriched, event

    def _soft_penalty(self, *, cand_cost: float, slot_cap: float | None, phase_key: str) -> tuple[float, dict]:
        if slot_cap is None or slot_cap <= 0:
            return 0.0, {"r": None, "slot_cap": slot_cap}

        eps = 1e-6
        r = float(cand_cost) / (float(slot_cap) + eps)

        # 分段惩罚参数（你可调）
        rho = 1.5
        lam = 0.3
        mu = 1.0

        if r <= 1.0:
            base = 0.0
        elif r <= rho:
            base = lam * (r - 1.0)
        else:
            base = lam * (rho - 1.0) + mu * (r - rho) * (r - rho)

        # phase 权重（避免 meal 被误杀）
        w = {"segment": 1.0, "stay": 1.2, "daily": 0.5}.get(phase_key, 1.0)
        penalty = w * base
        return penalty, {"r": r, "slot_cap": slot_cap, "base": base, "w": w}
