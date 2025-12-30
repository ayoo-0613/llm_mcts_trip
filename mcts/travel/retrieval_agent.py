from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mcts.travel.retrieval import (
    ActionFactory,
    BudgetAllocator,
    ConstraintNormalizer,
    DedupFilter,
    RelaxationController,
    filter_with_budget_relax,
)

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
        candidate_cap: int = 80,
        debug: bool = False,
        log_filter_usage: bool = False,
    ):
        self.kb = kb
        self.top_k = top_k
        self.candidate_cap = candidate_cap
        self.debug = debug
        self.log_filter_usage = log_filter_usage
        self._transport_cache: Dict[Tuple[str, str, bool], bool] = {}
        self._parsed_sig: Optional[str] = None
        self._city_pool: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._city_cuisine_coverage: Dict[str, set] = {}
        self._budget_scales = {"flight": 1.0, "stay": 1.0, "meal": 1.0, "attraction": 1.0}
        self._constraint_normalizer = ConstraintNormalizer()
        self._budget_allocator = BudgetAllocator(self._parsed_get, self._remaining_counts, self._budget_scales)
        self._relaxer = RelaxationController(self._budget_scales)
        self._action_factory = ActionFactory()

    def reset(self) -> None:
        self._transport_cache = {}
        self._parsed_sig = None
        self._city_pool = {}
        self._city_cuisine_coverage = {}
        self._budget_scales.clear()
        self._budget_scales.update({"flight": 1.0, "stay": 1.0, "meal": 1.0, "attraction": 1.0})

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
    ) -> List[Dict[str, Any]]:
        if not candidates or k <= 0:
            return []
        for start in range(0, len(candidates), k):
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
        out: List[Dict[str, Any]] = []
        for rest in restaurants:
            if cuisines:
                text = str(rest.get("cuisines_lc") or rest.get("cuisines") or "").lower()
                if not any(c in text for c in cuisines):
                    continue
            out.append(rest)
            if len(out) >= self.candidate_cap:
                break
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
        self._budget_scales.clear()
        self._budget_scales.update({"flight": 1.0, "stay": 1.0, "meal": 1.0, "attraction": 1.0})

        cons = self._constraints_from_parsed(parsed)
        cuisines = [c.lower() for c in (cons.get("cuisines") or []) if c]
        for city in self._candidate_cities(parsed):
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

    def _rank_candidate_cities(self, parsed: Any, cities: List[str], origin: str) -> List[str]:
        cons = self._constraints_from_parsed(parsed)
        required_cuisines = [c.lower() for c in (cons.get("cuisines") or []) if c]
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
            has_rest = bool(pool.get("meal"))
            has_att = bool(pool.get("attraction"))
            coverage = self._city_cuisine_coverage.get(city_norm, set())

            score = 0.0
            if has_stay:
                score += 1.0
            if has_rest:
                score += 0.6
            if has_att:
                score += 0.2
            if required_cuisines:
                score += 0.8 * (len(coverage) / float(max(1, len(set(required_cuisines)))))

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
            cuisines = [str(c).strip().lower() for c in (cons.get("cuisines") or []) if c]
            if not cuisines:
                return candidates
            filtered = []
            for rest in candidates:
                text = str(rest.get("cuisines_lc") or rest.get("cuisines") or "").lower()
                if any(c in text for c in cuisines):
                    filtered.append(rest)
            return filtered
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
        remaining_flights = sum(1 for idx, _, _ in segments if idx not in (getattr(state, "segment_modes", None) or {}))

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
            "flight": remaining_flights,
            "stay": remaining_stays,
            "stay_nights": remaining_nights,
            "meal": remaining_meals,
            "attraction": remaining_attractions,
        }

    # ----------------------------
    # Action builders (slot -> actions/payloads)
    # ----------------------------
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

    def _flight_candidates(self, parsed: Any, state: Any, slot: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        origin = getattr(slot, "origin", None)
        destination = getattr(slot, "destination", None)
        if not origin or not destination:
            return [], {}
        orig_norm = self.kb._normalize_city(origin)
        dest_norm = self.kb._normalize_city(destination)
        all_flights = list(getattr(self.kb, "_flight_buckets", {}).get((orig_norm, dest_norm), []))

        def _within_cap(flight: Dict[str, Any], limit: Optional[float]) -> bool:
            if limit is None:
                return True
            price = flight.get("price")
            if price is None:
                return False
            return float(price) <= limit

        budget_result = filter_with_budget_relax(
            parsed,
            state,
            "flight",
            all_flights,
            allocator=self._budget_allocator,
            relaxer=self._relaxer,
            filter_fn=_within_cap,
        )
        flights = list(budget_result.candidates)
        cap = budget_result.cap
        info = budget_result.info
        relaxed = budget_result.relaxed

        flights.sort(
            key=lambda f: (
                float(f.get("price") or 0.0),
                self._duration_minutes(f.get("duration")) or float("inf"),
            )
        )
        event = dict(info or {})
        event.update({"budget_cap": cap, "budget_relaxed": relaxed})
        return flights, event

    def _hotel_candidates(self, parsed: Any, state: Any, slot: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        city = getattr(slot, "city", None)
        if not city:
            return [], {}
        city_norm = self.kb._normalize_city(city)
        all_stays = list(self._city_pool.get(city_norm, {}).get("hotel", []) or [])
        all_stays = self._apply_constraints_to_candidates("hotel", all_stays, parsed)

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
        stays = list(all_stays)

        def _within_cap(stay: Dict[str, Any], limit: Optional[float]) -> bool:
            if limit is None:
                return True
            price = stay.get("price")
            if price is None:
                return True
            occ = stay.get("occupancy")
            try:
                occ_i = int(occ) if occ is not None else None
            except Exception:
                occ_i = None
            occ_i = max(1, occ_i) if occ_i else None
            rooms = math.ceil(people / float(occ_i or people))
            try:
                total_per_night = float(price) * rooms
            except Exception:
                return True
            return total_per_night <= limit

        budget_result = filter_with_budget_relax(
            parsed,
            state,
            "stay",
            all_stays,
            allocator=self._budget_allocator,
            relaxer=self._relaxer,
            filter_fn=_within_cap,
        )
        stays = list(budget_result.candidates)
        cap = budget_result.cap
        info = budget_result.info
        relaxed = budget_result.relaxed

        stays.sort(
            key=lambda s: (
                float(s.get("price") or 0.0),
                -(float(s.get("review") or 0.0)),
            )
        )
        event = dict(info or {})
        event.update({"budget_cap": cap, "budget_relaxed": relaxed, "planned_nights": nights})
        return stays, event

    def _meal_candidates(self, parsed: Any, state: Any, slot: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        city = getattr(slot, "city", None)
        if not city:
            return [], {}
        city_norm = self.kb._normalize_city(city)
        pool = list(self._city_pool.get(city_norm, {}).get("meal", []) or [])

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
        candidates = list(base_candidates)
        candidates = self._apply_constraints_to_candidates("meal", candidates, parsed)

        def _within_cap(rest: Dict[str, Any], limit: Optional[float]) -> bool:
            if limit is None:
                return True
            cost = rest.get("cost")
            if cost is None:
                return True
            return float(cost or 0.0) <= limit

        budget_result = filter_with_budget_relax(
            parsed,
            state,
            "meal",
            base_candidates,
            allocator=self._budget_allocator,
            relaxer=self._relaxer,
            filter_fn=_within_cap,
        )
        candidates = list(budget_result.candidates)
        cap = budget_result.cap
        info = budget_result.info
        relaxed = budget_result.relaxed

        def _rank_key(rest: Dict[str, Any]):
            text = str(rest.get("cuisines_lc") or rest.get("cuisines") or "").lower()
            match_missing = bool(missing and any(c in text for c in missing))
            return (
                0 if match_missing else 1,
                -(float(rest.get("rating") or 0.0)),
                float(rest.get("cost") or 0.0),
            )

        candidates.sort(key=_rank_key)
        event = dict(info or {})
        event.update(
            {
                "budget_cap": cap,
                "budget_relaxed": relaxed,
                "missing_cuisines": list(missing),
                "focus_missing": focus_missing,
                "cuisine_constraints": cuisines,
            }
        )
        return candidates, event

    def _attraction_candidates(self, parsed: Any, state: Any, slot: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        city = getattr(slot, "city", None)
        if not city:
            return [], {}
        city_norm = self.kb._normalize_city(city)
        candidates = list(self._city_pool.get(city_norm, {}).get("attraction", []) or [])
        return candidates, {}

    # ----------------------------
    # CITY phase (bundle selection)
    # ----------------------------
    def _score_city_bundle(
        self,
        parsed: Any,
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
        require_flight = bool(self._parsed_get(parsed, "require_flight", default=False))
        return_required = bool(self._parsed_get(parsed, "return_required", default=True))
        constraints = self._constraints_from_parsed(parsed)
        required_cuisines = [c.lower() for c in (constraints.get("cuisines") or []) if c]

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
            city_norm = self.kb._normalize_city(city)
            pool = self._city_pool.get(city_norm, {})
            has_stay = bool(pool.get("hotel"))
            has_rest = bool(pool.get("meal"))
            has_att = bool(pool.get("attraction"))
            score += 0.1 * has_stay + 0.1 * has_rest + 0.1 * has_att

        if required_cuisines:
            coverage = set()
            for city in seq:
                city_norm = self.kb._normalize_city(city)
                coverage |= self._city_cuisine_coverage.get(city_norm, set())
            if coverage:
                score += 0.2 * (len(coverage) / float(max(1, len(set(required_cuisines)))))
            else:
                score -= 0.2

        return score

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
                action = f"choose_city_bundle:[{','.join(seq)}]"
                payloads[action] = ("choose_city_bundle", seq, day_splits)
                actions.append(action)
                event.update({"strategy": "direct_city_bundle", "actions": 1})
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

        state_cities = list(self._parsed_get(parsed, "candidate_cities", default=[]) or [])
        if not state_cities:
            try:
                if hasattr(self.kb, "cities_in_state") and dest:
                    state_cities = list(self.kb.cities_in_state(dest) or [])
            except Exception:
                state_cities = []

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

        pool = [c for c in candidates if c not in prefix]
        pool_before = len(pool)
        pool = self._rank_candidate_cities(parsed, pool, origin)
        pool_limit = max(self.top_k * 4, remaining_needed * 4)
        if total_days in (5, 7) and remaining_needed > 1:
            pool_limit = max(pool_limit, remaining_needed * 6)
        pool = pool[:pool_limit]
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
                    parsed,
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
    def compute(self, parsed: Any, state: Any, slot: Any, *, user_query: str = "") -> SlotActionResult:
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
            if not self._flight_allowed(parsed):
                force = "taxi" if "taxi" in self._allowed_transport_modes(parsed) else "self-driving"
                actions, payloads = self._ground_fallback_actions(parsed, slot, force_mode=force)
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

            flights, event = self._flight_candidates(parsed, state, slot)
            candidates = self._select_valid_batch(flights, self.top_k)
            action_objs = self._action_factory.build(slot, candidates)
            actions, payloads = self._action_factory.to_actions_payloads(action_objs)
            if actions and len(actions) < self.top_k and "taxi" in self._allowed_transport_modes(parsed):
                extra_actions, extra_payloads = self._ground_fallback_actions(parsed, slot, force_mode="taxi")
                actions += extra_actions
                payloads.update(extra_payloads)
            if not actions:
                forced_mode = None
                if has_flight or has_taxi:
                    forced_mode = "taxi"
                    actions, payloads = self._ground_fallback_actions(parsed, slot, force_mode="taxi")
                elif has_self:
                    forced_mode = "self-driving"
                    actions, payloads = self._ground_fallback_actions(parsed, slot, force_mode="self-driving")
                else:
                    forced_mode = "taxi" if "taxi" in self._allowed_transport_modes(parsed) else "self-driving"
                    actions, payloads = self._ground_fallback_actions(parsed, slot, force_mode=forced_mode)
                return SlotActionResult(
                    actions=actions,
                    payloads=payloads,
                    candidates=[],
                    relaxed=False,
                    filt={},
                    policy_event={"forced_transport_mode": forced_mode, "flight_candidates_empty": True},
                    plan=None,
                    uncapped_filter=None,
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

        if stype == "hotel":
            candidates, event = self._hotel_candidates(parsed, state, slot)
            candidates = self._select_valid_batch(candidates, self.top_k)
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
