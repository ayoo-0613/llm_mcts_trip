from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple

from mcts.travel.retrieval.city_profile import CityProfileBuilder


class CityBundleBuilderV2:
    """Build city bundles with rule-based feasibility filtering."""

    def __init__(
        self,
        kb: Any,
        *,
        profile_builder: Optional[CityProfileBuilder] = None,
        max_combos: int = 5000,
        max_kept: int = 500,
    ):
        self.kb = kb
        self.profile_builder = profile_builder or CityProfileBuilder(kb)
        self.max_combos = max(100, int(max_combos))
        self.max_kept = max(50, int(max_kept))

    def build(
        self,
        *,
        parsed: Any,
        origin: str,
        candidates: List[str],
        city_target: int,
        total_days: int,
        people: int,
        constraints: Dict[str, Any],
        prefix: Optional[List[str]] = None,
        allow_repeat: bool = False,
        failure_memory: Optional[Any] = None,
        return_required: bool = True,
        require_accommodation: bool = True,
        min_transport_cost_fn: Optional[Callable[[str, str], Optional[float]]] = None,
        day_splits_fn: Optional[Callable[[int, List[str]], List[Dict[str, Any]]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        prefix = list(prefix or [])
        remaining_needed = max(0, int(city_target) - len(prefix))
        if remaining_needed <= 0:
            return [], {"combos_generated": 0, "bundles_kept": 0}

        if allow_repeat:
            iterator = itertools.product(candidates, repeat=remaining_needed)
        else:
            iterator = itertools.permutations(candidates, remaining_needed)

        combos_generated = 0
        excluded_bundles = 0
        hotel_pruned = 0
        attraction_pruned = 0
        meal_pruned = 0
        return_pruned = 0
        bundles: List[Dict[str, Any]] = []

        att_min = 1
        try:
            if isinstance(parsed, dict):
                att_min = int(parsed.get("attractions_per_day_min") or parsed.get("attractions_min") or 1)
        except Exception:
            att_min = 1
        att_min = max(0, int(att_min))
        meals_per_day = 3

        def _build_entry(seq: List[str], *, count_prune: bool, check_exclusion: bool) -> Optional[Dict[str, Any]]:
            nonlocal excluded_bundles, hotel_pruned, attraction_pruned, meal_pruned, return_pruned
            if not allow_repeat and len({self.kb._normalize_city(c) for c in seq if c}) != len(seq):
                return None
            bundle_key_set = self._bundle_key_set(seq)
            bundle_key_seq = self._bundle_key_seq(seq)
            if check_exclusion and failure_memory is not None and getattr(failure_memory, "is_excluded", None):
                if failure_memory.is_excluded(bundle_key_seq):
                    if count_prune:
                        excluded_bundles += 1
                    return None

            day_splits = (
                day_splits_fn(total_days, seq)
                if day_splits_fn is not None
                else self._compute_day_splits(total_days, seq)
            )
            day_to_city = self._day_to_city_from_splits(day_splits, total_days)
            nights_by_city = self._planned_nights_by_city(day_to_city, total_days)
            days_by_city: Dict[str, int] = {}
            for day in range(1, max(1, total_days) + 1):
                city = day_to_city.get(day)
                if not city:
                    continue
                days_by_city[city] = days_by_city.get(city, 0) + 1

            profiles: Dict[str, Dict[str, Any]] = {}
            hotel_feasible_all = True
            attraction_feasible_all = True
            meal_feasible_all = True
            min_nights_min = None
            for city in seq:
                profile = self.profile_builder.build(
                    city,
                    constraints,
                    people=people,
                )
                profiles[city] = profile
                if require_accommodation and not profile.get("hotel", {}).get("exists"):
                    hotel_feasible_all = False
                    break
                city_min_nights = profile.get("hotel", {}).get("min_nights_min")
                if city_min_nights is not None:
                    if min_nights_min is None or city_min_nights < min_nights_min:
                        min_nights_min = city_min_nights
                needed_nights = nights_by_city.get(city, 0)
                if require_accommodation and city_min_nights is not None:
                    required_nights = max(1, int(needed_nights))
                    if int(city_min_nights) > required_nights:
                        hotel_feasible_all = False
                        break

                required_attractions = days_by_city.get(city, 0) * att_min
                if required_attractions > 0:
                    available_attractions = int(profile.get("daily", {}).get("attraction_count") or 0)
                    if available_attractions < int(required_attractions):
                        attraction_feasible_all = False
                        break

                required_meals = days_by_city.get(city, 0) * meals_per_day
                if required_meals > 0:
                    available_meals = int(profile.get("daily", {}).get("meal_count") or 0)
                    if available_meals < int(required_meals):
                        meal_feasible_all = False
                        break

            if not hotel_feasible_all:
                if count_prune:
                    hotel_pruned += 1
                return None
            if not attraction_feasible_all:
                if count_prune:
                    attraction_pruned += 1
                return None
            if not meal_feasible_all:
                if count_prune:
                    meal_pruned += 1
                return None

            min_return_cost = 0.0
            return_leg_exists = True
            if return_required and origin and seq:
                if min_transport_cost_fn is None:
                    return_leg_exists = False
                else:
                    min_return_cost = min_transport_cost_fn(seq[-1], origin)
                    return_leg_exists = min_return_cost is not None
            if not return_leg_exists:
                if count_prune:
                    return_pruned += 1
                return None

            transport_cost, edges, edge_costs = self._transport_lower_bound(seq, origin, return_required, min_transport_cost_fn)
            stay_cost = self._stay_lower_bound(seq, nights_by_city, profiles)
            meal_cost = self._meal_lower_bound(day_to_city, profiles, people)

            bundle_lb_cost = None
            if transport_cost is not None and stay_cost is not None and meal_cost is not None:
                bundle_lb_cost = transport_cost + stay_cost + meal_cost

            city_features: Dict[str, Dict[str, Any]] = {}
            for city in seq:
                profile = profiles.get(city) or {}
                hotel = profile.get("hotel", {}) or {}
                daily = profile.get("daily", {}) or {}
                days = int(days_by_city.get(city, 0) or 0)
                nights = int(nights_by_city.get(city, 0) or 0)
                hotel_min_nights = hotel.get("min_nights_min")
                hotel_min_price = hotel.get("min_price")
                attraction_count = int(daily.get("attraction_count") or 0)
                meal_count = int(daily.get("meal_count") or 0)
                attr_required = days * int(att_min)
                meal_required = days * int(meals_per_day)
                attr_slack = attraction_count - attr_required
                meal_slack = meal_count - meal_required
                hotel_slack = None
                if hotel_min_nights is not None:
                    try:
                        hotel_slack = int(nights) - int(hotel_min_nights)
                    except Exception:
                        hotel_slack = None
                city_features[city] = {
                    "days": days,
                    "nights": nights,
                    "hotel_exists": bool(hotel.get("exists")),
                    "hotel_min_price": hotel_min_price,
                    "hotel_min_nights": hotel_min_nights,
                    "attraction_count": attraction_count,
                    "meal_count": meal_count,
                    "min_meal_cost": daily.get("min_meal_cost"),
                    "attr_required": attr_required,
                    "meal_required": meal_required,
                    "attr_slack": attr_slack,
                    "meal_slack": meal_slack,
                    "hotel_slack": hotel_slack,
                }

            meal_slacks = {c: f.get("meal_slack") for c, f in city_features.items() if f.get("meal_slack") is not None}
            attr_slacks = {c: f.get("attr_slack") for c, f in city_features.items() if f.get("attr_slack") is not None}
            hotel_prices = {c: f.get("hotel_min_price") for c, f in city_features.items() if f.get("hotel_min_price") is not None}
            bottleneck_meal_slack = min(meal_slacks.values()) if meal_slacks else None
            bottleneck_attr_slack = min(attr_slacks.values()) if attr_slacks else None
            bottleneck_hotel_price = max(hotel_prices.values()) if hotel_prices else None
            bottleneck_city_meal = min(meal_slacks, key=meal_slacks.get) if meal_slacks else None
            bottleneck_city_hotel = max(hotel_prices, key=hotel_prices.get) if hotel_prices else None

            bottleneck_edge = None
            if edge_costs:
                blocked_edges = [edge for edge, info in edge_costs.items() if info.get("blocked")]
                if blocked_edges:
                    bottleneck_edge = blocked_edges[0]
                else:
                    max_edge = None
                    max_cost = None
                    for edge, info in edge_costs.items():
                        cost_val = info.get("min_cost")
                        if cost_val is None:
                            continue
                        if max_cost is None or float(cost_val) > float(max_cost):
                            max_cost = cost_val
                            max_edge = edge
                    bottleneck_edge = max_edge

            bundle_risk = {
                "bottleneck_meal_slack": bottleneck_meal_slack,
                "bottleneck_attr_slack": bottleneck_attr_slack,
                "bottleneck_hotel_price": bottleneck_hotel_price,
                "bottleneck_city_meal": bottleneck_city_meal,
                "bottleneck_city_hotel": bottleneck_city_hotel,
                "bottleneck_edge": bottleneck_edge,
            }

            return {
                "bundle": seq,
                "day_splits": day_splits,
                "features": {
                    "hotel_feasible_all": hotel_feasible_all,
                    "return_leg_exists": return_leg_exists,
                    "min_return_cost": min_return_cost,
                    "bundle_lb_cost": bundle_lb_cost,
                    "stay_lb_cost": stay_cost,
                    "meal_lb_cost": meal_cost,
                    "transport_lb_cost": transport_cost,
                    "hotel_min_nights_min": min_nights_min,
                    "edges": edges,
                    "bundle_key": bundle_key_set,
                    "bundle_key_set": bundle_key_set,
                    "bundle_key_seq": bundle_key_seq,
                    "city_features": city_features,
                    "edge_costs": edge_costs,
                    "bundle_risk": bundle_risk,
                },
            }

        for combo in iterator:
            combos_generated += 1
            seq = prefix + list(combo)
            entry = _build_entry(seq, count_prune=True, check_exclusion=True)
            if entry is None:
                continue
            bundles.append(entry)
            if len(bundles) >= self.max_kept:
                break
            if combos_generated >= self.max_combos:
                break

        permutation_injected = 0
        if not allow_repeat and total_days in (5, 7) and len(bundles) < self.max_kept:
            top_m = min(10, len(bundles))
            existing_keys = {self._bundle_key_seq(entry.get("bundle") or []) for entry in bundles}
            for entry in bundles[:top_m]:
                seq = entry.get("bundle") or []
                if len(seq) != 2:
                    continue
                seq_rev = [seq[1], seq[0]]
                key_seq = self._bundle_key_seq(seq_rev)
                if key_seq in existing_keys:
                    continue
                rev_entry = _build_entry(seq_rev, count_prune=False, check_exclusion=True)
                if rev_entry is None:
                    continue
                bundles.append(rev_entry)
                existing_keys.add(key_seq)
                permutation_injected += 1
                if len(bundles) >= self.max_kept:
                    break

        event = {
            "combos_generated": combos_generated,
            "bundles_kept": len(bundles),
            "excluded_bundles": excluded_bundles,
            "hotel_pruned": hotel_pruned,
            "attraction_pruned": attraction_pruned,
            "meal_pruned": meal_pruned,
            "return_pruned": return_pruned,
            "permutation_injected": permutation_injected,
        }
        return bundles, event

    def _bundle_key(self, seq: List[str]) -> Tuple[str, ...]:
        return self._bundle_key_set(seq)

    def _bundle_key_set(self, seq: List[str]) -> Tuple[str, ...]:
        return tuple(sorted(self.kb._normalize_city(c) for c in seq if c))

    def _bundle_key_seq(self, seq: List[str]) -> Tuple[str, ...]:
        return tuple(self.kb._normalize_city(c) for c in seq if c)

    @staticmethod
    def _compute_day_splits(total_days: int, seq: List[str]) -> List[Dict[str, Any]]:
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
        if total_days and mapping:
            last = None
            for d in range(1, total_days + 1):
                if d in mapping:
                    last = mapping[d]
                elif last:
                    mapping[d] = last
        return mapping

    @staticmethod
    def _planned_nights_by_city(day_to_city: Dict[int, str], total_days: int) -> Dict[str, int]:
        nights_by_city: Dict[str, int] = {}
        for day in range(1, max(1, total_days)):
            city = day_to_city.get(day)
            if not city:
                continue
            nights_by_city[city] = nights_by_city.get(city, 0) + 1
        return nights_by_city

    def _transport_lower_bound(
        self,
        seq: List[str],
        origin: str,
        return_required: bool,
        min_transport_cost_fn: Optional[Callable[[str, str], Optional[float]]],
    ) -> Tuple[Optional[float], List[Tuple[str, str]], Dict[str, Dict[str, Any]]]:
        if min_transport_cost_fn is None:
            return None, [], {}
        edges_norm: List[Tuple[str, str]] = []
        edge_costs: Dict[str, Dict[str, Any]] = {}
        total = 0.0
        edges_raw: List[Tuple[str, str]] = []
        if origin and seq:
            edges_raw.append((origin, seq[0]))
            edges_norm.append((self.kb._normalize_city(origin), self.kb._normalize_city(seq[0])))
        for i in range(1, len(seq)):
            edges_raw.append((seq[i - 1], seq[i]))
            edges_norm.append((self.kb._normalize_city(seq[i - 1]), self.kb._normalize_city(seq[i])))
        if return_required and origin and seq:
            edges_raw.append((seq[-1], origin))
            edges_norm.append((self.kb._normalize_city(seq[-1]), self.kb._normalize_city(origin)))

        for (src_raw, dst_raw), (src_norm, dst_norm) in zip(edges_raw, edges_norm):
            cost = min_transport_cost_fn(src_raw, dst_raw)
            blocked = cost is None
            if cost is None:
                edge_costs[f"{src_raw}->{dst_raw}"] = {"min_cost": None, "blocked": True}
                continue
            try:
                cost_val = float(cost)
            except Exception:
                cost_val = float("inf")
            if cost_val == float("inf"):
                blocked = True
            edge_costs[f"{src_raw}->{dst_raw}"] = {"min_cost": (None if blocked else cost_val), "blocked": blocked}
            if blocked:
                continue
            total += cost_val
        if any(info.get("blocked") for info in edge_costs.values()):
            return None, edges_norm, edge_costs
        return total, edges_norm, edge_costs

    @staticmethod
    def _stay_lower_bound(
        seq: List[str],
        nights_by_city: Dict[str, int],
        profiles: Dict[str, Dict[str, Any]],
    ) -> Optional[float]:
        total = 0.0
        for city in seq:
            nights = nights_by_city.get(city, 0)
            if nights <= 0:
                continue
            profile = profiles.get(city) or {}
            price = profile.get("hotel", {}).get("min_price")
            if price is None:
                return None
            total += float(price) * float(nights)
        return total

    @staticmethod
    def _meal_lower_bound(
        day_to_city: Dict[int, str],
        profiles: Dict[str, Dict[str, Any]],
        people: int,
    ) -> Optional[float]:
        meals_per_day = 3
        total = 0.0
        for day, city in day_to_city.items():
            profile = profiles.get(city) or {}
            min_meal = profile.get("daily", {}).get("min_meal_cost")
            if min_meal is None:
                return None
            total += float(min_meal) * float(people) * float(meals_per_day)
        return total
