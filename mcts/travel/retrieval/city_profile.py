from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze(v) for v in value)
    return value


class CityProfileBuilder:
    """Rule-based city feature builder for feasibility and cost lower-bounds."""

    def __init__(self, kb: Any):
        self.kb = kb
        self._cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def build(
        self,
        city: str,
        constraints: Dict[str, Any],
        *,
        people: int = 1,
        pool: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        if not city:
            return {
                "city": city,
                "hotel": {"exists": False, "min_price": None, "min_nights_min": None},
                "daily": {"meal_count": 0, "attraction_count": 0, "min_meal_cost": None},
            }

        city_norm = self.kb._normalize_city(city)
        key = (city_norm, max(1, int(people or 1)), _freeze(constraints or {}))
        cached = self._cache.get(key)
        if cached is not None:
            return dict(cached)

        stays = None
        rests = None
        atts = None
        if pool:
            stays = list(pool.get("hotel") or [])
            rests = list(pool.get("meal") or [])
            atts = list(pool.get("attraction") or [])
        if stays is None:
            stays = self._filter_stays(city_norm, constraints)
        if rests is None:
            rests = self._filter_restaurants(city_norm, constraints)
        if atts is None:
            atts = self._filter_attractions(city_norm)

        people = max(1, int(people or 1))
        min_price = None
        min_nights = None
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
            if min_price is None or total_per_night < min_price:
                min_price = total_per_night
            mn = stay.get("minimum_nights")
            if mn is not None:
                try:
                    mn_i = int(mn)
                except Exception:
                    mn_i = None
                if mn_i is not None:
                    if min_nights is None or mn_i < min_nights:
                        min_nights = mn_i

        min_meal_cost = None
        for rest in rests:
            cost = rest.get("cost")
            if cost is None:
                continue
            try:
                cost_f = float(cost)
            except Exception:
                continue
            if min_meal_cost is None or cost_f < min_meal_cost:
                min_meal_cost = cost_f

        profile = {
            "city": city,
            "hotel": {
                "exists": bool(stays),
                "min_price": min_price,
                "min_nights_min": min_nights,
            },
            "daily": {
                "meal_count": len(rests),
                "attraction_count": len(atts),
                "min_meal_cost": min_meal_cost,
            },
        }
        self._cache[key] = dict(profile)
        return profile

    def _filter_stays(self, city_norm: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        stays = list(getattr(self.kb, "_accommodation_buckets", {}).get(city_norm, []))
        room_types = constraints.get("room_types") or []
        house_rules = constraints.get("house_rules") or []
        min_occ = constraints.get("min_occupancy")
        out: List[Dict[str, Any]] = []
        for stay in stays:
            price = stay.get("price")
            if price is None:
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
        return out

    def _filter_restaurants(self, city_norm: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        restaurants = list(getattr(self.kb, "_restaurant_buckets", {}).get(city_norm, []))
        cuisines = [str(c).strip().lower() for c in (constraints.get("cuisines") or []) if c]
        out: List[Dict[str, Any]] = []
        for rest in restaurants:
            if rest.get("cost") is None:
                continue
            if cuisines:
                text = str(rest.get("cuisines_lc") or rest.get("cuisines") or "").lower()
                if not any(c in text for c in cuisines):
                    continue
            out.append(rest)
        return out

    def _filter_attractions(self, city_norm: str) -> List[Dict[str, Any]]:
        return list(getattr(self.kb, "_attraction_buckets", {}).get(city_norm, []))

    @staticmethod
    def _room_type_matches(room_type: Optional[str], allowed: Iterable[str]) -> bool:
        allowed_list = [str(a).strip().lower() for a in allowed if a]
        if not allowed_list:
            return True
        rt = str(room_type or "").strip().lower()
        if not rt:
            return False
        normalized: List[str] = []
        for raw in allowed_list:
            if not raw:
                continue
            if raw in ("entire room", "entire home", "entire home/apt"):
                normalized.append("entire home/apt")
            elif raw == "not shared room":
                normalized.extend(["private room", "entire home/apt"])
            else:
                normalized.append(raw)
        return rt in set(normalized)

    @staticmethod
    def _violates_house_rule(rule: str, stay: Dict[str, Any]) -> bool:
        if not rule:
            return False
        rule = str(rule).strip().lower()
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
