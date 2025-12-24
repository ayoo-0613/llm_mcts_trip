from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _as_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x) for x in val if x is not None]
    return [str(val)]


def _as_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _as_int(val: Any) -> Optional[int]:
    try:
        if val is None:
            return None
        return int(val)
    except Exception:
        return None


@dataclass
class FlightFilter:
    origin: Optional[str] = None
    destination: Optional[str] = None
    date: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    earliest_depart: Optional[str] = None  # "HH:MM" 24h
    latest_depart: Optional[str] = None
    max_duration: Optional[float] = None  # minutes
    max_stops: Optional[int] = None
    sort_by: str = "price"  # price|duration|depart|arrive
    avoid_ids: List[str] = field(default_factory=list)


@dataclass
class HotelFilter:
    city: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_review: Optional[float] = None
    room_type: List[str] = field(default_factory=list)
    house_rules: List[str] = field(default_factory=list)
    min_occupancy: Optional[int] = None
    sort_by: str = "price"  # price|review
    avoid_ids: List[str] = field(default_factory=list)


@dataclass
class RestaurantFilter:
    city: Optional[str] = None
    cuisines: List[str] = field(default_factory=list)
    max_cost: Optional[float] = None
    min_rating: Optional[float] = None
    meal_type: Optional[str] = None
    sort_by: str = "rating"  # rating|cost
    avoid_ids: List[str] = field(default_factory=list)


@dataclass
class AttractionFilter:
    city: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    max_distance_km: Optional[float] = None
    sort_by: str = "rating"  # rating|distance|name
    avoid_ids: List[str] = field(default_factory=list)


DEFAULT_SORT = {
    "flight": "price",
    "hotel": "price",
    "restaurant": "rating",
    "attraction": "rating",
}


def _allowed_sort(filter_type: str) -> List[str]:
    if filter_type == "flight":
        return ["price", "duration", "depart", "arrive"]
    if filter_type == "hotel":
        return ["price", "review"]
    if filter_type == "restaurant":
        return ["rating", "cost"]
    if filter_type == "attraction":
        return ["rating", "distance", "name"]
    return []


def default_filter(filter_type: str, goal=None, state=None, slot=None) -> Dict[str, Any]:
    """Return a safe default filter for the given type."""
    filter_type = (filter_type or "").lower()
    if filter_type == "meal":
        filter_type = "restaurant"
    if filter_type == "flight":
        origin = getattr(slot, "origin", None) or getattr(slot, "from_city", None)
        destination = getattr(slot, "destination", None) or getattr(slot, "to_city", None)
        return FlightFilter(
            origin=origin,
            destination=destination,
            date=getattr(slot, "date", None) or getattr(goal, "start_date", None),
            max_price=getattr(goal, "budget", None),
            sort_by="price",
        ).__dict__
    if filter_type == "hotel":
        city = getattr(slot, "city", None) or getattr(goal, "destination", None)
        room_type: List[str] = []
        house_rules: List[str] = []
        if goal and hasattr(goal, "constraints"):
            stay_cons = (goal.constraints.get("stay", {}) or {}) if isinstance(goal.constraints, dict) else {}
            room_type = list(stay_cons.get("room_type") or [])
            house_rules = list(stay_cons.get("house_rules") or [])
        return HotelFilter(
            city=city,
            max_price=getattr(goal, "budget", None),
            room_type=room_type,
            house_rules=house_rules,
            min_occupancy=getattr(goal, "people_number", None),
            sort_by="price",
        ).__dict__
    if filter_type == "restaurant":
        city = getattr(slot, "city", None) or getattr(goal, "destination", None)
        cuisines = []
        if goal and hasattr(goal, "constraints"):
            cuisines = (goal.constraints.get("meal", {}) or {}).get("cuisines", [])
        return RestaurantFilter(
            city=city,
            cuisines=cuisines,
            max_cost=None,
            sort_by="rating",
        ).__dict__
    if filter_type == "attraction":
        city = getattr(slot, "city", None) or getattr(goal, "destination", None)
        return AttractionFilter(city=city, sort_by="rating").__dict__
    return {}


def validate_and_normalize(
    filt: Optional[Dict[str, Any]],
    filter_type: str,
    goal=None,
    state=None,
    slot=None,
) -> Dict[str, Any]:
    """Validate LLM output and fall back to defaults on errors."""
    filter_type = (filter_type or "").lower()
    if filter_type == "meal":
        filter_type = "restaurant"
    base = default_filter(filter_type, goal, state, slot)
    if not isinstance(filt, dict):
        return base

    clean: Dict[str, Any] = copy.deepcopy(base)

    if filter_type == "flight":
        clean["origin"] = str(filt.get("origin") or clean.get("origin") or "")
        clean["destination"] = str(filt.get("destination") or clean.get("destination") or "")
        clean["date"] = filt.get("date") or clean.get("date")
        clean["max_price"] = _as_float(filt.get("max_price")) or clean.get("max_price")
        clean["min_price"] = _as_float(filt.get("min_price"))
        clean["earliest_depart"] = filt.get("earliest_depart") or clean.get("earliest_depart")
        clean["latest_depart"] = filt.get("latest_depart") or clean.get("latest_depart")
        clean["max_duration"] = _as_float(filt.get("max_duration"))
        clean["max_stops"] = _as_int(filt.get("max_stops"))
        sort = str(filt.get("sort_by") or clean.get("sort_by") or DEFAULT_SORT["flight"]).lower()
        clean["sort_by"] = sort if sort in _allowed_sort("flight") else DEFAULT_SORT["flight"]
        clean["avoid_ids"] = list({str(x) for x in _as_list(filt.get("avoid_ids"))})
    elif filter_type == "hotel":
        clean["city"] = filt.get("city") or clean.get("city")
        clean["max_price"] = _as_float(filt.get("max_price")) or clean.get("max_price")
        clean["min_price"] = _as_float(filt.get("min_price"))
        clean["min_review"] = _as_float(filt.get("min_review"))
        clean["room_type"] = [s.lower() for s in _as_list(filt.get("room_type")) if s]
        clean["house_rules"] = [s.lower().replace(" ", "_") for s in _as_list(filt.get("house_rules")) if s]
        clean["min_occupancy"] = _as_int(filt.get("min_occupancy")) or clean.get("min_occupancy")
        sort = str(filt.get("sort_by") or clean.get("sort_by") or DEFAULT_SORT["hotel"]).lower()
        clean["sort_by"] = sort if sort in _allowed_sort("hotel") else DEFAULT_SORT["hotel"]
        clean["avoid_ids"] = list({str(x) for x in _as_list(filt.get("avoid_ids"))})
    elif filter_type == "restaurant":
        clean["city"] = filt.get("city") or clean.get("city")
        clean["cuisines"] = [s.lower() for s in _as_list(filt.get("cuisines")) if s]
        clean["max_cost"] = _as_float(filt.get("max_cost"))
        clean["min_rating"] = _as_float(filt.get("min_rating"))
        clean["meal_type"] = filt.get("meal_type") or clean.get("meal_type")
        sort = str(filt.get("sort_by") or clean.get("sort_by") or DEFAULT_SORT["restaurant"]).lower()
        clean["sort_by"] = sort if sort in _allowed_sort("restaurant") else DEFAULT_SORT["restaurant"]
        clean["avoid_ids"] = list({str(x) for x in _as_list(filt.get("avoid_ids"))})
    elif filter_type == "attraction":
        clean["city"] = filt.get("city") or clean.get("city")
        clean["categories"] = [s.lower() for s in _as_list(filt.get("categories")) if s]
        clean["max_distance_km"] = _as_float(filt.get("max_distance_km"))
        sort = str(filt.get("sort_by") or clean.get("sort_by") or DEFAULT_SORT["attraction"]).lower()
        clean["sort_by"] = sort if sort in _allowed_sort("attraction") else DEFAULT_SORT["attraction"]
        clean["avoid_ids"] = list({str(x) for x in _as_list(filt.get("avoid_ids"))})
    else:
        return base

    return clean
