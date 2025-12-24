from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


TP_DAILY_KEYS = (
    "days",
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
)


def _tp_place_str(place: Optional[Dict[str, Any]], city: Optional[str] = None) -> str:
    if not place:
        return "-"
    name = str(place.get("name") or "").strip()
    c = str(place.get("city") or city or "").strip()
    out = f"{name}, {c}".strip().strip(",")
    return out if out else "-"

def _tp_place_with_cost_str(
    place: Optional[Dict[str, Any]],
    *,
    city: Optional[str] = None,
    cost_key: str = "cost",
    cost_label: str = "Cost",
) -> str:
    base = _tp_place_str(place, city=city)
    if base == "-" or not place:
        return "-"
    cost = place.get(cost_key)
    if cost is None:
        return base
    try:
        cost_int = int(round(float(cost)))
    except Exception:
        return base
    return f"{base}; {cost_label}: {cost_int}"


def _tp_attraction_str(attractions: List[Optional[Dict[str, Any]]], city: Optional[str] = None) -> str:
    parts: List[str] = []
    for a in attractions or []:
        if not a:
            continue
        name = str(a.get("name") or "").strip()
        c = str(a.get("city") or city or "").strip()
        s = f"{name}, {c}".strip().strip(",")
        if s:
            parts.append(s)
    if not parts:
        return "-"
    # TravelPlanner evaluators parse attractions with `split(';')[:-1]`, so always
    # end with a trailing ';' to keep the last attraction visible to checks.
    return ";".join(parts) + ";"


def _tp_transport_str(seg: Optional[Dict[str, Any]], src: str, dst: str) -> str:
    if not seg:
        return "-"
    mode = seg.get("mode")
    detail = seg.get("detail", {}) if isinstance(seg, dict) else {}
    if mode == "flight" and isinstance(detail, dict):
        return (
            f"Flight Number: {detail.get('id', '-')}, from {src} to {dst}, "
            f"Departure Time: {detail.get('depart', '-')}, Arrival Time: {detail.get('arrive', '-')}"
        )
    if isinstance(detail, dict):
        if mode:
            mode_txt = str(mode).replace("_", "-")
            label = mode_txt.title()
            base = f"{label}, from {src} to {dst}"
            cost = detail.get("price", detail.get("cost", None))
            if cost is None:
                return base
            try:
                cost_int = int(round(float(cost)))
                return f"{base}, Cost: {cost_int}"
            except Exception:
                return f"{base}, Cost: {cost}"
    return str(seg)


def _segments_with_days(env) -> List[Tuple[int, str, str, Optional[int]]]:
    """
    Map env segments to the day they occur on (1-based), consistent with env._city_for_day.
    Returns list of (seg_idx, src, dst, transport_day).
    """
    state = env.state
    goal = env.goal
    total_days = getattr(goal, "duration_days", None) or getattr(env, "total_days", 0) or 0
    segments = env._segments(state)

    def _find_day_for_dst(dst_city: str) -> Optional[int]:
        for day in range(1, total_days + 1):
            if env._city_for_day(state, day) == dst_city:
                return day
        return None

    out: List[Tuple[int, str, str, Optional[int]]] = []
    for idx, src, dst in segments:
        transport_day: Optional[int] = None
        if dst == getattr(goal, "origin", None) and total_days:
            transport_day = total_days
        else:
            transport_day = _find_day_for_dst(dst)
        if transport_day is None:
            if src == getattr(goal, "origin", None):
                transport_day = 1
            elif dst == getattr(goal, "origin", None) and total_days:
                transport_day = total_days
            else:
                transport_day = min(idx + 1, total_days) if total_days else (idx + 1)
        out.append((idx, src, dst, transport_day))
    return out


def env_to_travelplanner_daily_plan(env) -> List[Dict[str, Any]]:
    state = env.state
    goal = env.goal
    total_days = int(getattr(goal, "duration_days", None) or getattr(env, "total_days", 0) or 0)
    total_days = max(1, total_days)

    seg_by_day: Dict[int, Tuple[str, str, Dict[str, Any]]] = {}
    for idx, src, dst, day in _segments_with_days(env):
        if day is None:
            continue
        seg = state.segment_modes.get(idx)
        if isinstance(seg, dict):
            seg_by_day[int(day)] = (src, dst, seg)

    daily: List[Dict[str, Any]] = []
    for day in range(1, total_days + 1):
        city = env._city_for_day(state, day) or getattr(goal, "destination", None) or "-"

        transport_str = "-"
        current_city = str(city)
        travel_dst: Optional[str] = None
        if day in seg_by_day:
            src, dst, seg = seg_by_day[day]
            transport_str = _tp_transport_str(seg, src, dst)
            current_city = f"from {src} to {dst}"
            travel_dst = str(dst)

        meals = (state.meals.get(day) or {}) if isinstance(getattr(state, "meals", None), dict) else {}
        breakfast = _tp_place_with_cost_str(meals.get("breakfast"), city=city, cost_key="cost", cost_label="Cost")
        lunch = _tp_place_with_cost_str(meals.get("lunch"), city=city, cost_key="cost", cost_label="Cost")
        dinner = _tp_place_with_cost_str(meals.get("dinner"), city=city, cost_key="cost", cost_label="Cost")

        atts_map = (state.attractions.get(day) or {}) if isinstance(getattr(state, "attractions", None), dict) else {}
        atts = [atts_map.get(k) for k in sorted(atts_map.keys())] if isinstance(atts_map, dict) else []
        attraction = _tp_attraction_str(atts, city=city)

        # Evaluation rules:
        # - day!=last requires accommodation not "-"
        # - accommodation city must match the *destination* city on travel days
        accommodation = "-"
        if day != total_days:
            accom_city = travel_dst or str(city)
            stay = None
            if isinstance(getattr(state, "city_stays", None), dict):
                stay = state.city_stays.get(accom_city)
            accommodation = _tp_place_str(stay, city=accom_city)

        rec = {
            "days": day,
            "current_city": str(current_city),
            "transportation": str(transport_str),
            "breakfast": str(breakfast),
            "attraction": str(attraction),
            "lunch": str(lunch),
            "dinner": str(dinner),
            "accommodation": str(accommodation),
        }
        daily.append(rec)

    return daily


def env_to_submission_record(idx: int, query: str, env) -> Dict[str, Any]:
    return {
        "idx": int(idx),
        "query": str(query or ""),
        "plan": env_to_travelplanner_daily_plan(env),
    }
