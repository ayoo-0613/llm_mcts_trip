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
    # TravelPlanner evaluation parses meals as "Name, City" (no trailing cost info).
    return _tp_place_str(place, city=city)


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
            mode_txt = str(mode).replace("_", "-").lower()
            if mode_txt == "self-driving":
                label = "Self-driving"
            else:
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


def _env_parsed_get(env, *keys: str, default: Any = None) -> Any:
    parsed = getattr(env, "goal_parsed", None) or {}
    if isinstance(parsed, dict):
        for key in keys:
            if key in parsed and parsed[key] is not None:
                return parsed[key]
    return default


def _segments_with_days(env) -> List[Tuple[int, str, str, Optional[int]]]:
    """
    Map env segments to the day they occur on (1-based), consistent with env._city_for_day.
    Returns list of (seg_idx, src, dst, transport_day).
    """
    state = env.state
    total_days = _env_parsed_get(env, "duration_days", "days", default=None) or getattr(env, "total_days", 0) or 0
    segments = env._segments(state)

    def _find_day_for_dst(dst_city: str) -> Optional[int]:
        for day in range(1, total_days + 1):
            if env._city_for_day(state, day) == dst_city:
                return day
        return None

    out: List[Tuple[int, str, str, Optional[int]]] = []
    for idx, src, dst in segments:
        transport_day: Optional[int] = None
        if dst == _env_parsed_get(env, "origin", "org", default=None) and total_days:
            transport_day = total_days
        else:
            transport_day = _find_day_for_dst(dst)
        if transport_day is None:
            if src == _env_parsed_get(env, "origin", "org", default=None):
                transport_day = 1
            elif dst == _env_parsed_get(env, "origin", "org", default=None) and total_days:
                transport_day = total_days
            else:
                transport_day = min(idx + 1, total_days) if total_days else (idx + 1)
        out.append((idx, src, dst, transport_day))
    return out


def env_to_travelplanner_daily_plan(env) -> List[Dict[str, Any]]:
    state = env.state
    total_days = int(_env_parsed_get(env, "duration_days", "days", default=None) or getattr(env, "total_days", 0) or 0)
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
        city = env._city_for_day(state, day) or _env_parsed_get(env, "destination", "dest", default=None) or "-"

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


def _format_time(minutes: int) -> str:
    minutes = int(minutes) % (24 * 60)
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _parse_flight_time(transportation: str, *, key: str) -> Optional[str]:
    # key: "Departure Time" | "Arrival Time"
    if not transportation:
        return None
    text = str(transportation)
    m = None
    try:
        import re

        m = re.search(rf"{re.escape(key)}:\\s*([0-9]{{1,2}}:[0-9]{{2}})", text, flags=re.IGNORECASE)
    except Exception:
        m = None
    if not m:
        return None
    return m.group(1)


def _hhmm_to_minutes(hhmm: str) -> Optional[int]:
    if not hhmm:
        return None
    try:
        h, m = hhmm.strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _tripcraft_place(place: Optional[Dict[str, Any]], *, city: Optional[str] = None) -> str:
    if not place:
        return "-"
    name = str(place.get("name") or "").strip()
    c = str(place.get("city") or city or "").strip()
    if name and c:
        return f"{name}, {c}"
    return name or "-"


def _tripcraft_attraction_str(attractions: List[Optional[Dict[str, Any]]], *, city: Optional[str] = None) -> str:
    # TripCraft feasibility scripts expect attractions to be split by ';' (no trailing ';')
    # and typically include city for sandbox validation.
    parts: List[str] = []
    for a in attractions or []:
        if not a:
            continue
        parts.append(_tripcraft_place(a, city=city))
    return "; ".join([p for p in parts if p and p != "-"]) if parts else "-"


def _poi_entry(
    *,
    kb: Any,
    kind: str,
    name: str,
    city: str,
    start_min: int,
    end_min: int,
) -> str:
    verb = "stay" if kind == "stay" else "visit"
    base = f"{name}, {verb} from {_format_time(start_min)} to {_format_time(end_min)}"
    if kb is None or not hasattr(kb, "nearest_transit"):
        return base
    transit = None
    try:
        transit = kb.nearest_transit(city=city, poi=name)
    except Exception:
        transit = None
    if not transit:
        return base
    stop = transit.get("nearest_stop_name")
    dist = transit.get("nearest_stop_distance")
    try:
        dist_f = float(dist)
    except Exception:
        dist_f = None
    if stop and dist_f is not None:
        return f"{base}, nearest transit: {stop}, {dist_f:.2f}m away"
    if stop:
        return f"{base}, nearest transit: {stop}"
    return base


def env_to_tripcraft_daily_plan(env) -> List[Dict[str, Any]]:
    """
    TripCraft expects additional keys per day:
    - event
    - point_of_interest_list

    This exporter constructs a heuristic daily PoI timeline consistent with the
    selected meals/attractions/accommodations and attaches nearest-transit info
    if the KB provides it.
    """
    kb = getattr(env, "kb", None)
    state = env.state
    total_days = int(_env_parsed_get(env, "duration_days", "days", default=None) or getattr(env, "total_days", 0) or 0)
    total_days = max(1, total_days)

    seg_by_day: Dict[int, Tuple[str, str, Dict[str, Any]]] = {}
    for idx, src, dst, day in _segments_with_days(env):
        if day is None:
            continue
        seg = state.segment_modes.get(idx)
        if isinstance(seg, dict):
            seg_by_day[int(day)] = (src, dst, seg)

    daily: List[Dict[str, Any]] = []
    prev_accommodation_name: Optional[str] = None
    prev_accommodation_city: Optional[str] = None

    for day in range(1, total_days + 1):
        city = env._city_for_day(state, day) or _env_parsed_get(env, "destination", "dest", default=None) or "-"

        transport_str = "-"
        current_city = str(city)
        travel_dst: Optional[str] = None
        travel_src: Optional[str] = None
        if day in seg_by_day:
            src, dst, seg = seg_by_day[day]
            transport_str = _tp_transport_str(seg, src, dst)
            current_city = f"from {src} to {dst}"
            travel_src = str(src)
            travel_dst = str(dst)

        meals = (state.meals.get(day) or {}) if isinstance(getattr(state, "meals", None), dict) else {}
        breakfast_place = meals.get("breakfast")
        lunch_place = meals.get("lunch")
        dinner_place = meals.get("dinner")

        breakfast = _tripcraft_place(breakfast_place, city=city)
        lunch = _tripcraft_place(lunch_place, city=city)
        dinner = _tripcraft_place(dinner_place, city=city)

        atts_map = (state.attractions.get(day) or {}) if isinstance(getattr(state, "attractions", None), dict) else {}
        atts = [atts_map.get(k) for k in sorted(atts_map.keys())] if isinstance(atts_map, dict) else []
        attraction = _tripcraft_attraction_str(atts, city=city)

        accommodation = "-"
        accom_city = travel_dst or str(city)
        stay = None
        if day != total_days:
            if isinstance(getattr(state, "city_stays", None), dict):
                stay = state.city_stays.get(accom_city)
            accommodation = _tripcraft_place(stay, city=accom_city)

        # ----------------------------
        # Build point_of_interest_list
        # ----------------------------
        # TripCraft prompt + evaluators assume:
        # - PoI list is an ordered timeline
        # - contains only accommodation/restaurants/attractions
        # - includes "stay/visit from HH:MM to HH:MM"
        # - day starts/ends at accommodation when accommodation is provided
        pois: List[str] = []

        # TripCraft requires:
        # - normal day: PoI list starts/ends with (current) accommodation if provided
        # - transition day (handled in their evaluator): day3/day5 start with prev accommodation and end with current
        # We approximate by:
        # - always starting with prev accommodation if this is a travel day in 5/7d and we have it
        # - otherwise starting with current day's accommodation if available
        start_anchor_name = None
        start_anchor_city = None

        if prev_accommodation_name and prev_accommodation_city and travel_dst and total_days in (5, 7) and day in (3, 5):
            start_anchor_name = prev_accommodation_name
            start_anchor_city = prev_accommodation_city
        elif accommodation != "-" and "," in accommodation:
            start_anchor_name = accommodation.rsplit(",", 1)[0].strip()
            start_anchor_city = accommodation.rsplit(",", 1)[1].strip()
        elif prev_accommodation_name and prev_accommodation_city:
            start_anchor_name = prev_accommodation_name
            start_anchor_city = prev_accommodation_city

        # Time anchors
        base_start = 9 * 60
        arr = _parse_flight_time(transport_str, key="Arrival Time")
        arr_min = _hhmm_to_minutes(arr) if arr else None
        if day == 1 and arr_min is not None:
            base_start = max(base_start, arr_min + 30)

        # Start accommodation stay window
        if start_anchor_name and start_anchor_city:
            pois.append(
                _poi_entry(
                    kb=kb,
                    kind="stay",
                    name=start_anchor_name,
                    city=start_anchor_city,
                    start_min=base_start,
                    end_min=base_start + 30,
                )
            )

        t = base_start + 40

        def _add_meal(place_str: str, minutes: int) -> Optional[Tuple[str, str]]:
            nonlocal t
            if place_str == "-" or "," not in place_str:
                return None
            nm, ct = place_str.rsplit(",", 1)
            nm = nm.strip()
            ct = ct.strip()
            start = max(t, minutes)
            end = start + 45
            pois.append(_poi_entry(kb=kb, kind="visit", name=nm, city=ct, start_min=start, end_min=end))
            t = end + 30
            return (nm, ct)

        def _parse_attractions(attraction_str: str) -> List[Tuple[str, str]]:
            if not attraction_str or attraction_str == "-":
                return []
            out: List[Tuple[str, str]] = []
            items = [x.strip() for x in attraction_str.split(";") if x.strip()]
            for item in items:
                nm = item
                ct = str(city)
                if "," in item:
                    nm, ct = item.rsplit(",", 1)
                    nm = nm.strip()
                    ct = ct.strip()
                nm = nm.strip()
                ct = ct.strip()
                if not nm:
                    continue
                out.append((nm, ct or str(city)))
            # de-dup while preserving order
            seen = set()
            dedup: List[Tuple[str, str]] = []
            for nm, ct in out:
                key = (nm.lower(), ct.lower())
                if key in seen:
                    continue
                seen.add(key)
                dedup.append((nm, ct))
            return dedup

        def _add_attraction_items(items: List[Tuple[str, str]], start_hint: int) -> None:
            nonlocal t
            if not items:
                return
            start = max(t, start_hint)
            for nm, ct in items:
                end = start + 90
                pois.append(_poi_entry(kb=kb, kind="visit", name=nm, city=ct, start_min=start, end_min=end))
                start = end + 30
            t = start

        attraction_items = _parse_attractions(attraction)
        # Put the first attraction block before lunch and the rest after lunch (no duplication).
        pre = attraction_items[:1]
        post = attraction_items[1:]

        _add_meal(breakfast, 10 * 60)
        _add_attraction_items(pre, 11 * 60 + 30)
        _add_meal(lunch, 14 * 60)
        _add_attraction_items(post, 15 * 60 + 30)
        _add_meal(dinner, 19 * 60 + 30)

        # End accommodation (if day has accommodation)
        if accommodation != "-" and "," in accommodation:
            nm, ct = accommodation.rsplit(",", 1)
            nm = nm.strip()
            ct = ct.strip()
            end_start = max(t, 22 * 60)
            pois.append(_poi_entry(kb=kb, kind="stay", name=nm, city=ct, start_min=end_start, end_min=8 * 60))

        # Match TripCraft examples: semicolon-separated with spaces, ends with a period.
        point_of_interest_list = "; ".join([p for p in pois if p]).strip()
        if point_of_interest_list and not point_of_interest_list.endswith("."):
            point_of_interest_list += "."

        rec = {
            "days": day,
            "current_city": str(current_city),
            "transportation": str(transport_str),
            "breakfast": str(breakfast),
            "attraction": str(attraction),
            "lunch": str(lunch),
            "dinner": str(dinner),
            "accommodation": str(accommodation),
            "event": "-",
            "point_of_interest_list": str(point_of_interest_list),
        }
        daily.append(rec)

        # Keep track of last non-empty accommodation for transition-day anchoring.
        if accommodation != "-" and "," in accommodation:
            prev_accommodation_name = accommodation.rsplit(",", 1)[0].strip()
            prev_accommodation_city = accommodation.rsplit(",", 1)[1].strip()

    return daily


def env_to_tripcraft_record(idx: int, query_json: Dict[str, Any], persona: str, env) -> Dict[str, Any]:
    return {
        "idx": int(idx),
        "JSON": dict(query_json or {}),
        "persona": str(persona or ""),
        "plan": env_to_tripcraft_daily_plan(env),
    }


def env_to_submission_record(idx: int, query: str, env) -> Dict[str, Any]:
    return {
        "idx": int(idx),
        "query": str(query or ""),
        "plan": env_to_travelplanner_daily_plan(env),
    }
