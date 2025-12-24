import re
from typing import Any, Dict, List, Optional

HOUSE_RULES = {"parties", "smoking", "children under 10", "pets", "visitors"}
TRANSPORT_MODES = {"flight", "taxi", "self-driving"}
CANON_ROOM_TYPES = {"shared room", "private room", "entire home/apt"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def route_preferences(
    raw_prefs: Optional[List[str]],
    transport_allow: Optional[List[str]] = None,
    transport_forbid: Optional[List[str]] = None,
    people_number: Optional[int] = None,
) -> Dict[str, Any]:
    raw_prefs = raw_prefs or []
    allow = [_norm(x) for x in (transport_allow or [])] or None
    forbid = [_norm(x) for x in (transport_forbid or [])]

    cons = {
        "meal": {"cuisines": [], "must_have": [], "avoid": []},
        "stay": {"house_rules": [], "room_type": [], "min_occupancy": None},
        "transport": {"allow": allow, "forbid": forbid},
        "other": [],
    }
    if people_number:
        try:
            cons["stay"]["min_occupancy"] = int(people_number)
        except Exception:
            cons["stay"]["min_occupancy"] = None

    for p in raw_prefs:
        t = _norm(p)
        if not t:
            continue

        # room type hints
        # Dataset uses: "Shared room", "Private room", "Entire home/apt"
        if "not shared room" in t or ("not" in t and "shared" in t and "room" in t):
            cons["stay"]["room_type"].extend(["private room", "entire home/apt"])
            continue
        if "shared room" in t:
            cons["stay"]["room_type"].append("shared room")
            continue
        if "private room" in t:
            cons["stay"]["room_type"].append("private room")
            continue
        if "entire home/apt" in t or "entire home" in t or "entire room" in t:
            cons["stay"]["room_type"].append("entire home/apt")
            continue

        # house rules → canonical tokens
        if "party" in t:
            cons["stay"]["house_rules"].append("parties")
            continue
        if "children under 10" in t or ("children" in t and "under" in t and "10" in t):
            cons["stay"]["house_rules"].append("children_under_10")
            continue
        if "smoking" in t:
            cons["stay"]["house_rules"].append("smoking")
            continue
        if "pet" in t:
            cons["stay"]["house_rules"].append("pets")
            continue
        if "visitor" in t:
            cons["stay"]["house_rules"].append("visitors")
            continue

        # transport forbid/allow hints
        if "no flight" in t or "without flight" in t:
            cons["transport"]["forbid"].append("flight")
            continue
        if "no self-driving" in t or "no driving" in t or "without driving" in t:
            cons["transport"]["forbid"].append("self-driving")
            continue

        # default: treat as cuisine keyword
        cons["meal"]["cuisines"].append(t)

    cons["stay"]["house_rules"] = sorted(set(cons["stay"]["house_rules"]))
    cons["stay"]["room_type"] = [x for x in sorted(set(cons["stay"]["room_type"])) if x in CANON_ROOM_TYPES]
    cons["meal"]["cuisines"] = [
        x for x in sorted(set(cons["meal"]["cuisines"])) if x and x not in HOUSE_RULES
    ]
    cons["transport"]["forbid"] = sorted(set([x for x in cons["transport"]["forbid"] if x in TRANSPORT_MODES]))

    return cons
