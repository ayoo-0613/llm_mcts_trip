from __future__ import annotations

from typing import Any, Dict, FrozenSet, List

ENV_SPEC_VERSION = "envspec.v1"

TOP_LEVEL_KEYS: FrozenSet[str] = frozenset(
    {
        "version",
        "goal",
        "constraints",
        "retrieval",
        "reward_cfg_overrides",
    }
)

TRANSPORT_MODES: FrozenSet[str] = frozenset({"flight", "taxi", "self-driving"})

# Keep these aligned with the TravelPlanner query parsing prompt.
HOUSE_RULES: FrozenSet[str] = frozenset({"parties", "smoking", "children under 10", "pets", "visitors"})
ROOM_TYPES: FrozenSet[str] = frozenset({"entire room", "private room", "shared room", "not shared room"})
CUISINES: FrozenSet[str] = frozenset({"Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", "French"})

# Contract: meal slots are fixed in the environment; EnvSpec must not override them.
MEAL_SLOTS: List[str] = ["breakfast", "lunch", "dinner"]


def envspec_skeleton() -> Dict[str, Any]:
    """
    Stable JSON skeleton for EnvSpec v1. LLM should only fill values, not add fields.

    Notes:
    - `constraints` may be null, but the key must exist for stability.
    - `reward_cfg_overrides` is optional; keep as {} when unused.
    """
    return {
        "version": ENV_SPEC_VERSION,
        "goal": {
            "origin": None,
            "destination": None,
            "start_date": None,
            "duration_days": None,
            "budget": None,
            "people_number": 1,
            "visiting_city_number": 1,
            "preferences": [],
            "must_visit_cities": [],
            "fixed_city_order": [],
            "priority_cities": [],
        },
        "constraints": {
            "transport": {"allow": None, "forbid": None},
            "meal": {"cuisines": None},
            "stay": {"room_type": None, "house_rule": None},
            "daily": {"attractions_per_day_min": None, "attractions_per_day_max": None},
            "city": {"candidate_cities": None},
        },
        "retrieval": {"top_k": None, "candidate_cap": None},
        "reward_cfg_overrides": {},
    }

