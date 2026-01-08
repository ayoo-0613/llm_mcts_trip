from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple

from mcts.travel.envspec.schema import (
    CUISINES,
    ENV_SPEC_VERSION,
    HOUSE_RULES,
    ROOM_TYPES,
    TOP_LEVEL_KEYS,
    TRANSPORT_MODES,
    envspec_skeleton,
)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_GOAL_KEYS = frozenset(envspec_skeleton().get("goal", {}).keys())
_CONSTRAINT_KEYS = frozenset(envspec_skeleton().get("constraints", {}).keys())
_TRANSPORT_KEYS = frozenset(envspec_skeleton().get("constraints", {}).get("transport", {}).keys())
_MEAL_KEYS = frozenset(envspec_skeleton().get("constraints", {}).get("meal", {}).keys())
_STAY_KEYS = frozenset(envspec_skeleton().get("constraints", {}).get("stay", {}).keys())
_DAILY_KEYS = frozenset(envspec_skeleton().get("constraints", {}).get("daily", {}).keys())
_CITY_KEYS = frozenset(envspec_skeleton().get("constraints", {}).get("city", {}).keys())
_RETRIEVAL_KEYS = frozenset(envspec_skeleton().get("retrieval", {}).keys())


def validate_envspec(spec: dict) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(spec, dict):
        return False, ["spec must be a dict"]

    extra = [k for k in spec.keys() if k not in TOP_LEVEL_KEYS]
    if extra:
        errors.append(f"unknown top-level keys: {sorted(extra)}")

    version = spec.get("version")
    if version != ENV_SPEC_VERSION:
        errors.append(f"version must be '{ENV_SPEC_VERSION}'")

    goal = spec.get("goal")
    if not isinstance(goal, dict):
        errors.append("goal must be an object")
        return False, errors
    extra_goal = [k for k in goal.keys() if k not in _GOAL_KEYS]
    if extra_goal:
        errors.append(f"unknown goal keys: {sorted(extra_goal)}")

    if not str(goal.get("origin") or "").strip():
        errors.append("goal.origin required")
    if not str(goal.get("destination") or "").strip():
        errors.append("goal.destination required")

    days = goal.get("duration_days")
    if days is None:
        errors.append("goal.duration_days required")
    else:
        try:
            days_i = int(days)
            if days_i <= 0:
                errors.append("goal.duration_days must be > 0")
        except Exception:
            errors.append("goal.duration_days must be an integer")

    start_date = goal.get("start_date")
    if start_date is not None and str(start_date).strip():
        if not _DATE_RE.match(str(start_date).strip()):
            errors.append("goal.start_date must be 'YYYY-MM-DD' or null")

    people = goal.get("people_number")
    if people is not None:
        try:
            people_i = int(people)
            if people_i <= 0:
                errors.append("goal.people_number must be > 0")
        except Exception:
            errors.append("goal.people_number must be an integer")

    budget = goal.get("budget")
    if budget is not None and str(budget).strip():
        try:
            float(budget)
        except Exception:
            errors.append("goal.budget must be a number or null")

    def _check_str_list(path: str, val: Any) -> None:
        if val is None:
            return
        if not isinstance(val, list):
            errors.append(f"{path} must be a list or null")
            return
        for i, v in enumerate(val):
            if not str(v or "").strip():
                errors.append(f"{path}[{i}] must be a non-empty string")

    _check_str_list("goal.preferences", goal.get("preferences"))
    _check_str_list("goal.must_visit_cities", goal.get("must_visit_cities"))
    _check_str_list("goal.fixed_city_order", goal.get("fixed_city_order"))
    _check_str_list("goal.priority_cities", goal.get("priority_cities"))

    constraints = spec.get("constraints")
    if constraints is not None and not isinstance(constraints, dict):
        errors.append("constraints must be an object or null")
    if isinstance(constraints, dict):
        extra_blocks = [k for k in constraints.keys() if k not in _CONSTRAINT_KEYS]
        if extra_blocks:
            errors.append(f"unknown constraints blocks: {sorted(extra_blocks)}")

        transport = constraints.get("transport") or {}
        if not isinstance(transport, dict):
            errors.append("constraints.transport must be an object")
        else:
            extra_t = [k for k in transport.keys() if k not in _TRANSPORT_KEYS]
            if extra_t:
                errors.append(f"unknown constraints.transport keys: {sorted(extra_t)}")
            allow = transport.get("allow")
            forbid = transport.get("forbid")
            if allow is not None:
                _check_str_list("constraints.transport.allow", allow)
                if isinstance(allow, list):
                    bad = [m for m in allow if str(m).lower() not in TRANSPORT_MODES]
                    if bad:
                        errors.append(f"constraints.transport.allow invalid modes: {sorted(set(bad))}")
            if forbid is not None:
                _check_str_list("constraints.transport.forbid", forbid)
                if isinstance(forbid, list):
                    bad = [m for m in forbid if str(m).lower() not in TRANSPORT_MODES]
                    if bad:
                        errors.append(f"constraints.transport.forbid invalid modes: {sorted(set(bad))}")

        meal = constraints.get("meal") or {}
        if not isinstance(meal, dict):
            errors.append("constraints.meal must be an object")
        else:
            extra_m = [k for k in meal.keys() if k not in _MEAL_KEYS]
            if extra_m:
                errors.append(f"unknown constraints.meal keys: {sorted(extra_m)}")
            cuisines = meal.get("cuisines")
            if cuisines is not None:
                _check_str_list("constraints.meal.cuisines", cuisines)
                if isinstance(cuisines, list):
                    bad = [c for c in cuisines if str(c) not in CUISINES]
                    if bad:
                        errors.append(f"constraints.meal.cuisines invalid: {sorted(set(bad))}")

        stay = constraints.get("stay") or {}
        if not isinstance(stay, dict):
            errors.append("constraints.stay must be an object")
        else:
            extra_s = [k for k in stay.keys() if k not in _STAY_KEYS]
            if extra_s:
                errors.append(f"unknown constraints.stay keys: {sorted(extra_s)}")
            room_type = stay.get("room_type")
            if room_type is not None and str(room_type).strip():
                if str(room_type) not in ROOM_TYPES:
                    errors.append(f"constraints.stay.room_type invalid: {room_type}")
            house_rule = stay.get("house_rule")
            if house_rule is not None and str(house_rule).strip():
                if str(house_rule) not in HOUSE_RULES:
                    errors.append(f"constraints.stay.house_rule invalid: {house_rule}")

        daily = constraints.get("daily") or {}
        if not isinstance(daily, dict):
            errors.append("constraints.daily must be an object")
        else:
            extra_d = [k for k in daily.keys() if k not in _DAILY_KEYS]
            if extra_d:
                errors.append(f"unknown constraints.daily keys: {sorted(extra_d)}")
            for k in ("attractions_per_day_min", "attractions_per_day_max"):
                v = daily.get(k)
                if v is None:
                    continue
                try:
                    int(v)
                except Exception:
                    errors.append(f"constraints.daily.{k} must be an integer or null")

        city = constraints.get("city") or {}
        if not isinstance(city, dict):
            errors.append("constraints.city must be an object")
        else:
            extra_c = [k for k in city.keys() if k not in _CITY_KEYS]
            if extra_c:
                errors.append(f"unknown constraints.city keys: {sorted(extra_c)}")
            _check_str_list("constraints.city.candidate_cities", city.get("candidate_cities"))

    retrieval = spec.get("retrieval")
    if retrieval is not None and not isinstance(retrieval, dict):
        errors.append("retrieval must be an object or null")
    if isinstance(retrieval, dict):
        extra_r = [k for k in retrieval.keys() if k not in _RETRIEVAL_KEYS]
        if extra_r:
            errors.append(f"unknown retrieval keys: {sorted(extra_r)}")
        for k in ("top_k", "candidate_cap"):
            v = retrieval.get(k)
            if v is None:
                continue
            try:
                int(v)
            except Exception:
                errors.append(f"retrieval.{k} must be an integer or null")

    reward_overrides = spec.get("reward_cfg_overrides")
    if reward_overrides is not None and not isinstance(reward_overrides, dict):
        errors.append("reward_cfg_overrides must be an object or null")
    if isinstance(reward_overrides, dict) and reward_overrides:
        try:
            from mcts.travel.env_agent import DEFAULT_REWARD_CFG as _DEFAULT_REWARD_CFG  # local import
        except Exception:
            _DEFAULT_REWARD_CFG = None
        if isinstance(_DEFAULT_REWARD_CFG, dict):
            extra_keys = [k for k in reward_overrides.keys() if k not in _DEFAULT_REWARD_CFG]
            if extra_keys:
                errors.append(f"reward_cfg_overrides unknown keys: {sorted(extra_keys)}")
        for k, v in reward_overrides.items():
            if v is None:
                continue
            try:
                float(v)
            except Exception:
                errors.append(f"reward_cfg_overrides.{k} must be a number or null")

    return (len(errors) == 0), errors


def normalize_envspec(spec: dict) -> dict:
    """
    Normalize EnvSpec into a stable, schema-shaped dict.

    - Fills missing top-level blocks with defaults.
    - Coerces basic scalar types where safe.
    - Does not attempt to infer missing origin/destination (validation enforces).
    """
    base = envspec_skeleton()
    if not isinstance(spec, dict):
        return base

    out = copy.deepcopy(base)
    for k in TOP_LEVEL_KEYS:
        if k in spec:
            out[k] = copy.deepcopy(spec.get(k))

    out["version"] = ENV_SPEC_VERSION

    goal = out.get("goal") if isinstance(out.get("goal"), dict) else {}
    goal_norm = copy.deepcopy(base["goal"])
    for k in goal_norm:
        if k in goal:
            goal_norm[k] = goal.get(k)

    for k in ("origin", "destination", "start_date"):
        v = goal_norm.get(k)
        goal_norm[k] = str(v).strip() if v is not None and str(v).strip() else None

    for k, default in (("duration_days", None), ("people_number", 1), ("visiting_city_number", 1)):
        v = goal_norm.get(k)
        if v is None:
            goal_norm[k] = default
            continue
        try:
            goal_norm[k] = int(v)
        except Exception:
            goal_norm[k] = default

    budget = goal_norm.get("budget")
    if budget is None or (isinstance(budget, str) and not budget.strip()):
        goal_norm["budget"] = None
    else:
        try:
            goal_norm["budget"] = float(budget)
        except Exception:
            goal_norm["budget"] = None

    def _norm_list(val: Any) -> List[str]:
        if val is None:
            return []
        if not isinstance(val, list):
            val = [val]
        out_list: List[str] = []
        seen = set()
        for v in val:
            s = str(v or "").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out_list.append(s)
        return out_list

    for k in ("preferences", "must_visit_cities", "fixed_city_order", "priority_cities"):
        goal_norm[k] = _norm_list(goal_norm.get(k))

    out["goal"] = goal_norm

    constraints = out.get("constraints")
    if constraints is None:
        constraints = {}
    if not isinstance(constraints, dict):
        constraints = {}
    cons_norm = copy.deepcopy(base["constraints"])
    for block in cons_norm:
        if block in constraints:
            cons_norm[block] = copy.deepcopy(constraints.get(block))
    for block in cons_norm:
        if cons_norm[block] is None:
            cons_norm[block] = {}
        if not isinstance(cons_norm[block], dict):
            cons_norm[block] = {}
    transport = cons_norm.get("transport") or {}
    allow = transport.get("allow")
    forbid = transport.get("forbid")
    if allow is not None:
        transport["allow"] = [str(m).strip().lower() for m in (allow if isinstance(allow, list) else [allow]) if m]
    if forbid is not None:
        transport["forbid"] = [str(m).strip().lower() for m in (forbid if isinstance(forbid, list) else [forbid]) if m]
    cons_norm["transport"] = transport

    meal = cons_norm.get("meal") or {}
    cuisines = meal.get("cuisines")
    if cuisines is not None:
        meal["cuisines"] = [str(c).strip() for c in (cuisines if isinstance(cuisines, list) else [cuisines]) if c]
    cons_norm["meal"] = meal

    stay = cons_norm.get("stay") or {}
    for k in ("room_type", "house_rule"):
        v = stay.get(k)
        stay[k] = str(v).strip() if v is not None and str(v).strip() else None
    cons_norm["stay"] = stay

    daily = cons_norm.get("daily") or {}
    for k in ("attractions_per_day_min", "attractions_per_day_max"):
        v = daily.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            daily[k] = None
            continue
        try:
            daily[k] = int(v)
        except Exception:
            daily[k] = None
    cons_norm["daily"] = daily

    city = cons_norm.get("city") or {}
    cand = city.get("candidate_cities")
    if cand is not None:
        city["candidate_cities"] = _norm_list(cand)
    cons_norm["city"] = city

    out["constraints"] = cons_norm

    retrieval = out.get("retrieval")
    if retrieval is None:
        retrieval = {}
    if not isinstance(retrieval, dict):
        retrieval = {}
    ret_norm = copy.deepcopy(base["retrieval"])
    for k in ret_norm:
        if k in retrieval:
            ret_norm[k] = retrieval.get(k)
    for k in ("top_k", "candidate_cap"):
        v = ret_norm.get(k)
        if v is None:
            ret_norm[k] = None
            continue
        try:
            ret_norm[k] = int(v)
        except Exception:
            ret_norm[k] = None
    out["retrieval"] = ret_norm

    rwd = out.get("reward_cfg_overrides")
    if rwd is None:
        rwd = {}
    if not isinstance(rwd, dict):
        rwd = {}
    out["reward_cfg_overrides"] = dict(rwd)

    return out


def compile_envspec(spec: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ok, errors = validate_envspec(spec)
    if not ok:
        raise ValueError("Invalid EnvSpec: " + "; ".join(errors))
    spec_n = normalize_envspec(spec)

    goal = spec_n["goal"]
    constraints = spec_n.get("constraints") or {}
    retrieval = spec_n.get("retrieval") or {}
    reward_overrides = spec_n.get("reward_cfg_overrides") or {}

    goal_parsed: Dict[str, Any] = {
        "origin": goal.get("origin"),
        "destination": goal.get("destination"),
        "start_date": goal.get("start_date"),
        "duration_days": goal.get("duration_days"),
        "budget": goal.get("budget"),
        "people_number": goal.get("people_number"),
        "visiting_city_number": goal.get("visiting_city_number") or 1,
        "preferences": list(goal.get("preferences") or []),
        "must_visit_cities": list(goal.get("must_visit_cities") or []),
        "fixed_city_order": list(goal.get("fixed_city_order") or []),
        "priority_cities": list(goal.get("priority_cities") or []),
        "constraints": copy.deepcopy(constraints) if isinstance(constraints, dict) else {},
    }

    transport = (constraints.get("transport") or {}) if isinstance(constraints, dict) else {}
    forbid = transport.get("forbid")
    if isinstance(forbid, list) and forbid:
        goal_parsed["transport_forbid"] = [str(m).lower() for m in forbid if m]

    daily = (constraints.get("daily") or {}) if isinstance(constraints, dict) else {}
    if daily.get("attractions_per_day_min") is not None:
        goal_parsed["attractions_per_day_min"] = int(daily["attractions_per_day_min"])
    if daily.get("attractions_per_day_max") is not None:
        goal_parsed["attractions_per_day_max"] = int(daily["attractions_per_day_max"])

    meal = (constraints.get("meal") or {}) if isinstance(constraints, dict) else {}
    cuisines = meal.get("cuisines")
    if isinstance(cuisines, list) and cuisines:
        # Keep `_matches_preference` working even if callers ignore `constraints`.
        seen = {p.lower() for p in goal_parsed.get("preferences") or []}
        for c in cuisines:
            if not c:
                continue
            if c.lower() in seen:
                continue
            goal_parsed["preferences"].append(str(c))
            seen.add(str(c).lower())

    city = (constraints.get("city") or {}) if isinstance(constraints, dict) else {}
    candidate_cities = city.get("candidate_cities")
    if isinstance(candidate_cities, list) and candidate_cities:
        goal_parsed["candidate_cities"] = list(candidate_cities)

    env_kwargs: Dict[str, Any] = {}
    if retrieval.get("top_k") is not None:
        env_kwargs["top_k"] = int(retrieval["top_k"])
    if retrieval.get("candidate_cap") is not None:
        env_kwargs["candidate_cap"] = int(retrieval["candidate_cap"])

    if reward_overrides:
        try:
            from mcts.travel.env_agent import DEFAULT_REWARD_CFG as _DEFAULT_REWARD_CFG  # local import
        except Exception:
            _DEFAULT_REWARD_CFG = {}
        merged = dict(_DEFAULT_REWARD_CFG or {})
        for k, v in reward_overrides.items():
            if k in merged and v is not None:
                merged[k] = float(v)
        env_kwargs["reward_cfg"] = merged

    return goal_parsed, env_kwargs
