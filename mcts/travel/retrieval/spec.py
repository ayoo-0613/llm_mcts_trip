from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def parsed_get(parsed: Any, *keys: str, default: Any = None) -> Any:
    if not isinstance(parsed, dict):
        return default
    for key in keys:
        if key in parsed and parsed[key] is not None:
            return parsed[key]
    return default


def listify(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    if isinstance(val, (set, tuple)):
        return [str(v) for v in val if v is not None]
    return [str(val)]


@dataclass(frozen=True)
class QueryConstraints:
    cuisines: List[str] = field(default_factory=list)
    house_rules: List[str] = field(default_factory=list)
    room_types: List[str] = field(default_factory=list)
    min_occupancy: Optional[int] = None
    allow_modes: List[str] = field(default_factory=list)
    forbid_modes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cuisines": list(self.cuisines),
            "house_rules": list(self.house_rules),
            "room_types": list(self.room_types),
            "min_occupancy": self.min_occupancy,
            "allow_modes": list(self.allow_modes),
            "forbid_modes": list(self.forbid_modes),
        }


@dataclass(frozen=True)
class QuerySpec:
    origin: Optional[str]
    destination: Optional[str]
    start_date: Any
    duration_days: Any
    budget: Any
    visiting_city_number: Any
    people_number: Any
    candidate_cities: List[str]
    must_visit_cities: List[str]
    priority_cities: List[str]
    constraints: QueryConstraints
    raw: Any = None

    def signature(self) -> str:
        parts = [
            self.origin or "",
            self.destination or "",
            str(self.start_date or ""),
            str(self.duration_days or ""),
            str(self.budget or ""),
            str(self.visiting_city_number or ""),
            "|".join(sorted(self.constraints.cuisines)),
            "|".join(sorted(self.constraints.house_rules)),
            "|".join(sorted(self.constraints.room_types)),
            "|".join(sorted(self.constraints.allow_modes)),
            "|".join(sorted(self.constraints.forbid_modes)),
            "|".join(sorted(self.candidate_cities)),
            "|".join(sorted(self.must_visit_cities)),
            "|".join(sorted(self.priority_cities)),
        ]
        return "|".join(parts)


class ConstraintNormalizer:
    def to_constraints(self, parsed: Any) -> QueryConstraints:
        cons = parsed_get(parsed, "constraints", default=None)
        cons = cons if isinstance(cons, dict) else {}
        meal = dict(cons.get("meal", {}) or {})
        stay = dict(cons.get("stay", {}) or {})
        transport = dict(cons.get("transport", {}) or {})

        local_constraint = parsed_get(parsed, "local_constraint", default=None) or {}
        house_rules = listify(
            stay.get("house_rules")
            or local_constraint.get("house rule")
            or parsed_get(parsed, "house_rule", default=None)
        )
        room_types = listify(
            stay.get("room_type")
            or local_constraint.get("room type")
            or parsed_get(parsed, "room_type", default=None)
        )
        cuisines = listify(
            meal.get("cuisines")
            or local_constraint.get("cuisine")
            or parsed_get(parsed, "cuisine", default=None)
        )
        min_occ = stay.get("min_occupancy")
        if min_occ is None:
            try:
                min_occ = int(parsed_get(parsed, "people_number", default=None) or 1)
            except Exception:
                min_occ = None

        allow_modes = listify(
            transport.get("allow")
            or parsed_get(parsed, "transport_allowed_modes", "transport_allow", default=None)
        )
        forbid_modes = listify(
            transport.get("forbid")
            or parsed_get(parsed, "transport_forbidden_modes", "transport_forbid", "transport_forbidden", default=None)
        )
        transportation = listify(parsed_get(parsed, "transportation", default=None) or local_constraint.get("transportation"))
        for raw in transportation:
            low = raw.strip().lower()
            if "no flight" in low:
                forbid_modes.append("flight")
            if "no self-driving" in low:
                forbid_modes.append("self-driving")
            if "no taxi" in low:
                forbid_modes.append("taxi")

        return QueryConstraints(
            cuisines=[c.strip() for c in cuisines if c and str(c).strip()],
            house_rules=[r.strip() for r in house_rules if r and str(r).strip()],
            room_types=[r.strip() for r in room_types if r and str(r).strip()],
            min_occupancy=min_occ,
            allow_modes=[m.strip().lower() for m in allow_modes if m],
            forbid_modes=[m.strip().lower() for m in forbid_modes if m],
        )

    def to_spec(self, parsed: Any) -> QuerySpec:
        constraints = self.to_constraints(parsed)
        return QuerySpec(
            origin=parsed_get(parsed, "origin", "org", default=None),
            destination=parsed_get(parsed, "destination", "dest", default=None),
            start_date=parsed_get(parsed, "start_date", default=""),
            duration_days=parsed_get(parsed, "duration_days", "days", default=""),
            budget=parsed_get(parsed, "budget", default=""),
            visiting_city_number=parsed_get(parsed, "visiting_city_number", default=""),
            people_number=parsed_get(parsed, "people_number", default=None),
            candidate_cities=list(parsed_get(parsed, "candidate_cities", default=[]) or []),
            must_visit_cities=list(parsed_get(parsed, "must_visit_cities", default=[]) or []),
            priority_cities=list(parsed_get(parsed, "priority_cities", default=[]) or []),
            constraints=constraints,
            raw=parsed,
        )
