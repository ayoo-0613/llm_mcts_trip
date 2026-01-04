from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _stable_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return json.dumps(str(obj), ensure_ascii=True)


def fingerprint_goal(goal_parsed: Optional[Dict[str, Any]]) -> str:
    goal_parsed = goal_parsed or {}
    key_obj = {
        "origin": goal_parsed.get("origin") or goal_parsed.get("org"),
        "destination": goal_parsed.get("destination") or goal_parsed.get("dest"),
        "days": goal_parsed.get("duration_days") or goal_parsed.get("days"),
        "visiting_city_number": goal_parsed.get("visiting_city_number"),
        "people_number": goal_parsed.get("people_number"),
        "budget": goal_parsed.get("budget"),
        "local_constraint": goal_parsed.get("local_constraint"),
        "constraints": goal_parsed.get("constraints"),
    }
    raw = _stable_json(key_obj)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def fingerprint_bundle(bundle: Any) -> str:
    try:
        seq = list(bundle or [])
    except Exception:
        seq = [str(bundle)]
    norm = [str(c).strip().lower() for c in seq if str(c).strip()]
    raw = _stable_json(norm)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def phase_key(phase: str) -> str:
    p = str(phase or "").strip().lower()
    if p in ("flight", "segment"):
        return "segment"
    if p in ("hotel", "stay"):
        return "stay"
    if p in ("meal", "attraction", "daily"):
        return "daily"
    if p in ("city",):
        return "city"
    return p


@dataclass
class FailureGradient:
    hard_exclusions: List[Dict[str, Any]] = field(default_factory=list)   # [{phase, rule, ttl, ...}]
    soft_penalties: List[Dict[str, Any]] = field(default_factory=list)    # [{phase, feature, delta, ...}]
    retrieval_patches: List[Dict[str, Any]] = field(default_factory=list) # [{phase, patch, ttl?}]
    bundle_override: Optional[List[str]] = None
    scope: Dict[str, Any] = field(default_factory=dict)                   # {goal_fp, phase, slot_fp, bundle_fp}
    confidence: float = 0.0
    reason: Optional[str] = None


class FailureGradientBuilder:
    """
    Build a FailureGradient from (failure_signal + report + refiner output).
    This is intentionally conservative: rules are enum-like strings only.
    """

    ALLOWED_RULES = {
        "NO_FLIGHT",
        "NO_SELF_DRIVING",
        "NO_TAXI",
        "AVOID_CITY",
        "BLOCK_EDGE",
        "EXCLUDE_BUNDLE_SET",
        "EXCLUDE_BUNDLE_SEQ",
    }
    ALLOWED_FEATURES = {
        "budget_risk",
        "cost_ratio",
        "soft_penalty",
        "edge_cost",
        "hotel_price",
    }

    @classmethod
    def build(
        cls,
        *,
        failure_signal: Optional[Dict[str, Any]],
        report: Optional[Dict[str, Any]],
        refine_out: Optional[Dict[str, Any]],
        state: Any,
        goal_parsed: Optional[Dict[str, Any]],
        slot_fp: Optional[str] = None,
    ) -> FailureGradient:
        report = report or {}
        failure_signal = failure_signal or {}
        phase = phase_key(report.get("failed_phase") or failure_signal.get("phase") or "")

        goal_fp = fingerprint_goal(goal_parsed or {})
        bundle = report.get("city_bundle") or failure_signal.get("bundle") or list(getattr(state, "city_sequence", []) or [])
        bundle_fp = fingerprint_bundle(bundle)

        raw_grad = None
        if isinstance(refine_out, dict):
            raw_grad = refine_out.get("gradient")
        gradient = FailureGradient(
            bundle_override=(refine_out.get("bundle_override") if isinstance(refine_out, dict) else None),
            confidence=float(refine_out.get("confidence") or 0.0) if isinstance(refine_out, dict) else 0.0,
            reason=(refine_out.get("reason") if isinstance(refine_out, dict) else None),
            scope={
                "goal_fp": goal_fp,
                "phase": phase,
                "slot_fp": slot_fp,
                "bundle_fp": bundle_fp,
            },
        )

        parsed = cls._validate_gradient(raw_grad)
        if parsed is None:
            parsed = cls._fallback_gradient(report, failure_signal)
        gradient.hard_exclusions = parsed.get("hard_exclusions", []) or []
        gradient.soft_penalties = parsed.get("soft_penalties", []) or []
        gradient.retrieval_patches = parsed.get("retrieval_patches", []) or []
        return gradient

    @classmethod
    def _validate_gradient(cls, raw: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        out = {"hard_exclusions": [], "soft_penalties": [], "retrieval_patches": []}
        for item in raw.get("hard_exclusions", []) or []:
            if not isinstance(item, dict):
                continue
            rule = str(item.get("rule") or "").strip().upper()
            if rule not in cls.ALLOWED_RULES:
                continue
            ph = phase_key(item.get("phase") or "")
            ttl = item.get("ttl")
            try:
                ttl_i = int(ttl) if ttl is not None else 1
            except Exception:
                ttl_i = 1
            d = dict(item)
            d["rule"] = rule
            d["phase"] = ph
            d["ttl"] = max(1, ttl_i)
            out["hard_exclusions"].append(d)

        for item in raw.get("soft_penalties", []) or []:
            if not isinstance(item, dict):
                continue
            feature = str(item.get("feature") or "").strip()
            if feature not in cls.ALLOWED_FEATURES:
                continue
            ph = phase_key(item.get("phase") or "")
            try:
                delta = float(item.get("delta") or 0.0)
            except Exception:
                continue
            out["soft_penalties"].append({"phase": ph, "feature": feature, "delta": delta})

        for item in raw.get("retrieval_patches", []) or []:
            if not isinstance(item, dict):
                continue
            ph = phase_key(item.get("phase") or "")
            patch = item.get("patch")
            if not isinstance(patch, dict):
                continue
            ttl = item.get("ttl")
            try:
                ttl_i = int(ttl) if ttl is not None else 1
            except Exception:
                ttl_i = 1
            out["retrieval_patches"].append({"phase": ph, "patch": dict(patch), "ttl": max(1, ttl_i)})
        return out

    @staticmethod
    def _fallback_gradient(report: Dict[str, Any], failure_signal: Dict[str, Any]) -> Dict[str, Any]:
        phase = phase_key(report.get("failed_phase") or failure_signal.get("phase") or "")
        dead_city = report.get("dead_end_city") or failure_signal.get("city")
        subtype = str(report.get("dead_end_subtype") or failure_signal.get("subtype") or "").lower()
        reason = str(report.get("dead_end_reason") or failure_signal.get("reason") or "").lower()

        if phase == "segment":
            return {
                "hard_exclusions": [],
                "soft_penalties": [{"phase": "segment", "feature": "cost_ratio", "delta": 0.5}],
                "retrieval_patches": [{"phase": "segment", "patch": {"topk": 30, "prefer": "low_cost"}, "ttl": 1}],
            }

        if phase == "stay":
            hard = []
            if dead_city and subtype in ("min_nights", "stay_unavailable", "stay_constraints"):
                hard.append({"phase": "stay", "rule": "AVOID_CITY", "city": dead_city, "ttl": 1})
            return {
                "hard_exclusions": hard,
                "soft_penalties": [{"phase": "stay", "feature": "hotel_price", "delta": 0.6}],
                "retrieval_patches": [{"phase": "stay", "patch": {"topk": 50, "prefer": "low_cost"}, "ttl": 1}],
            }

        if phase == "daily":
            if subtype == "missing_cuisine" or reason == "constraint_mismatch":
                return {
                    "hard_exclusions": [],
                    "soft_penalties": [{"phase": "daily", "feature": "budget_risk", "delta": 0.2}],
                    "retrieval_patches": [{"phase": "daily", "patch": {"topk": 50}, "ttl": 1}],
                }

        return {"hard_exclusions": [], "soft_penalties": [], "retrieval_patches": []}

