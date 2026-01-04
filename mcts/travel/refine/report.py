from __future__ import annotations

from typing import Any, Dict, List, Optional


class FailureReportBuilder:
    @staticmethod
    def build(
        *,
        goal_parsed: Optional[Dict[str, Any]],
        state: Any,
        failure_signal: Optional[Dict[str, Any]],
        violations: List[str],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        goal_parsed = goal_parsed or {}
        try:
            days = int(goal_parsed.get("duration_days") or goal_parsed.get("days") or 0)
        except Exception:
            days = 0
        if days not in (5, 7):
            return None

        bundle = failure_signal.get("bundle") if failure_signal else None
        if not bundle:
            bundle = list(getattr(state, "city_sequence", None) or [])

        cur_feat: Dict[str, Any] = {}
        for entry in candidates or []:
            if (entry.get("bundle") or []) == bundle:
                cur_feat = entry.get("features") or {}
                break

        counts = failure_signal.get("counts") if failure_signal else None
        if not isinstance(counts, dict):
            counts = {}
        returned = counts.get("returned")
        if returned is None:
            returned = counts.get("topk")
        return {
            "failed_phase": (failure_signal.get("phase") if failure_signal else "unknown"),
            "subtype": (failure_signal.get("subtype") if failure_signal else None),
            "violations": violations,
            "city_bundle": bundle,
            "current_bundle_features": cur_feat,
            "current_bottleneck": cur_feat.get("bundle_risk") if isinstance(cur_feat, dict) else None,
            "budget_context": failure_signal.get("budget_context") if failure_signal else None,
            "counts": counts,
            "dead_end_city": failure_signal.get("city") if failure_signal else None,
            "dead_end_reason": failure_signal.get("reason") if failure_signal else None,
            "dead_end_counts": counts,
            "dead_end_budget_context": failure_signal.get("budget_context") if failure_signal else None,
            "dead_end_subtype": failure_signal.get("subtype") if failure_signal else None,
            "dead_end_after_nonbudget": counts.get("after_nonbudget"),
            "dead_end_after_budget": counts.get("after_budget"),
            "dead_end_returned": returned,
            "constraints_snapshot": {
                "budget": goal_parsed.get("budget"),
                "return_required": goal_parsed.get("return_required"),
                "require_accommodation": goal_parsed.get("require_accommodation"),
                "local_constraint": goal_parsed.get("local_constraint"),
                "visiting_city_number": goal_parsed.get("visiting_city_number"),
            },
        }
