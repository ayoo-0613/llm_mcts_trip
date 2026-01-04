from __future__ import annotations

from typing import Any, Dict, List, Optional


class FailureSignalExtractor:
    """Extract structured failure signals from dead-end metadata."""

    @staticmethod
    def extract(
        *,
        filter_events: List[Dict[str, Any]],
        goal_parsed: Optional[Dict[str, Any]] = None,
        state: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        if not filter_events:
            return None

        event = None
        dead_end = None
        for item in reversed(filter_events):
            if not isinstance(item, dict):
                continue
            if item.get("dead_end_meta") is not None:
                event = item
                dead_end = item.get("dead_end_meta") or {}
                break
        if event is None:
            return None

        phase = event.get("phase") or dead_end.get("phase")
        budget_subtype = event.get("budget_subtype")
        segment_role = event.get("segment_role")
        dominant = dead_end.get("dominant_nonbudget_filter")
        reason = dead_end.get("reason")

        subtype = None
        if phase in ("segment", "flight"):
            subtype = budget_subtype
        elif phase in ("hotel", "stay"):
            subtype = dominant
        else:
            subtype = budget_subtype or dominant

        bundle = dead_end.get("city_bundle")
        if not bundle and state is not None:
            bundle = list(getattr(state, "city_sequence", None) or [])

        signal = {
            "phase": phase,
            "subtype": subtype,
            "city": dead_end.get("city"),
            "bundle": bundle,
            "counts": dead_end.get("counts") or event.get("counts"),
            "budget_context": dead_end.get("budget_context") or event.get("budget_context"),
        }
        if reason:
            signal["reason"] = reason
        if segment_role:
            signal["segment_role"] = segment_role
        if goal_parsed:
            origin = goal_parsed.get("origin") or goal_parsed.get("org")
            if origin:
                signal["origin"] = origin
        return signal
