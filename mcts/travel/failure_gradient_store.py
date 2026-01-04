from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcts.travel.failure_gradient import FailureGradient, phase_key


@dataclass
class _StoredGradient:
    gradient: FailureGradient
    ttl: int = 1


class FailureGradientStore:
    """
    Store FailureGradient objects with TTL and provide merged queries.
    """

    def __init__(self):
        self._items: List[_StoredGradient] = []

    def add(self, gradient: FailureGradient, ttl: Optional[int] = None) -> None:
        ttl_i = None
        if ttl is not None:
            try:
                ttl_i = int(ttl)
            except Exception:
                ttl_i = None
        if ttl_i is None:
            # Default TTL: max across explicit TTLs if present, otherwise 1.
            ttl_candidates = []
            for item in (gradient.hard_exclusions or []):
                if isinstance(item, dict) and item.get("ttl") is not None:
                    try:
                        ttl_candidates.append(int(item.get("ttl")))
                    except Exception:
                        pass
            for item in (gradient.retrieval_patches or []):
                if isinstance(item, dict) and item.get("ttl") is not None:
                    try:
                        ttl_candidates.append(int(item.get("ttl")))
                    except Exception:
                        pass
            ttl_i = max(ttl_candidates) if ttl_candidates else 1
        ttl_i = max(1, int(ttl_i))
        self._items.append(_StoredGradient(gradient=gradient, ttl=ttl_i))

    def tick(self) -> None:
        kept: List[_StoredGradient] = []
        for item in self._items:
            ttl_next = int(item.ttl or 0) - 1
            if ttl_next > 0:
                item.ttl = ttl_next
                kept.append(item)
        self._items = kept

    def query(
        self,
        *,
        goal_fp: str,
        phase: str,
        slot_fp: Optional[str] = None,
    ) -> Dict[str, Any]:
        ph = phase_key(phase)
        out = {"hard_exclusions": [], "soft_penalties": [], "retrieval_patches": []}
        for item in self._items:
            grad = item.gradient
            scope = grad.scope or {}
            if scope.get("goal_fp") != goal_fp:
                continue
            gph = phase_key(scope.get("phase") or "")
            if gph and gph != ph:
                continue
            if slot_fp and scope.get("slot_fp") and scope.get("slot_fp") != slot_fp:
                continue
            for ex in grad.hard_exclusions or []:
                if isinstance(ex, dict) and phase_key(ex.get("phase") or "") == ph:
                    out["hard_exclusions"].append(dict(ex))
            for sp in grad.soft_penalties or []:
                if isinstance(sp, dict) and phase_key(sp.get("phase") or "") == ph:
                    out["soft_penalties"].append(dict(sp))
            for rp in grad.retrieval_patches or []:
                if isinstance(rp, dict) and phase_key(rp.get("phase") or "") == ph:
                    out["retrieval_patches"].append(dict(rp))

        # Merge retrieval patches (later overrides earlier)
        merged_patch: Dict[str, Any] = {}
        for item in out["retrieval_patches"]:
            patch = item.get("patch")
            if isinstance(patch, dict):
                merged_patch.update(patch)
        out["retrieval_patch"] = merged_patch
        return out

