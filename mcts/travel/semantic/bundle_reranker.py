from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


class SemanticBundleReranker:
    """Bundle reranker that only reorders/filters candidate bundles."""

    def __init__(
        self,
        *,
        max_keep: int = 5,
        cost_scale: float = 0.001,
        return_scale: float = 0.001,
    ):
        self.max_keep = max(1, int(max_keep))
        self.cost_scale = float(cost_scale)
        self.return_scale = float(return_scale)

    def rerank(
        self,
        *,
        constraints: Dict[str, Any],
        failure_memory: Optional[Any],
        bundles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for item in bundles:
            seq = list(item.get("bundle") or [])
            if not seq:
                continue
            features = dict(item.get("features") or {})
            bundle_key = features.get("bundle_key") or seq
            if failure_memory is not None and getattr(failure_memory, "is_excluded", None):
                if failure_memory.is_excluded(bundle_key):
                    continue
            score = 0.0
            lb_cost = features.get("bundle_lb_cost")
            if lb_cost is None:
                score -= 1000.0
            else:
                score -= float(lb_cost) * self.cost_scale
            min_return = features.get("min_return_cost")
            if min_return is not None:
                score -= float(min_return) * self.return_scale
            score -= self._failure_penalty(seq, features.get("edges"), failure_memory)
            ranked.append({"bundle": seq, "score": score})

        ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return ranked[: self.max_keep]

    def _failure_penalty(
        self,
        bundle: Iterable[str],
        edges: Optional[List[Tuple[str, str]]],
        failure_memory: Optional[Any],
    ) -> float:
        if failure_memory is None:
            return 0.0
        norm = getattr(failure_memory, "normalize_city", None)
        if not callable(norm):
            norm = lambda c: str(c).strip().lower()
        penalty = 0.0
        penalized_cities = getattr(failure_memory, "penalized_cities", {})
        for city in bundle:
            penalty += float(penalized_cities.get(norm(city), 0.0) or 0.0)
        penalized_edges = getattr(failure_memory, "penalized_edges", {})
        if edges:
            for edge in edges:
                penalty += float(penalized_edges.get(edge, 0.0) or 0.0)
        return penalty
