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
        risk_scale: float = 1.0,
    ):
        self.max_keep = max(1, int(max_keep))
        self.cost_scale = float(cost_scale)
        self.return_scale = float(return_scale)
        self.risk_scale = float(risk_scale)

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
            bundle_key = features.get("bundle_key_seq") or features.get("bundle_key") or seq
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
            score -= self._bundle_risk_penalty(features) * self.risk_scale
            score -= self._failure_penalty(seq, features.get("edges"), failure_memory)
            ranked.append({"bundle": seq, "score": score})

        ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return ranked[: self.max_keep]

    @staticmethod
    def _bundle_risk_penalty(features: Dict[str, Any]) -> float:
        risk = features.get("bundle_risk") if isinstance(features, dict) else None
        if not isinstance(risk, dict):
            return 0.0

        penalty = 0.0

        meal_slack = risk.get("bottleneck_meal_slack")
        if meal_slack is not None:
            try:
                v = float(meal_slack)
                if v <= 0:
                    penalty += 30.0
                else:
                    penalty += 10.0 / (v + 1.0)
            except Exception:
                pass

        attr_slack = risk.get("bottleneck_attr_slack")
        if attr_slack is not None:
            try:
                v = float(attr_slack)
                if v <= 0:
                    penalty += 10.0
                else:
                    penalty += 4.0 / (v + 1.0)
            except Exception:
                pass

        hotel_price = risk.get("bottleneck_hotel_price")
        if hotel_price is not None:
            try:
                penalty += float(hotel_price) * 0.002
            except Exception:
                pass

        edge_costs = features.get("edge_costs") if isinstance(features, dict) else None
        if isinstance(edge_costs, dict) and edge_costs:
            blocked = any(bool(info.get("blocked")) for info in edge_costs.values() if isinstance(info, dict))
            if blocked:
                penalty += 50.0
            max_edge = None
            for info in edge_costs.values():
                if not isinstance(info, dict):
                    continue
                if info.get("blocked"):
                    continue
                cost = info.get("min_cost")
                if cost is None:
                    continue
                try:
                    val = float(cost)
                except Exception:
                    continue
                if max_edge is None or val > max_edge:
                    max_edge = val
            if max_edge is not None:
                penalty += float(max_edge) * 0.001

        return float(penalty)

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
