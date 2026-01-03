from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple


class FailureMemory:
    """Track failure-driven exclusions and penalties across attempts."""

    def __init__(self):
        self.excluded_bundles: Dict[Tuple[str, ...], int] = {}
        self.penalized_cities: Dict[str, float] = {}
        self.penalized_edges: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def normalize_city(city: Any) -> str:
        return str(city or "").strip().lower()

    def bundle_key(self, bundle: Iterable[str]) -> Tuple[str, ...]:
        return tuple(sorted(self.normalize_city(c) for c in bundle if c))

    def is_excluded(self, bundle_or_key: Any) -> bool:
        if isinstance(bundle_or_key, tuple):
            key = bundle_or_key
        else:
            key = self.bundle_key(bundle_or_key or [])
        ttl = self.excluded_bundles.get(key, 0)
        return int(ttl or 0) > 0

    def exclude_bundle(self, bundle: Iterable[str], ttl: int = 1) -> None:
        key = self.bundle_key(bundle)
        cur = int(self.excluded_bundles.get(key, 0) or 0)
        self.excluded_bundles[key] = max(cur, int(ttl or 0))

    def penalize_city(self, city: str, weight: float = 1.0) -> None:
        if not city:
            return
        key = self.normalize_city(city)
        self.penalized_cities[key] = float(self.penalized_cities.get(key, 0.0) or 0.0) + float(weight or 0.0)

    def penalize_edge(self, src: str, dst: str, weight: float = 1.0) -> None:
        if not src or not dst:
            return
        key = (self.normalize_city(src), self.normalize_city(dst))
        self.penalized_edges[key] = float(self.penalized_edges.get(key, 0.0) or 0.0) + float(weight or 0.0)

    def tick(self) -> None:
        expired = []
        for key, ttl in self.excluded_bundles.items():
            ttl_next = int(ttl or 0) - 1
            if ttl_next <= 0:
                expired.append(key)
            else:
                self.excluded_bundles[key] = ttl_next
        for key in expired:
            self.excluded_bundles.pop(key, None)

    def update(self, signal: Optional[Dict[str, Any]]) -> None:
        if not signal:
            return
        phase = str(signal.get("phase") or "").lower()
        subtype = str(signal.get("subtype") or "").lower()
        bundle = signal.get("bundle") or []
        city = signal.get("city")
        segment_role = signal.get("segment_role")
        origin = signal.get("origin")

        if phase in ("segment", "flight") and subtype == "violate_reserve":
            self.exclude_bundle(bundle, ttl=2)
            if segment_role == "return" and bundle and origin:
                self.penalize_edge(bundle[-1], origin, weight=1.0)
            return

        if phase in ("hotel", "stay") and subtype == "min_nights":
            self.exclude_bundle(bundle, ttl=1)
            if city:
                self.penalize_city(city, weight=1.0)
