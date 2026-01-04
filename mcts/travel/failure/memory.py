from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple


class FailureMemory:
    """Track failure-driven exclusions and penalties across attempts."""

    def __init__(self):
        self.excluded_bundles_set: Dict[Tuple[str, ...], int] = {}
        self.excluded_bundles_seq: Dict[Tuple[str, ...], int] = {}
        self.excluded_bundles = self.excluded_bundles_set
        self.penalized_cities: Dict[str, float] = {}
        self.penalized_edges: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def normalize_city(city: Any) -> str:
        return str(city or "").strip().lower()

    def bundle_key(self, bundle: Iterable[str]) -> Tuple[str, ...]:
        return self.bundle_key_set(bundle)

    def bundle_key_set(self, bundle: Iterable[str]) -> Tuple[str, ...]:
        return tuple(sorted(self.normalize_city(c) for c in bundle if c))

    def bundle_key_seq(self, bundle: Iterable[str]) -> Tuple[str, ...]:
        return tuple(self.normalize_city(c) for c in bundle if c)

    def is_excluded(self, bundle_or_key: Any) -> bool:
        if isinstance(bundle_or_key, tuple):
            key_seq = tuple(self.normalize_city(c) for c in bundle_or_key if c)
        else:
            key_seq = self.bundle_key_seq(bundle_or_key or [])
        key_set = tuple(sorted(key_seq))
        return self._ttl_active(self.excluded_bundles_seq, key_seq) or self._ttl_active(self.excluded_bundles_set, key_set)

    def is_excluded_set(self, bundle_or_key: Any) -> bool:
        if isinstance(bundle_or_key, tuple):
            key = tuple(sorted(self.normalize_city(c) for c in bundle_or_key if c))
        else:
            key = self.bundle_key_set(bundle_or_key or [])
        return self._ttl_active(self.excluded_bundles_set, key)

    def is_excluded_seq(self, bundle_or_key: Any) -> bool:
        if isinstance(bundle_or_key, tuple):
            key = tuple(self.normalize_city(c) for c in bundle_or_key if c)
        else:
            key = self.bundle_key_seq(bundle_or_key or [])
        return self._ttl_active(self.excluded_bundles_seq, key)

    def exclude_bundle(self, bundle: Iterable[str], ttl: int = 1) -> None:
        self.exclude_bundle_set(bundle, ttl=ttl)

    def exclude_bundle_set(self, bundle: Iterable[str], ttl: int = 1) -> None:
        key = self.bundle_key_set(bundle)
        cur = int(self.excluded_bundles_set.get(key, 0) or 0)
        self.excluded_bundles_set[key] = max(cur, int(ttl or 0))

    def exclude_bundle_seq(self, bundle: Iterable[str], ttl: int = 1) -> None:
        key = self.bundle_key_seq(bundle)
        cur = int(self.excluded_bundles_seq.get(key, 0) or 0)
        self.excluded_bundles_seq[key] = max(cur, int(ttl or 0))

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
        self._tick_dict(self.excluded_bundles_set)
        self._tick_dict(self.excluded_bundles_seq)

    def update(self, signal: Optional[Dict[str, Any]]) -> None:
        if not signal:
            return
        phase = str(signal.get("phase") or "").lower()
        subtype = str(signal.get("subtype") or "").lower()
        reason = str(signal.get("reason") or "").lower()
        bundle = signal.get("bundle") or []
        city = signal.get("city")
        segment_role = signal.get("segment_role")
        origin = signal.get("origin")

        if phase in ("segment", "flight"):
            self.exclude_bundle_seq(bundle, ttl=2)
            if segment_role == "return" and bundle and origin:
                self.penalize_edge(bundle[-1], origin, weight=1.0)
            return

        city_level_subtypes = {
            "min_nights",
            "stay_constraints",
            "stay_unavailable",
            "missing_cuisine",
            "meal_constraints",
        }
        if phase in ("hotel", "stay", "meal", "daily", "attraction"):
            if subtype in city_level_subtypes or reason in ("constraint_mismatch", "scarcity"):
                self.exclude_bundle_set(bundle, ttl=1)
                if city:
                    self.penalize_city(city, weight=1.0)
                return

    @staticmethod
    def _ttl_active(store: Dict[Tuple[str, ...], int], key: Tuple[str, ...]) -> bool:
        ttl = store.get(key, 0)
        return int(ttl or 0) > 0

    @staticmethod
    def _tick_dict(store: Dict[Tuple[str, ...], int]) -> None:
        expired = []
        for key, ttl in store.items():
            ttl_next = int(ttl or 0) - 1
            if ttl_next <= 0:
                expired.append(key)
            else:
                store[key] = ttl_next
        for key in expired:
            store.pop(key, None)
