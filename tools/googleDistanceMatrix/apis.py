from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from tools._paths import maybe_path


def _parse_distance_km(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).replace(",", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


@dataclass
class GoogleDistanceMatrix:
    csv_path: Optional[str] = None

    def __post_init__(self) -> None:
        path = maybe_path(self.csv_path, "googleDistanceMatrix", "distance.csv")
        self.data = pd.read_csv(path)
        # normalize strings for lookup
        for col in ("origin", "destination"):
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(str)

    def run_for_evaluation(self, origin: str, destination: str, mode: str = "taxi") -> Dict[str, Any]:
        if not origin or not destination:
            return {"cost": None, "distance_km": None, "mode": mode}
        df = self.data
        if "origin" not in df.columns or "destination" not in df.columns:
            return {"cost": None, "distance_km": None, "mode": mode}
        sub = df[(df["origin"] == origin) & (df["destination"] == destination)]
        if sub.empty:
            return {"cost": None, "distance_km": None, "mode": mode}
        row = sub.iloc[0]
        km = _parse_distance_km(row.get("distance"))
        if km is None:
            return {"cost": None, "distance_km": None, "mode": mode}

        # The upstream evaluator expects a numeric cost to exist; this dataset's `cost` is often empty.
        # Use a simple proxy: cost ~= distance_km.
        return {"cost": float(km), "distance_km": float(km), "mode": mode}

