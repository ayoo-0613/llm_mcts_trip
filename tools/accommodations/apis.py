from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from tools._paths import maybe_path


@dataclass
class Accommodations:
    csv_path: Optional[str] = None

    def __post_init__(self) -> None:
        path = maybe_path(self.csv_path, "accommodations", "clean_accommodations_2022.csv")
        self.data = pd.read_csv(path)

