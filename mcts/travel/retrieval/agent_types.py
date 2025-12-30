from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SlotActionResult:
    actions: List[str]
    payloads: Dict[str, Tuple]
    candidates: List[Dict[str, Any]]
    relaxed: bool
    filt: Dict[str, Any]
    policy_event: Dict[str, Any]
    plan: Optional[Any] = None
    uncapped_filter: Optional[Dict[str, Any]] = None

