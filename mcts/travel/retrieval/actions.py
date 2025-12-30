from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class Action:
    text: str
    payload: Tuple


class ActionFactory:
    def build(self, slot: Any, candidates: List[dict]) -> List[Action]:
        actions: List[Action] = []
        stype = getattr(slot, "type", None)
        if stype == "finish":
            actions.append(Action(text="finish", payload=("finish",)))
            return actions
        if not candidates:
            return actions
        if stype == "flight":
            seg_idx = getattr(slot, "seg", None)
            for flight in candidates:
                seg_val = seg_idx if seg_idx is not None else -1
                price_val = float(flight.get("price", 0) or 0.0)
                text = (
                    f"move:seg{seg_val}:flight:{flight.get('id')} {flight.get('origin')}->{flight.get('destination')} "
                    f"${price_val:.0f} {flight.get('depart', '?')}-{flight.get('arrive', '?')}"
                )
                actions.append(Action(text=text, payload=("segment_mode", seg_val, "flight", flight)))
        elif stype == "hotel":
            slot_city = getattr(slot, "city", None)
            for stay in candidates:
                city = slot_city or stay.get("city")
                price_val = float(stay.get("price", 0) or 0.0)
                text = (
                    f"stay:{city}:{stay.get('id')} {stay.get('name')} "
                    f"{stay.get('room_type')} ${price_val:.0f}"
                )
                actions.append(Action(text=text, payload=("stay_city", city, stay)))
        elif stype == "meal":
            day = getattr(slot, "day", None)
            meal_type = getattr(slot, "meal_type", None)
            for rest in candidates:
                cost_val = float(rest.get("cost", 0) or 0.0)
                text = (
                    f"eat:d{day}:{meal_type}:{rest.get('id')} "
                    f"{rest.get('name')} {rest.get('cuisines')} ${cost_val:.0f} rating {rest.get('rating', 0)}"
                )
                actions.append(Action(text=text, payload=("meal", day, meal_type, rest)))
        elif stype == "attraction":
            day = getattr(slot, "day", None)
            slot_name = getattr(slot, "meal_type", None) or "spot"
            slot_city = getattr(slot, "city", None)
            for att in candidates:
                city = att.get("city") or slot_city
                text = f"visit:d{day}:{slot_name}:{att.get('id')} {att.get('name')} @ {city}"
                actions.append(Action(text=text, payload=("attraction", day, slot_name, att)))
        elif stype == "finish":
            actions.append(Action(text="finish", payload=("finish",)))
        return actions

    def to_actions_payloads(self, actions: List[Action]) -> Tuple[List[str], dict]:
        texts = [action.text for action in actions]
        payloads = {action.text: action.payload for action in actions}
        return texts, payloads
