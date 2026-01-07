import argparse
import json
from typing import Any, Dict, List


REQUIRED_DAY_KEYS = [
    "days",
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
    "event",
    "point_of_interest_list",
]


def _fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def _as_str(val: Any) -> str:
    return "" if val is None else str(val)


def _check_record(rec: Dict[str, Any]) -> None:
    if not isinstance(rec.get("idx"), int):
        _fail("missing/invalid idx (expected int)")
    if not isinstance(rec.get("JSON"), dict):
        _fail("missing/invalid JSON (expected object)")
    if not isinstance(rec.get("plan"), list):
        _fail("missing/invalid plan (expected list)")

    plan = rec["plan"]
    for day_idx, day in enumerate(plan, start=1):
        if not isinstance(day, dict):
            _fail(f"idx={rec['idx']} day#{day_idx}: day entry is not object")
        missing = [k for k in REQUIRED_DAY_KEYS if k not in day]
        if missing:
            _fail(f"idx={rec['idx']} day#{day_idx}: missing keys {missing}")
        if int(day.get("days")) != day_idx:
            _fail(f"idx={rec['idx']} day#{day_idx}: `days` field mismatch ({day.get('days')})")

        poi = _as_str(day.get("point_of_interest_list"))
        if poi != "-" and poi:
            # Must contain at least one time segment marker.
            if " from " not in poi or " to " not in poi:
                _fail(f"idx={rec['idx']} day#{day_idx}: point_of_interest_list missing 'from/to' times")

        accom = _as_str(day.get("accommodation"))
        if accom and accom != "-" and "," in accom and poi:
            accom_name = accom.rsplit(",", 1)[0].strip()
            first = poi.split(";")[0]
            last = poi.split(";")[-1]
            if accom_name and accom_name not in first:
                _fail(f"idx={rec['idx']} day#{day_idx}: poi does not start with accommodation ({accom_name})")
            if accom_name and accom_name not in last:
                _fail(f"idx={rec['idx']} day#{day_idx}: poi does not end with accommodation ({accom_name})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True, help="TripCraft-format JSONL file")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    checked = 0
    with open(args.plans, "r", encoding="utf-8") as f:
        for line in f:
            if args.limit is not None and checked >= int(args.limit):
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            _check_record(rec)
            checked += 1
    print(f"[OK] checked {checked} record(s)")


if __name__ == "__main__":
    main()

