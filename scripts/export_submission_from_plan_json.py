from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


def _infer_idx0_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.match(r"^(\d{3,})_", base)
    if not m:
        return None
    try:
        # Saved per-sample files are typically named with 001/002/...; keep it 1-based by default.
        return max(1, int(m.group(1)))
    except Exception:
        return None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _coerce_day_key(plan: List[Dict[str, Any]], *, day_key: str) -> List[Dict[str, Any]]:
    if day_key == "days":
        return plan
    if day_key != "day":
        raise ValueError("--day-key must be 'days' or 'day'")
    out: List[Dict[str, Any]] = []
    for rec in plan:
        if not isinstance(rec, dict):
            continue
        rec2 = dict(rec)
        if "days" in rec2 and "day" not in rec2:
            rec2["day"] = rec2["days"]
            rec2.pop("days", None)
        out.append(rec2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a TravelPlanner submission JSONL record from a saved per-query plan JSON."
    )
    parser.add_argument("--input-json", required=True, help="Path to a saved plan JSON (e.g., mas_plans_out/001_*.json).")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--idx", type=int, default=None, help="Override idx (1-based). Defaults to filename prefix NNN.")
    parser.add_argument("--query", default=None, help="Override query. Defaults to input JSON 'query'.")
    parser.add_argument(
        "--day-key",
        default="days",
        choices=["days", "day"],
        help="Field name for the day index in each daily plan item (evaluation usually expects 'days').",
    )
    args = parser.parse_args()

    data = _load_json(args.input_json)
    query = args.query if args.query is not None else str(data.get("query") or "")
    idx0 = args.idx if args.idx is not None else _infer_idx0_from_filename(args.input_json)
    if idx0 is None:
        raise ValueError("Cannot infer idx from filename; pass --idx explicitly.")

    plan = data.get("submission_plan")
    if not isinstance(plan, list):
        raise ValueError(
            "Input JSON does not contain 'submission_plan' (list). "
            "Re-generate plans with the updated scripts/run_travel_batch.py."
        )

    plan = _coerce_day_key(plan, day_key=args.day_key)

    rec = {"idx": int(idx0), "query": query, "plan": plan}
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
