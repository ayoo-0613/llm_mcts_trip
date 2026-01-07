import argparse
import ast
import json
from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_eval(text: Any) -> Any:
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text
    s = str(text).strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return text


def _row_to_query_json(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "org": row.get("org"),
        "dest": row.get("dest"),
        "days": int(row.get("days")) if row.get("days") is not None else None,
        "visiting_city_number": int(row.get("visiting_city_number")) if row.get("visiting_city_number") is not None else None,
        "date": _safe_eval(row.get("date")),
        "people_number": int(row.get("people_number")) if row.get("people_number") is not None else None,
        "local_constraint": _safe_eval(row.get("local_constraint")) or {},
        "budget": float(row.get("budget")) if row.get("budget") is not None else None,
        "query": row.get("query"),
        "level": row.get("level"),
    }
    # Keep keys stable for downstream eval scripts.
    out["date"] = out["date"] if isinstance(out["date"], list) else []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="TripCraft query CSV (e.g., tripcraft_3day.csv)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--use-annotation-plan",
        action="store_true",
        help="If set, populate `plan` from the CSV's `annotation_plan` column; else output empty plans.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(df.to_dict(orient="records"), start=1):
            query_json = _row_to_query_json(row)
            persona = str(row.get("persona") or "")
            plan: List[Dict[str, Any]] = []
            if args.use_annotation_plan:
                plan_raw = _safe_eval(row.get("annotation_plan"))
                if isinstance(plan_raw, list):
                    plan = plan_raw
            rec = {
                "idx": i,
                "JSON": query_json,
                "persona": persona,
                "plan": plan,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

