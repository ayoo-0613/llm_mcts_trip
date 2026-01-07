import argparse
import ast
import json
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


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
    out["date"] = out["date"] if isinstance(out["date"], list) else []
    return out


def _build_given_information(row: Dict[str, Any]) -> str:
    cols = [c for c in row.keys() if str(c).startswith("reference_information")]
    cols.sort()
    blocks: List[str] = []
    for col in cols:
        val = _safe_eval(row.get(col))
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    desc = str(item.get("Description") or "").strip()
                    content = str(item.get("Content") or "").strip()
                    if desc or content:
                        blocks.append((desc + "\n" + content).strip())
                elif item is not None:
                    blocks.append(str(item))
        elif val is not None:
            blocks.append(str(val))
    return "\n\n".join([b for b in blocks if b])


PLANNER_INSTRUCTION_OG = """You are a proficient planner. Based on the provided information, query and persona, please give a detailed travel plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plans should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the "Current City" section as in the example (i.e., from A to B). Include events happening on that day, if any. Provide a Point of Interest List, which is an ordered list of places visited throughout the day. This list should include only accommodations, attractions, or restaurants and their starting and ending timestamps. Each day must start and end with the accommodation where the traveler is staying.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """


PLANNER_INSTRUCTION_PARAMETER_INFO = """You are a proficient planner. Based on the provided information, query and persona, please give a detailed travel plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plans should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the "Current City" section as in the example (i.e., from A to B). Include events happening on that day, if any. Provide a Point of Interest List, which is an ordered list of places visited throughout the day. This list should include accommodations, attractions, or restaurants and their starting and ending timestamps. Each day must start and end with the accommodation where the traveler is staying. Breakfast is ideally scheduled at 9:40 AM and lasts about 50 minutes. Lunch is best planned for 2:20 PM, with a duration of around an hour. Dinner should take place at 8:45 PM, lasting approximately 1 hour and 15 minutes. Laidback Travelers typically explore one attraction per day and sometimes opt for more, while Adventure Seekers often visit 2 or 3 attractions, occasionally exceeding that number.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """


def _call_ollama_generate(base_url: str, model: str, prompt: str, timeout: float = 120.0) -> Optional[str]:
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "temperature": 0.0, "stream": False}
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
    except Exception:
        return None
    if resp.status_code >= 400:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    for key in ("response", "text", "output", "content"):
        if key in data:
            return data[key]
    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. deepseek-r1:14b")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--prompt", choices=["og", "param"], default="param")
    args = ap.parse_args()

    df = pd.read_csv(args.queries_csv)
    rows = df.to_dict(orient="records")[: max(0, int(args.limit))]

    template = PLANNER_INSTRUCTION_PARAMETER_INFO if args.prompt == "param" else PLANNER_INSTRUCTION_OG
    json_instructions = (
        "\n\nReturn ONLY a JSON object with a single key `plan`.\n"
        "`plan` must be a list of day objects, each containing exactly these keys:\n"
        "`days`, `current_city`, `transportation`, `breakfast`, `attraction`, `lunch`, `dinner`, "
        "`accommodation`, `event`, `point_of_interest_list`.\n"
        "Use '-' when information is unnecessary. Do not include markdown.\n"
    )

    with open(args.out, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            query_json = _row_to_query_json(row)
            persona = str(row.get("persona") or "")
            given_info = _build_given_information(row)
            prompt = template.format(text=given_info, query=str(query_json.get("query") or ""), persona=persona) + json_instructions
            raw = _call_ollama_generate(args.base_url, args.model, prompt, timeout=float(args.timeout))

            plan: List[Dict[str, Any]] = []
            parsed = _extract_json(raw or "")
            if parsed and isinstance(parsed.get("plan"), list):
                plan = [x for x in parsed["plan"] if isinstance(x, dict)]

            rec = {
                "idx": int(idx),
                "JSON": query_json,
                "persona": persona,
                "plan": plan,
                "raw": raw,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

