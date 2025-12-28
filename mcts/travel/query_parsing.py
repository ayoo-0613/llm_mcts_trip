import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

__all__ = [
    "call_local_llm",
    "parse_nl_query",
    "normalize_parsed_query",
    "fallback_parse",
    "extract_query_text",
    "load_queries",
]


def call_local_llm(base_url: str, model: str, prompt: str, timeout: float = 60.0) -> Optional[str]:
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
    except Exception:
        return None
    if resp.status_code >= 400:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    for key in ("response", "text", "output", "content"):
        if key in data:
            return data[key]
    return None


def parse_nl_query(nl_query: str, base_url: str, model: str, timeout: float = 60.0) -> Dict[str, Any]:
    prompt = (
        "Please assist me in extracting valid information from a given natural language text and reconstructing it "
        "in JSON format, as demonstrated in the following example.\n"
        'In the JSON, "org" denotes the departure city. "dest" denotes the destination city. "days" denotes the total '
        'number of travel days. When "days" exceeds 3, "visiting_city_number" specifies the number of cities to be covered '
        'in the destination state. "date" includes the detailed date to visit.\n'
        'In addition, "local_constraint" contains four possible constraints. Possible options of "house rule" includes '
        '["parties", "smoking", "children under 10", "pets", "visitors"]. Possible options of "cuisine" includes '
        '["Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", "French"]. Possible options of "house type" '
        'includes ["entire room", "private room", "shared room", "not shared room"]. Possible options of "transportation" '
        'includes ["no flight", "no self-driving"]. If neither are mentioned in the text, make the value to be null.\n'
        "You can only assign null to local constraints if it is needed. Other fields must have values.\n"
        "Here are three examples:\n"
        "-----EXAMPLE 1-----\n"
        "Text: {Please help me plan a trip from St. Petersburg to Rockford spanning 3 days from March 16th to March 18th, 2022. "
        "The travel should be planned for a single person with a budget of $1,700.}\n"
        "JSON:\n"
        "{\n"
        '    "org": "St. Petersburg", \n'
        '    "dest": "Rockford", \n'
        '    "days": 3, \n'
        '    "visiting_city_number": 1, \n'
        '    "date": ["2022-03-16", "2022-03-17", "2022-03-18"], \n'
        '    "people_number": 1, \n'
        '    "local_constraint": {\n'
        '                            "house rule": null, \n'
        '                            "cuisine": null, \n'
        '                            "room type": null, \n'
        '                            "transportation": null\n'
        "                        }, \n"
        '    "budget": 1700\n'
        "}\n"
        "-----EXAMPLE 2-----\n"
        "Text: {Please create a 3-day travel itinerary for 2 people beginning in Fort Lauderdale and ending in Milwaukee from the "
        "8th to the 10th of March, 2022. Our travel budget is set at $1,100. We'd love to experience both American and Chinese cuisines "
        "during our journey.}\n"
        "JSON:\n"
        "{\n"
        '    "org": "Fort Lauderdale", \n'
        '    "dest": "Milwaukee", \n'
        '    "days": 3, \n'
        '    "visiting_city_number": 1, \n'
        '    "date": ["2022-03-08", "2022-03-09", "2022-03-10"], \n'
        '    "people_number": 2, \n'
        '    "local_constraint": {\n'
        '        "house rule": null, \n'
        '        "cuisine": ["American", "Chinese"], \n'
        '        "room type": null, \n'
        '        "transportation": null\n'
        "        }, \n"
        '    "budget": 1100\n'
        "}\n"
        "-----EXAMPLE 3-----\n"
        "Text: {Can you create a 5-day travel itinerary for a group of 3, departing from Atlanta and visiting 2 cities in Minnesota from "
        "March 3rd to March 7th, 2022? We have a budget of $7,900. We require accommodations that allow parties and should ideally be entire "
        "rooms. Although we don't plan to self-drive, we would like the flexibility to host parties.}\n"
        "JSON:\n"
        "{\n"
        '    "org": "Atlanta", \n'
        '    "dest": "Minnesota", \n'
        '    "days": 5, \n'
        '    "visiting_city_number": 2, \n'
        '    "date": ["2022-03-03", "2022-03-04", "2022-03-05", "2022-03-06", "2022-03-07"], \n'
        '    "people_number": 3, \n'
        '    "local_constraint": {\n'
        '        "house rule": "parties", \n'
        '        "cuisine": null, \n'
        '        "room type": "entire room", \n'
        '        "transportation": "no self-driving"\n'
        "    }, \n"
        '    "budget": 7900\n'
        "}\n"
        "-----EXAMPLES END-----\n"
        "Text:\n"
        "{"
        + nl_query
        + "}\n"
        "JSON:\n"
        "Output ONLY JSON.\n"
    )
    raw = call_local_llm(base_url, model, prompt, timeout=timeout)
    if not raw:
        return {}
    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(raw[start : end + 1])
        except Exception:
            parsed = {}
    return normalize_parsed_query(parsed)


def normalize_parsed_query(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize various parser schemas into the internal keys used by the travel pipeline.

    Supports:
    - legacy schema: origin/destination/start_date/duration_days/.../preferences
    - TravelPlanner-style schema: org/dest/days/date/local_constraint/...
    """

    def _as_list(val: Any) -> List[Any]:
        if val is None:
            return []
        if isinstance(val, list):
            return val
        return [val]

    out: Dict[str, Any] = dict(parsed or {})

    # Accept TravelPlanner-style keys.
    if out.get("origin") is None and out.get("org") is not None:
        out["origin"] = out.get("org")
    if out.get("destination") is None and out.get("dest") is not None:
        out["destination"] = out.get("dest")

    if out.get("duration_days") is None and out.get("days") is not None:
        out["duration_days"] = out.get("days")

    dates = out.get("date") or out.get("dates")
    if out.get("start_date") is None and dates:
        try:
            date_list = _as_list(dates)
            if date_list:
                out["start_date"] = date_list[0]
        except Exception:
            pass
    if out.get("duration_days") is None and dates:
        try:
            out["duration_days"] = len(_as_list(dates))
        except Exception:
            pass

    # Preferences and local constraints.
    preferences: List[str] = []
    for p in _as_list(out.get("preferences")):
        if p is not None and str(p).strip():
            preferences.append(str(p).strip())

    lc = out.get("local_constraint") or {}
    if isinstance(lc, dict):
        for key in ("cuisine",):
            for v in _as_list(lc.get(key)):
                if v is not None and str(v).strip():
                    preferences.append(str(v).strip())
        for key in ("house rule", "room type", "house type"):
            for v in _as_list(lc.get(key)):
                if v is not None and str(v).strip():
                    preferences.append(str(v).strip())

    # Transportation constraints: map "no flight"/"no self-driving" to internal forbid list.
    forbid: List[str] = []
    for v in _as_list(out.get("transport_forbid")) + _as_list(out.get("transport_forbidden")):
        if v is not None and str(v).strip():
            forbid.append(str(v).strip().lower())
    if isinstance(lc, dict):
        for v in _as_list(lc.get("transportation")):
            t = str(v or "").strip().lower()
            if not t:
                continue
            if "no flight" in t:
                forbid.append("flight")
            if "no self-driving" in t or "no self driving" in t:
                forbid.append("self-driving")

    # De-dup while preserving order.
    seen = set()
    preferences_clean: List[str] = []
    for p in preferences:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        preferences_clean.append(p)
    out["preferences"] = preferences_clean

    seen_forbid = set()
    forbid_clean: List[str] = []
    for m in forbid:
        if m in seen_forbid:
            continue
        seen_forbid.add(m)
        forbid_clean.append(m)
    if forbid_clean:
        out["transport_forbid"] = forbid_clean

    # Do not inject defaults; keep original sparsity.
    return out


def fallback_parse(nl_query: str) -> Dict[str, Any]:
    text = nl_query
    out: Dict[str, Any] = {}

    # origin/destination patterns
    patterns = [
        r"from\s+([A-Za-z .'-]+)\s+to\s+([A-Za-z .'-]+)",
        r"beginning in\s+([A-Za-z .'-]+)\s+and heading to\s+([A-Za-z .'-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            out["origin"] = m.group(1).strip()
            out["destination"] = m.group(2).strip()
            break

    # budget
    m = re.search(r"budget[^$]*\$?([\d,\.]+)", text, flags=re.IGNORECASE)
    if m:
        try:
            out["budget"] = float(m.group(1).replace(",", ""))
        except Exception:
            pass

    # duration days
    m = re.search(r"(\d+)\s*-\s*day|\b(\d+)\s*day", text, flags=re.IGNORECASE)
    if m:
        val = m.group(1) or m.group(2)
        if val:
            try:
                out["duration_days"] = int(val)
            except Exception:
                pass

    # dates
    month_pat = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    date_pat = rf"{month_pat}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?"
    date_matches = list(re.finditer(date_pat, text, flags=re.IGNORECASE))
    iso_dates: List[str] = []
    month_to_num = {m: i for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"], start=1)}
    for dm in date_matches:
        month = dm.group(1).lower()
        day = int(dm.group(2))
        year = dm.group(3)
        year_val = int(year) if year else datetime.now().year
        month_num = month_to_num.get(month)
        if month_num:
            try:
                iso_dates.append(datetime(year_val, month_num, day).strftime("%Y-%m-%d"))
            except Exception:
                continue
    if iso_dates:
        out["start_date"] = iso_dates[0]
        if len(iso_dates) >= 2 and "duration_days" not in out:
            try:
                d0 = datetime.fromisoformat(iso_dates[0])
                d1 = datetime.fromisoformat(iso_dates[1])
                out["duration_days"] = (d1 - d0).days + 1
            except Exception:
                pass
    return out


def extract_query_text(entry: Dict[str, Any]) -> str:
    for key in ("query", "instruction", "prompt", "input", "user_query", "text"):
        if key in entry and entry[key]:
            return str(entry[key])
    return ""


def load_queries(args) -> List[str]:
    if load_dataset is None:
        raise ImportError("datasets package is required to load remote queries. Install via pip install datasets")
    split = args.set_type if args.set_type in ("train", "test", "validation") else "train"
    config = split if args.dataset_id == "osunlp/TravelPlanner" else None
    if config:
        ds = load_dataset(args.dataset_id, config, split=split)
    else:
        ds = load_dataset(args.dataset_id, split=split)
    return [extract_query_text(x) for x in ds]


_call_local_llm = call_local_llm
_parse_nl_query = parse_nl_query
_normalize_parsed_query = normalize_parsed_query
_fallback_parse = fallback_parse
_extract_query_text = extract_query_text
_load_queries = load_queries
