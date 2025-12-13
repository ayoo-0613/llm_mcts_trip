import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import requests

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys  # noqa: E402

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcts.mcts.mcts import MCTSAgent  # noqa: E402
from mcts.travel.knowledge_base import TripGoal, TravelKnowledgeBase  # noqa: E402
from mcts.travel.llm_policy import TravelLLMPolicy  # noqa: E402
from mcts.travel.travel_env import TravelEnv  # noqa: E402
from mcts.travel.phase_plan import PhasePlanGenerator  # noqa: E402
from mcts.travel.preference_router import route_preferences  # noqa: E402


def _call_local_llm(base_url: str, model: str, prompt: str, timeout: float = 60.0) -> Optional[str]:
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


def _parse_nl_query(nl_query: str, base_url: str, model: str, timeout: float = 60.0) -> Dict[str, Any]:
    prompt = (
        "Extract a JSON with fields: origin, destination, start_date, duration_days (int), budget (number), "
        "visiting_city_number (int), must_visit_cities (array of strings), priority_cities (array of strings), "
        "fixed_city_order (array of strings or null), transport_allow (array of strings or null), "
        "transport_forbid (array of strings or null), preferences (array of strings), people_number (int or null). "
        "Allowed transport values: \"flight\", \"taxi\", \"self-driving\". "
        "Destination can be a state/region rather than a single city; keep the stated destination verbatim. "
        "Do NOT invent a city order; only set fixed_city_order when the user explicitly specifies an order like "
        "\"first A then B\". Otherwise leave fixed_city_order as null. "
        "If the user forbids flights or self-driving, add them to transport_forbid. "
        "If the user explicitly restricts to some modes, put them in transport_allow; otherwise use null. "
        "Use null for any missing scalar. Output ONLY JSON.\n\n"
        "Example 1:\n"
        "Query: We will fly from Indianapolis to Colorado for about a week, want to visit three cities there, bring our dog, try Mexican food, no self-driving. Budget 15000, around March 11 2022.\n"
        "JSON: {\"origin\": \"Indianapolis\", \"destination\": \"Colorado\", \"start_date\": \"2022-03-11\", \"duration_days\": 7, \"budget\": 15000, \"visiting_city_number\": 3, \"must_visit_cities\": [], \"priority_cities\": [], \"fixed_city_order\": null, \"transport_allow\": null, \"transport_forbid\": [\"self-driving\"], \"preferences\": [\"Mexican\"], \"people_number\": null}\n\n"
        "Example 2:\n"
        "Query: First go to Salt Lake City then Moab from Houston for 3 days on March 23 2022, no flights, two people, like sushi.\n"
        "JSON: {\"origin\": \"Houston\", \"destination\": \"Utah\", \"start_date\": \"2022-03-23\", \"duration_days\": 3, \"budget\": null, \"visiting_city_number\": 2, \"must_visit_cities\": [\"Salt Lake City\",\"Moab\"], \"priority_cities\": [], \"fixed_city_order\": [\"Salt Lake City\",\"Moab\"], \"transport_allow\": null, \"transport_forbid\": [\"flight\"], \"preferences\": [\"sushi\"], \"people_number\": 2}\n\n"
        f"Query: {nl_query}"
    )
    raw = _call_local_llm(base_url, model, prompt, timeout=timeout)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start : end + 1])
        except Exception:
            pass
    return {}


def _fallback_parse(nl_query: str) -> Dict[str, Any]:
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


def _parse_date_safe(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    fmts = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m-%d %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


def _structured_plan(env: TravelEnv) -> Dict[str, List[str]]:
    state = env.state
    goal = env.goal

    # 目的地城市列表：优先使用已选城市序列，其次 fixed_city_order，最后 fallback 到 destination
    cities = list(
        state.city_sequence
        or goal.fixed_city_order
        or ([goal.destination] if goal.destination else [])
    )

    # ----------------------------
    # 1. 交通方式（先根据状态中已经选好的 segment_modes）
    # ----------------------------
    segments = env._segments(state)  # [(idx, src, dst), ...]
    transport_methods: List[str] = []

    for idx, src, dst in segments:
        seg = state.segment_modes.get(idx)
        if seg:
            detail = seg.get("detail", {})
            mode = seg.get("mode")
            if mode == "flight" and isinstance(detail, dict):
                transport_methods.append(
                    f"Flight {detail.get('id', '?')}, from {src} to {dst}, "
                    f"Departure Time: {detail.get('depart', '?')}, "
                    f"Arrival Time: {detail.get('arrive', '?')}, "
                    f"cost: {detail.get('price', detail.get('cost', '?'))}"
                )
            else:
                cost = None
                if isinstance(detail, dict):
                    cost = detail.get("cost")
                transport_methods.append(
                    f"{mode}, from {src} to {dst}, cost: {cost if cost is not None else '?'}"
                )
        else:
            transport_methods.append(f"{src}->{dst} missing mode")

    # ----------------------------
    # 2. 交通日期：根据“每天所在城市”推断每段交通发生的日期
    # ----------------------------
    transport_dates: List[str] = []
    if goal.start_date:
        start_dt = _parse_date_safe(goal.start_date)
        if start_dt:
            total_days = goal.duration_days or env.total_days

            for idx, src, dst in segments:
                transport_day = None  # 1-based day index

                # 特例：最后一段返回 origin，优先对齐到行程最后一天
                if dst == goal.origin and total_days:
                    transport_day = total_days
                else:
                    # 一般情况：找到第一个 day，使 day 所在城市 == 该段的目标城市 dst
                    for day in range(1, total_days + 1):
                        city_d = env._city_for_day(state, day)
                        if city_d == dst:
                            transport_day = day
                            break

                # 兜底逻辑（极端情况下找不到 dst 对应的 day）
                if transport_day is None:
                    if src == goal.origin:
                        # 出发段：默认第一天
                        transport_day = 1
                    elif dst == goal.origin and total_days:
                        # 返回段：默认最后一天
                        transport_day = total_days
                    else:
                        # 其他情况：用段编号近似映射到天数，但不超过 total_days
                        if total_days:
                            transport_day = min(idx + 1, total_days)
                        else:
                            transport_day = idx + 1  # 没有 total_days 信息时的兜底

                date_str = (start_dt + timedelta(days=transport_day - 1)).strftime("%Y-%m-%d")
                transport_dates.append(date_str)

    # ----------------------------
    # 3. 餐厅去重列表
    # ----------------------------
    restaurants_set: List[str] = []
    seen_rest = set()
    for day in range(1, env.total_days + 1):
        for slot, meal in state.meals.get(day, {}).items():
            if meal and meal["id"] not in seen_rest:
                restaurants_set.append(f"{meal['name']}, {meal.get('city', goal.destination)}")
                seen_rest.add(meal["id"])

    # ----------------------------
    # 4. 景点去重列表
    # ----------------------------
    attractions_set: List[str] = []
    seen_att = set()
    for day_map in state.attractions.values():
        for att in day_map.values():
            if att and att["id"] not in seen_att:
                attractions_set.append(f"{att['name']}, {att.get('city')}")
                seen_att.add(att["id"])

    # ----------------------------
    # 5. 每个城市的住宿
    # ----------------------------
    accommodations_list: List[str] = []
    for city in cities:
        stay = state.city_stays.get(city)
        if stay:
            accommodations_list.append(f"{stay['name']}, {city}")
        else:
            accommodations_list.append(f"missing, {city}")

    return {
        "destination_cities": cities,
        "transportation_dates": transport_dates,
        "transportation_methods": transport_methods,
        "restaurants": restaurants_set,
        "attractions": attractions_set,
        "accommodations": accommodations_list,
    }

def _state_hint(kb: TravelKnowledgeBase, name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    norm = kb._normalize_city(name)
    return kb.state_norm_map.get(norm) or kb.city_to_state_norm.get(norm)


def _goal_diagnostics(kb: TravelKnowledgeBase, goal: TripGoal) -> Dict[str, Any]:
    dest_state = _state_hint(kb, goal.destination)
    origin_state = _state_hint(kb, goal.origin)
    state_city_pool = kb.get_cities_for_state(dest_state) if dest_state else []
    must_states = {c: kb.get_state_for_city(c) or _state_hint(kb, c) for c in goal.must_visit_cities}
    pri_states = {c: kb.get_state_for_city(c) or _state_hint(kb, c) for c in goal.priority_cities}
    candidate_states = {c: kb.get_state_for_city(c) or _state_hint(kb, c) for c in goal.candidate_cities}
    return {
        "destination_state": dest_state,
        "origin_state": origin_state,
        "state_city_pool": state_city_pool,
        "candidate_cities": goal.candidate_cities,
        "must_city_states": must_states,
        "priority_city_states": pri_states,
        "candidate_city_states": candidate_states,
    }


def _extract_query_text(entry: Dict[str, Any]) -> str:
    for key in ("query", "instruction", "prompt", "input", "user_query", "text"):
        if key in entry and entry[key]:
            return str(entry[key])
    return ""


def _load_queries(args) -> List[str]:
    if load_dataset is None:
        raise ImportError("datasets package is required to load remote queries. Install via pip install datasets")
    split = args.set_type if args.set_type in ("train", "test", "validation") else "train"
    config = split if args.dataset_id == "osunlp/TravelPlanner" else None
    if config:
        ds = load_dataset(args.dataset_id, config, split=split)
    else:
        ds = load_dataset(args.dataset_id, split=split)
    return [_extract_query_text(x) for x in ds]


def _build_goal(parsed: Dict[str, Any], args, kb: TravelKnowledgeBase) -> TripGoal:
    visit_num = parsed.get("visiting_city_number")
    if not visit_num or visit_num <= 0:
        visit_num = args.visiting_city_number or 1
    must_cities = parsed.get("must_visit_cities") or args.must_city or []
    priority_cities = parsed.get("priority_cities") or args.priority_city or []
    fixed_city_order = parsed.get("fixed_city_order") or args.fixed_city_order or []
    candidate_cities = parsed.get("candidate_cities") or args.candidate_city or []
    if not (parsed.get("visiting_city_number") or args.visiting_city_number) and fixed_city_order:
        visit_num = len(fixed_city_order)

    allow_modes = parsed.get("transport_allow") or args.allow_transport or []
    forbid_modes = parsed.get("transport_forbid") or parsed.get("transport_forbidden") or args.forbid_transport or []
    raw_prefs = parsed.get("preferences") or args.preferences or []
    people_n = parsed.get("people_number") or 1

    constraints = route_preferences(
        raw_prefs=raw_prefs,
        transport_allow=allow_modes,
        transport_forbid=forbid_modes,
        people_number=people_n,
    )

    def _norm_modes(modes):
        if modes is None:
            return []
        if isinstance(modes, str):
            return [modes]
        return list(modes)

    allow_modes = [m.lower() for m in _norm_modes(allow_modes)] or None
    forbid_modes = [m.lower() for m in _norm_modes(forbid_modes)]
    require_flight = (not args.no_flight) and ("flight" not in forbid_modes)

    if kb and not candidate_cities and (parsed.get("destination") or args.destination):
        dest_hint = parsed.get("destination") or args.destination
        candidate_cities = kb.get_candidate_cities(
            destination_hint=dest_hint,
            must_visit=must_cities,
            priority=priority_cities,
            top_k=max(args.top_k * 2, visit_num),
        )

    return TripGoal(
        origin=parsed.get("origin") or args.origin,
        destination=parsed.get("destination") or args.destination,
        start_date=parsed.get("start_date") or args.start_date,
        duration_days=parsed.get("duration_days") or args.days,
        budget=parsed.get("budget") or args.budget,
        require_flight=require_flight,
        require_accommodation=not args.no_stay,
        num_restaurants=parsed.get("restaurants") or args.restaurants,
        num_attractions=args.attractions,
        preferences=raw_prefs,
        constraints=constraints,
        visiting_city_number=visit_num,
        must_visit_cities=must_cities,
        priority_cities=priority_cities,
        candidate_cities=candidate_cities,
        fixed_city_order=fixed_city_order,
        transport_allowed_modes=allow_modes,
        transport_forbidden_modes=forbid_modes,
        return_required=True,
        meals_per_day=3,
        attractions_per_day_min=parsed.get("attractions_min") or 1,
        attractions_per_day_max=parsed.get("attractions_max") or 1,
        notes=args.notes or parsed.get("notes"),
    )

    
def _run_single(goal: TripGoal, kb: TravelKnowledgeBase, policy: TravelLLMPolicy, args, raw_query: str = ""):
    # Reuse the same LLM endpoint as NL parser for filter generation (no extra CLI needed).
    plan_llm = None
    model_for_plan = args.parser_model or args.local_model
    base_for_plan = args.local_base
    if args.use_llm_filters and model_for_plan and base_for_plan:
        def _plan_call(prompt: str, base=base_for_plan, model=model_for_plan):
            return _call_local_llm(base, model, prompt, timeout=args.parser_timeout)
        plan_llm = _plan_call

    phase_planner = PhasePlanGenerator(llm=plan_llm, enable=args.use_llm_filters)

    env = TravelEnv(
        kb,
        goal,
        max_steps=args.max_episode_len,
        top_k=args.top_k,
        debug=args.debug,
        candidate_cap=args.candidate_cap,
        use_llm_filters=args.use_llm_filters,
        relax_max_tries=args.relax_max_tries,
        user_query=raw_query,
        log_filter_usage=args.log_filter_usage,
        phase_planner=phase_planner,
    )
    mcts_args = argparse.Namespace(
        exploration_constant=args.exploration_constant,
        bonus_constant=args.bonus_constant,
        max_depth=args.max_depth,
        simulation_per_act=args.simulation_per_act,
        discount_factor=args.discount_factor,
        simulation_num=args.simulation_num,
        uct_type=args.uct_type,
        round=0,
        seed=args.seed,
        model=args.local_model,
        debug=args.debug,
        use_llm_prior=args.use_llm_prior,
    )
    agent = MCTSAgent(
        mcts_args,
        env,
        policy=policy,
        uct_type=args.uct_type,
        use_llm=args.use_llm_prior != "none",
    )

    obs, valid_actions = env.reset()
    history = list(env.base_history)
    done = False
    plan_actions = []
    for step in range(args.max_episode_len):
        action = agent.search(obs, history, step, valid_actions, done)
        if action is None:
            print("[WARN] MCTS returned None, terminating early")
            break
        obs, reward, done, history, valid_actions = env.apply_action(action)
        plan_actions.append(action)
        if done:
            break
    success = env.is_success(env.state)
    return {
        "success": success,
        "actions": plan_actions,
        "cost": env.state.cost,
        "violations": env.state.violations,
        "state": env.state,
        "goal": goal,
        "env": env,
        "filter_usage": list(env.filter_events),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Batch run travel MCTS over queries from a dataset or local CSV.")
    parser.add_argument("--set-type", default="train", choices=["train", "validation", "test"],
                        help="HF dataset split to load.")
    parser.add_argument("--dataset-id", default="osunlp/TravelPlanner", help="HF dataset id to load queries from.")
    parser.add_argument("--origin", default=None, help="Override origin city for all queries.")
    parser.add_argument("--destination", default=None, help="Override destination city/state for all queries.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--visiting-city-number", type=int, default=None, help="Target number of cities to visit.")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--restaurants", type=int, default=3)
    parser.add_argument("--attractions", type=int, default=2)
    parser.add_argument("--preference", action="append", dest="preferences", default=[])
    parser.add_argument("--no-flight", action="store_true", help="Do not require a flight.")
    parser.add_argument("--no-stay", action="store_true", help="Do not require accommodation.")
    parser.add_argument("--must-city", action="append", dest="must_city", default=[],
                        help="City that must appear in the plan (can be repeated).")
    parser.add_argument("--priority-city", action="append", dest="priority_city", default=[],
                        help="Preferred city to include (can be repeated).")
    parser.add_argument("--fixed-city-order", action="append", dest="fixed_city_order", default=[],
                        help="Explicit city order; list cities in order by repeating the flag.")
    parser.add_argument("--candidate-city", action="append", dest="candidate_city", default=[],
                        help="Preset candidate city list (overrides search).")
    parser.add_argument("--allow-transport", action="append", dest="allow_transport", default=[],
                        help="Allow only these transport modes (flight, taxi, self-driving).")
    parser.add_argument("--forbid-transport", action="append", dest="forbid_transport", default=[],
                        help="Ban these transport modes (flight, taxi, self-driving).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1, help="Max number of queries to process.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into query list (0-based).")
    # MCTS params
    parser.add_argument("--exploration-constant", type=float, default=8.0)
    parser.add_argument("--bonus-constant", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-episode-len", type=int, default=40)
    parser.add_argument("--simulation-per-act", type=int, default=1)
    parser.add_argument("--simulation-num", type=int, default=50)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--candidate-cap", type=int, default=80, help="Max candidates returned per slot.")
    parser.add_argument(
        "--use-llm-filters",
        dest="use_llm_filters",
        action="store_true",
        help="Enable LLM filter generation for candidate retrieval.",
    )
    parser.add_argument(
        "--no-llm-filters",
        dest="use_llm_filters",
        action="store_false",
        help="Disable LLM filter generation (use defaults only).",
    )
    parser.add_argument(
        "--use-llm-prior",
        choices=["root", "all", "none"],
        default="root",
        help="Apply LLM policy priors at root, all nodes, or disable.",
    )
    parser.add_argument(
        "--relax-max-tries",
        type=int,
        default=6,
        help="Max relaxation attempts when KB query returns empty candidates.",
    )
    parser.add_argument(
        "--log-filter-usage",
        action="store_true",
        help="Print filters and whether LLM/cache was used when building candidates.",
    )
    parser.add_argument("--uct-type", default="PUCT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    # LLM params
    parser.add_argument("--local-model", default=None, help="Local LLM for action priors and NL parsing.")
    parser.add_argument("--local-base", default=os.getenv("LOCAL_LLM_BASE", "http://localhost:11434"))
    parser.add_argument("--device", default="cpu", help="cpu/mps/cuda:0 etc.")
    parser.add_argument("--parser-model", default=None, help="Parser model (defaults to --local-model).")
    parser.add_argument("--parser-timeout", type=float, default=180.0)
    parser.add_argument("--database-root", default="database")
    parser.add_argument("--save-dir", default="plans_out", help="Directory to save generated plans as JSON.")
    parser.add_argument("--notes", default=None, help="Optional notes saved with each plan.")
    parser.set_defaults(use_llm_filters=True)
    return parser.parse_args()


def main():
    args = parse_args()
    queries = _load_queries(args)
    if not queries:
        raise RuntimeError("No queries loaded. Check dataset path or network.")
    subset = queries[args.start_index: args.start_index + args.limit]

    parser_model = args.parser_model or args.local_model or "deepseek-r1:14b"
    kb = TravelKnowledgeBase(args.database_root)
    policy = TravelLLMPolicy(device=args.device, model_path=args.local_model, embedding_model="all-MiniLM-L6-v2")
    os.makedirs(args.save_dir, exist_ok=True)

    for idx, q in enumerate(subset, start=args.start_index + 1):
        # 先输出原始 query，方便观察
        print(f"\n=== Query {idx} (raw) === {q}")
        parsed = _parse_nl_query(q, args.local_base, parser_model, timeout=args.parser_timeout)
        print(f"LLM parsed JSON: {json.dumps(parsed, ensure_ascii=False)}")
        if not parsed.get("origin") or not parsed.get("destination"):
            fallback = _fallback_parse(q)
            parsed = {**fallback, **parsed}
        if not parsed.get("origin") or not parsed.get("destination"):
            print(f"[{idx}] skip (parse failed): {q}")
            continue
        goal = _build_goal(parsed, args, kb)
        # Debug: show candidate city list after state expansion
        print(f"[DEBUG] Goal candidate_cities: {goal.candidate_cities}")
        t0 = time.perf_counter()
        result = _run_single(goal, kb, policy, args, raw_query=q)
        elapsed = time.perf_counter() - t0
        structured = _structured_plan(result["env"])
        diagnostics = _goal_diagnostics(kb, goal)
        filename = f"{idx:03d}_{goal.origin}_to_{goal.destination}_{goal.start_date or 'na'}_{goal.duration_days or 'days'}d.json"
        safe_name = filename.replace(" ", "_").replace("/", "-")
        out_path = os.path.join(args.save_dir, safe_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "query": q,
                    "parsed": parsed,
                    "goal": goal.as_text(),
                    "actions": result["actions"],
                    "success": result["success"],
                    "cost": result["cost"],
                    "violations": result["violations"],
                    "structured_plan": structured,
                    "filter_usage": result.get("filter_usage", []),
                    "elapsed_seconds": elapsed,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n=== Query {idx} ===")
        print(f"Raw: {q}")
        print(f"Parsed goal: {goal.as_text()}")
        print(f"Success: {result['success']} | Cost: {result['cost']:.2f} | Violations: {result['violations']}")
        print(f"Actions: {result['actions']}")
        print(f"Dest state hint: {diagnostics['destination_state']} | Candidate cities seeded: {diagnostics['candidate_cities']}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
