import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, Optional, List

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcts.mcts.mcts import MCTSAgent
from mcts.travel.knowledge_base import TravelKnowledgeBase
from mcts.travel.travel_env import TravelEnv
from mcts.travel.agents import SemanticAgent
from mcts.travel.query_parsing import normalize_parsed_query


def _goal_get(goal: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(goal, dict):
        for key in keys:
            if key in goal and goal[key] is not None:
                return goal[key]
        return default
    for key in keys:
        if hasattr(goal, key):
            val = getattr(goal, key)
            if val is not None:
                return val
    return default


def _merge_args_into_parsed(parsed: Dict[str, Any], args) -> Dict[str, Any]:
    out = dict(parsed or {})
    if out.get("origin") is None and getattr(args, "origin", None):
        out["origin"] = args.origin
    if out.get("destination") is None and getattr(args, "destination", None):
        out["destination"] = args.destination
    if out.get("start_date") is None and getattr(args, "start_date", None):
        out["start_date"] = args.start_date
    if out.get("duration_days") is None and getattr(args, "days", None):
        out["duration_days"] = args.days
    if out.get("visiting_city_number") is None and getattr(args, "visiting_city_number", None):
        out["visiting_city_number"] = args.visiting_city_number
    if out.get("budget") is None and getattr(args, "budget", None):
        out["budget"] = args.budget
    return out


def _print_plan(env: TravelEnv, success: bool, actions):
    state = env.state
    goal = env.goal_parsed or {}
    budget = _goal_get(goal, "budget", default="n/a")
    print(f"Success: {success}")
    print(f"Actions taken: {actions}")
    print(f"Total cost: {state.cost:.2f} / {budget}")
    print(f"Violations: {state.violations or 'none'}")

    city_seq = state.city_sequence or (_goal_get(goal, "fixed_city_order", default=[]) or [])
    print(f"Cities: {city_seq or 'missing'}")
    segments = env._segments(state)
    for idx, src, dst in segments:
        seg = state.segment_modes.get(idx)
        if seg:
            detail = seg.get("detail", {})
            mode = seg.get("mode")
            if isinstance(detail, dict) and "id" in detail:
                print(f"Segment {idx}: {mode} {detail['id']} {src}->{dst}")
            else:
                print(f"Segment {idx}: {mode} {src}->{dst}")
        else:
            print(f"Segment {idx}: {src}->{dst} missing mode")

    if _goal_get(goal, "require_accommodation", default=True) and city_seq:
        for city in city_seq:
            stay = state.city_stays.get(city)
            if stay:
                print(f"Stay in {city}: {stay['name']} ({stay['room_type']}) ${stay['price']:.0f}")
            else:
                print(f"Stay in {city}: missing")
    elif _goal_get(goal, "require_accommodation", default=True):
        print("Stay: missing")

    for day in range(1, env.total_days + 1):
        city = env._city_for_day(state, day)
        print(f"Day {day} ({city}):")
        meals = state.meals[day]
        for slot in env.meal_slots:
            meal = meals.get(slot)
            txt = f"{slot}: {meal['name']} ({meal['cuisines']}) ${meal['cost']:.0f} rating {meal['rating']}" if meal else f"{slot}: missing"
            print(f"  {txt}")
        atts = state.attractions[day]
        att_lines = []
        for slot in env.attraction_slots:
            att = atts.get(slot)
            att_lines.append(f"{slot}: {att['name']}" if att else f"{slot}: empty")
        print("  Attractions -> " + "; ".join(att_lines))

    summary = _structured_plan(env)
    print("\nStructured plan:")
    print(f"Destination cities: {summary['destination_cities']}")
    print(f"Transportation dates: {summary['transportation_dates']}")
    print(f"Transportation methods: {summary['transportation_methods']}")
    print(f"Restaurants: {summary['restaurants']}")
    print(f"Attractions: {summary['attractions']}")
    print(f"Accommodations: {summary['accommodations']}")


def _call_local_llm(base_url: str, model: str, prompt: str, timeout: float = 60.0) -> Optional[str]:
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False,
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
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
    goal = env.goal_parsed or {}
    destination = _goal_get(goal, "destination", "dest", default=None)
    fixed_order = _goal_get(goal, "fixed_city_order", default=[]) or []
    cities = list(state.city_sequence or fixed_order or ([destination] if destination else []))

    segments = env._segments(state)
    transport_methods: List[str] = []
    for idx, src, dst in segments:
        seg = state.segment_modes.get(idx)
        if seg:
            detail = seg.get("detail", {})
            mode = seg.get("mode")
            if mode == "flight" and isinstance(detail, dict):
                transport_methods.append(
                    f"Flight {detail.get('id', '?')}, from {src} to {dst}, "
                    f"Departure Time: {detail.get('depart', '?')}, Arrival Time: {detail.get('arrive', '?')}, "
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

    transport_dates: List[str] = []
    start_date = _goal_get(goal, "start_date", default=None)
    if start_date:
        start_dt = _parse_date_safe(start_date)
        if start_dt:
            for i in range(len(transport_methods)):
                transport_dates.append((start_dt + timedelta(days=i)).strftime("%Y-%m-%d"))

    restaurants_set = []
    seen_rest = set()
    for day in range(1, env.total_days + 1):
        for slot, meal in state.meals.get(day, {}).items():
            if meal and meal["id"] not in seen_rest:
                restaurants_set.append(f"{meal['name']}, {meal.get('city', destination)}")
                seen_rest.add(meal["id"])

    attractions_set = []
    seen_att = set()
    for day_map in state.attractions.values():
        for att in day_map.values():
            if att and att["id"] not in seen_att:
                attractions_set.append(f"{att['name']}, {att.get('city')}")
                seen_att.add(att["id"])

    accommodations_list = []
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


def _extract_query_text(entry: Dict[str, Any]) -> str:
    for key in ("query", "instruction", "prompt", "input", "user_query", "text", "nl_query"):
        if key in entry and entry[key]:
            return str(entry[key])
    return ""


def _load_goal_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # If the file contains a list, take the first element for single-run use.
    if isinstance(data, list):
        data = data[0] if data else {}
    # Prefer already parsed goal fields if present.
    if isinstance(data, dict):
        if isinstance(data.get("parsed"), dict):
            parsed = data["parsed"]
        elif isinstance(data.get("goal"), dict):
            parsed = data["goal"]
        else:
            parsed = data
    else:
        parsed = data
    parsed["__query_text"] = _extract_query_text(data)
    return parsed


def _load_goal_json_inline(text: str) -> Dict[str, Any]:
    """Parse inline JSON (debug helper)."""
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _parse_nl_query(nl_query: str, base_url: str, model: str, timeout: float = 60.0) -> Dict[str, Any]:
    prompt = (
        "Extract a JSON with fields: origin, destination, start_date, duration_days (int), budget (number), "
        "visiting_city_number (int), must_visit_cities (array of strings), priority_cities (array of strings), "
        "fixed_city_order (array of strings or null), transport_allow (array of strings or null), "
        "transport_forbid (array of strings or null), preferences (array of strings), people_number (int or null). "
        "Allowed transport values: \"flight\", \"taxi\", \"self-driving\". "
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
        import json
        return json.loads(raw)
    except Exception:
        return {}


def _default_output_path(parsed: Dict[str, Any], args) -> str:
    """Consistent filename under plans_out/<split>/<idx>_origin_to_dest_date_daysd.json."""
    directory = os.path.join("plans_out", args.set_type)
    origin = parsed.get("origin") or parsed.get("org") or "origin"
    dest = parsed.get("destination") or parsed.get("dest") or "dest"
    start_date = parsed.get("start_date") or "na"
    days = parsed.get("duration_days") or parsed.get("days") or "days"
    name = f"{args.query_index:03d}_{origin}_to_{dest}_{start_date}_{days}d.json"
    safe_name = name.replace(" ", "_").replace("/", "-")
    return os.path.join(directory, safe_name)


def _save_plan(
    output_path: str,
    query_text: Optional[str],
    parsed: Dict[str, Any],
    actions: List[str],
    success: bool,
    env: TravelEnv,
    elapsed_seconds: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    structured = _structured_plan(env)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "query": query_text,
                "parsed": {k: v for k, v in parsed.items() if k != "__query_text"},
                "actions": actions,
                "success": success,
                "cost": env.state.cost,
                "violations": env.state.violations,
                "structured_plan": structured,
                "filter_usage": getattr(env, "filter_events", []),
                "elapsed_seconds": elapsed_seconds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run MCTS over the travel dataset with a local LLM policy.")
    parser.add_argument("--origin", help="Origin city for flights.")
    parser.add_argument("--destination", help="Destination city.")
    parser.add_argument("--start-date", default=None, help="Optional start date string.")
    parser.add_argument("--days", type=int, default=None, help="Trip duration in days.")
    parser.add_argument("--visiting-city-number", type=int, default=None, help="Target number of cities to visit.")
    parser.add_argument("--budget", type=float, default=None, help="Overall budget in USD.")
    parser.add_argument("--restaurants", type=int, default=1, help="Number of restaurants to include.")
    parser.add_argument("--attractions", type=int, default=1, help="Number of attractions to include.")
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
    parser.add_argument("--preference", action="append", dest="preferences", default=[],
                        help="Cuisine or style preference. Can be passed multiple times.")
    parser.add_argument("--notes", default=None, help="Free-form notes passed to the policy.")
    parser.add_argument("--nl-query", default=None, help="Natural language trip request to be parsed by local LLM.")
    parser.add_argument("--local-base", default=os.getenv("LOCAL_LLM_BASE", "http://localhost:11434"),
                        help="Local LLM base URL for NL parsing (Ollama style).")
    parser.add_argument("--database-root", default="database", help="Path to the tabular dataset.")
    parser.add_argument("--top-k", type=int, default=30, help="Limit of candidates per category.")
    parser.add_argument("--goal-json", default=None, help="Path to JSON file with parsed trip fields.")
    parser.add_argument("--parsed-json", default=None, help="Inline JSON string with parsed trip fields (debug).")
    parser.add_argument("--output-path", default=None, help="Path to save the generated plan JSON.")
    parser.add_argument("--set-type", default="train", choices=["train", "validation", "test"],
                        help="Output split folder (train/validation/test).")
    parser.add_argument("--query-index", type=int, default=0,
                        help="Index number used in the default output filename.")

    # MCTS params
    parser.add_argument("--exploration-constant", type=float, default=8.0)
    parser.add_argument("--bonus-constant", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-episode-len", type=int, default=30)
    parser.add_argument("--simulation-per-act", type=int, default=1)
    parser.add_argument("--simulation-num", type=int, default=30)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--candidate-cap", type=int, default=80, help="Max candidates returned per slot.")
    parser.add_argument(
        "--use-llm-filters",
        dest="use_llm_filters",
        action="store_true",
        help="Deprecated (no-op): LLM filter generation removed.",
    )
    parser.add_argument(
        "--no-llm-filters",
        dest="use_llm_filters",
        action="store_false",
        help="Deprecated (no-op): LLM filter generation removed.",
    )
    parser.add_argument(
        "--use-llm-prior",
        choices=["root", "all", "none"],
        default="root",
        help="Apply LLM priors at root, all nodes, or disable.",
    )
    parser.add_argument(
        "--relax-max-tries",
        type=int,
        default=6,
        help="Deprecated (no-op): relaxation removed.",
    )
    parser.add_argument(
        "--log-filter-usage",
        action="store_true",
        help="Print candidate retrieval details and cache usage.",
    )
    parser.add_argument("--uct-type", default="PUCT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Print debug info for root stats and rewards.")

    # LLM params
    parser.add_argument("--local-model", default=None, help="Path/name of a local transformers model for scoring actions.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name for similarity scoring.")
    parser.add_argument("--device", default="mps", help="Device string for LLM/embeddings, e.g. cuda:0, mps, or cpu.")
    parser.add_argument("--parser-model", default=None, help="Model name/path for NL parsing (defaults to --local-model).")
    parser.add_argument("--parser-timeout", type=float, default=60.0, help="Timeout (s) for NL parser call.")
    parser.set_defaults(use_llm_filters=False)
    return parser.parse_args()


def main():
    args = parse_args()
    parsed: Dict[str, Any] = {}
    query_text: Optional[str] = None
    if args.parsed_json:
        parsed = _load_goal_json_inline(args.parsed_json)
        query_text = parsed.pop("__query_text", None)
        print(f"[DEBUG] Parsed JSON (inline): {json.dumps(parsed, ensure_ascii=False)}")
    elif args.goal_json:
        parsed = _load_goal_json(args.goal_json)
        query_text = parsed.pop("__query_text", None)
    elif args.nl_query:
        parser_model = args.parser_model or args.local_model or "deepseek-r1:14b"
        parsed = _parse_nl_query(args.nl_query, args.local_base, parser_model, timeout=args.parser_timeout)
        print(f"[DEBUG] Parsed JSON (LLM): {json.dumps(parsed, ensure_ascii=False)}")
        query_text = query_text or args.nl_query
    else:
        # no parsing source; rely on CLI args
        parsed = {}

    parsed = normalize_parsed_query(parsed)
    parsed = _merge_args_into_parsed(parsed, args)
    if not parsed.get("origin") or not parsed.get("destination"):
        raise ValueError("Origin and destination must be provided via JSON, arguments, or NL query.")

    kb = TravelKnowledgeBase(args.database_root)
    semantic = SemanticAgent(embedding_model=args.embedding_model)

    env = TravelEnv(
        kb,
        max_steps=args.max_episode_len,
        top_k=args.top_k,
        debug=args.debug,
        candidate_cap=args.candidate_cap,
        user_query=query_text or args.nl_query or "",
        log_filter_usage=args.log_filter_usage,
        goal_parsed=parsed,
    )
    policy = semantic.build_policy(args)

    mcts_args = SimpleNamespace(
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

    t0 = time.perf_counter()
    obs, valid_actions = env.reset()
    history = list(env.base_history)
    done = False
    plan_actions = []

    print("Goal:", env.get_goal())
    for step in range(args.max_episode_len):
        action = agent.search(obs, history, step, valid_actions, done)
        obs, reward, done, history, valid_actions = env.apply_action(action)
        plan_actions.append(action)
        print(f"Step {step}: {action}")
        print(f"Reward: {reward:.2f} | Cost: {env.state.cost:.2f} | Violations: {env.state.violations}")
        if args.debug and agent.root:
            stats = agent.root_statistics()
            if stats:
                print("Root stats (top):")
                for s in stats:
                    print(f"  {s['action']} -> Q={s['Q']:.3f}, N={s['N']}")
        print(f"Observation: {obs}")
        if done:
            print("Episode finished.")
            break

    success = env.is_success(env.state)
    elapsed = time.perf_counter() - t0
    print("\nFinal plan:")
    _print_plan(env, success, plan_actions)

    output_path = args.output_path
    if not output_path:
        output_path = _default_output_path(parsed, args)
    if output_path:
        _save_plan(output_path, query_text, parsed, plan_actions, success, env, elapsed_seconds=elapsed)
        print(f"\nSaved plan JSON to: {output_path} | Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
