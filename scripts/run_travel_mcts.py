import argparse
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, Optional, List

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcts.mcts.mcts import MCTSAgent
from mcts.travel.knowledge_base import TravelKnowledgeBase, TripGoal
from mcts.travel.travel_env import TravelEnv
from mcts.travel.llm_policy import TravelLLMPolicy


def _print_plan(env: TravelEnv, success: bool, actions):
    state = env.state
    goal = env.goal
    budget = goal.budget if goal.budget is not None else "n/a"
    print(f"Success: {success}")
    print(f"Actions taken: {actions}")
    print(f"Total cost: {state.cost:.2f} / {budget}")
    print(f"Violations: {state.violations or 'none'}")

    city_seq = state.city_sequence or goal.fixed_city_order
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

    if goal.require_accommodation and city_seq:
        for city in city_seq:
            stay = state.city_stays.get(city)
            if stay:
                print(f"Stay in {city}: {stay['name']} ({stay['room_type']}) ${stay['price']:.0f}")
            else:
                print(f"Stay in {city}: missing")
    elif goal.require_accommodation:
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
    goal = env.goal
    cities = list(state.city_sequence or goal.fixed_city_order or ([goal.destination] if goal.destination else []))

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
    if goal.start_date:
        start_dt = _parse_date_safe(goal.start_date)
        if start_dt:
            for i in range(len(transport_methods)):
                transport_dates.append((start_dt + timedelta(days=i)).strftime("%Y-%m-%d"))

    restaurants_set = []
    seen_rest = set()
    for day in range(1, env.total_days + 1):
        for slot, meal in state.meals.get(day, {}).items():
            if meal and meal["id"] not in seen_rest:
                restaurants_set.append(f"{meal['name']}, {meal.get('city', goal.destination)}")
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


def _parse_nl_query(nl_query: str, base_url: str, model: str, timeout: float = 60.0) -> Dict[str, Any]:
    prompt = (
        "Extract ONLY a JSON with fields: org, dest, days (int), visiting_city_number (int), "
        "date (array of ISO dates, length=days), people_number (int), "
        "local_constraint (object with keys: \"house rule\", \"cuisine\" (array or null), \"room type\", \"transportation\"), "
        "budget (number). Use null for missing scalars/arrays. Do not add extra fields.\n\n"
        "Example:\n"
        "Query: Could you arrange a 3-day solo trip for me starting from Ontario and heading to Honolulu spanning from March 4th to March 6th, 2022, with a total budget of $3,200?\n"
        "JSON: {\"org\": \"Ontario\", \"dest\": \"Honolulu\", \"days\": 3, \"visiting_city_number\": 1, \"date\": [\"2022-03-04\",\"2022-03-05\",\"2022-03-06\"], \"people_number\": 1, \"local_constraint\": {\"house rule\": null, \"cuisine\": null, \"room type\": null, \"transportation\": null}, \"budget\": 3200}\n\n"
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
    parser.add_argument("--top-k", type=int, default=5, help="Limit of candidates per category.")

    # MCTS params
    parser.add_argument("--exploration-constant", type=float, default=8.0)
    parser.add_argument("--bonus-constant", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-episode-len", type=int, default=30)
    parser.add_argument("--simulation-per-act", type=int, default=1)
    parser.add_argument("--simulation-num", type=int, default=50)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--uct-type", default="PUCT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Print debug info for root stats and rewards.")

    # LLM params
    parser.add_argument("--local-model", default="deepseek-r1:14b", help="Path/name of a local transformers model for scoring actions.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name for similarity scoring.")
    parser.add_argument("--device", default="mps", help="Device string for LLM/embeddings, e.g. cuda:0, mps, or cpu.")
    parser.add_argument("--parser-model", default=None, help="Model name/path for NL parsing (defaults to --local-model).")
    parser.add_argument("--parser-timeout", type=float, default=60.0, help="Timeout (s) for NL parser call.")
    parser.add_argument("--plan-llm-lambda", type=float, default=0.0, help="Weight for plan-level LLM rollout scoring (terminal).")
    return parser.parse_args()


def main():
    args = parse_args()
    parsed = {}
    if args.nl_query:
        parser_model = args.parser_model or args.local_model or "deepseek-r1:14b"
        parsed = _parse_nl_query(args.nl_query, args.local_base, parser_model, timeout=args.parser_timeout)

    visit_num = parsed.get("visiting_city_number") or parsed.get("visiting_city_number".replace("_", "")) or args.visiting_city_number or 1
    origin_parsed = parsed.get("origin") or parsed.get("org")
    destination_parsed = parsed.get("destination") or parsed.get("dest")
    start_date_parsed = parsed.get("start_date") or (parsed.get("date")[0] if parsed.get("date") else None)
    duration_parsed = parsed.get("duration_days") or parsed.get("days")
    must_cities = parsed.get("must_visit_cities") or args.must_city or []
    priority_cities = parsed.get("priority_cities") or args.priority_city or []
    fixed_city_order = parsed.get("fixed_city_order") or args.fixed_city_order or []
    candidate_cities = parsed.get("candidate_cities") or args.candidate_city or []
    if not (parsed.get("visiting_city_number") or args.visiting_city_number) and fixed_city_order:
        visit_num = len(fixed_city_order)
    local_transport = None
    local_constraint = parsed.get("local_constraint") or {}
    if isinstance(local_constraint, dict):
        local_transport = local_constraint.get("transportation")
        local_room_type = local_constraint.get("room type")
        local_house_rule = local_constraint.get("house rule")
        local_cuisine = local_constraint.get("cuisine") or []
    else:
        local_transport = None
        local_room_type = None
        local_house_rule = None
        local_cuisine = []

    allow_modes = parsed.get("transport_allow") or args.allow_transport or []
    forbid_modes = parsed.get("transport_forbid") or parsed.get("transport_forbidden") or args.forbid_transport or []
    if local_transport:
        if isinstance(local_transport, str):
            if "no flight" in local_transport:
                forbid_modes = forbid_modes or []
                forbid_modes.append("flight")
            if "no self-driving" in local_transport or "no driving" in local_transport:
                forbid_modes = forbid_modes or []
                forbid_modes.append("self-driving")
        if isinstance(local_transport, list):
            for t in local_transport:
                if isinstance(t, str) and "no flight" in t:
                    forbid_modes.append("flight")
                if isinstance(t, str) and ("no self-driving" in t or "no driving" in t):
                    forbid_modes.append("self-driving")
    def _norm_modes(modes):
        if modes is None:
            return []
        if isinstance(modes, str):
            return [modes]
        return list(modes)
    allow_modes = [m.lower() for m in _norm_modes(allow_modes)] or None
    forbid_modes = [m.lower() for m in _norm_modes(forbid_modes)]
    require_flight = (not args.no_flight) and ("flight" not in forbid_modes)

    goal = TripGoal(
        origin=origin_parsed or args.origin,
        destination=destination_parsed or args.destination,
        start_date=start_date_parsed or args.start_date,
        duration_days=duration_parsed or args.days,
        budget=parsed.get("budget") or args.budget,
        require_flight=require_flight,
        require_accommodation=not args.no_stay,
        visiting_city_number=visit_num,
        num_restaurants=args.restaurants,
        num_attractions=args.attractions,
        preferences=parsed.get("preferences") or local_cuisine or args.preferences,
        people_number=parsed.get("people_number") or parsed.get("people") or parsed.get("num_people") or 1,
        must_visit_cities=must_cities,
        priority_cities=priority_cities,
        candidate_cities=candidate_cities,
        fixed_city_order=fixed_city_order,
        transport_allowed_modes=allow_modes,
        transport_forbidden_modes=forbid_modes,
        room_type=local_room_type,
        house_rule=local_house_rule,
        required_cuisines=local_cuisine if isinstance(local_cuisine, list) else [],
        notes=args.notes,
        return_required=True,
        meals_per_day=3,
        attractions_per_day_min=parsed.get("attractions_min") or 1,
        attractions_per_day_max=parsed.get("attractions_max") or 1,
    )
    if not goal.origin or not goal.destination:
        raise ValueError("Origin and destination must be provided via arguments or NL query.")

    kb = TravelKnowledgeBase(args.database_root)
    env = TravelEnv(kb, goal, max_steps=args.max_episode_len, top_k=args.top_k)
    policy = TravelLLMPolicy(device=args.device, model_path=args.local_model, embedding_model=args.embedding_model)

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
        plan_llm_lambda=args.plan_llm_lambda,
    )
    agent = MCTSAgent(mcts_args, env, policy=policy, uct_type=args.uct_type, use_llm=True)

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
    print("\nFinal plan:")
    _print_plan(env, success, plan_actions)


if __name__ == "__main__":
    main()
