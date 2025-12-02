import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional

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

    if state.outbound_flight:
        f = state.outbound_flight
        print(f"Outbound: {f['id']} {f['origin']}->{f['destination']} depart {f['depart']} arrive {f['arrive']} ${f['price']:.0f}")
    else:
        print("Outbound: missing")

    if goal.return_required:
        if state.return_flight:
            f = state.return_flight
            print(f"Return: {f['id']} {f['origin']}->{f['destination']} depart {f['depart']} arrive {f['arrive']} ${f['price']:.0f}")
        else:
            print("Return: missing")

    if state.accommodation:
        s = state.accommodation
        print(f"Stay: {s['name']} in {s['city']} ({s['room_type']}) ${s['price']:.0f}")
    elif goal.require_accommodation:
        print("Stay: missing")

    for day in range(1, env.total_days + 1):
        print(f"Day {day}:")
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


def _parse_nl_query(nl_query: str, base_url: str, model: str, timeout: float = 60.0) -> Dict[str, Any]:
    prompt = (
        "Extract a JSON with fields: origin, destination, start_date, duration_days (int), "
        "budget (number), preferences (array of strings), restaurants (int), attractions_min (int), attractions_max (int). "
        "Use null if missing. Return ONLY JSON.\n"
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
    parser.add_argument("--budget", type=float, default=None, help="Overall budget in USD.")
    parser.add_argument("--restaurants", type=int, default=1, help="Number of restaurants to include.")
    parser.add_argument("--attractions", type=int, default=1, help="Number of attractions to include.")
    parser.add_argument("--no-flight", action="store_true", help="Do not require a flight.")
    parser.add_argument("--no-stay", action="store_true", help="Do not require accommodation.")
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
    parser.add_argument("--local-model", default=None, help="Path/name of a local transformers model for scoring actions.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name for similarity scoring.")
    parser.add_argument("--device", default="cpu", help="Device string for LLM/embeddings, e.g. cuda:0, mps, or cpu.")
    parser.add_argument("--parser-model", default=None, help="Model name/path for NL parsing (defaults to --local-model).")
    parser.add_argument("--parser-timeout", type=float, default=60.0, help="Timeout (s) for NL parser call.")
    return parser.parse_args()


def main():
    args = parse_args()
    parsed = {}
    if args.nl_query:
        parser_model = args.parser_model or args.local_model or "deepseek-r1:14b"
        parsed = _parse_nl_query(args.nl_query, args.local_base, parser_model, timeout=args.parser_timeout)

    goal = TripGoal(
        origin=parsed.get("origin") or args.origin,
        destination=parsed.get("destination") or args.destination,
        start_date=parsed.get("start_date") or args.start_date,
        duration_days=parsed.get("duration_days") or args.days,
        budget=parsed.get("budget") or args.budget,
        require_flight=not args.no_flight,
        require_accommodation=not args.no_stay,
        num_restaurants=args.restaurants,
        num_attractions=args.attractions,
        preferences=parsed.get("preferences") or args.preferences,
        notes=args.notes,
        return_required=True,
        meals_per_day=3,
        attractions_per_day_min=parsed.get("attractions_min") or 2,
        attractions_per_day_max=parsed.get("attractions_max") or 3,
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
