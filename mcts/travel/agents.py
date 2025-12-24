from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from mcts.mcts.mcts import MCTSAgent
from mcts.travel.filtering import PhasePlanGenerator
from mcts.travel.knowledge_base import TripGoal, TravelKnowledgeBase
from mcts.travel.llm_policy import TravelLLMPolicy
from mcts.travel.preference_router import route_preferences
from mcts.travel.retrieval_agent import RetrievalAgent
from mcts.travel.travel_env import TravelEnv as EnvAgent

__all__ = [
    "SemanticAgent",
    "RetrievalAgent",
    "EnvAgent",
    "MCTSPlanningAgent",
]


class SemanticAgent:
    """Semantic understanding agent: NL prefs -> structured constraints, plus LLM-based helpers."""

    def __init__(self, *, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model

    def build_policy(self, args) -> TravelLLMPolicy:
        return TravelLLMPolicy(
            device=getattr(args, "device", "cpu"),
            model_path=getattr(args, "local_model", None),
            embedding_model=self.embedding_model,
            local_base=getattr(args, "local_base", None),
            model_name=getattr(args, "local_model", None),
        )

    def build_phase_planner(self, args, llm_call: Optional[Any] = None) -> PhasePlanGenerator:
        enable = bool(getattr(args, "use_llm_filters", False))
        return PhasePlanGenerator(llm_client=llm_call, enable=enable)

    def build_goal(self, parsed: Dict[str, Any], args, kb: Optional[TravelKnowledgeBase]) -> TripGoal:
        visit_num = parsed.get("visiting_city_number")
        if not visit_num or visit_num <= 0:
            visit_num = getattr(args, "visiting_city_number", None) or 1

        must_cities = parsed.get("must_visit_cities") or getattr(args, "must_city", None) or []
        priority_cities = parsed.get("priority_cities") or getattr(args, "priority_city", None) or []
        fixed_city_order = parsed.get("fixed_city_order") or getattr(args, "fixed_city_order", None) or []
        candidate_cities = parsed.get("candidate_cities") or getattr(args, "candidate_city", None) or []
        if not (parsed.get("visiting_city_number") or getattr(args, "visiting_city_number", None)) and fixed_city_order:
            visit_num = len(fixed_city_order)

        allow_modes = parsed.get("transport_allow") or getattr(args, "allow_transport", None) or []
        forbid_modes = (
            parsed.get("transport_forbid")
            or parsed.get("transport_forbidden")
            or getattr(args, "forbid_transport", None)
            or []
        )
        raw_prefs = parsed.get("preferences") or getattr(args, "preferences", None) or []
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
        require_flight = (not getattr(args, "no_flight", False)) and ("flight" not in forbid_modes)

        if kb and not candidate_cities and (parsed.get("destination") or getattr(args, "destination", None)):
            dest_hint = parsed.get("destination") or getattr(args, "destination", None)
            candidate_cities = kb.get_candidate_cities(
                destination_hint=dest_hint,
                must_visit=must_cities,
                priority=priority_cities,
                top_k=max(getattr(args, "top_k", 5) * 2, visit_num),
            )

        return TripGoal(
            origin=parsed.get("origin") or getattr(args, "origin", None),
            destination=parsed.get("destination") or getattr(args, "destination", None),
            start_date=parsed.get("start_date") or getattr(args, "start_date", None),
            duration_days=parsed.get("duration_days") or getattr(args, "days", 3),
            budget=parsed.get("budget") or getattr(args, "budget", None),
            require_flight=require_flight,
            require_accommodation=not getattr(args, "no_stay", False),
            num_restaurants=parsed.get("restaurants") or getattr(args, "restaurants", 3),
            num_attractions=getattr(args, "attractions", 2),
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
            notes=getattr(args, "notes", None) or parsed.get("notes"),
        )


class MCTSPlanningAgent:
    """Thin orchestrator over the generic MCTSAgent for the travel environment."""

    def __init__(self, args, env: EnvAgent, policy: Optional[TravelLLMPolicy]):
        mcts_args = argparse.Namespace(
            exploration_constant=getattr(args, "exploration_constant", 8.0),
            bonus_constant=getattr(args, "bonus_constant", 1.0),
            max_depth=getattr(args, "max_depth", 16),
            simulation_per_act=getattr(args, "simulation_per_act", 1),
            discount_factor=getattr(args, "discount_factor", 0.95),
            simulation_num=getattr(args, "simulation_num", 50),
            uct_type=getattr(args, "uct_type", "PUCT"),
            round=0,
            seed=getattr(args, "seed", 0),
            model=getattr(args, "local_model", None),
            debug=getattr(args, "debug", False),
            use_llm_prior=getattr(args, "use_llm_prior", "all"),
        )
        self.agent = MCTSAgent(
            mcts_args,
            env,
            policy=policy,
            uct_type=getattr(args, "uct_type", "PUCT"),
            use_llm=getattr(args, "use_llm_prior", "all") != "none",
        )

    def run(self, *, max_episode_len: int) -> Dict[str, Any]:
        env = self.agent.env
        self.agent.prior_logs = []
        obs, valid_actions = env.reset()
        history = list(getattr(env, "base_history", []) or [])
        done = False
        plan_actions: List[str] = []

        for step in range(max_episode_len):
            action = self.agent.search(obs, history, step, valid_actions, done)
            if action is None:
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
            "env": env,
            "filter_usage": list(getattr(env, "filter_events", [])),
            "prior_usage": list(getattr(self.agent, "prior_logs", [])),
        }
