from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from mcts.mcts.mcts import MCTSAgent
from mcts.travel.llm_policy import TravelLLMPolicy
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
