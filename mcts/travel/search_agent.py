from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from mcts.mcts.mcts import MCTSAgent
from mcts.travel.failure import FailureMemory, FailureSignalExtractor
from mcts.travel.semantic.llm_policy import TravelLLMPolicy

__all__ = ["SearchAgent"]


class SearchAgent:
    """Thin orchestrator over the generic MCTSAgent for the travel environment."""

    def __init__(self, args, env: Any, policy: Optional[TravelLLMPolicy]):
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

    def run(self, *, max_episode_len: int, max_attempts: int = 1) -> Dict[str, Any]:
        env = self.agent.env
        failure_memory = FailureMemory()
        last_result: Optional[Dict[str, Any]] = None

        for attempt in range(max(1, int(max_attempts))):
            if attempt > 0:
                failure_memory.tick()
            if hasattr(env, "set_failure_memory"):
                env.set_failure_memory(failure_memory)
            else:
                try:
                    env.retrieval_agent.set_failure_memory(failure_memory)
                except Exception:
                    pass

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
            result = {
                "success": success,
                "actions": plan_actions,
                "cost": env.state.cost,
                "violations": env.state.violations,
                "state": env.state,
                "env": env,
                "filter_usage": list(getattr(env, "filter_events", [])),
                "prior_usage": list(getattr(self.agent, "prior_logs", [])),
                "attempt": attempt + 1,
            }
            if success:
                return result

            failure_signal = FailureSignalExtractor.extract(
                filter_events=result.get("filter_usage") or [],
                goal_parsed=getattr(env, "goal_parsed", None),
                state=env.state,
            )
            result["failure_signal"] = failure_signal
            last_result = result
            if failure_signal:
                failure_memory.update(failure_signal)

        return last_result or {
            "success": False,
            "actions": [],
            "cost": 0.0,
            "violations": [],
            "state": None,
            "env": env,
            "filter_usage": [],
            "prior_usage": [],
            "attempt": max_attempts,
        }
