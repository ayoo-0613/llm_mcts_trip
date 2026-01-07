from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from mcts.mcts.mcts import MCTSAgent
from mcts.travel.failure import FailureMemory, FailureSignalExtractor
from mcts.travel.failure_gradient import FailureGradientBuilder
from mcts.travel.failure_gradient_store import FailureGradientStore
from mcts.travel.refine import FailureReportBuilder, LLMBundleRefiner
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
        self.llm_base = getattr(args, "local_base", None)
        self.llm_model = getattr(args, "local_model", None)
        self.llm_timeout = getattr(args, "llm_timeout", 60.0)

    def run(self, *, max_episode_len: int, max_attempts: int = 1) -> Dict[str, Any]:
        env = self.agent.env
        failure_memory = FailureMemory()
        gradient_store = FailureGradientStore()
        last_result: Optional[Dict[str, Any]] = None
        llm_refinements: List[Dict[str, Any]] = []
        gradients: List[Dict[str, Any]] = []

        for attempt in range(max(1, int(max_attempts))):
            if attempt > 0:
                failure_memory.tick()
                gradient_store.tick()
            if hasattr(env, "set_failure_memory"):
                env.set_failure_memory(failure_memory)
            else:
                try:
                    env.retrieval_agent.set_failure_memory(failure_memory)
                except Exception:
                    pass
            if hasattr(env, "set_failure_gradient_store"):
                env.set_failure_gradient_store(gradient_store)
            else:
                try:
                    env.failure_gradient_store = gradient_store
                except Exception:
                    pass

            refined_used = False
            while True:
                self.agent.prior_logs = []
                obs, valid_actions = env.reset()
                history = list(getattr(env, "base_history", []) or [])
                done = False
                plan_actions: List[str] = []

                for step in range(max_episode_len):
                    print("\n[MCTS INPUT]")
                    print(" obs:", obs)
                    print(" history:", history)
                    print(" valid_actions:", valid_actions)
                    print(" step:", step)
                    print(" done:", done)
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
                    if llm_refinements:
                        result["llm_refinements"] = list(llm_refinements)
                    return result

                failure_signal = FailureSignalExtractor.extract(
                    filter_events=result.get("filter_usage") or [],
                    goal_parsed=getattr(env, "goal_parsed", None),
                    state=env.state,
                )
                result["failure_signal"] = failure_signal
                candidates = list(getattr(env, "city_bundle_candidates", []) or [])
                if not candidates and hasattr(env, "retrieval_agent"):
                    candidates = list(getattr(env.retrieval_agent, "city_bundle_candidates", []) or [])
                report = FailureReportBuilder.build(
                    goal_parsed=getattr(env, "goal_parsed", None) or {},
                    state=env.state,
                    failure_signal=failure_signal,
                    violations=list(getattr(env.state, "violations", []) or []),
                    candidates=candidates,
                )
                refine_out = None
                gradient_added = False
                if not refined_used and report and candidates and self.llm_base and self.llm_model:
                    refiner = LLMBundleRefiner(base_url=self.llm_base, model=self.llm_model, timeout=self.llm_timeout)
                    refine_out = refiner.refine(report=report, candidates=candidates)
                    if refine_out is not None:
                        new_bundle = refine_out.get("bundle_override") or refine_out.get("bundle")
                        refinement = {
                            "attempt": attempt + 1,
                            "report": report,
                            "new_bundle": new_bundle,
                            "reason": refine_out.get("reason"),
                            "raw": refine_out.get("raw"),
                            "parsed": refine_out.get("parsed"),
                            "error": refine_out.get("error"),
                            "error_detail": refine_out.get("error_detail"),
                            "status_code": refine_out.get("status_code"),
                            "endpoint": refine_out.get("endpoint"),
                            "gradient": refine_out.get("gradient") if isinstance(refine_out, dict) else None,
                        }
                        if new_bundle and hasattr(env, "set_forced_city_bundle"):
                            env.set_forced_city_bundle(new_bundle)
                            refinement["applied"] = True
                            refinement["rerun"] = True
                        else:
                            refinement["applied"] = False
                            refinement["rerun"] = False
                        llm_refinements.append(refinement)
                        result["llm_refinement"] = refinement
                        # Build/store gradient before potential immediate rerun so it is consumable.
                        try:
                            slot = getattr(env, "last_slot", None)
                            slot_fp = slot.signature() if slot is not None and hasattr(slot, "signature") else None
                            gradient = FailureGradientBuilder.build(
                                failure_signal=failure_signal,
                                report=report,
                                refine_out=refine_out,
                                state=env.state,
                                goal_parsed=getattr(env, "goal_parsed", None) or {},
                                slot_fp=slot_fp,
                            )
                            gradient_store.add(gradient)
                            gradients.append(
                                {
                                    "attempt": attempt + 1,
                                    "scope": gradient.scope,
                                    "confidence": gradient.confidence,
                                    "hard_exclusions": gradient.hard_exclusions,
                                    "soft_penalties": gradient.soft_penalties,
                                    "retrieval_patches": gradient.retrieval_patches,
                                    "bundle_override": gradient.bundle_override,
                                }
                            )
                            result["failure_gradient"] = gradients[-1]
                            gradient_added = True
                        except Exception:
                            pass
                        if failure_signal:
                            failure_memory.update(failure_signal)
                        if refinement["applied"]:
                            refined_used = True
                            continue

                # Build and store failure gradient (even if LLM was not used / failed)
                if not gradient_added:
                    try:
                        slot = getattr(env, "last_slot", None)
                        slot_fp = slot.signature() if slot is not None and hasattr(slot, "signature") else None
                        gradient = FailureGradientBuilder.build(
                            failure_signal=failure_signal,
                            report=report,
                            refine_out=refine_out,
                            state=env.state,
                            goal_parsed=getattr(env, "goal_parsed", None) or {},
                            slot_fp=slot_fp,
                        )
                        gradient_store.add(gradient)
                        gradients.append(
                            {
                                "attempt": attempt + 1,
                                "scope": gradient.scope,
                                "confidence": gradient.confidence,
                                "hard_exclusions": gradient.hard_exclusions,
                                "soft_penalties": gradient.soft_penalties,
                                "retrieval_patches": gradient.retrieval_patches,
                                "bundle_override": gradient.bundle_override,
                            }
                        )
                        result["failure_gradient"] = gradients[-1]
                    except Exception:
                        pass
                last_result = result
                if failure_signal:
                    failure_memory.update(failure_signal)
                break

        if last_result is None:
            return {
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
        if llm_refinements:
            last_result["llm_refinements"] = list(llm_refinements)
        if gradients:
            last_result["failure_gradients"] = list(gradients)
        return last_result
