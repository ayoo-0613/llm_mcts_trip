from __future__ import annotations

import unittest

from mcts.travel.env.knowledge_base import TravelKnowledgeBase
from mcts.travel.env_agent import TravelEnv
from mcts.travel.envspec.compiler import compile_envspec
from mcts.travel.envspec.factory import build_env_from_query
from mcts.travel.envspec.schema import envspec_skeleton


class TestEnvSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = TravelKnowledgeBase("database", keep_raw_frames=False)

    def test_envspec_compile_minimal(self):
        spec = envspec_skeleton()
        spec["goal"].update(
            {
                "origin": "Atlanta",
                "destination": "Milwaukee",
                "start_date": "2022-03-08",
                "duration_days": 3,
                "budget": 1100,
                "people_number": 2,
            }
        )
        goal_parsed, env_kwargs = compile_envspec(spec)
        self.assertEqual(goal_parsed["origin"], "Atlanta")
        self.assertEqual(goal_parsed["destination"], "Milwaukee")
        self.assertEqual(goal_parsed["duration_days"], 3)
        self.assertEqual(goal_parsed["people_number"], 2)
        self.assertIn("constraints", goal_parsed)
        self.assertIsInstance(env_kwargs, dict)

    def test_envspec_transport_forbid_affects_allowed_modes(self):
        spec = envspec_skeleton()
        spec["goal"].update(
            {
                "origin": "Atlanta",
                "destination": "Milwaukee",
                "start_date": "2022-03-08",
                "duration_days": 3,
                "budget": 1100,
                "people_number": 1,
            }
        )
        spec["constraints"]["transport"]["forbid"] = ["flight"]
        goal_parsed, _ = compile_envspec(spec)

        env = TravelEnv(self.kb, goal_parsed=goal_parsed)
        self.assertNotIn("flight", env._allowed_transport_modes())

    def test_envspec_constraints_meal_affects_preference_match(self):
        spec = envspec_skeleton()
        spec["goal"].update(
            {
                "origin": "Atlanta",
                "destination": "Milwaukee",
                "start_date": "2022-03-08",
                "duration_days": 3,
                "budget": 1100,
                "people_number": 1,
                "preferences": [],
            }
        )
        spec["constraints"]["meal"]["cuisines"] = ["Chinese"]
        goal_parsed, _ = compile_envspec(spec)

        env = TravelEnv(self.kb, goal_parsed=goal_parsed)
        self.assertTrue(env._matches_preference({"cuisines": "Chinese"}))

    def test_envspec_fallback_on_invalid_json(self):
        def bad_spec_generator(nl_query: str, base_url: str, model: str, timeout: float):
            return {"version": "envspec.v0", "goal": {}, "constraints": None, "retrieval": None, "reward_cfg_overrides": {}}

        def fallback_parser(nl_query: str, base_url: str, model: str, timeout: float):
            return {"origin": "Atlanta", "destination": "Milwaukee", "duration_days": 3, "people_number": 1}

        env = build_env_from_query(
            "dummy query",
            self.kb,
            use_envspec=True,
            llm_cfg={"base_url": "http://localhost:11434", "model": "noop", "timeout": 1.0},
            env_defaults={"max_steps": 5, "top_k": 2, "candidate_cap": 10},
            spec_generator=bad_spec_generator,
            fallback_parser=fallback_parser,
        )
        self.assertEqual(env.goal_parsed.get("origin"), "Atlanta")
        self.assertEqual(env.goal_parsed.get("destination"), "Milwaukee")


if __name__ == "__main__":
    unittest.main()
