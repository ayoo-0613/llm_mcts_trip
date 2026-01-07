"""
Travel planning agents.

This package previously imported all agent modules at import time. Some optional
dependencies (e.g., embedding/transformer stacks) can crash in constrained
environments when imported eagerly.

To keep `import mcts.travel` lightweight and robust, we expose the public API
via lazy attribute access.
"""

from __future__ import annotations

from typing import Any

__all__ = ["EnvAgent", "RetrievalAgent", "SearchAgent", "SemanticAgent"]


def __getattr__(name: str) -> Any:  # PEP 562
    if name == "EnvAgent":
        from mcts.travel.env_agent import TravelEnv as EnvAgent

        return EnvAgent
    if name == "RetrievalAgent":
        from mcts.travel.retrieval_agent import RetrievalAgent

        return RetrievalAgent
    if name == "SearchAgent":
        from mcts.travel.search_agent import SearchAgent

        return SearchAgent
    if name == "SemanticAgent":
        from mcts.travel.semantic_agent import SemanticAgent

        return SemanticAgent
    raise AttributeError(name)
