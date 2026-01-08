"""
Travel planning agents.

Keep this module import-light: heavyweight deps (e.g., torch) are only imported
when the corresponding symbols are accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["EnvAgent", "RetrievalAgent", "SearchAgent", "SemanticAgent"]

if TYPE_CHECKING:
    from mcts.travel.env_agent import TravelEnv as EnvAgent
    from mcts.travel.retrieval_agent import RetrievalAgent
    from mcts.travel.search_agent import SearchAgent
    from mcts.travel.semantic_agent import SemanticAgent


def __getattr__(name: str) -> Any:
    if name == "EnvAgent":
        from mcts.travel.env_agent import TravelEnv as EnvAgent  # local import

        return EnvAgent
    if name == "RetrievalAgent":
        from mcts.travel.retrieval_agent import RetrievalAgent  # local import

        return RetrievalAgent
    if name == "SearchAgent":
        from mcts.travel.search_agent import SearchAgent  # local import

        return SearchAgent
    if name == "SemanticAgent":
        from mcts.travel.semantic_agent import SemanticAgent  # local import

        return SemanticAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
