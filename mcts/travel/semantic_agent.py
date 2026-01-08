from __future__ import annotations

from typing import Any, Dict, Optional

from mcts.travel.semantic.query_parsing import normalize_parsed_query, parse_nl_query

__all__ = ["SemanticAgent"]


class SemanticAgent:
    """
    Semantic understanding agent:
    - NL query -> normalized parsed JSON
    - optionally build an LLM/embedding policy for MCTS priors
    """

    def __init__(self, *, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model

    def parse(self, nl_query: str, *, base_url: str, model: str, timeout: float = 60.0) -> Dict[str, Any]:
        return parse_nl_query(nl_query, base_url, model, timeout=timeout)

    def normalize(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        return normalize_parsed_query(parsed)

    def build_policy(self, args) -> Optional[Any]:
        # Lazy import: torch/transformers are optional, and may be unavailable in some sandboxes.
        from mcts.travel.semantic.llm_policy import TravelLLMPolicy

        return TravelLLMPolicy(
            device=getattr(args, "device", "cpu"),
            model_path=getattr(args, "local_model", None),
            embedding_model=self.embedding_model,
            local_base=getattr(args, "local_base", None),
            model_name=getattr(args, "local_model", None),
        )
