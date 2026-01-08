from __future__ import annotations

from typing import Any, TYPE_CHECKING

__all__ = [
    "SemanticBundleReranker",
    "TravelLLMPolicy",
    "call_local_llm",
    "extract_query_text",
    "fallback_parse",
    "load_queries",
    "normalize_parsed_query",
    "parse_nl_query",
]

if TYPE_CHECKING:
    from .bundle_reranker import SemanticBundleReranker
    from .llm_policy import TravelLLMPolicy


def __getattr__(name: str) -> Any:
    # Keep package import-light: torch dependencies are only loaded on demand.
    if name == "TravelLLMPolicy":
        from .llm_policy import TravelLLMPolicy

        return TravelLLMPolicy
    if name == "SemanticBundleReranker":
        from .bundle_reranker import SemanticBundleReranker

        return SemanticBundleReranker

    if name in {
        "call_local_llm",
        "extract_query_text",
        "fallback_parse",
        "load_queries",
        "normalize_parsed_query",
        "parse_nl_query",
    }:
        from . import query_parsing as _qp

        return getattr(_qp, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
