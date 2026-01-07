"""
Semantic utilities for the travel domain.

Keep this package import lightweight: some optional ML dependencies (torch,
sentence-transformers, transformers) may not be usable in constrained
environments. We therefore avoid importing them eagerly.
"""

from __future__ import annotations

from typing import Any

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


def __getattr__(name: str) -> Any:  # PEP 562
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
        from . import query_parsing as qp

        return getattr(qp, name)
    raise AttributeError(name)
