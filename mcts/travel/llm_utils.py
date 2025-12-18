from __future__ import annotations

"""
Compatibility shim for LLM helpers.

The concrete implementation now lives in mcts.travel.filtering.call_llm.
"""

from mcts.travel.filtering import call_llm

__all__ = ["call_llm"]
