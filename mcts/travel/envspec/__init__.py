from __future__ import annotations

from typing import Any

from mcts.travel.envspec.schema import ENV_SPEC_VERSION

__all__ = [
    "ENV_SPEC_VERSION",
    "build_env_from_query",
    "compile_envspec",
    "generate_envspec_from_nl",
    "normalize_envspec",
    "validate_envspec",
]


def __getattr__(name: str) -> Any:
    if name in {"compile_envspec", "normalize_envspec", "validate_envspec"}:
        from mcts.travel.envspec import compiler as _compiler

        return getattr(_compiler, name)
    if name == "generate_envspec_from_nl":
        from mcts.travel.envspec.generator import generate_envspec_from_nl

        return generate_envspec_from_nl
    if name == "build_env_from_query":
        from mcts.travel.envspec.factory import build_env_from_query

        return build_env_from_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
