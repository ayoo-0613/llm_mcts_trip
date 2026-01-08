from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from mcts.travel.env_agent import TravelEnv
from mcts.travel.envspec.compiler import compile_envspec
from mcts.travel.envspec.generator import generate_envspec_from_nl
from mcts.travel.semantic.query_parsing import normalize_parsed_query, parse_nl_query


def build_env_from_query(
    nl_query: str,
    kb: Any,
    *,
    use_envspec: bool,
    llm_cfg: Optional[Dict[str, Any]] = None,
    env_defaults: Optional[Dict[str, Any]] = None,
    goal_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    spec_generator: Callable[[str, str, str, float], Dict[str, Any]] = generate_envspec_from_nl,
    fallback_parser: Callable[[str, str, str, float], Dict[str, Any]] = parse_nl_query,
) -> TravelEnv:
    """
    Unified entrypoint for building TravelEnv from NL query.

    - EnvSpec path: NL -> LLM -> EnvSpec -> validate/normalize/compile -> (goal_parsed, env_kwargs)
    - Fallback path: legacy parse_nl_query() (kept intact)
    """
    llm_cfg = dict(llm_cfg or {})
    env_defaults = dict(env_defaults or {})

    base_url = str(llm_cfg.get("base_url") or llm_cfg.get("local_base") or "http://localhost:11434")
    model = str(llm_cfg.get("model") or llm_cfg.get("parser_model") or llm_cfg.get("local_model") or "")
    timeout = float(llm_cfg.get("timeout") or llm_cfg.get("parser_timeout") or 60.0)

    goal_parsed: Dict[str, Any] = {}
    env_kwargs: Dict[str, Any] = {}
    used_envspec = False

    if use_envspec:
        try:
            spec = spec_generator(nl_query, base_url, model, timeout)
            goal_parsed, env_kwargs = compile_envspec(spec)
            used_envspec = True
        except Exception:
            goal_parsed = {}
            env_kwargs = {}
            used_envspec = False

    if not goal_parsed:
        goal_parsed = fallback_parser(nl_query, base_url, model, timeout)

    goal_parsed = normalize_parsed_query(goal_parsed)
    if goal_transform is not None:
        goal_parsed = goal_transform(goal_parsed)

    merged_env_kwargs = dict(env_defaults)
    merged_env_kwargs.update(env_kwargs)

    env = TravelEnv(
        kb,
        user_query=str(nl_query or ""),
        goal_parsed=goal_parsed,
        **merged_env_kwargs,
    )
    # Non-invasive runtime signal for debugging/acceptance checks.
    setattr(env, "parser_mode", "envspec" if used_envspec else "legacy")
    return env
