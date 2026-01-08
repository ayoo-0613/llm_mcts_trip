from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mcts.travel.envspec.schema import (
    CUISINES,
    ENV_SPEC_VERSION,
    HOUSE_RULES,
    ROOM_TYPES,
    TRANSPORT_MODES,
    envspec_skeleton,
)
from mcts.travel.semantic.query_parsing import call_local_llm


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def generate_envspec_from_nl(
    nl_query: str,
    base_url: str,
    model: str,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    NL query -> EnvSpec JSON (declarative, schema-shaped).

    This function does not validate or compile; callers should run
    `normalize_envspec` + `validate_envspec` + `compile_envspec`.
    """
    skeleton = envspec_skeleton()
    prompt = (
        "You are a JSON generator for a travel environment specification.\n"
        "Output ONLY a single JSON object. Do not output any extra text.\n"
        "You MUST keep the JSON structure exactly the same as the provided skeleton: do not add/remove/rename fields.\n"
        f'The field `version` MUST be exactly "{ENV_SPEC_VERSION}".\n'
        "Fill the values based on the user query.\n"
        "If a value is unknown, use null (do not omit fields).\n"
        "Allowed enums:\n"
        f"- transport modes: {sorted(TRANSPORT_MODES)}\n"
        f"- cuisines: {sorted(CUISINES)}\n"
        f"- room types: {sorted(ROOM_TYPES)}\n"
        f"- house rules: {sorted(HOUSE_RULES)}\n"
        "User query:\n"
        f"{nl_query}\n\n"
        "JSON skeleton (fill values only):\n"
        f"{json.dumps(skeleton, ensure_ascii=False, indent=2)}\n"
    )
    raw = call_local_llm(base_url, model, prompt, timeout=timeout)
    obj = _extract_json_object(raw or "")
    return obj or {}

