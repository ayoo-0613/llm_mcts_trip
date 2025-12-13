from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

try:
    from openai import OpenAI  # optional: only needed for deepseek backend
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None
import requests


logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """Unified candidate wrapper passed to selectors."""

    id: str
    kind: str
    meta: dict
    payload: object


class ActionSelector:
    """Interface for selecting a subset of candidates."""

    def select(
        self,
        observation: str,
        goal: str,
        phase: str,
        candidates: Sequence[Candidate],
        k: int,
        relaxed: bool = False,
    ) -> List[str]:
        raise NotImplementedError


class RuleSelector(ActionSelector):
    """No-LLM fallback: keep deterministic, slice-by-order."""

    def select(
        self,
        observation: str,
        goal: str,
        phase: str,
        candidates: Sequence[Candidate],
        k: int,
        relaxed: bool = False,
    ) -> List[str]:
        return [c.id for c in list(candidates)[:k]]


class LLMSelector(ActionSelector):
    """
    Call a local/OpenAI-compatible endpoint to pick candidate IDs.
    The endpoint must accept POST {model, prompt, temperature, stream=False}
    and return JSON with a text field or {"response": "..."}.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.0,
        fallback: Optional[ActionSelector] = None,
        backend: str = "local",  # "local" (Ollama-style) or "deepseek" (OpenAI-compatible)
    ):
        self.model = model or "local"
        # choose default base by backend if not provided
        if base_url is None:
            base_url = "https://api.deepseek.com" if backend == "deepseek" else "http://localhost:11434"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.fallback = fallback or RuleSelector()
        self.backend = backend

    def select(
        self,
        observation: str,
        goal: str,
        phase: str,
        candidates: Sequence[Candidate],
        k: int,
        relaxed: bool = False,
    ) -> List[str]:
        if not candidates:
            return []

        prompt = self._build_prompt(goal, observation, phase, candidates, k, relaxed)
        try:
            text = self._call_llm(prompt)
            selected = self._parse_ids(text)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning("LLMSelector failed (%s), falling back to rule", exc)
            selected = []

        if not selected:
            return self.fallback.select(observation, goal, phase, candidates, k, relaxed)

        # Respect k and pool membership.
        pool_ids = {c.id for c in candidates}
        filtered = [cid for cid in selected if cid in pool_ids]
        if len(filtered) < k:
            # Pad with rule-based order to hit k when possible.
            ordered_ids = [c.id for c in candidates]
            for cid in ordered_ids:
                if cid not in filtered:
                    filtered.append(cid)
                if len(filtered) >= k:
                    break
        return filtered[:k]

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "deepseek":
            if OpenAI is None:
                raise RuntimeError("openai package not installed; install via `pip install openai` for deepseek backend.")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable for deepseek backend.")
            client = OpenAI(api_key=api_key, base_url=self.base_url)
            resp = client.chat.completions.create(
                model=self.model or "deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                stream=False,
            )
            return resp.choices[0].message.content

        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
        }
        resp = requests.post(endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            return resp.text
        for key in ("response", "text", "output", "content"):
            if key in data:
                return data[key]
        # If the server directly returns a JSON object with selected_ids, pass it through.
        if isinstance(data, dict):
            return json.dumps(data)
        return ""

    @staticmethod
    def _build_prompt(
        goal: str,
        observation: str,
        phase: str,
        candidates: Sequence[Candidate],
        k: int,
        relaxed: bool,
    ) -> str:
        # Keep prompt concise; rely on meta info only.
        lines = []
        for idx, c in enumerate(candidates, start=1):
            summary = json.dumps(c.meta, ensure_ascii=False)
            lines.append(f"{idx}. id={c.id} kind={c.kind} meta={summary}")

        return (
            "You are selecting the best next actions for a travel planner. "
            "Return a JSON object with a key \"selected_ids\" listing up to "
            f"{k} candidate ids in priority order. Do not invent ids.\n"
            f"Goal: {goal}\n"
            f"Phase: {phase} | Relaxed: {relaxed}\n"
            f"Observation: {observation}\n"
            "Candidates:\n" + "\n".join(lines) + "\nOutput JSON only."
        )

    @staticmethod
    def _parse_ids(text: str) -> List[str]:
        if not text:
            return []
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("selected_ids"), Iterable):
                return [str(x) for x in data["selected_ids"]]
        except Exception:
            pass
        return []
