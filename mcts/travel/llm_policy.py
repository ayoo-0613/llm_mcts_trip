import re
from typing import List, Sequence, Tuple

import numpy as np
import requests

try:
    import torch  # optional; only needed for embeddings / MPS/CUDA
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer, util as st_utils
except ImportError:  # embeddings are optional
    SentenceTransformer = None
    st_utils = None

try:
    from transformers import pipeline
except ImportError:  # transformers is optional
    pipeline = None


class TravelLLMPolicy:
    """Lightweight policy scorer that can run on local LLMs (pipeline/HTTP) or embeddings."""

    def __init__(self, device: str = "mps", model_path: str = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_base: str = None, llm_timeout: float = 30.0, debug: bool = False):
        # Normalize device strings so "gpu" maps to the first CUDA device if available.
        if device.lower() == "gpu":
            device = "cuda:0"
        elif device.lower().startswith("mps"):
            device = "mps"
        elif device.lower().startswith("cpu"):
            device = "cpu"
        self.device = device
        self.llm_base = llm_base
        self.llm_timeout = llm_timeout
        self.model_name = model_path
        self.debug = debug
        self.generator = None
        if model_path and pipeline is not None:
            try:
                # Prefer explicit device on mac (mps) or cpu; otherwise fall back to auto for CUDA.
                if device and device.startswith("cuda"):
                    self.generator = pipeline(
                        "text-generation",
                        model=model_path,
                        device_map="auto",
                    )
                else:
                    self.generator = pipeline(
                        "text-generation",
                        model=model_path,
                        device=self.device,
                    )
            except Exception:
                self.generator = None

        if SentenceTransformer is None or torch is None:
            self.embedder = None
        else:
            try:
                self.embedder = SentenceTransformer(embedding_model).to(device)
            except Exception:
                self.embedder = None

    def score_actions(self, history: List[str], observation: str,
                      valid_actions: Sequence[str], goal: str,
                      round_idx: int = 0, discount_factor: float = 0.95
                      ) -> Tuple[np.ndarray, float]:
        if not valid_actions:
            return np.array([]), 0.0

        if self.generator is not None:
            scores = self._score_with_generator(history, observation, valid_actions, goal)
            if self.debug:
                print("[LLMPolicy] scored with local pipeline" if scores is not None else "[LLMPolicy] pipeline scoring failed")
        else:
            scores = None

        if (scores is None or len(scores) != len(valid_actions)) and self.llm_base:
            scores = self._score_with_remote(history, observation, valid_actions, goal)
            if self.debug:
                print("[LLMPolicy] scored with remote LLM" if scores is not None else "[LLMPolicy] remote scoring failed")

        if scores is None or len(scores) != len(valid_actions):
            scores = self._score_with_embeddings(history, observation, valid_actions, goal)
            if self.debug:
                print("[LLMPolicy] scored with embeddings" if scores is not None else "[LLMPolicy] embedding scoring failed; returning zeros")

        if scores is None or len(scores) != len(valid_actions):
            scores = np.zeros(len(valid_actions))

        probs = self._to_probs(scores)
        predicted_reward = float(np.max(scores)) if len(scores) else 0.0
        return probs, predicted_reward

    def _score_with_embeddings(self, history: List[str], observation: str,
                               valid_actions: Sequence[str], goal: str) -> np.ndarray:
        if self.embedder is None or st_utils is None:
            return np.zeros(len(valid_actions))

        context = goal + " " + observation + " " + " ".join(history)
        context_embedding = self.embedder.encode(
            context, convert_to_tensor=True, device=self.device, show_progress_bar=False
        )
        action_embeddings = self.embedder.encode(
            list(valid_actions), convert_to_tensor=True, device=self.device, show_progress_bar=False
        )
        cos_scores = st_utils.pytorch_cos_sim(context_embedding, action_embeddings)[0]
        return cos_scores.detach().cpu().numpy()

    def _score_with_generator(self, history: List[str], observation: str,
                              valid_actions: Sequence[str], goal: str) -> np.ndarray:
        prompt = self._build_prompt(history, observation, valid_actions, goal)
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=96,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.6,
                return_full_text=False,
            )
            text = outputs[0]["generated_text"]
        except TypeError:
            outputs = self.generator(
                prompt,
                max_new_tokens=96,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.6,
            )
            text = outputs[0]["generated_text"]
        except Exception:
            return None

        numbers = re.findall(r"[01](?:\\.\\d+)?", text)
        scores = [float(n) for n in numbers[:len(valid_actions)]]
        if len(scores) != len(valid_actions):
            return None
        return np.array(scores)

    def _score_with_remote(self, history: List[str], observation: str,
                           valid_actions: Sequence[str], goal: str) -> np.ndarray:
        prompt = self._build_prompt(history, observation, valid_actions, goal)
        text = self._call_local_llm(prompt)
        if not text:
            return None
        numbers = re.findall(r"[01](?:\\.\\d+)?", text)
        scores = [float(n) for n in numbers[:len(valid_actions)]]
        if len(scores) != len(valid_actions):
            return None
        return np.array(scores)

    @staticmethod
    def _build_prompt(history: List[str], observation: str,
                      valid_actions: Sequence[str], goal: str) -> str:
        action_lines = [f"{idx+1}. {act}" for idx, act in enumerate(valid_actions)]
        history_text = "; ".join(history) if history else "None yet"
        prompt = (
            "You are ranking the best next actions for a travel planning agent.\n"
            f"Goal: {goal}\n"
            f"History: {history_text}\n"
            f"Observation: {observation}\n"
            "Rate each candidate from 0 to 1 in the given order, comma separated only with numbers.\n"
            "Candidates:\n" + "\n".join(action_lines) + "\nScores:"
        )
        return prompt

    @staticmethod
    def _to_probs(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return np.array([])
        scores = scores - np.max(scores)
        exp = np.exp(scores)
        return exp / np.sum(exp)

    # -------- Plan-level scoring for terminal rollouts --------
    def score_plan(self, goal: str, history: List[str], observation: str) -> float:
        """Score a complete plan (terminal rollout) using the generator or remote LLM if available."""
        if self.generator is None and not self.llm_base:
            return 0.0
        plan_text = "; ".join(history) if history else "None"
        prompt = (
            "You are evaluating a travel itinerary. Give a single score between 0 and 1 "
            "for how reasonable and complete the plan is.\n"
            f"Goal: {goal}\n"
            f"Observation: {observation}\n"
            f"Actions: {plan_text}\n"
            "Score:"
        )
        text = None
        if self.generator is not None:
            try:
                outputs = self.generator(
                    prompt,
                    max_new_tokens=8,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.0,
                    return_full_text=False,
                )
                text = outputs[0]["generated_text"]
            except TypeError:
                outputs = self.generator(
                    prompt,
                    max_new_tokens=8,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.0,
                )
                text = outputs[0]["generated_text"]
            except Exception:
                text = None
        if text is None:
            text = self._call_local_llm(prompt)
            if text is None:
                return 0.0
            if self.debug:
                print("[LLMPolicy] plan scored via remote LLM")

        m = re.search(r"([01](?:\\.\\d+)?)", text)
        if not m:
            return 0.0
        try:
            return float(m.group(1))
        except Exception:
            return 0.0

    def _call_local_llm(self, prompt: str) -> str:
        """Call a local LLM server (e.g., Ollama) for scoring."""
        if not self.llm_base or not self.model_name:
            return None
        endpoint = f"{self.llm_base.rstrip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=self.llm_timeout)
            if resp.status_code >= 400:
                return None
            data = resp.json()
        except Exception:
            return None
        if self.debug:
            print(f"[LLMPolicy] POST {endpoint} -> {resp.status_code}")
        for key in ("response", "text", "output", "content"):
            if key in data and isinstance(data[key], str):
                return data[key]
        return None
