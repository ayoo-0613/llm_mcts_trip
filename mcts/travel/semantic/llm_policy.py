import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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

try:
    import requests  # optional; used for local HTTP LLM
except ImportError:
    requests = None


class TravelLLMPolicy:
    """Lightweight policy scorer that can run on local LLMs or embeddings."""

    def __init__(self, device: str = "cpu", model_path: str = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 local_base: str = None,
                 model_name: str = None):
        # Normalize device strings so "gpu" maps to the first CUDA device if available.
        if device.lower() == "gpu":
            device = "cuda:0"
        elif device.lower().startswith("mps"):
            device = "mps"
        elif device.lower().startswith("cpu"):
            device = "cpu"
        self.device = device
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

        self.prior_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self.local_base = local_base
        self.model_name = model_name or model_path

    def score_actions(self, history: List[str], observation: str,
                      valid_actions: Sequence[str], goal: str,
                      round_idx: int = 0, discount_factor: float = 0.95
                      ) -> Tuple[np.ndarray, float]:
        if not valid_actions:
            return np.array([]), 0.0

        if self.generator is not None:
            scores = self._score_with_generator(history, observation, valid_actions, goal)
        else:
            scores = None

        if scores is None or len(scores) != len(valid_actions):
            scores = self._score_with_embeddings(history, observation, valid_actions, goal)

        if scores is None or len(scores) != len(valid_actions):
            scores = np.zeros(len(valid_actions))

        probs = self._to_probs(scores)
        predicted_reward = float(np.max(scores)) if len(scores) else 0.0
        return probs, predicted_reward

    # -------- Slot-aware policy prior --------
    def score_prior(
        self,
        obs: str,
        slot: Any,
        actions: Sequence[str],
        payloads: Optional[Dict[str, Tuple]] = None,
        history: Optional[List[str]] = None,
        goal: Optional[Any] = None,
        state_signature: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """Compute priors for a batch of actions with slot/context awareness."""
        if not actions:
            return np.array([]), 0.0

        payloads = payloads or {}
        cache_key = self._prior_cache_key(obs, slot, actions, state_signature)
        if cache_key in self.prior_cache:
            return self.prior_cache[cache_key], 0.0

        scores = None

        # Try listwise local LLM scoring first if configured.
        llm_scores = self._score_with_llm_http(obs, slot, actions, payloads, goal, history)
        if llm_scores is not None and len(llm_scores) == len(actions):
            scores = llm_scores

        if scores is None:
            scores = self._heuristic_scores(slot, actions, payloads, goal)

        try:
            priors = self._to_probs(np.array(scores, dtype=np.float32))
        except Exception:
            priors = self._to_probs(np.zeros(len(actions), dtype=np.float32))

        self.prior_cache[cache_key] = priors
        return priors, 0.0

    def _score_with_embeddings(self, history: List[str], observation: str,
                               valid_actions: Sequence[str], goal: str) -> np.ndarray:
        if self.embedder is None or st_utils is None:
            return np.zeros(len(valid_actions))

        goal_txt = goal if isinstance(goal, str) else str(goal)
        context = goal_txt + " " + observation + " " + " ".join(history)
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
        goal_txt = goal if isinstance(goal, str) else str(goal)
        prompt = self._build_prompt(history, observation, valid_actions, goal_txt)
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

    # -------- HTTP LLM scoring (listwise) --------
    def _score_with_llm_http(
        self,
        obs: str,
        slot: Any,
        actions: Sequence[str],
        payloads: Dict[str, Tuple],
        goal: Any = None,
        history: Optional[List[str]] = None,
    ) -> Optional[List[float]]:
        if self.local_base is None or self.model_name is None or requests is None:
            return None
        prompt = self._build_prior_prompt(obs, slot, actions, payloads, goal, history)
        try:
            resp = requests.post(
                f"{self.local_base.rstrip('/')}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response") or data.get("text") or data.get("output") or data.get("content") or ""
        except Exception:
            return None
        try:
            obj = json.loads(text)
            priors = obj.get("priors") or obj.get("scores")
            if isinstance(priors, list):
                scores = [float(item.get("score", item.get("p", 0))) if isinstance(item, dict) else float(item) for item in priors]
                return scores
        except Exception:
            pass
        # fallback: try to parse numbers directly
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(numbers) >= len(actions):
            try:
                return [float(x) for x in numbers[: len(actions)]]
            except Exception:
                return None
        return None

    def _build_prior_prompt(
        self,
        obs: str,
        slot: Any,
        actions: Sequence[str],
        payloads: Dict[str, Tuple],
        goal: Any,
        history: Optional[List[str]],
    ) -> str:
        slot_txt = ""
        if slot is not None:
            if hasattr(slot, "signature"):
                try:
                    slot_txt = slot.signature()
                except Exception:
                    slot_txt = str(slot)
            else:
                slot_txt = str(slot)
        goal_txt = ""
        if goal is not None:
            if hasattr(goal, "as_text"):
                goal_txt = goal.as_text()
            else:
                goal_txt = str(goal)
        hist_txt = "; ".join(history) if history else "None"
        action_lines = []
        for idx, act in enumerate(actions):
            payload = payloads.get(act)
            extra = ""
            if payload:
                try:
                    extra = json.dumps(payload, ensure_ascii=False)
                except Exception:
                    extra = str(payload)
            action_lines.append(f"{idx}. {act} || {extra}")
        prompt = (
            "You are ranking travel-planning actions for the current slot.\n"
            f"Goal: {goal_txt}\n"
            f"Observation: {obs}\n"
            f"Slot: {slot_txt}\n"
            f"History: {hist_txt}\n"
            "Return JSON with a 'priors' list of objects [{\"id\": idx, \"score\": number}]. "
            "Higher score means better. Do not normalize.\n"
            "Candidates:\n"
            + "\n".join(action_lines)
            + "\nJSON:"
        )
        return prompt

    # -------- Heuristic priors (slot-aware) --------
    @staticmethod
    def _prior_cache_key(obs: str, slot: Any, actions: Sequence[str], state_signature: Optional[str]):
        slot_sig = None
        if hasattr(slot, "signature"):
            try:
                slot_sig = slot.signature()
            except Exception:
                slot_sig = None
        if slot_sig is None and slot is not None:
            slot_sig = str(slot)
        sig = state_signature
        if sig is None:
            try:
                sig = hash(obs)
            except Exception:
                sig = None
        return (sig, slot_sig, tuple(actions))

    def _heuristic_scores(
        self,
        slot: Any,
        actions: Sequence[str],
        payloads: Dict[str, Tuple],
        goal: Optional[Any] = None,
    ) -> List[float]:
        slot_type = getattr(slot, "type", None) if slot is not None else None
        scores: List[float] = []
        for idx, act in enumerate(actions):
            payload = payloads.get(act)
            if payload:
                kind = payload[0]
                if kind == "segment_mode":
                    score = self._score_segment(payload)
                elif kind == "stay_city":
                    score = self._score_stay(payload)
                elif kind == "meal":
                    score = self._score_meal(payload, goal)
                elif kind == "attraction":
                    score = self._score_attraction(payload)
                elif kind == "choose_city_bundle":
                    score = self._score_city_bundle(payload)
                else:
                    score = 0.0
            else:
                score = -5.0 if slot_type == "finish" else 0.0
            if slot_type == "finish":
                score = min(score, -2.0)
            # Small position bias to keep original ordering stable.
            score -= 0.001 * idx
            scores.append(score)
        return scores

    @staticmethod
    def _score_segment(payload: Tuple) -> float:
        try:
            _, _, mode, detail = payload
        except ValueError:
            return 0.0
        detail = detail or {}
        base = 1.0 if mode == "flight" else 0.3
        try:
            price_val = float(detail.get("price", detail.get("cost", 0.0)) or 0.0)
        except Exception:
            price_val = 0.0
        base -= price_val * 0.01
        duration = detail.get("duration")
        if duration is not None:
            try:
                base -= float(duration) / 600.0
            except Exception:
                pass
        if detail.get("fallback_nonflight"):
            base -= 1.5
        return base

    @staticmethod
    def _score_stay(payload: Tuple) -> float:
        stay = payload[2] if len(payload) > 2 else {}
        try:
            price_val = float(stay.get("price") or 0.0)
        except Exception:
            price_val = 0.0
        try:
            review_val = float(stay.get("review") or 0.0)
        except Exception:
            review_val = 0.0
        return 0.3 + review_val * 0.6 - price_val * 0.02

    @staticmethod
    def _score_meal(payload: Tuple, goal: Optional[Any]) -> float:
        rest = payload[3] if len(payload) > 3 else {}
        try:
            rating_val = float(rest.get("rating") or 0.0)
        except Exception:
            rating_val = 0.0
        try:
            cost_val = float(rest.get("cost") or 0.0)
        except Exception:
            cost_val = 0.0
        cuisines_text = str(rest.get("cuisines", "")).lower()
        pref_bonus = 0.0
        prefs: List[str] = []
        if goal is not None:
            if isinstance(goal, dict):
                meal_cons = (goal.get("constraints", {}) or {}).get("meal", {}) if isinstance(goal.get("constraints"), dict) else {}
                prefs = meal_cons.get("cuisines", []) or []
                if not prefs:
                    prefs = goal.get("preferences", []) or []
            else:
                if hasattr(goal, "constraints"):
                    meal_cons = (getattr(goal, "constraints", {}) or {}).get("meal", {}) or {}
                    prefs = meal_cons.get("cuisines", []) or []
                if not prefs and hasattr(goal, "preferences"):
                    prefs = getattr(goal, "preferences", []) or []
        for pref in prefs:
            if pref and str(pref).lower() in cuisines_text:
                pref_bonus = 0.3
                break
        return rating_val * 0.6 - cost_val * 0.01 + pref_bonus

    @staticmethod
    def _score_attraction(payload: Tuple) -> float:
        attr = payload[3] if len(payload) > 3 else {}
        name = str(attr.get("name") or "")
        return 0.1 if name else 0.0

    @staticmethod
    def _score_city_bundle(payload: Tuple) -> float:
        seq = payload[1] if len(payload) > 1 else []
        return float(len(seq) or 0)

    # -------- Plan-level scoring for terminal rollouts --------
    def score_plan(self, goal: str, history: List[str], observation: str) -> float:
        """Score a complete plan (terminal rollout) using the generator if available."""
        if self.generator is None:
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
            return 0.0

        m = re.search(r"([01](?:\\.\\d+)?)", text)
        if not m:
            return 0.0
        try:
            return float(m.group(1))
        except Exception:
            return 0.0
