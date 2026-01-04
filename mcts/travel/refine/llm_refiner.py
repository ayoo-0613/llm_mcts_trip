from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError:
    requests = None


class LLMBundleRefiner:
    """
    Refiner upgrade:
      - "Generate bundle" -> "Select candidate_id" (eliminates no_valid_bundle).
      - Two-stage: rule-based hard gate + LLM ranking/selection.
      - Deterministic fallback scorer if LLM fails or outputs invalid id.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        max_candidates: int = 40,
        gated_pool_size: int = 16,
        enable_hard_gate: bool = True,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_candidates = int(max_candidates)
        self.gated_pool_size = int(gated_pool_size)
        self.enable_hard_gate = bool(enable_hard_gate)

    # -----------------------------
    # Public API
    # -----------------------------
    def refine(
        self,
        *,
        report: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Returns:
          - None if disabled/insufficient inputs
          - dict with either:
              {"bundle": [...], "candidate_id": int, "reason": "...", "endpoint": "...", "raw": "...", "parsed": {...},
               "gated": {...}}  # success
            or
              {"error": "...", "reason": "...", ...}  # caller may fallback
        """
        if not self.base_url or not self.model:
            return None
        if not report or not candidates:
            return None

        # Limit candidate volume
        candidates = list(candidates[: max(1, self.max_candidates)])

        # Stage 1: rule-based hard gate (build a smaller, safer pool)
        gated = self._hard_gate(report, candidates) if self.enable_hard_gate else self._pass_through_gate(candidates)
        gated_candidates = gated["candidates"]

        # If gating kills everything, fall back to full candidate list
        if not gated_candidates:
            gated = {
                "mode": "disabled_or_empty",
                "candidates": candidates,
                "excluded": [],
                "notes": ["hard_gate_empty_fallback_to_all"],
            }
            gated_candidates = candidates

        # Stage 2: LLM selects candidate_id (NOT a bundle string/list)
        prompt = self._build_prompt_select_id(report, gated_candidates, gated)
        response = self._call_llm(prompt)
        raw = response.get("raw")

        if not raw:
            # deterministic fallback on gated candidates
            fb = self._fallback_pick(report, gated_candidates)
            return {
                "error": response.get("error") or "llm_no_response",
                "error_detail": response.get("error_detail"),
                "status_code": response.get("status_code"),
                "endpoint": response.get("endpoint"),
                "raw": None,
                "parsed": None,
                "fallback": fb,
                "gated": self._gate_summary(gated),
                "bundle_override": (fb.get("bundle") if isinstance(fb, dict) else None),
                "gradient": self._fallback_gradient(report, candidates),
                "confidence": 0.0,
            }

        parsed = self._extract_json(raw)
        candidate_id = self._parse_candidate_id(parsed, raw, len(gated_candidates))
        reason = self._extract_reason(parsed, raw)
        confidence = self._parse_confidence(parsed)
        gradient = self._parse_gradient(parsed) or self._fallback_gradient(report, candidates)

        if candidate_id is None:
            fb = self._fallback_pick(report, gated_candidates)
            return {
                "error": "invalid_candidate_id",
                "reason": reason,
                "endpoint": response.get("endpoint"),
                "raw": raw,
                "parsed": parsed,
                "fallback": fb,
                "gated": self._gate_summary(gated),
                "bundle_override": (fb.get("bundle") if isinstance(fb, dict) else None),
                "gradient": gradient,
                "confidence": confidence,
            }

        chosen = gated_candidates[int(candidate_id)]
        bundle = chosen.get("bundle") or []

        # Code-level hard validations (still important even when selecting IDs)
        cur = report.get("city_bundle") or report.get("bundle") or report.get("current_bundle") or []
        dead_city = report.get("dead_end_city") or report.get("city") or ""
        phase = str(report.get("failed_phase") or report.get("phase") or "").lower()

        # 1) Reject same bundle as current (order-sensitive)
        if cur and self._normalize_bundle(bundle) == self._normalize_bundle(cur):
            fb = self._fallback_pick(report, [c for i, c in enumerate(gated_candidates) if i != candidate_id] or gated_candidates)
            return {
                "error": "same_bundle_as_current",
                "reason": reason,
                "endpoint": response.get("endpoint"),
                "raw": raw,
                "parsed": parsed,
                "candidate_id": candidate_id,
                "bundle": bundle,
                "fallback": fb,
                "gated": self._gate_summary(gated),
                "bundle_override": (fb.get("bundle") if isinstance(fb, dict) else None),
                "gradient": gradient,
                "confidence": confidence,
            }

        # 2) If hotel/stay dead-end, avoid dead_end_city when possible
        if dead_city and phase in ("hotel", "stay"):
            dead = str(dead_city).strip().lower()
            if dead and any(str(c).strip().lower() == dead for c in bundle):
                # if any alternative exists without dead city, take fallback
                alt = self._first_without_dead_city(gated_candidates, dead, cur=cur)
                if alt is not None:
                    return {
                        "bundle": alt["bundle"],
                        "candidate_id": alt["candidate_id"],
                        "reason": f"{reason or ''} (override: avoid dead_end_city={dead_city})".strip(),
                        "endpoint": response.get("endpoint"),
                        "raw": raw,
                        "parsed": parsed,
                        "gated": self._gate_summary(gated),
                        "override": "avoid_dead_end_city",
                        "bundle_override": alt["bundle"],
                        "gradient": gradient,
                        "confidence": confidence,
                    }
                # otherwise, allow through (no alternative)

        return {
            "bundle": bundle,
            "candidate_id": candidate_id,
            "reason": reason,
            "endpoint": response.get("endpoint"),
            "raw": raw,
            "parsed": parsed,
            "gated": self._gate_summary(gated),
            "bundle_override": bundle,
            "gradient": gradient,
            "confidence": confidence,
        }

    # -----------------------------
    # Stage 1: Hard Gate (rule-based)
    # -----------------------------
    def _pass_through_gate(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"mode": "none", "candidates": candidates, "excluded": [], "notes": []}

    def _hard_gate(self, report: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Conservative gating:
          - Prefer to exclude obviously repeating dead-ends.
          - Never mutate constraints; only chooses among candidates.
        """
        cur = report.get("city_bundle") or report.get("bundle") or report.get("current_bundle") or []
        cur_norm = self._normalize_bundle(cur)
        phase = str(report.get("failed_phase") or report.get("phase") or "").lower()
        dead_city = (report.get("dead_end_city") or report.get("city") or "")
        dead_norm = str(dead_city).strip().lower()
        reason = str(report.get("dead_end_reason") or report.get("reason") or "").lower()
        budget_subtype = str(report.get("dead_end_subtype") or report.get("subtype") or "").lower()

        excluded: List[Dict[str, Any]] = []
        kept: List[Dict[str, Any]] = []
        notes: List[str] = []

        # Helpers for feature access
        def feats(c: Dict[str, Any]) -> Dict[str, Any]:
            return c.get("features") or {}

        def has_dead_city(bundle: List[str]) -> bool:
            if not dead_norm:
                return False
            return any(str(x).strip().lower() == dead_norm for x in (bundle or []))

        def blocked_edge_exists(c: Dict[str, Any]) -> bool:
            ec = (feats(c).get("edge_costs") or {})
            for _, v in ec.items():
                if isinstance(v, dict) and v.get("blocked"):
                    return True
            return False

        def bottleneck_meal_slack(c: Dict[str, Any]) -> Optional[int]:
            br = (feats(c).get("bundle_risk") or {})
            val = br.get("bottleneck_meal_slack")
            return int(val) if isinstance(val, int) else None

        # First pass: exclude identical bundle (order-sensitive)
        for c in candidates:
            bundle = c.get("bundle") or []
            if cur_norm and self._normalize_bundle(bundle) == cur_norm:
                excluded.append({"bundle": bundle, "why": "same_as_current"})
            else:
                kept.append(c)

        candidates = kept
        kept = []

        # Second pass: phase-specific gating
        for c in candidates:
            bundle = c.get("bundle") or []
            why: List[str] = []

            # hotel/stay dead-end: strongly prefer avoiding dead_end_city if any alternatives exist
            if phase in ("hotel", "stay") and dead_norm:
                if has_dead_city(bundle):
                    why.append("contains_dead_end_city")

            # segment/flight dead-end: avoid blocked edges
            if phase in ("segment", "flight"):
                if blocked_edge_exists(c):
                    why.append("blocked_edge")

            # duplication-type dead-end: prefer higher meal slack (avoid tiny pools)
            # (If you have explicit duplication reason, use it; otherwise heuristics by reason text.)
            if "dup" in reason or "duplicate" in reason:
                ms = bottleneck_meal_slack(c)
                if ms is not None and ms < 1:
                    why.append("low_meal_slack")

            # budget reserve/over_budget: downweight extreme LB; do not hard-exclude by default
            if ("budget" in reason) or (budget_subtype in ("over_budget", "violate_reserve")):
                # keep; scoring handles preference
                pass

            if why:
                excluded.append({"bundle": bundle, "why": ",".join(why)})
            else:
                kept.append(c)

        # If we excluded too aggressively (e.g., all contain dead_end_city), relax by keeping all.
        if not kept:
            notes.append("hard_gate_overfiltered_relax")
            kept = candidates  # use pre-phase-filter list

        # Finally: shrink pool size but keep diversity (simple spread by first/second city)
        pruned = self._diversity_prune(kept, k=max(1, self.gated_pool_size))
        notes.append(f"gated_pool_size={len(pruned)}")

        return {"mode": "hard_gate", "candidates": pruned, "excluded": excluded, "notes": notes}

    def _diversity_prune(self, candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        Simple diversity: group by (bundle[0], bundle[1]) key and take best by fallback score per group.
        """
        if not candidates:
            return []
        k = max(1, int(k))
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for c in candidates:
            b = c.get("bundle") or []
            a = str(b[0]).strip().lower() if len(b) > 0 else ""
            d = str(b[1]).strip().lower() if len(b) > 1 else ""
            key = (a, d)
            groups.setdefault(key, []).append(c)

        # pick best per group by deterministic score (no report here; purely structure)
        picked: List[Dict[str, Any]] = []
        for _, items in groups.items():
            items_sorted = sorted(items, key=lambda x: self._cheap_struct_score(x))
            picked.append(items_sorted[0])

        # if still too many, keep best k by cheap score
        picked_sorted = sorted(picked, key=lambda x: self._cheap_struct_score(x))
        return picked_sorted[:k]

    def _cheap_struct_score(self, cand: Dict[str, Any]) -> float:
        """
        Very cheap structural score used only for diversity pruning:
          prefer lower bundle_lb_cost (if present).
        """
        feats = cand.get("features") or {}
        v = feats.get("bundle_lb_cost")
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
        return 1e18

    def _gate_summary(self, gated: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mode": gated.get("mode"),
            "notes": gated.get("notes") or [],
            "excluded_n": len(gated.get("excluded") or []),
            "candidates_n": len(gated.get("candidates") or []),
            "excluded_sample": (gated.get("excluded") or [])[:5],
        }

    # -----------------------------
    # Stage 2: LLM chooses candidate_id
    # -----------------------------
    def _build_prompt_select_id(self, report: Dict[str, Any], candidates: List[Dict[str, Any]], gated: Dict[str, Any]) -> str:
        """
        LLM must output:
          {"candidate_id": <int>, "reason": "<short>", "confidence": <float>, "gradient": {...}}
        candidate_id is index into the provided candidates list (0..N-1).
        """
        cur_bundle = report.get("city_bundle") or report.get("bundle") or report.get("current_bundle") or []
        dead_city = report.get("dead_end_city") or report.get("city") or None
        failed_phase = report.get("failed_phase") or report.get("phase") or None

        # Keep report compact
        failure_report = {
            "failed_phase": failed_phase,
            "dead_end_city": dead_city,
            "dead_end_reason": report.get("dead_end_reason") or report.get("reason"),
            "dead_end_counts": report.get("dead_end_counts") or report.get("counts"),
            "budget_context": report.get("dead_end_budget_context") or report.get("budget_context"),
            "current_bundle": cur_bundle,
            "current_bottleneck": report.get("current_bottleneck"),
            "violations": report.get("violations"),
        }

        def slim_city_features(cf: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for city, v in (cf or {}).items():
                if not isinstance(v, dict):
                    continue
                out[city] = {
                    "days": v.get("days"),
                    "nights": v.get("nights"),
                    "hotel_min_price": v.get("hotel_min_price"),
                    "hotel_slack": v.get("hotel_slack"),
                    "meal_slack": v.get("meal_slack"),
                    "min_meal_cost": v.get("min_meal_cost"),
                    "attr_slack": v.get("attr_slack"),
                }
            return out

        def slim_edge_costs(ec: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in (ec or {}).items():
                if isinstance(v, dict):
                    out[k] = {"min_cost": v.get("min_cost"), "blocked": v.get("blocked")}
            return out

        cand_payload: List[Dict[str, Any]] = []
        for idx, c in enumerate(candidates):
            feats = c.get("features") or {}
            cand_payload.append(
                {
                    "id": idx,
                    "bundle": c.get("bundle") or [],
                    "bundle_risk": feats.get("bundle_risk"),
                    "city_summary": slim_city_features(feats.get("city_features") or {}),
                    "edge_summary": slim_edge_costs(feats.get("edge_costs") or {}),
                    "lb_costs": {
                        "transport_lb_cost": feats.get("transport_lb_cost"),
                        "stay_lb_cost": feats.get("stay_lb_cost"),
                        "meal_lb_cost": feats.get("meal_lb_cost"),
                        "bundle_lb_cost": feats.get("bundle_lb_cost"),
                    },
                    "feasibility": {
                        "return_leg_exists": feats.get("return_leg_exists"),
                        "hotel_feasible_all": feats.get("hotel_feasible_all"),
                    },
                }
            )

        payload = {
            "task": "Select the best candidate_id for the next MCTS attempt and output a FailureGradient. Do NOT invent cities; choose only by id.",
            "hard_rules": [
                f"candidate_id must be an integer in [0, {max(0, len(candidates) - 1)}].",
                "Do NOT output a city list. Output candidate_id only.",
                "Return JSON only with keys: candidate_id, reason, confidence, gradient.",
                "Prefer avoiding dead_end_city for hotel/stay failures if feasible.",
                "Prefer candidates with no blocked edges for segment/flight failures.",
                "Use bundle_risk + city_summary + edge_summary + lb_costs to decide.",
                "gradient.hard_exclusions rule must be one of: NO_FLIGHT, NO_SELF_DRIVING, NO_TAXI, AVOID_CITY, BLOCK_EDGE, EXCLUDE_BUNDLE_SET, EXCLUDE_BUNDLE_SEQ.",
                "gradient.soft_penalties feature must be one of: budget_risk, cost_ratio, soft_penalty, edge_cost, hotel_price.",
            ],
            "failure_report": failure_report,
            "gating_summary": self._gate_summary(gated),
            "candidates": cand_payload,
            "output_schema": {
                "candidate_id": "int",
                "reason": "short string",
                "confidence": "float in [0,1]",
                "gradient": {
                    "hard_exclusions": "list[{phase, rule, ttl, city?, edge?}]",
                    "soft_penalties": "list[{phase, feature, delta}]",
                    "retrieval_patches": "list[{phase, patch, ttl}]",
                },
            },
        }

        return (
            "You are selecting among candidates.\n"
            "Return JSON only.\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )

    def _parse_candidate_id(self, parsed: Optional[Any], raw: str, n: int) -> Optional[int]:
        if n <= 0:
            return None
        # If dict with candidate_id
        if isinstance(parsed, dict):
            cid = parsed.get("candidate_id")
            if isinstance(cid, int) and 0 <= cid < n:
                return cid
            # sometimes numeric string
            if isinstance(cid, str) and cid.strip().isdigit():
                val = int(cid.strip())
                if 0 <= val < n:
                    return val

        # fallback: search in raw
        if raw:
            m = re.search(r"candidate_id\s*[:=]\s*(\d+)", raw, flags=re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 0 <= val < n:
                    return val
            # last resort: first integer in text
            m2 = re.search(r"\b(\d+)\b", raw)
            if m2:
                val = int(m2.group(1))
                if 0 <= val < n:
                    return val
        return None

    @staticmethod
    def _parse_confidence(parsed: Optional[Any]) -> float:
        if isinstance(parsed, dict):
            c = parsed.get("confidence")
            if c is not None:
                try:
                    v = float(c)
                    return max(0.0, min(1.0, v))
                except Exception:
                    return 0.0
        return 0.0

    @staticmethod
    def _parse_gradient(parsed: Optional[Any]) -> Optional[Dict[str, Any]]:
        if isinstance(parsed, dict):
            grad = parsed.get("gradient")
            if isinstance(grad, dict):
                return grad
        return None

    @staticmethod
    def _fallback_gradient(report: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        phase = str(report.get("failed_phase") or report.get("phase") or "").lower()
        dead_city = report.get("dead_end_city") or report.get("city")
        subtype = str(report.get("dead_end_subtype") or report.get("subtype") or "").lower()
        reason = str(report.get("dead_end_reason") or report.get("reason") or "").lower()

        if phase in ("segment", "flight"):
            hard = []
            # If return leg budget issues are common, bias against expensive edges.
            hard.append({"phase": "segment", "rule": "EXCLUDE_BUNDLE_SEQ", "ttl": 1})
            return {
                "hard_exclusions": hard,
                "soft_penalties": [{"phase": "segment", "feature": "cost_ratio", "delta": 0.4}],
                "retrieval_patches": [{"phase": "segment", "patch": {"topk": 30, "prefer": "low_cost"}, "ttl": 1}],
            }

        if phase in ("hotel", "stay"):
            hard = []
            if dead_city and subtype in ("min_nights", "stay_unavailable", "stay_constraints"):
                hard.append({"phase": "stay", "rule": "AVOID_CITY", "city": dead_city, "ttl": 1})
            return {
                "hard_exclusions": hard,
                "soft_penalties": [{"phase": "stay", "feature": "hotel_price", "delta": 0.6}],
                "retrieval_patches": [{"phase": "stay", "patch": {"topk": 50, "prefer": "low_cost"}, "ttl": 1}],
            }

        if phase in ("meal", "daily", "attraction"):
            if subtype == "missing_cuisine" or reason == "constraint_mismatch":
                return {
                    "hard_exclusions": [],
                    "soft_penalties": [{"phase": "daily", "feature": "budget_risk", "delta": 0.2}],
                    "retrieval_patches": [{"phase": "daily", "patch": {"topk": 50}, "ttl": 1}],
                }

        return {"hard_exclusions": [], "soft_penalties": [], "retrieval_patches": []}

    # -----------------------------
    # Deterministic fallback
    # -----------------------------
    def _fallback_pick(self, report: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deterministic fallback selection:
          - phase-aware penalties
          - then choose min score
        """
        cur = report.get("city_bundle") or report.get("bundle") or report.get("current_bundle") or []
        cur_norm = self._normalize_bundle(cur)
        phase = str(report.get("failed_phase") or report.get("phase") or "").lower()
        dead_city = (report.get("dead_end_city") or report.get("city") or "")
        dead_norm = str(dead_city).strip().lower()

        best_idx = None
        best_score = 1e30
        best_reason = "fallback_min_score"

        for i, c in enumerate(candidates):
            b = c.get("bundle") or []
            if cur_norm and self._normalize_bundle(b) == cur_norm:
                continue

            feats = c.get("features") or {}
            score = 0.0

            # primary: low bundle lower-bound cost
            lb = feats.get("bundle_lb_cost")
            if isinstance(lb, (int, float)) and math.isfinite(lb):
                score += float(lb)
            else:
                score += 1e6

            # phase-specific: avoid dead city for hotel/stay if possible
            if phase in ("hotel", "stay") and dead_norm:
                if any(str(x).strip().lower() == dead_norm for x in b):
                    score += 1e6  # heavy penalty

            # segment/flight: avoid blocked edges
            if phase in ("segment", "flight"):
                ec = feats.get("edge_costs") or {}
                if any(isinstance(v, dict) and v.get("blocked") for v in ec.values()):
                    score += 1e6

            # duplication: prefer higher meal slack
            br = feats.get("bundle_risk") or {}
            ms = br.get("bottleneck_meal_slack")
            if isinstance(ms, int):
                score += max(0.0, 10.0 - float(ms)) * 1000.0  # smaller slack -> higher penalty

            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            best_idx = 0

        return {
            "candidate_id": best_idx,
            "bundle": candidates[best_idx].get("bundle") or [],
            "score": best_score,
            "reason": best_reason,
        }

    def _first_without_dead_city(self, candidates: List[Dict[str, Any]], dead_norm: str, cur: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        cur_norm = self._normalize_bundle(cur or [])
        for idx, c in enumerate(candidates):
            b = c.get("bundle") or []
            if cur_norm and self._normalize_bundle(b) == cur_norm:
                continue
            if not any(str(x).strip().lower() == dead_norm for x in b):
                return {"candidate_id": idx, "bundle": b}
        return None

    # -----------------------------
    # LLM call
    # -----------------------------
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if requests is None:
            return {"raw": None, "error": "requests_unavailable"}
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=self.timeout)
        except Exception as exc:
            return {
                "raw": None,
                "error": "llm_request_failed",
                "error_detail": str(exc),
                "endpoint": endpoint,
            }
        if resp.status_code >= 400:
            return {
                "raw": None,
                "error": "llm_http_error",
                "status_code": resp.status_code,
                "error_detail": (resp.text or "")[:1000],
                "endpoint": endpoint,
            }
        try:
            data = resp.json()
        except Exception:
            return {
                "raw": None,
                "error": "llm_invalid_json",
                "error_detail": (resp.text or "")[:1000],
                "endpoint": endpoint,
            }
        text = data.get("response") or data.get("text") or data.get("output") or data.get("content")
        if not text:
            return {"raw": None, "error": "llm_empty_response", "endpoint": endpoint}
        return {"raw": text, "endpoint": endpoint}

    # -----------------------------
    # Parsing helpers
    # -----------------------------
    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        if not text:
            return None
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except Exception:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_bundle(seq: Sequence[Any]) -> Tuple[str, ...]:
        return tuple(str(c).strip().lower() for c in seq if str(c).strip())

    @staticmethod
    def _extract_reason(parsed: Optional[Any], raw: str) -> Optional[str]:
        if isinstance(parsed, dict):
            reason = parsed.get("reason")
            if isinstance(reason, str) and reason.strip():
                return reason.strip()
        if raw:
            match = re.search(r"reason\s*[:：]\s*(.+)", raw, flags=re.IGNORECASE)
            if match:
                reason = match.group(1).strip()
                if reason:
                    return reason
        return None
