import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcts.mcts.mcts import MCTSAgent
from mcts.travel.env.tripcraft_knowledge_base import TripCraftKnowledgeBase
from mcts.travel.env.submission import env_to_tripcraft_record
from mcts.travel.env_agent import TravelEnv
from mcts.travel.semantic.query_parsing import normalize_parsed_query


def _safe_eval(text: Any) -> Any:
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text
    s = str(text).strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return text


def _row_to_query_json(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "org": row.get("org"),
        "dest": row.get("dest"),
        "days": int(row.get("days")) if row.get("days") is not None else None,
        "visiting_city_number": int(row.get("visiting_city_number")) if row.get("visiting_city_number") is not None else None,
        "date": _safe_eval(row.get("date")),
        "people_number": int(row.get("people_number")) if row.get("people_number") is not None else None,
        "local_constraint": _safe_eval(row.get("local_constraint")) or {},
        "budget": float(row.get("budget")) if row.get("budget") is not None else None,
        "query": row.get("query"),
        "level": row.get("level"),
    }
    out["date"] = out["date"] if isinstance(out["date"], list) else []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tripcraft-db", default="Tripcraft/Tripcraft/TripCraft_database")
    ap.add_argument("--queries-csv", required=True, help="TripCraft query CSV (3/5/7 day)")
    ap.add_argument("--out", required=True, help="Output JSONL in TripCraft eval format")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="Print per-sample progress to stdout.")

    # MCTS settings
    ap.add_argument("--max-episode-len", type=int, default=40)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--exploration-constant", type=float, default=8.0)
    ap.add_argument("--bonus-constant", type=float, default=1.0)
    ap.add_argument("--max-depth", type=int, default=16)
    ap.add_argument("--simulation-per-act", type=int, default=1)
    ap.add_argument("--simulation-num", type=int, default=50)
    ap.add_argument("--discount-factor", type=float, default=0.95)
    ap.add_argument("--uct-type", type=str, default="PUCT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")

    # TripCraft-specific toggles
    ap.add_argument(
        "--require-flight",
        action="store_true",
        help="If set, enforce flight for first+last segment (slower: loads flights table).",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-cap", type=int, default=80)
    args = ap.parse_args()

    kb = TripCraftKnowledgeBase(args.tripcraft_db, load_flights=bool(args.require_flight), keep_raw_frames=True)

    df = pd.read_csv(args.queries_csv)
    rows = df.to_dict(orient="records")
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    if not rows:
        print(f"[WARN] No rows loaded from {args.queries_csv}")
        print(f"[OK] wrote 0 record(s) to {args.out}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            if args.verbose:
                q = str(row.get("query") or "")
                q_preview = (q[:120] + "...") if len(q) > 120 else q
                print(f"[INFO] idx={i}/{len(rows)} query={q_preview}")
            query_json = _row_to_query_json(row)
            persona = str(row.get("persona") or "")

            parsed = normalize_parsed_query(query_json)
            parsed["require_flight"] = bool(args.require_flight)
            parsed.setdefault("attractions_per_day_min", 1)
            parsed.setdefault("restaurants_per_day_min", 3)

            env = TravelEnv(
                knowledge_base=kb,
                max_steps=int(args.max_episode_len),
                top_k=int(args.top_k),
                candidate_cap=int(args.candidate_cap),
                debug=bool(args.debug),
                user_query=str(query_json.get("query") or ""),
                goal_parsed=parsed,
            )

            mcts_args = SimpleNamespace(
                exploration_constant=float(args.exploration_constant),
                bonus_constant=float(args.bonus_constant),
                max_depth=int(args.max_depth),
                simulation_per_act=int(args.simulation_per_act),
                discount_factor=float(args.discount_factor),
                simulation_num=int(args.simulation_num),
                uct_type=str(args.uct_type),
                round=0,
                seed=int(args.seed),
                model=None,
                debug=bool(args.debug),
                use_llm_prior="none",
            )
            agent = MCTSAgent(
                mcts_args,
                env,
                policy=None,
                uct_type=str(args.uct_type),
                use_llm=False,
            )

            obs, valid_actions = env.reset()
            history = list(getattr(env, "base_history", []) or [])
            done = False
            for step in range(int(args.max_episode_len)):
                action = agent.search(obs, history, step, valid_actions, done)
                if action is None:
                    break
                obs, reward, done, history, valid_actions = env.apply_action(action)
                if done:
                    break

            rec = env_to_tripcraft_record(i, query_json, persona, env)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(rows)} record(s) to {args.out}")


if __name__ == "__main__":
    main()
