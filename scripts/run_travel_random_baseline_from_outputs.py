import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import gc
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcts.travel.knowledge_base import TravelKnowledgeBase  # noqa: E402
from mcts.travel.submission import env_to_submission_record  # noqa: E402
from mcts.travel.travel_env import TravelEnv  # noqa: E402
from mcts.travel.query_parsing import normalize_parsed_query  # noqa: E402


def _infer_idx_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.match(r"^(\d{3,})_", base)
    if not m:
        return None
    try:
        return max(1, int(m.group(1)))
    except Exception:
        return None


def _iter_output_json_paths(input_dir: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(input_dir):
        if not name.endswith(".json"):
            continue
        if name == "mas_submission.json":
            continue
        paths.append(os.path.join(input_dir, name))
    # Sort by inferred idx first, then by name for stability.
    def _key(p: str) -> Tuple[int, str]:
        idx = _infer_idx_from_filename(p)
        return (idx if idx is not None else 10**9, os.path.basename(p))
    return sorted(paths, key=_key)


def _parse_date_safe(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    fmts = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m-%d %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


def _build_date_list(start_date: Optional[str], days: int) -> List[str]:
    start_dt = _parse_date_safe(start_date)
    if not start_dt or days <= 0:
        return []
    return [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def _default_local_constraint() -> Dict[str, Any]:
    return {
        "house rule": None,
        "cuisine": None,
        "room type": None,
        "transportation": None,
    }


def _infer_room_type_from_preferences(prefs: Any) -> Optional[str]:
    if not prefs:
        return None
    if isinstance(prefs, str):
        prefs_list = [prefs]
    else:
        try:
            prefs_list = list(prefs)
        except Exception:
            return None
    prefs_norm = " ".join(str(x).lower() for x in prefs_list if x is not None)
    if "private room" in prefs_norm:
        return "private room"
    if "shared room" in prefs_norm:
        return "shared room"
    if "entire room" in prefs_norm or "entire home" in prefs_norm or "entire home/apt" in prefs_norm:
        return "entire room"
    if "not shared room" in prefs_norm:
        return "not shared room"
    return None


def _infer_transport_constraint(parsed: Dict[str, Any]) -> Optional[str]:
    forbid = parsed.get("transport_forbid") or parsed.get("transport_forbidden") or parsed.get("transport_forbidden_modes")
    if isinstance(forbid, str):
        forbid_list = [forbid]
    else:
        try:
            forbid_list = list(forbid or [])
        except Exception:
            forbid_list = []
    forbid_norm = {str(x).lower() for x in forbid_list}
    if "flight" in forbid_norm:
        return "no flight"
    if "self-driving" in forbid_norm or "self driving" in forbid_norm:
        return "no self-driving"
    return None


def normalize_question_from_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a saved `parsed` dict into the question schema expected by
    evaluation/{commonsense_constraint,hard_constraint}.py.
    """
    parsed = dict(parsed or {})

    org = parsed.get("org") or parsed.get("origin")
    dest = parsed.get("dest") or parsed.get("destination")
    days = parsed.get("days") or parsed.get("duration_days") or 0
    try:
        days_int = int(days)
    except Exception:
        days_int = 0
    days_int = max(1, days_int)

    date_list = parsed.get("date")
    if not isinstance(date_list, list) or not date_list:
        date_list = _build_date_list(parsed.get("start_date"), days_int)

    budget = parsed.get("budget")
    if budget is None:
        budget_val: float = 1e18
    else:
        try:
            budget_val = float(budget)
        except Exception:
            budget_val = 1e18

    people = parsed.get("people_number")
    try:
        people_int = int(people) if people is not None else 1
    except Exception:
        people_int = 1
    people_int = max(1, people_int)

    lc = parsed.get("local_constraint")
    if not isinstance(lc, dict):
        lc = _default_local_constraint()
        room_type = _infer_room_type_from_preferences(parsed.get("preferences"))
        transport = _infer_transport_constraint(parsed)
        if room_type is not None:
            lc["room type"] = room_type
        if transport is not None:
            lc["transportation"] = transport
    else:
        lc = {**_default_local_constraint(), **lc}

    # Evaluation code expects `cuisine` to be a list or None.
    if lc.get("cuisine") is not None and not isinstance(lc.get("cuisine"), list):
        lc["cuisine"] = [lc["cuisine"]]

    return {
        "org": org,
        "dest": dest,
        "days": days_int,
        "date": date_list,
        "budget": budget_val,
        "people_number": people_int,
        "visiting_city_number": parsed.get("visiting_city_number") or 1,
        "local_constraint": lc,
    }


def _merge_args_into_parsed(parsed: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(parsed or {})
    if out.get("origin") is None and getattr(args, "origin", None):
        out["origin"] = args.origin
    if out.get("destination") is None and getattr(args, "destination", None):
        out["destination"] = args.destination
    if out.get("start_date") is None and getattr(args, "start_date", None):
        out["start_date"] = args.start_date
    if out.get("duration_days") is None and getattr(args, "days", None):
        out["duration_days"] = args.days
    if out.get("visiting_city_number") is None and getattr(args, "visiting_city_number", None):
        out["visiting_city_number"] = args.visiting_city_number
    if out.get("budget") is None and getattr(args, "budget", None):
        out["budget"] = args.budget
    return out


def _run_random_episode(env: TravelEnv, rng: random.Random, *, max_episode_len: int) -> Dict[str, Any]:
    obs, valid_actions = env.reset()
    history = list(getattr(env, "base_history", []) or [])
    done = False
    plan_actions: List[str] = []

    for _step in range(max_episode_len):
        if not valid_actions:
            break
        action = rng.choice(valid_actions)
        obs, reward, done, history, valid_actions = env.apply_action(action)
        plan_actions.append(action)
        if done:
            break

    return {
        "success": env.is_success(env.state),
        "actions": plan_actions,
        "cost": env.state.cost,
        "violations": env.state.violations,
        "env": env,
    }


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Random slot-filling baseline: load `parsed` from saved output JSONs, "
            "query the same KB + phase framework, but choose actions uniformly at random."
        )
    )
    p.add_argument("--input-dir", default="output_mcts", help="Directory with per-sample output JSONs (with `parsed`).")
    p.add_argument("--database-root", default="database")
    p.add_argument("--output-jsonl", default="output_mcts/random_baseline_submission.jsonl")
    p.add_argument("--questions-json", default="output_mcts/questions_from_parsed.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of samples to process.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-len", type=int, default=40)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-cap", type=int, default=80)
    p.add_argument("--relax-max-tries", type=int, default=6)
    p.add_argument("--visiting-city-number", type=int, default=None)
    p.add_argument("--restaurants", type=int, default=3)
    p.add_argument("--attractions", type=int, default=2)
    p.add_argument("--no-flight", action="store_true")
    p.add_argument("--no-stay", dest="no_stay", action="store_true")
    p.add_argument("--origin", default=None)
    p.add_argument("--destination", default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--use-llm-filters", dest="use_llm_filters", action="store_true",
                   help="Deprecated (no-op): LLM filter generation removed.")
    p.add_argument("--no-llm-filters", dest="use_llm_filters", action="store_false",
                   help="Deprecated (no-op): LLM filter generation removed.")
    p.set_defaults(use_llm_filters=False)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--log-filter-usage", action="store_true")
    p.add_argument(
        "--compare-jsonl",
        default="output_mcts/mas_submission.json",
        help="Optional existing submission JSONL (or JSONL-with-.json) to evaluate for comparison.",
    )
    p.add_argument("--eval", dest="do_eval", action="store_true", help="Run evaluator after writing plans.")
    p.add_argument("--no-eval", dest="do_eval", action="store_false", help="Skip evaluator (write files only).")
    p.set_defaults(do_eval=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    out_path = args.output_jsonl
    questions_path = args.questions_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    paths = _iter_output_json_paths(input_dir)
    if args.limit is not None:
        paths = paths[: max(0, int(args.limit))]
    if not paths:
        raise RuntimeError(f"No per-sample JSON files found in {input_dir}")

    # Build idx -> question map (for evaluating both baselines and any compare JSONL).
    questions_by_idx: Dict[int, Dict[str, Any]] = {}
    query_by_idx: Dict[int, str] = {}
    parsed_by_idx: Dict[int, Dict[str, Any]] = {}
    for path in paths:
        idx = _infer_idx_from_filename(path)
        if idx is None:
            continue
        data = _read_json(path)
        parsed = data.get("parsed") or {}
        if not isinstance(parsed, dict):
            parsed = {}
        questions_by_idx[idx] = normalize_question_from_parsed(parsed)
        query_by_idx[idx] = str(data.get("query") or "")
        parsed_by_idx[idx] = parsed

    # Write questions JSON for evaluation/eval.py (expects list ordered by idx_base=1).
    max_idx = max(questions_by_idx) if questions_by_idx else 0
    questions_list: List[Dict[str, Any]] = []
    for i in range(1, max_idx + 1):
        q = questions_by_idx.get(i)
        if not q:
            raise RuntimeError(f"Missing parsed question for idx={i}; cannot build {questions_path}.")
        questions_list.append(q)
    os.makedirs(os.path.dirname(questions_path) or ".", exist_ok=True)
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(questions_list, f, ensure_ascii=False, indent=2)

    mode = "w" if args.overwrite else "a"

    kb = TravelKnowledgeBase(args.database_root, keep_raw_frames=False)

    total = 0
    t0 = time.perf_counter()

    with open(out_path, mode, encoding="utf-8") as fp:
        for path in paths:
            idx = _infer_idx_from_filename(path)
            if idx is None:
                continue

            q_text = query_by_idx.get(idx, "")
            parsed = parsed_by_idx.get(idx, {})

            plan: List[Dict[str, Any]] = []
            result: Optional[Dict[str, Any]] = None
            try:
                parsed = normalize_parsed_query(parsed)
                parsed = _merge_args_into_parsed(parsed, args)
                env = TravelEnv(
                    kb,
                    max_steps=args.max_episode_len,
                    top_k=args.top_k,
                    debug=args.debug,
                    candidate_cap=args.candidate_cap,
                    user_query=q_text,
                    log_filter_usage=args.log_filter_usage,
                    goal_parsed=parsed,
                )

                # Per-sample RNG: deterministic w.r.t (seed, idx), so reruns are stable.
                rng = random.Random(int(args.seed) + int(idx) * 1000003)
                result = _run_random_episode(env, rng, max_episode_len=args.max_episode_len)
                plan = env_to_submission_record(idx=idx, query=q_text, env=result["env"])["plan"]
            except Exception as e:
                if args.debug:
                    print(f"[WARN] idx={idx} failed to generate plan: {e}")
                plan = []

            rec = {"idx": int(idx), "query": str(q_text), "plan": plan}
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total += 1

            if total % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f"[PROGRESS] {total}/{len(paths)} elapsed={elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    if total:
        print(f"[RANDOM] wrote={out_path} total={total} elapsed={elapsed:.1f}s")

    if not args.do_eval:
        return

    # Free as much KB memory as possible before running evaluator (it loads large CSVs too).
    del kb
    gc.collect()

    def _run_eval(plans_path: str, label: str) -> None:
        cmd = [
            sys.executable,
            os.path.join(ROOT, "evaluation", "eval.py"),
            "--plans",
            plans_path,
            "--questions-file",
            questions_path,
            "--idx-base",
            "1",
        ]
        print(f"[EVAL] {label}: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

    _run_eval(out_path, "random-baseline")
    if args.compare_jsonl and os.path.exists(args.compare_jsonl):
        _run_eval(args.compare_jsonl, "compare")


if __name__ == "__main__":
    main()
