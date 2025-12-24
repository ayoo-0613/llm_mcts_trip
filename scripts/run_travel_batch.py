import argparse
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys  # noqa: E402

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcts.travel.knowledge_base import TripGoal, TravelKnowledgeBase  # noqa: E402
from mcts.travel.agents import EnvAgent, MCTSPlanningAgent, SemanticAgent  # noqa: E402
from mcts.travel.submission import env_to_submission_record  # noqa: E402
from mcts.travel.query_parsing import (
    call_local_llm,
    fallback_parse,
    load_queries,
    parse_nl_query,
)  # noqa: E402


def _parse_date_safe(date_str: str) -> Optional[datetime]:
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


def _structured_plan(env: EnvAgent) -> Dict[str, List[str]]:
    state = env.state
    goal = env.goal

    # 目的地城市列表：优先使用已选城市序列，其次 fixed_city_order，最后 fallback 到 destination
    cities = list(
        state.city_sequence
        or goal.fixed_city_order
        or ([goal.destination] if goal.destination else [])
    )

    # ----------------------------
    # 1. 交通方式（先根据状态中已经选好的 segment_modes）
    # ----------------------------
    segments = env._segments(state)  # [(idx, src, dst), ...]
    transport_methods: List[str] = []

    for idx, src, dst in segments:
        seg = state.segment_modes.get(idx)
        if seg:
            detail = seg.get("detail", {})
            mode = seg.get("mode")
            if mode == "flight" and isinstance(detail, dict):
                transport_methods.append(
                    f"Flight {detail.get('id', '?')}, from {src} to {dst}, "
                    f"Departure Time: {detail.get('depart', '?')}, "
                    f"Arrival Time: {detail.get('arrive', '?')}, "
                    f"cost: {detail.get('price', detail.get('cost', '?'))}"
                )
            else:
                cost = None
                if isinstance(detail, dict):
                    cost = detail.get("cost")
                transport_methods.append(
                    f"{mode}, from {src} to {dst}, cost: {cost if cost is not None else '?'}"
                )
        else:
            transport_methods.append(f"{src}->{dst} missing mode")

    # ----------------------------
    # 2. 交通日期：根据“每天所在城市”推断每段交通发生的日期
    # ----------------------------
    transport_dates: List[str] = []
    if goal.start_date:
        start_dt = _parse_date_safe(goal.start_date)
        if start_dt:
            total_days = goal.duration_days or env.total_days

            for idx, src, dst in segments:
                transport_day = None  # 1-based day index

                # 特例：最后一段返回 origin，优先对齐到行程最后一天
                if dst == goal.origin and total_days:
                    transport_day = total_days
                else:
                    # 一般情况：找到第一个 day，使 day 所在城市 == 该段的目标城市 dst
                    for day in range(1, total_days + 1):
                        city_d = env._city_for_day(state, day)
                        if city_d == dst:
                            transport_day = day
                            break

                # 兜底逻辑（极端情况下找不到 dst 对应的 day）
                if transport_day is None:
                    if src == goal.origin:
                        # 出发段：默认第一天
                        transport_day = 1
                    elif dst == goal.origin and total_days:
                        # 返回段：默认最后一天
                        transport_day = total_days
                    else:
                        # 其他情况：用段编号近似映射到天数，但不超过 total_days
                        if total_days:
                            transport_day = min(idx + 1, total_days)
                        else:
                            transport_day = idx + 1  # 没有 total_days 信息时的兜底

                date_str = (start_dt + timedelta(days=transport_day - 1)).strftime("%Y-%m-%d")
                transport_dates.append(date_str)

    # ----------------------------
    # 3. 餐厅去重列表
    # ----------------------------
    restaurants_set: List[str] = []
    seen_rest = set()
    for day in range(1, env.total_days + 1):
        for slot, meal in state.meals.get(day, {}).items():
            if meal and meal["id"] not in seen_rest:
                restaurants_set.append(f"{meal['name']}, {meal.get('city', goal.destination)}")
                seen_rest.add(meal["id"])

    # ----------------------------
    # 4. 景点去重列表
    # ----------------------------
    attractions_set: List[str] = []
    seen_att = set()
    for day_map in state.attractions.values():
        for att in day_map.values():
            if att and att["id"] not in seen_att:
                attractions_set.append(f"{att['name']}, {att.get('city')}")
                seen_att.add(att["id"])

    # ----------------------------
    # 5. 每个城市的住宿
    # ----------------------------
    accommodations_list: List[str] = []
    for city in cities:
        stay = state.city_stays.get(city)
        if stay:
            accommodations_list.append(f"{stay['name']}, {city}")
        else:
            accommodations_list.append(f"missing, {city}")

    return {
        "destination_cities": cities,
        "transportation_dates": transport_dates,
        "transportation_methods": transport_methods,
        "restaurants": restaurants_set,
        "attractions": attractions_set,
        "accommodations": accommodations_list,
    }

def _state_hint(kb: TravelKnowledgeBase, name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    norm = kb._normalize_city(name)
    return kb.state_norm_map.get(norm) or kb.city_to_state_norm.get(norm)


def _goal_diagnostics(kb: TravelKnowledgeBase, goal: TripGoal) -> Dict[str, Any]:
    dest_state = _state_hint(kb, goal.destination)
    origin_state = _state_hint(kb, goal.origin)
    state_city_pool = kb.get_cities_for_state(dest_state) if dest_state else []
    must_states = {c: kb.get_state_for_city(c) or _state_hint(kb, c) for c in goal.must_visit_cities}
    pri_states = {c: kb.get_state_for_city(c) or _state_hint(kb, c) for c in goal.priority_cities}
    candidate_states = {c: kb.get_state_for_city(c) or _state_hint(kb, c) for c in goal.candidate_cities}
    return {
        "destination_state": dest_state,
        "origin_state": origin_state,
        "state_city_pool": state_city_pool,
        "candidate_cities": goal.candidate_cities,
        "must_city_states": must_states,
        "priority_city_states": pri_states,
        "candidate_city_states": candidate_states,
    }


def _build_goal(parsed: Dict[str, Any], args, kb: TravelKnowledgeBase, semantic: SemanticAgent) -> TripGoal:
    return semantic.build_goal(parsed, args, kb)

    
def _run_single(goal: TripGoal, kb: TravelKnowledgeBase, policy: Optional[Any], args, *, raw_query: str = "", semantic: SemanticAgent):
    # Reuse the same LLM endpoint as NL parser for filter generation (no extra CLI needed).
    plan_llm = None
    model_for_plan = args.parser_model or args.local_model
    base_for_plan = args.local_base
    if args.use_llm_filters and model_for_plan and base_for_plan:
        def _plan_call(prompt: str, base=base_for_plan, model=model_for_plan):
            return call_local_llm(base, model, prompt, timeout=args.parser_timeout)
        plan_llm = _plan_call

    phase_planner = semantic.build_phase_planner(args, llm_call=plan_llm)

    env = EnvAgent(
        kb,
        goal,
        max_steps=args.max_episode_len,
        top_k=args.top_k,
        debug=args.debug,
        candidate_cap=args.candidate_cap,
        relax_max_tries=args.relax_max_tries,
        user_query=raw_query,
        log_filter_usage=args.log_filter_usage,
        phase_planner=phase_planner,
    )
    planner = MCTSPlanningAgent(args, env, policy)
    result = planner.run(max_episode_len=args.max_episode_len)
    result["goal"] = goal
    result["state"] = env.state
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Batch run travel MCTS over queries from a dataset or local CSV.")
    parser.add_argument("--set-type", default="train", choices=["train", "validation", "test"],
                        help="HF dataset split to load.")
    parser.add_argument("--dataset-id", default="osunlp/TravelPlanner", help="HF dataset id to load queries from.")
    parser.add_argument("--origin", default=None, help="Override origin city for all queries.")
    parser.add_argument("--destination", default=None, help="Override destination city/state for all queries.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--visiting-city-number", type=int, default=None, help="Target number of cities to visit.")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--restaurants", type=int, default=3)
    parser.add_argument("--attractions", type=int, default=2)
    parser.add_argument("--preference", action="append", dest="preferences", default=[])
    parser.add_argument("--no-flight", action="store_true", help="Do not require a flight.")
    parser.add_argument("--no-stay", action="store_true", help="Do not require accommodation.")
    parser.add_argument("--must-city", action="append", dest="must_city", default=[],
                        help="City that must appear in the plan (can be repeated).")
    parser.add_argument("--priority-city", action="append", dest="priority_city", default=[],
                        help="Preferred city to include (can be repeated).")
    parser.add_argument("--fixed-city-order", action="append", dest="fixed_city_order", default=[],
                        help="Explicit city order; list cities in order by repeating the flag.")
    parser.add_argument("--candidate-city", action="append", dest="candidate_city", default=[],
                        help="Preset candidate city list (overrides search).")
    parser.add_argument("--allow-transport", action="append", dest="allow_transport", default=[],
                        help="Allow only these transport modes (flight, taxi, self-driving).")
    parser.add_argument("--forbid-transport", action="append", dest="forbid_transport", default=[],
                        help="Ban these transport modes (flight, taxi, self-driving).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1, help="Max number of queries to process.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into query list (0-based).")
    # MCTS params
    parser.add_argument("--exploration-constant", type=float, default=8.0)
    parser.add_argument("--bonus-constant", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-episode-len", type=int, default=40)
    parser.add_argument("--simulation-per-act", type=int, default=1)
    parser.add_argument("--simulation-num", type=int, default=50)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--candidate-cap", type=int, default=80, help="Max candidates returned per slot.")
    parser.add_argument(
        "--use-llm-filters",
        dest="use_llm_filters",
        action="store_true",
        help="Enable LLM filter generation for candidate retrieval.",
    )
    parser.add_argument(
        "--no-llm-filters",
        dest="use_llm_filters",
        action="store_false",
        help="Disable LLM filter generation (use defaults only).",
    )
    parser.add_argument(
        "--use-llm-prior",
        choices=["root", "all", "none"],
        default="root",
        help="Apply LLM policy priors at root, all nodes, or disable.",
    )
    parser.add_argument(
        "--relax-max-tries",
        type=int,
        default=6,
        help="Max relaxation attempts when KB query returns empty candidates.",
    )
    parser.add_argument(
        "--log-filter-usage",
        action="store_true",
        help="Print filters and whether LLM/cache was used when building candidates.",
    )
    parser.add_argument("--uct-type", default="PUCT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    # LLM params
    parser.add_argument("--local-model", default=None, help="Local LLM for action priors and NL parsing.")
    parser.add_argument("--local-base", default=os.getenv("LOCAL_LLM_BASE", "http://localhost:11434"))
    parser.add_argument("--device", default="mps", help="cpu/mps/cuda:0 etc.")
    parser.add_argument("--parser-model", default=None, help="Parser model (defaults to --local-model).")
    parser.add_argument("--parser-timeout", type=float, default=180.0)
    parser.add_argument("--database-root", default="database")
    parser.add_argument("--save-dir", default="plans_out", help="Directory to save generated plans as JSON.")
    parser.add_argument(
        "--output-format",
        default="debug",
        choices=["debug", "submission", "both"],
        help="Output format: debug (per-sample JSON), submission (JSONL only), both.",
    )
    parser.add_argument(
        "--submission-jsonl",
        default=None,
        help="Optional TravelPlanner submission JSONL path (one sample per line with idx/query/plan).",
    )
    parser.add_argument(
        "--submission-overwrite",
        action="store_true",
        help="Overwrite submission JSONL instead of appending.",
    )
    parser.add_argument("--notes", default=None, help="Optional notes saved with each plan.")
    parser.set_defaults(use_llm_filters=False)
    return parser.parse_args()


def main():
    args = parse_args()
    queries = load_queries(args)
    if not queries:
        raise RuntimeError("No queries loaded. Check dataset path or network.")
    subset = queries[args.start_index: args.start_index + args.limit]

    parser_model = args.parser_model or args.local_model or "deepseek-r1:14b"
    kb = TravelKnowledgeBase(args.database_root)
    semantic = SemanticAgent()
    policy = semantic.build_policy(args)
    # 如果开启 use_llm_filters，则输出到 save_dir/llm_filters 子文件夹
    output_dir = args.save_dir
    if getattr(args, "use_llm_filters", False):
        output_dir = os.path.join(args.save_dir, "llm_filters")
    os.makedirs(output_dir, exist_ok=True)

    submission_fp = None
    wants_submission = args.output_format in ("submission", "both")
    if wants_submission:
        if args.submission_jsonl:
            submission_path = args.submission_jsonl
        else:
            end_idx = args.start_index + max(0, len(subset) - 1)
            submission_path = f"submission_{args.set_type}_{args.start_index:03d}_{end_idx:03d}.jsonl"
        if not os.path.isabs(submission_path):
            submission_path = os.path.join(output_dir, submission_path)
        # Only sanitize the filename, not the directory separators.
        submission_dir = os.path.dirname(submission_path) or "."
        submission_name = os.path.basename(submission_path).replace(" ", "_").replace("/", "-")
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, submission_name)
        mode = "w" if args.submission_overwrite else "a"
        submission_fp = open(submission_path, mode, encoding="utf-8")

    for idx, q in enumerate(subset, start=args.start_index + 1):
        # 先输出原始 query，方便观察
        print(f"\n=== Query {idx} (raw) === {q}")
        parsed = parse_nl_query(q, args.local_base, parser_model, timeout=args.parser_timeout)
        print(f"LLM parsed JSON: {json.dumps(parsed, ensure_ascii=False)}")
        if not parsed.get("origin") or not parsed.get("destination"):
            fallback = fallback_parse(q)
            parsed = {**fallback, **parsed}
        if not parsed.get("origin") or not parsed.get("destination"):
            print(f"[{idx}] skip (parse failed): {q}")
            continue
        goal = _build_goal(parsed, args, kb, semantic)
        # Debug: show candidate city list after state expansion
        print(f"[DEBUG] Goal candidate_cities: {goal.candidate_cities}")
        t0 = time.perf_counter()
        result = _run_single(goal, kb, policy, args, raw_query=q, semantic=semantic)
        elapsed = time.perf_counter() - t0
        structured = _structured_plan(result["env"])
        diagnostics = _goal_diagnostics(kb, goal)
        filename = f"{idx:03d}_{goal.origin}_to_{goal.destination}_{goal.start_date or 'na'}_{goal.duration_days or 'days'}d.json"
        safe_name = filename.replace(" ", "_").replace("/", "-")
        out_path = os.path.join(output_dir, safe_name)
        idx_out = idx  # 1-based index for submission
        wants_debug = args.output_format in ("debug", "both")
        if wants_debug:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "query": q,
                        "parsed": parsed,
                        "goal": goal.as_text(),
                        "actions": result["actions"],
                        "success": result["success"],
                        "cost": result["cost"],
                        "violations": result["violations"],
                        "structured_plan": structured,
                        "submission_plan": env_to_submission_record(idx=idx_out, query=q, env=result["env"])["plan"],
                        "filter_usage": result.get("filter_usage", []),
                        "prior_usage": result.get("prior_usage", []),
                        "elapsed_seconds": elapsed,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        print(f"\n=== Query {idx} ===")
        print(f"Raw: {q}")
        print(f"Parsed goal: {goal.as_text()}")
        print(f"Success: {result['success']} | Cost: {result['cost']:.2f} | Violations: {result['violations']}")
        print(f"Actions: {result['actions']}")
        print(f"Dest state hint: {diagnostics['destination_state']} | Candidate cities seeded: {diagnostics['candidate_cities']}")
        print(f"Elapsed: {elapsed:.2f}s")
        if wants_debug:
            print(f"Saved to: {out_path}")

        if wants_submission and submission_fp is not None:
            rec = env_to_submission_record(
                idx=idx_out,
                query=q,
                env=result["env"],
            )
            submission_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if submission_fp is not None:
        submission_fp.close()


if __name__ == "__main__":
    main()
