import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

# Make local evaluation modules importable regardless of CWD.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from commonsense_constraint import evaluation as commonsense_eval
from hard_constraint import evaluation as hard_eval


def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def count_true_false(data):
    """Count the number of true and false values in a list."""
    true_count = data.count(True)
    false_count = data.count(False)
    return true_count, false_count

def statistics(commonsense_statistic):
    """Generate statistics for each level and day in the given data with a different structure."""
    result = {level: {day: {} for day in commonsense_statistic[level]} for level in commonsense_statistic}
    
    for level, days in commonsense_statistic.items():
        for day, dicts in days.items():
            for dct in dicts:
                if dct:
                    for key, data in dct.items():
                        true_count, false_count = count_true_false(data)
                        if key not in result[level][day]:
                            result[level][day][key] = {"true": 0, "false": 0}
                        result[level][day][key]["true"] += true_count
                        result[level][day][key]["false"] += false_count
                
    return result

def paper_term_mapping(commonsense_constraint_record, hard_constraint_record):
    mapping_dict = {'is_valid_information_in_current_city':'Within Current City','is_valid_information_in_sandbox':'Within Sandbox','is_reasonable_visiting_city':'Reasonable City Route','is_valid_restaurants':'Diverse Restaurants','is_valid_transportation':'Non-conf. Transportation','is_valid_attractions':'Diverse Attractions','is_valid_accommodation':'Minimum Nights Stay','is_not_absent':'Complete Information','valid_cost':'Budget','valid_room_rule':'Room Rule','valid_cuisine':'Cuisine','valid_room_type':'Room Type','valid_transportation':'Transportation'}
    remap_commonsense_constraint_record = {level:{day:{} for day in [3,5,7]} for level in ['easy','medium','hard']} 
    remap_hard_constraint_record = {level:{day:{} for day in [3,5,7]} for level in ['easy','medium','hard']} 
    for level in commonsense_constraint_record:
        for day in commonsense_constraint_record[level]:
            remap_commonsense_constraint_record[level][day] = {mapping_dict[key] : val for key,val in commonsense_constraint_record[level][day].items()}
            remap_hard_constraint_record[level][day] = {mapping_dict[key] : val for key,val in hard_constraint_record[level][day].items()}
    return remap_commonsense_constraint_record, remap_hard_constraint_record


def _safe_eval_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        try:
            return ast.literal_eval(obj)
        except Exception:
            return obj
    return obj


def _normalize_question(q: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(q or {})
    q = _safe_eval_obj(q)
    if isinstance(q, str):
        q = _safe_eval_obj(q)
    if isinstance(q, dict):
        if isinstance(q.get("local_constraint"), str):
            q["local_constraint"] = _safe_eval_obj(q["local_constraint"])
    return q


def _load_questions_from_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(_normalize_question(json.loads(line)))
    return items


def _load_questions_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [_normalize_question(x) for x in data]
    if isinstance(data, dict):
        return [_normalize_question(data)]
    raise ValueError("Unsupported questions JSON format")


def _load_questions_from_datasets(set_type: str, dataset_id: str, offline: bool = True) -> List[Dict[str, Any]]:
    if offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("datasets is required for --set-type loading; pass --questions-file instead") from e

    if set_type not in ("train", "validation", "test"):
        raise ValueError("--set-type must be train/validation/test")
    # TravelPlanner uses configs named by split.
    ds = load_dataset(dataset_id, set_type, split=set_type, download_mode="reuse_cache_if_exists")
    return [_normalize_question(x) for x in ds]


def _iter_plan_records(path: str) -> Iterable[Dict[str, Any]]:
    def _iter_jsonl(fp) -> Iterable[Dict[str, Any]]:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

    # Prefer extension hint.
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            yield from _iter_jsonl(f)
        return

    # Heuristic: many users save JSONL with a .json extension.
    # Detect multiple JSON objects separated by newlines and fall back to JSONL parsing.
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
        f.seek(0)
        head_lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
        looks_like_jsonl = (
            len(head_lines) >= 2
            and head_lines[0].startswith("{")
            and head_lines[1].startswith("{")
        )
        if looks_like_jsonl:
            yield from _iter_jsonl(f)
            return
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Last resort: try JSONL anyway (gives clearer per-line errors).
            f.seek(0)
            yield from _iter_jsonl(f)
            return
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        yield data
        return
    raise ValueError("Unsupported plans format; expected JSONL or JSON object/list")


def _constraint_pass(info_box: Optional[Dict[str, Tuple[Any, Any]]]) -> bool:
    if not info_box:
        return False
    for key, pair in info_box.items():
        if not pair:
            continue
        ok = pair[0]
        if ok is not None and ok is False:
            return False
    return True


def eval_local(
    *,
    plans_path: str,
    questions: List[Dict[str, Any]],
    idx_base: int = 1,
    limit: Optional[int] = None,
    only_idxs: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    total = 0
    delivered = 0
    commonsense_pass = 0
    hard_pass = 0
    final_pass = 0

    per_sample: List[Dict[str, Any]] = []

    for rec in _iter_plan_records(plans_path):
        if limit is not None and total >= limit:
            break
        if not isinstance(rec, dict):
            continue
        idx = rec.get("idx")
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except Exception:
            continue
        if only_idxs and idx_int not in only_idxs:
            continue

        q_idx = idx_int - idx_base
        if q_idx < 0 or q_idx >= len(questions):
            if verbose:
                print(f"[WARN] idx={idx_int} out of range for questions (idx_base={idx_base})")
            continue
        q = questions[q_idx]
        plan = rec.get("plan") or []

        total += 1
        has_plan = bool(plan)
        if has_plan:
            delivered += 1
            commonsense_info = commonsense_eval(q, plan)
        else:
            commonsense_info = None

        hard_info = None
        if commonsense_info and commonsense_info.get("is_not_absent", (False,))[0] and commonsense_info.get(
            "is_valid_information_in_sandbox", (False,)
        )[0]:
            hard_info = hard_eval(q, plan)

        c_pass = _constraint_pass(commonsense_info)
        h_pass = _constraint_pass(hard_info)
        f_pass = c_pass and h_pass

        commonsense_pass += int(c_pass)
        hard_pass += int(h_pass)
        final_pass += int(f_pass)

        per_sample.append(
            {
                "idx": idx_int,
                "delivery": has_plan,
                "commonsense_pass": c_pass,
                "hard_pass": h_pass,
                "final_pass": f_pass,
            }
        )

        if verbose:
            print(
                f"idx={idx_int} delivery={has_plan} commonsense={c_pass} hard={h_pass} final={f_pass}"
            )

    summary = {
        "total": total,
        "delivery_rate": (delivered / total) if total else 0.0,
        "commonsense_pass_rate": (commonsense_pass / total) if total else 0.0,
        "hard_pass_rate": (hard_pass / total) if total else 0.0,
        "final_pass_rate": (final_pass / total) if total else 0.0,
        "per_sample": per_sample,
    }
    return summary


def eval_score(set_type: str, file_path: str):

    # Legacy batch eval (HF download); kept for compatibility.
    from datasets import load_dataset  # type: ignore
    if set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train',download_mode="reuse_cache_if_exists")['train']
    elif set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation',download_mode="reuse_cache_if_exists")['validation']

    
    query_data_list = [x for x in query_data_list]
    hardConstraint_statistic= {level:{day:[] for day in [3,5,7]} for level in ['easy','medium','hard']} 
    commonsenseConstraint_statistic = {level:{day:[] for day in [3,5,7]} for level in ['easy','medium','hard']} 
    tested_plans = load_line_json_data(file_path)
    delivery_cnt = 0
    plan_constraint_store = []
    for idx in tqdm(range(0,len(query_data_list))):
        query_data = query_data_list[idx]
        tested_plan = tested_plans[idx]
        if type(query_data) == str:
            query_data = eval(query_data)
        if type(tested_plan) == str:
            tested_plan = eval(tested_plan)
        if type(query_data['local_constraint']) == str:
            query_data['local_constraint'] = eval(query_data['local_constraint'])

        if tested_plan['plan']:
            delivery_cnt += 1
            commonsense_info_box = commonsense_eval(query_data,tested_plan['plan'])
        else:
            commonsense_info_box = None

        if commonsense_info_box and commonsense_info_box['is_not_absent'][0] and commonsense_info_box['is_valid_information_in_sandbox'][0]:
            hard_info_box = hard_eval(query_data,tested_plan['plan'])
        else:
            hard_info_box = None

        plan_constraint_store.append({'commonsense_constraint':commonsense_info_box,'hard_constraint':hard_info_box})

        commonsenseConstraint_statistic[query_data['level']][query_data['days']].append(commonsense_info_box)
        hardConstraint_statistic[query_data['level']][query_data['days']].append(hard_info_box)

    constraint_record = {key: {day: {'house rule':0, 'cuisine':0, 'room type':0, 'transportation':0} for day in [3,5,7]} for key in ['medium','hard']}
    constraint_mapping = {'house rule':'valid_room_rule','cuisine':'valid_cuisine','room type':'valid_room_type','transportation':'valid_transportation'}
    mapping_constraint_record = {key: {day: {'valid_room_rule':0, 'valid_cuisine':0, 'valid_room_type':0, 'valid_transportation':0} for day in [3,5,7]} for key in ['medium','hard']}
    count_record = {key:{day:0 for day in [3,5,7]} for key in ['easy','medium','hard']}

    for unit in query_data_list:
        count_record[unit['level']][unit['days']] += 1
        for key in constraint_record['medium'][3]:
            if unit['local_constraint'][key] != None:
                constraint_record[unit['level']][unit['days']][key] += 1
                mapping_constraint_record[unit['level']][unit['days']][constraint_mapping[key]] += 1
    
    commonsenseConstraint_statistic_processed = statistics(commonsenseConstraint_statistic)
    hardConstraint_statistic_processed = statistics(hardConstraint_statistic)


    data_record = {key:{day:[] for day in [3,5,7]} for key in ['easy','medium','hard']}

    constraint_dis_record = {"commonsense":{"pass":0,"total":0},"hard":{"pass":0,"total":0}}
    constraint_count = {key:{day:{} for day in [3,5,7]} for key in ['easy','medium','hard']}

    for constraint in ['commonsense','hard']:
        if constraint == 'commonsense':
            constraint_statistic = commonsenseConstraint_statistic_processed
        elif constraint == 'hard':
            constraint_statistic = hardConstraint_statistic_processed

        key_dict = {'commonsense':['is_valid_information_in_current_city','is_valid_information_in_sandbox','is_reasonable_visiting_city','is_valid_restaurants','is_valid_transportation','is_valid_attractions','is_valid_accommodation','is_not_absent'],'hard':['valid_cost','valid_room_rule','valid_cuisine','valid_room_type','valid_transportation']}
        
        for key in constraint_statistic:
            for key2 in constraint_statistic[key]:
                if key2 == -1:
                    print(constraint_statistic[key])
                    exit(0)
                for key3 in key_dict[constraint]:
                    data_record[key][key2].append('0/0')
                    if key3 in constraint_statistic[key][key2]:
                        constraint_dis_record[constraint]['pass'] += constraint_statistic[key][key2][key3]['true']
                        if constraint == 'hard':
                            if key == 'hard' and key3 in ['valid_room_rule','valid_cuisine','valid_room_type','valid_transportation']:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{mapping_constraint_record[key][key2][key3]}"
                                constraint_dis_record[constraint]['total'] += mapping_constraint_record[key][key2][key3]
                                hardConstraint_statistic_processed[key][key2][key3]['total'] = mapping_constraint_record[key][key2][key3]
                            elif key == 'medium' and key3 in ['valid_room_rule','valid_cuisine','valid_room_type']:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{mapping_constraint_record[key][key2][key3]}"
                                constraint_dis_record[constraint]['total'] += mapping_constraint_record[key][key2][key3]
                                hardConstraint_statistic_processed[key][key2][key3]['total'] = mapping_constraint_record[key][key2][key3]
                            else:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{count_record[key][key2]}"
                                if key3 in ['valid_cost','valid_visitng_city_number','valid_days']:
                                    constraint_dis_record[constraint]['total'] += count_record[key][key2]
                                    constraint_count[key][key2][key3] = count_record[key][key2]
                                    hardConstraint_statistic_processed[key][key2][key3]['total'] = count_record[key][key2]
                        else:
                            data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{count_record[key][key2]}"
                            constraint_dis_record[constraint]['total'] += count_record[key][key2]
                            constraint_count[key][key2][key3] = count_record[key][key2]
                            commonsenseConstraint_statistic_processed[key][key2][key3]['total'] =  count_record[key][key2]
    final_all_cnt = 0
    final_commonsense_cnt = 0
    final_hardConstraint_cnt = 0
    final_all_cnt_map = {level:0 for level in ['easy','medium','hard']}
    for idx in (range(0,len(query_data_list))):
        if plan_constraint_store[idx]['commonsense_constraint']:
            final_commonsense_pass = True
            final_hardConstraint_pass = True
            for item in plan_constraint_store[idx]['commonsense_constraint']:
                if plan_constraint_store[idx]['commonsense_constraint'][item][0] is not None and not plan_constraint_store[idx]['commonsense_constraint'][item][0]:
                    final_commonsense_pass = False
                    break
            if plan_constraint_store[idx]['hard_constraint'] is None:
                continue
            for item in plan_constraint_store[idx]['hard_constraint']:
                if plan_constraint_store[idx]['hard_constraint'][item][0] is not None and  plan_constraint_store[idx]['hard_constraint'][item][0] == False:
                    final_hardConstraint_pass = False
                    break
                
            if final_commonsense_pass:
                final_commonsense_cnt += 1
            if final_hardConstraint_pass:
                final_hardConstraint_cnt += 1
            if final_commonsense_pass and final_hardConstraint_pass:
                final_all_cnt += 1
                final_all_cnt_map[query_data_list[idx]['level']] += 1

    result = {}

    remap_commonsense_constraint_record, remap_hard_constraint_record = paper_term_mapping(commonsenseConstraint_statistic_processed, hardConstraint_statistic_processed)

    if set_type == 'train':
        result['Delivery Rate'] = delivery_cnt / 45
        result['Commonsense Constraint Micro Pass Rate'] = constraint_dis_record['commonsense']['pass'] / 360
        result['Commonsense Constraint Macro Pass Rate'] = final_commonsense_cnt / 45
        result['Hard Constraint Micro Pass Rate'] = constraint_dis_record['hard']['pass'] / 105
        result['Hard Constraint Macro Pass Rate'] = final_hardConstraint_cnt / 45
        result['Final Pass Rate'] = final_all_cnt / 45

    elif set_type == 'validation':
        result['Delivery Rate'] = delivery_cnt / 180
        result['Commonsense Constraint Micro Pass Rate'] = constraint_dis_record['commonsense']['pass'] / 1440
        result['Commonsense Constraint Macro Pass Rate'] = final_commonsense_cnt / 180
        result['Hard Constraint Micro Pass Rate'] = constraint_dis_record['hard']['pass'] / 420
        result['Hard Constraint Macro Pass Rate'] = final_hardConstraint_cnt / 180
        result['Final Pass Rate'] = final_all_cnt / 180

    elif set_type == 'test':
        result['Delivery Rate'] = delivery_cnt / 1000
        result['Commonsense Constraint Micro Pass Rate'] = constraint_dis_record['commonsense']['pass'] / 8000
        result['Commonsense Constraint Macro Pass Rate'] = final_commonsense_cnt / 1000
        result['Hard Constraint Micro Pass Rate'] = constraint_dis_record['hard']['pass'] / 2290
        result['Hard Constraint Macro Pass Rate'] = final_hardConstraint_cnt / 1000
        result['Final Pass Rate'] = final_all_cnt / 1000
    

    return result, {"Commonsense Constraint":remap_commonsense_constraint_record, "Hard Constraint":remap_hard_constraint_record}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=str, default=None, help="Submission JSONL/JSON path with {idx, query, plan}.")
    parser.add_argument("--questions-file", type=str, default=None, help="Local questions JSON/JSONL (dataset entries).")
    parser.add_argument("--set-type", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--dataset-id", type=str, default="osunlp/TravelPlanner")
    parser.add_argument("--offline", action="store_true", help="Load dataset from local HF cache only.")
    parser.add_argument("--idx-base", type=int, default=1, help="Idx base in plan file (1 for your submission, 0 for some baselines).")
    parser.add_argument("--idx", type=int, action="append", default=None, help="Evaluate only these idx values (repeatable).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--legacy-eval-file", type=str, default=None, help="Use legacy eval_score() on full split (network/cache).")
    args = parser.parse_args()

    if args.legacy_eval_file:
        scores, detailed_scores = eval_score(args.set_type, file_path=args.legacy_eval_file)
        for key in scores:
            print(f"{key}: {scores[key]*100}%")
        print("------------------")
        print(detailed_scores)
        print("------------------")
        raise SystemExit(0)

    if not args.plans:
        raise SystemExit("Missing --plans. Provide a submission JSONL/JSON file.")

    if args.questions_file:
        if args.questions_file.endswith(".jsonl"):
            questions = _load_questions_from_jsonl(args.questions_file)
        else:
            questions = _load_questions_from_json(args.questions_file)
    else:
        questions = _load_questions_from_datasets(args.set_type, args.dataset_id, offline=args.offline)

    summary = eval_local(
        plans_path=args.plans,
        questions=questions,
        idx_base=args.idx_base,
        limit=args.limit,
        only_idxs=args.idx,
        verbose=args.verbose,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, ensure_ascii=False, indent=2))
