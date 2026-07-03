import argparse
import ast
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts.phase0_probe import DEFAULT_SPLIT_SEED, build_base_prompt, load_exp2, read_json, write_json
from scripts.v16_d2_prep import collect_eval_touches, collect_training_touches


PARALLEL_CATEGORIES = ["parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"]
SIMPLE_PYTHON_CATEGORY = "simple_python"
DEFAULT_CATEGORIES = PARALLEL_CATEGORIES + [SIMPLE_PYTHON_CATEGORY]
DEFAULT_SEED = 20260704

PARALLEL_PROMPT_TEMPLATE = """You are a function-calling model. Choose every function call required by the user from the function document.

Output format:
[{{"name": "<function_name>", "arguments": {{"<argument_name>": <argument_value>}}}}, ...]

Rules:
- Return exactly one JSON array.
- Each array item is one function call object.
- Do not explain.

User query:
{query}

Function document:
{function_doc}

Return only the JSON array."""


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_sample(rows, count, seed, label):
    if count >= len(rows):
        return list(rows)
    keyed = []
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{label}:{row['episode_id']}".encode("utf-8")).hexdigest()
        keyed.append((digest, row))
    return [row for _, row in sorted(keyed, key=lambda item: item[0])[:count]]


def query_text(row):
    question = row.get("question")
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        messages = question[0] if question and isinstance(question[0], list) else question
        return "\n".join(message.get("content", "") for message in messages if isinstance(message, dict))
    return ""


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def parse_call(call):
    if isinstance(call, dict) and "name" in call and "arguments" in call:
        return {"name": call["name"], "arguments": dict(call.get("arguments") or {})}
    if isinstance(call, dict) and len(call) == 1:
        name, arguments = next(iter(call.items()))
        return {"name": name, "arguments": dict(arguments or {})}
    if not isinstance(call, str):
        raise ValueError(f"Unsupported call shape: {call!r}")
    expression = ast.parse(call, mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ValueError(f"Expected a function call expression: {call!r}")
    arguments = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("Variadic keyword arguments are not supported")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return {"name": call_name(expression.func), "arguments": arguments}


def parse_model_output_multi(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return [parse_call(item) for item in obj]

    calls = []
    consumed = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if any(start <= index < stop for start, stop in consumed):
            continue
        try:
            calls.append(parse_call(obj))
            consumed.append((index, index + end))
        except Exception:
            continue
    if calls:
        return calls

    line_calls = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*\s*\(.*\)", line, re.S):
            line_calls.append(parse_call(line))
    if line_calls:
        return line_calls
    raise ValueError("No parseable parallel function calls found")


def canonical_gt_call(call):
    parsed = parse_call(call)
    return {"name": parsed["name"], "accepted_arguments": dict(parsed.get("arguments") or {})}


def canonical_gt_list(ground_truth):
    if not isinstance(ground_truth, list):
        raise ValueError(f"Expected parallel ground truth list: {ground_truth!r}")
    return [canonical_gt_call(call) for call in ground_truth]


def has_concrete(value):
    if value == "":
        return False
    if isinstance(value, dict):
        return any(has_concrete(nested) for nested in value.values())
    return True


def value_matches(value, accepted):
    if isinstance(accepted, dict):
        if not isinstance(value, dict):
            return False
        required = {
            key
            for key, nested in accepted.items()
            if any(has_concrete(candidate) for candidate in (nested if isinstance(nested, list) else [nested]))
        }
        if set(value) != required:
            return False
        return all(matches_accepted(value[key], accepted[key]) for key in required)
    return value == accepted


def matches_accepted(value, accepted):
    values = accepted if isinstance(accepted, list) else [accepted]
    non_empty = [candidate for candidate in values if candidate != ""]
    if not non_empty:
        return value in ("", None)
    return any(value_matches(value, candidate) for candidate in non_empty)


def single_call_success(prediction, ground_truth):
    if prediction.get("name") != ground_truth["name"]:
        return False
    required = {
        key
        for key, accepted in ground_truth["accepted_arguments"].items()
        if any(has_concrete(candidate) for candidate in (accepted if isinstance(accepted, list) else [accepted]))
    }
    arguments = prediction.get("arguments") or {}
    if set(arguments) != required:
        return False
    return all(matches_accepted(arguments[key], ground_truth["accepted_arguments"][key]) for key in required)


def multiset_equal(left, right):
    return Counter(left) == Counter(right)


def parallel_call_success(predictions, ground_truth_calls):
    if len(predictions) != len(ground_truth_calls):
        return False
    used = [False] * len(predictions)

    def backtrack(gt_index):
        if gt_index == len(ground_truth_calls):
            return True
        gt = ground_truth_calls[gt_index]
        for pred_index, prediction in enumerate(predictions):
            if used[pred_index]:
                continue
            if single_call_success(prediction, gt):
                used[pred_index] = True
                if backtrack(gt_index + 1):
                    return True
                used[pred_index] = False
        return False

    return backtrack(0)


def parallel_router_success(predictions, ground_truth_calls):
    return len(predictions) == len(ground_truth_calls) and multiset_equal(
        [prediction.get("name") for prediction in predictions],
        [gt["name"] for gt in ground_truth_calls],
    )


def render_parallel_prompt(row):
    return PARALLEL_PROMPT_TEMPLATE.format(
        query=row.get("query", ""),
        function_doc=json.dumps(row.get("function_pool", []), ensure_ascii=False),
    )


def load_raw_categories(raw_root, categories, recorded_ids):
    raw_root = Path(raw_root)
    output = []
    for category in categories:
        question_path = raw_root / f"BFCL_v4_{category}.json"
        answer_path = raw_root / "possible_answer" / f"BFCL_v4_{category}.json"
        answers = {row["id"]: row for row in read_jsonl(answer_path)}
        for row in read_jsonl(question_path):
            if category == SIMPLE_PYTHON_CATEGORY and row["id"] in recorded_ids:
                continue
            answer = answers[row["id"]]
            merged = {
                "episode_id": row["id"],
                "category": category,
                "segment": "simple_python_unrecorded" if category == SIMPLE_PYTHON_CATEGORY else "parallel_family",
                "query": query_text(row),
                "function_pool": list(row.get("function") or []),
                "ground_truth": answer["ground_truth"],
            }
            if category in PARALLEL_CATEGORIES:
                merged["ground_truth_calls"] = canonical_gt_list(answer["ground_truth"])
            else:
                gt_call = canonical_gt_list(answer["ground_truth"])[0]
                merged["ground_truth_call"] = gt_call
            output.append(merged)
    output.sort(key=lambda item: (item["segment"], item["category"], item["episode_id"]))
    return output


def touched_sets(root, baseline, split_seed):
    root = Path(root)
    training_touches = collect_training_touches(root)
    eval_touches = collect_eval_touches(root)
    training_ids = set().union(*training_touches.values()) if training_touches else set()
    eval_ids = set().union(*eval_touches.values()) if eval_touches else set()
    s00 = read_json(root / "results/s00_inventory.json")
    old400 = set(s00["regression_sets"]["r_success_eval"])
    d_val = set()
    d_heldout = set()
    evo = load_exp2("../EDG-EXP2-struct")
    multiple_failures = [item for item in baseline["failures"] if item.get("split") == "multiple"]
    split = evo.deterministic_split(multiple_failures, seed=split_seed)
    d_val.update(row["episode_id"] for row in split["validation"])
    heldout_path = root / "results/s01_heldout_pass16_partition.json"
    if heldout_path.exists():
        d_heldout.update(row["episode_id"] for row in read_json(heldout_path).get("episodes", []))
    capped = set()
    capped_path = root / "results/v13_capped_pool_arena.json"
    if capped_path.exists():
        capped.update(read_json(capped_path).get("episode_ids", []))
    return {
        "training": training_ids,
        "all_eval_or_diagnostic": eval_ids,
        "old400": old400,
        "capped106": capped,
        "d_val": d_val,
        "d_heldout": d_heldout,
    }


def summarize(records):
    summary = {}
    for segment in sorted({row["segment"] for row in records}):
        rows = [row for row in records if row["segment"] == segment]
        success = sum(row.get("call_success") is True for row in rows)
        summary[segment] = {"n": len(rows), "success": success, "success_rate": success / len(rows) if rows else 0.0}
    by_category = {}
    for category in sorted({row["category"] for row in records}):
        rows = [row for row in records if row["category"] == category]
        success = sum(row.get("call_success") is True for row in rows)
        by_category[category] = {"n": len(rows), "success": success, "success_rate": success / len(rows) if rows else 0.0}
    return summary, by_category


def build_arena(root, records, baseline, args):
    clean_successes = [row for row in records if row.get("call_success") is True]
    selected = stable_sample(clean_successes, min(args.arena_size, len(clean_successes)), args.seed, "v17_sibling_arena")
    ids = {row["episode_id"] for row in selected}
    touches = touched_sets(root, baseline, args.split_seed)
    asserts = {
        "arena_intersect_all_training_rows": len(ids & touches["training"]),
        "arena_intersect_old400": len(ids & touches["old400"]),
        "arena_intersect_capped106": len(ids & touches["capped106"]),
        "arena_intersect_d_val": len(ids & touches["d_val"]),
        "arena_intersect_d_heldout": len(ids & touches["d_heldout"]),
    }
    if any(asserts.values()):
        raise AssertionError(asserts)
    distribution = Counter(row["segment"] for row in selected)
    category_distribution = Counter(row["category"] for row in selected)
    payload = {
        "probe": "v17_sibling_arena",
        "source": "parallel-family never-touched plus simple_python unrecorded candidate successes",
        "seed": args.seed,
        "requested_arena_size": args.arena_size,
        "available_clean_successes": len(clean_successes),
        "selected_size": len(selected),
        "status": "needs_v17_decision_tree_adjudication",
        "segment_distribution": dict(sorted(distribution.items())),
        "category_distribution": dict(sorted(category_distribution.items())),
        "asserts": asserts,
        "episode_ids": [row["episode_id"] for row in selected],
        "records": selected,
    }
    write_json(root / args.arena_output, payload)
    return payload


def evaluate(args):
    from scripts.lora_phase0 import generate_one, load_model_for_eval, load_tokenizer

    root = Path(args.root)
    baseline = read_json(args.failures)
    recorded_ids = {row["episode_id"] for row in baseline["records"]}
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    rows = load_raw_categories(args.raw_root, categories, recorded_ids)
    existing = []
    if args.resume and (root / args.output).exists():
        existing = read_json(root / args.output).get("records", [])
    seen = {row["episode_id"] for row in existing}
    records = list(existing)
    tokenizer = load_tokenizer(args.model_id)
    model = load_model_for_eval(args.model_id, adapter_dir=None, qlora=not args.no_qlora)
    for index, row in enumerate(rows, start=1):
        if row["episode_id"] in seen:
            continue
        prompt = render_parallel_prompt(row) if row["category"] in PARALLEL_CATEGORIES else build_base_prompt(
            {
                "query": row["query"],
                "function_pool": row["function_pool"],
            },
            baseline["prompt_template"],
        )
        started = time.time()
        raw = generate_one(model, tokenizer, prompt, args.max_new_tokens)
        latency_ms = int((time.time() - started) * 1000)
        parse_error = None
        if row["category"] in PARALLEL_CATEGORIES:
            try:
                prediction = parse_model_output_multi(raw)
            except Exception as error:
                prediction = []
                parse_error = str(error)
            router = parallel_router_success(prediction, row["ground_truth_calls"])
            call = parallel_call_success(prediction, row["ground_truth_calls"])
            validator = call
        else:
            try:
                prediction = parse_model_output_multi(raw)
            except Exception as error:
                prediction = []
                parse_error = str(error)
            if len(prediction) == 1:
                router = prediction[0].get("name") == row["ground_truth_call"]["name"]
                call = single_call_success(prediction[0], row["ground_truth_call"])
                validator = call
            else:
                router = False
                validator = False
                call = False
        record = {
            **row,
            "predicted_calls": prediction,
            "raw_model_output": raw,
            "parse_error": parse_error,
            "router_success": router,
            "validator_success": validator,
            "call_success": call,
            "latency_ms": latency_ms,
        }
        records.append(record)
        if len(records) % args.save_every == 0:
            write_payload(root, records, baseline, args)
        if index == 1 or index % args.progress_every == 0 or index == len(rows):
            print(f"[v17-base] {index}/{len(rows)} {row['category']} {row['episode_id']} call={call}", flush=True)
    payload = write_payload(root, records, baseline, args)
    if args.arena_output:
        build_arena(root, records, baseline, args)
    return payload


def write_payload(root, records, baseline, args):
    summary, by_category = summarize(records)
    simple_python_ids = [
        row["episode_id"] for row in records if row["category"] == SIMPLE_PYTHON_CATEGORY
    ]
    payload = {
        "probe": "v17_sibling_base_eval",
        "judge_scope": "parallel-family differential judge plus existing single-call-compatible simple_python subset",
        "model_id": args.model_id,
        "categories": [item.strip() for item in args.categories.split(",") if item.strip()],
        "generation_params": {"do_sample": False, "temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "parallel_prompt_template": PARALLEL_PROMPT_TEMPLATE,
        "summary": summary,
        "by_category": by_category,
        "simple_python_unrecorded_episode_ids": simple_python_ids,
        "simple_python_unrecorded_count": len(simple_python_ids),
        "records": records,
    }
    write_json(root / args.output, payload)
    return payload


def judge_smoke():
    gt = canonical_gt_list([
        {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
        {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
    ])
    pred = parse_model_output_multi('[{"name":"spotify.play","arguments":{"artist":"Maroon 5","duration":15}}, {"name":"spotify.play","arguments":{"artist":"Taylor Swift","duration":20}}]')
    assert parallel_call_success(pred, gt)
    assert parallel_router_success(pred, gt)
    wrong_count = parse_model_output_multi('[{"name":"spotify.play","arguments":{"artist":"Taylor Swift","duration":20}}]')
    assert not parallel_call_success(wrong_count, gt)
    wrong_arg = parse_model_output_multi('[{"name":"spotify.play","arguments":{"artist":"Taylor Swift","duration":21}}, {"name":"spotify.play","arguments":{"artist":"Maroon 5","duration":15}}]')
    assert not parallel_call_success(wrong_arg, gt)
    nested_gt = canonical_gt_list([
        {"ChaDri.change_drink": {"drink_id": ["123"], "new_preferences": [{"size": ["large"], "temperature": ["hot"]}]}},
    ])
    nested_pred = [{"name": "ChaDri.change_drink", "arguments": {"drink_id": "123", "new_preferences": {"size": "large", "temperature": "hot"}}}]
    assert parallel_call_success(nested_pred, nested_gt)
    print(json.dumps({"judge_smoke": "ok"}, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description="v1.17 sibling arena with parallel-family differential judge")
    parser.add_argument("--root", default=".")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--raw-root", default="../EDG-BFCL/deps/bfcl_eval_pkg/bfcl_eval/data")
    parser.add_argument("--model-id", required=False)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--output", default="results/v17_sibling_base_eval.json")
    parser.add_argument("--arena-output", default="results/v17_sibling_arena.json")
    parser.add_argument("--arena-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-smoke", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.judge_smoke:
        judge_smoke()
        return
    if not args.model_id:
        raise SystemExit("--model-id is required unless --judge-smoke is used")
    result = evaluate(args)
    print(json.dumps({"summary": result["summary"], "by_category": result["by_category"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
