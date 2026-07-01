import ast
import json
import shutil
from collections import Counter
from pathlib import Path


BFCL_VERSION = "bfcl-eval 2026.3.23 / BFCL_v4"
MULTIPLE_CATEGORIES = ["multiple", "live_multiple"]
SIMPLE_CATEGORIES = ["simple_python", "simple_java", "simple_javascript", "live_simple"]
ALL_CATEGORIES = MULTIPLE_CATEGORIES + SIMPLE_CATEGORIES


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_raw_data(source_data_dir, raw_dir, categories=ALL_CATEGORIES):
    source = Path(source_data_dir)
    raw = Path(raw_dir)
    copied = []
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "possible_answer").mkdir(parents=True, exist_ok=True)
    for category in categories:
        name = f"BFCL_v4_{category}.json"
        for src, dst in [
            (source / name, raw / name),
            (source / "possible_answer" / name, raw / "possible_answer" / name),
        ]:
            if not src.exists():
                raise FileNotFoundError(src)
            if not dst.exists():
                shutil.copy2(src, dst)
                copied.append(str(dst))
    return copied


def query_text(row):
    question = row.get("question")
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        messages = question[0] if question and isinstance(question[0], list) else question
        return "\n".join(message.get("content", "") for message in messages if isinstance(message, dict))
    return ""


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def parse_bfcl_call(call):
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
    return {"name": _call_name(expression.func), "arguments": arguments}


def _canonical_ground_truth_object(ground_truth):
    if isinstance(ground_truth, dict) and "name" in ground_truth and "accepted_arguments" in ground_truth:
        return {"name": ground_truth["name"], "accepted_arguments": dict(ground_truth["accepted_arguments"])}
    if isinstance(ground_truth, list):
        if len(ground_truth) != 1:
            raise ValueError(f"Expected one ground-truth call, got {len(ground_truth)}")
        ground_truth = ground_truth[0]
    if isinstance(ground_truth, str):
        parsed = parse_bfcl_call(ground_truth)
        return {"name": parsed["name"], "accepted_arguments": {key: [value] for key, value in parsed["arguments"].items()}}
    if not isinstance(ground_truth, dict) or len(ground_truth) != 1:
        raise ValueError(f"Unsupported ground-truth shape: {ground_truth!r}")
    name, arguments = next(iter(ground_truth.items()))
    return {"name": name, "accepted_arguments": dict(arguments or {})}


def canonical_ground_truth(row_or_ground_truth):
    if isinstance(row_or_ground_truth, dict) and "ground_truth" in row_or_ground_truth:
        return _canonical_ground_truth_object(row_or_ground_truth["ground_truth"])
    return _canonical_ground_truth_object(row_or_ground_truth)


def _first_concrete_value(accepted):
    values = accepted if isinstance(accepted, list) else [accepted]
    for value in values:
        if value == "":
            continue
        if isinstance(value, dict):
            nested_values = {
                key: concrete
                for key, nested in value.items()
                for concrete in [_first_concrete_value(nested)]
                if concrete != ""
            }
            return nested_values if nested_values else ""
        return value
    return ""


def concrete_arguments(ground_truth):
    gt = canonical_ground_truth(ground_truth)
    return {key: _first_concrete_value(value) for key, value in gt["accepted_arguments"].items() if _first_concrete_value(value) != ""}


def _matches_accepted(value, accepted):
    values = accepted if isinstance(accepted, list) else [accepted]
    non_empty = [candidate for candidate in values if candidate != ""]
    if not non_empty:
        return value in ("", None)
    return any(_value_matches(value, candidate) for candidate in non_empty)


def _has_concrete(value):
    if value == "":
        return False
    if isinstance(value, dict):
        return any(_has_concrete(nested) for nested in value.values())
    return True


def _value_matches(value, accepted):
    if isinstance(accepted, dict):
        if not isinstance(value, dict):
            return False
        required = {
            key
            for key, nested in accepted.items()
            if any(_has_concrete(v) for v in (nested if isinstance(nested, list) else [nested]))
        }
        if set(value) != required:
            return False
        return all(_matches_accepted(value[key], accepted[key]) for key in required)
    return value == accepted


def ast_arguments_match(prediction, ground_truth):
    parsed = parse_bfcl_call(prediction)
    gt = canonical_ground_truth(ground_truth)
    required = {
        key
        for key, accepted in gt["accepted_arguments"].items()
        if any(_has_concrete(value) for value in (accepted if isinstance(accepted, list) else [accepted]))
    }
    if set(parsed["arguments"]) != required:
        return False
    return all(_matches_accepted(parsed["arguments"][key], gt["accepted_arguments"][key]) for key in required)


def router_success(prediction, ground_truth):
    try:
        return parse_bfcl_call(prediction)["name"] == canonical_ground_truth(ground_truth)["name"]
    except Exception:
        return False


def validator_success(prediction, ground_truth):
    try:
        return router_success(prediction, ground_truth) and ast_arguments_match(prediction, ground_truth)
    except Exception:
        return False


def call_success(prediction, ground_truth):
    return router_success(prediction, ground_truth) and validator_success(prediction, ground_truth)


def ast_check(prediction, ground_truth):
    return call_success(prediction, ground_truth)


def function_names(function_pool):
    return [item["name"] for item in function_pool if isinstance(item, dict) and item.get("name")]


def build_wrong_function_prediction(row, ground_truth):
    gt = canonical_ground_truth(ground_truth)
    wrong_name = next((name for name in function_names(row.get("function_pool", row.get("function", []))) if name != gt["name"]), None)
    if wrong_name is None:
        wrong_name = f"{gt['name']}__wrong"
    return {"name": wrong_name, "arguments": concrete_arguments(gt)}


def _mutate_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "__wrong"
    if isinstance(value, dict) and value:
        mutated = dict(value)
        first = next(iter(mutated))
        mutated[first] = _mutate_value(mutated[first])
        return mutated
    return "__wrong"


def build_wrong_argument_prediction(ground_truth):
    gt = canonical_ground_truth(ground_truth)
    args = concrete_arguments(gt)
    if args:
        first = next(iter(args))
        args[first] = _mutate_value(args[first])
    else:
        args["unexpected"] = "wrong"
    return {"name": gt["name"], "arguments": args}


def load_categories(raw_dir, categories):
    raw = Path(raw_dir)
    loaded = {}
    for category in categories:
        questions = read_jsonl(raw / f"BFCL_v4_{category}.json")
        answers = {row["id"]: row for row in read_jsonl(raw / "possible_answer" / f"BFCL_v4_{category}.json")}
        rows = []
        for row in questions:
            merged = dict(row)
            merged["ground_truth"] = answers[row["id"]]["ground_truth"]
            merged["category"] = category
            merged["split"] = "multiple" if category in MULTIPLE_CATEGORIES else "simple"
            merged["query"] = query_text(row)
            merged["function_pool"] = list(row.get("function") or [])
            merged["ground_truth_call"] = canonical_ground_truth(merged)
            merged["ground_truth_concrete_arguments"] = concrete_arguments(merged)
            rows.append(merged)
        loaded[category] = rows
    return loaded


def load_bfcl_splits(raw_dir):
    category_rows = load_categories(raw_dir, ALL_CATEGORIES)
    return {
        "multiple": [row for category in MULTIPLE_CATEGORIES for row in category_rows[category]],
        "simple": [row for category in SIMPLE_CATEGORIES for row in category_rows[category]],
    }


def summarize_pool_sizes(rows):
    sizes = [len(row["function_pool"]) for row in rows]
    ordered = sorted(sizes)
    return {
        "min": min(sizes) if sizes else None,
        "median": ordered[len(ordered) // 2] if sizes else None,
        "mean": sum(sizes) / len(sizes) if sizes else None,
        "max": max(sizes) if sizes else None,
        "histogram": dict(sorted(Counter(sizes).items())),
    }


def summarize_field_coverage(rows):
    coverage = {
        "query": 0,
        "ground_truth_function_name": 0,
        "ground_truth_arguments": 0,
        "function_pool": 0,
    }
    for row in rows:
        coverage["query"] += int(bool(row.get("query")))
        coverage["ground_truth_function_name"] += int(bool(row.get("ground_truth_call", {}).get("name")))
        coverage["ground_truth_arguments"] += int(isinstance(row.get("ground_truth_call", {}).get("accepted_arguments"), dict))
        coverage["function_pool"] += int(bool(row.get("function_pool")))
    return coverage
