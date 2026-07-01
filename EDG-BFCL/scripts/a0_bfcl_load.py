import argparse
import ast
import json
import shutil
import sys
from pathlib import Path


BFCL_VERSION = "bfcl-eval 2026.3.23 / BFCL_v4"
CONNECTED_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "live_simple",
]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def parse_bfcl_call(call):
    if isinstance(call, dict) and "name" in call and "arguments" in call:
        return {"name": call["name"], "arguments": dict(call["arguments"])}
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
            raise ValueError("Variadic keyword arguments are not supported in BFCL simple AST probe")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return {"name": _call_name(expression.func), "arguments": arguments}


def _canonical_ground_truth(ground_truth):
    if isinstance(ground_truth, list):
        if len(ground_truth) != 1:
            raise ValueError(f"Expected one ground-truth call for simple category, got {len(ground_truth)}")
        ground_truth = ground_truth[0]
    if isinstance(ground_truth, str):
        parsed = parse_bfcl_call(ground_truth)
        return {"name": parsed["name"], "accepted_arguments": {key: [value] for key, value in parsed["arguments"].items()}}
    if not isinstance(ground_truth, dict) or len(ground_truth) != 1:
        raise ValueError(f"Unsupported ground-truth shape: {ground_truth!r}")
    name, arguments = next(iter(ground_truth.items()))
    return {"name": name, "accepted_arguments": dict(arguments or {})}


def extract_ground_truth(row):
    canonical = _canonical_ground_truth(row["ground_truth"])
    arguments = {}
    for key, accepted in canonical["accepted_arguments"].items():
        accepted_values = accepted if isinstance(accepted, list) else [accepted]
        non_empty = [value for value in accepted_values if value != ""]
        if non_empty:
            arguments[key] = non_empty[0]
    return {"name": canonical["name"], "arguments": arguments}


def _accepted_value_matches(value, accepted_values):
    values = accepted_values if isinstance(accepted_values, list) else [accepted_values]
    non_empty = [candidate for candidate in values if candidate != ""]
    if not non_empty:
        return value in ("", None)
    return value in non_empty


def ast_check(prediction, ground_truth):
    parsed = parse_bfcl_call(prediction)
    if "accepted_arguments" not in ground_truth:
        ground_truth = {
            "name": ground_truth["name"],
            "accepted_arguments": {key: [value] for key, value in ground_truth.get("arguments", {}).items()},
        }
    if parsed["name"] != ground_truth["name"]:
        return False
    required_keys = {
        key
        for key, accepted in ground_truth["accepted_arguments"].items()
        if any(value != "" for value in (accepted if isinstance(accepted, list) else [accepted]))
    }
    if set(parsed["arguments"]) != required_keys:
        return False
    return all(
        _accepted_value_matches(parsed["arguments"][key], ground_truth["accepted_arguments"][key])
        for key in required_keys
    )


def query_text(row):
    question = row.get("question")
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        messages = question[0] if question and isinstance(question[0], list) else question
        return "\n".join(message.get("content", "") for message in messages if isinstance(message, dict))
    return ""


def summarize_category_rows(category, rows):
    coverage = {
        "query": 0,
        "ground_truth_function_name": 0,
        "ground_truth_arguments": 0,
        "function_doc": 0,
    }
    for row in rows:
        coverage["query"] += int(bool(query_text(row)))
        try:
            gt = _canonical_ground_truth(row["ground_truth"])
            coverage["ground_truth_function_name"] += int(bool(gt["name"]))
            coverage["ground_truth_arguments"] += int(isinstance(gt["accepted_arguments"], dict))
        except Exception:
            pass
        coverage["function_doc"] += int(bool(row.get("function")))
    return {"category": category, "n_episodes": len(rows), "field_coverage": coverage}


def ensure_raw_data(source_data_dir, raw_dir, categories=CONNECTED_CATEGORIES):
    source = Path(source_data_dir)
    raw = Path(raw_dir)
    copied = []
    for category in categories:
        question_name = f"BFCL_v4_{category}.json"
        answer_name = f"BFCL_v4_{category}.json"
        question_src = source / question_name
        answer_src = source / "possible_answer" / answer_name
        question_dst = raw / question_name
        answer_dst = raw / "possible_answer" / answer_name
        if not question_dst.exists():
            question_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(question_src, question_dst)
            copied.append(str(question_dst))
        if not answer_dst.exists():
            answer_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(answer_src, answer_dst)
            copied.append(str(answer_dst))
    return copied


def load_connected_categories(raw_dir, categories=CONNECTED_CATEGORIES):
    raw = Path(raw_dir)
    loaded = {}
    for category in categories:
        questions = _read_jsonl(raw / f"BFCL_v4_{category}.json")
        answers = {row["id"]: row for row in _read_jsonl(raw / "possible_answer" / f"BFCL_v4_{category}.json")}
        merged = []
        for row in questions:
            item = dict(row)
            item["ground_truth"] = answers[row["id"]]["ground_truth"]
            item["category"] = category
            merged.append(item)
        loaded[category] = merged
    return loaded


def main(argv=None):
    parser = argparse.ArgumentParser(description="A0 BFCL load and offline AST sanity checks")
    parser.add_argument("--source-data-dir", default="deps/bfcl_eval_pkg/bfcl_eval/data")
    parser.add_argument("--raw-dir", default="data/raw/bfcl")
    parser.add_argument("--output", default="results/a0_bfcl_load.json")
    args = parser.parse_args(argv)

    copied = ensure_raw_data(args.source_data_dir, args.raw_dir)
    loaded = load_connected_categories(args.raw_dir)
    summaries = [summarize_category_rows(category, rows) for category, rows in loaded.items()]
    known_gt = {"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}
    ast_self_check = {
        "known_correct": ast_check('calculate_triangle_area(base=10, height=5, unit="units")', known_gt),
        "known_wrong": ast_check('calculate_triangle_area(base=10, height=6, unit="units")', known_gt),
    }
    assert ast_self_check["known_correct"] is True
    assert ast_self_check["known_wrong"] is False
    for summary in summaries:
        assert summary["n_episodes"] > 0
        for value in summary["field_coverage"].values():
            assert value == summary["n_episodes"], summary
    result = {
        "bfcl_version": BFCL_VERSION,
        "connected_categories": CONNECTED_CATEGORIES,
        "copied_files": copied,
        "category_summaries": summaries,
        "total_episodes": sum(summary["n_episodes"] for summary in summaries),
        "ast_self_check": ast_self_check,
        "excluded_or_deferred_categories": {
            "multiple": "excluded: multiple candidate functions, not single function doc",
            "parallel": "excluded: multiple ground-truth calls",
            "parallel_multiple": "excluded: multiple functions and multiple calls",
            "irrelevance": "excluded: no possible_answer AST target",
            "live_relevance": "excluded: relevance detection, no single function call target",
            "sql/rest/exec/chatable": "deferred/special: unused or special evaluation structure in package data",
            "web_search/memory/multi_turn": "excluded: agentic or multi-turn, outside single-turn single-function scope",
        },
    }
    _write_json(args.output, result)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
