import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.a0_bfcl_load import (
    CONNECTED_CATEGORIES,
    _canonical_ground_truth,
    _write_json,
    ast_check,
    load_connected_categories,
    parse_bfcl_call,
)


def router_success(prediction, ground_truth):
    try:
        return parse_bfcl_call(prediction)["name"] == ground_truth["name"]
    except Exception:
        return False


def validator_success(prediction, ground_truth):
    return router_success(prediction, ground_truth) and ast_check(prediction, ground_truth)


def call_success(prediction, ground_truth):
    return router_success(prediction, ground_truth) and validator_success(prediction, ground_truth)


def _first_value(accepted_values):
    values = accepted_values if isinstance(accepted_values, list) else [accepted_values]
    non_empty = [value for value in values if value != ""]
    return non_empty[0] if non_empty else ""


def _format_value(value):
    return repr(value)


def _prediction_string(name, arguments):
    args = ", ".join(f"{key}={_format_value(value)}" for key, value in arguments.items())
    return f"{name}({args})"


def _prediction_object(name, arguments):
    return {"name": name, "arguments": arguments}


def _correct_arguments(ground_truth):
    return {
        key: _first_value(accepted_values)
        for key, accepted_values in ground_truth["accepted_arguments"].items()
        if _first_value(accepted_values) != ""
    }


def _wrong_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int) and not isinstance(value, bool):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "__wrong"
    if isinstance(value, list):
        return value + ["__wrong"]
    if isinstance(value, dict):
        changed = dict(value)
        changed["__wrong"] = True
        return changed
    return "__wrong"


def _wrong_arguments(arguments):
    changed = dict(arguments)
    if changed:
        first_key = next(iter(changed))
        changed[first_key] = _wrong_value(changed[first_key])
    else:
        changed["unexpected_argument"] = "__wrong"
    return changed


def _partial_arguments(arguments):
    changed = dict(arguments)
    if changed:
        first_key = next(iter(changed))
        changed.pop(first_key)
    else:
        changed["unexpected_argument"] = "__partial"
    return changed


def build_prediction_cases(rows):
    cases = []
    for row in rows:
        ground_truth = _canonical_ground_truth(row["ground_truth"])
        correct_arguments = _correct_arguments(ground_truth)
        base = {
            "episode_id": row["id"],
            "category": row.get("category"),
            "ground_truth": ground_truth,
        }
        cases.append(
            {
                **base,
                "case_type": "correct",
                "prediction": _prediction_object(ground_truth["name"], correct_arguments),
            }
        )
        cases.append(
            {
                **base,
                "case_type": "function_correct_arguments_wrong",
                "prediction": _prediction_object(ground_truth["name"], _wrong_arguments(correct_arguments)),
            }
        )
        cases.append(
            {
                **base,
                "case_type": "function_wrong",
                "prediction": _prediction_object(f"{ground_truth['name']}__wrong", correct_arguments),
            }
        )
        cases.append(
            {
                **base,
                "case_type": "function_correct_arguments_partial",
                "prediction": _prediction_object(ground_truth["name"], _partial_arguments(correct_arguments)),
            }
        )
    return cases


def evaluate_cases(cases):
    matrix = defaultdict(lambda: {"n": 0, "router_true": 0, "router_false": 0, "validator_true": 0, "validator_false": 0})
    examples = {}
    router_values = []
    validator_values = []
    for case in cases:
        router = router_success(case["prediction"], case["ground_truth"])
        validator = validator_success(case["prediction"], case["ground_truth"])
        router_values.append(router)
        validator_values.append(validator)
        bucket = matrix[case["case_type"]]
        bucket["n"] += 1
        bucket["router_true"] += int(router)
        bucket["router_false"] += int(not router)
        bucket["validator_true"] += int(validator)
        bucket["validator_false"] += int(not validator)
        examples.setdefault(
            case["case_type"],
            {
                "episode_id": case["episode_id"],
                "category": case["category"],
                "prediction": case["prediction"],
                "ground_truth": case["ground_truth"],
                "router_success": router,
                "validator_success": validator,
            },
        )
    return {
        "matrix": dict(matrix),
        "examples": examples,
        "asserts": {
            "router_non_degenerate": any(router_values) and not all(router_values),
            "validator_non_degenerate": any(validator_values) and not all(validator_values),
            "correct_all_router_true": matrix["correct"]["router_false"] == 0,
            "function_wrong_all_router_false": matrix["function_wrong"]["router_true"] == 0,
            "arguments_wrong_router_true_validator_false": matrix["function_correct_arguments_wrong"]["router_false"] == 0
            and matrix["function_correct_arguments_wrong"]["validator_true"] == 0,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="A1 BFCL router/validator signal non-degeneracy check")
    parser.add_argument("--raw-dir", default="data/raw/bfcl")
    parser.add_argument("--output", default="results/a1_signal_check.json")
    args = parser.parse_args(argv)

    loaded = load_connected_categories(args.raw_dir, CONNECTED_CATEGORIES)
    rows = [row for category_rows in loaded.values() for row in category_rows]
    cases = build_prediction_cases(rows)
    result = {
        "signal_definitions": {
            "router_success": "ast_function_name(prediction) == ast_function_name(ground_truth)",
            "validator_success": "router_success and ast_arguments_match(prediction, ground_truth)",
            "call_success": "router_success and validator_success",
        },
        "n_episodes": len(rows),
        "n_prediction_cases": len(cases),
        **evaluate_cases(cases),
    }
    for name, value in result["asserts"].items():
        assert value is True, name
    _write_json(args.output, result)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
