import argparse
import json
from collections import defaultdict

from scripts.bfcl_common import (
    build_wrong_argument_prediction,
    build_wrong_function_prediction,
    call_success,
    concrete_arguments,
    ensure_raw_data,
    load_bfcl_splits,
    router_success,
    validator_success,
    write_json,
)


def _empty_counts():
    return {"n": 0, "router_true": 0, "router_false": 0, "validator_true": 0, "validator_false": 0}


def _add_case(matrix, case_name, prediction, gt):
    router = router_success(prediction, gt)
    validator = validator_success(prediction, gt)
    counts = matrix[case_name]
    counts["n"] += 1
    counts["router_true"] += int(router)
    counts["router_false"] += int(not router)
    counts["validator_true"] += int(validator)
    counts["validator_false"] += int(not validator)


def _matrix_for_rows(rows):
    matrix = defaultdict(_empty_counts)
    for row in rows:
        gt = row["ground_truth_call"]
        correct = {"name": gt["name"], "arguments": concrete_arguments(gt)}
        _add_case(matrix, "correct", correct, gt)
        _add_case(matrix, "function_correct_arguments_wrong", build_wrong_argument_prediction(gt), gt)
        _add_case(matrix, "function_wrong", build_wrong_function_prediction(row, gt), gt)
    matrix = dict(matrix)
    return {
        "matrix": matrix,
        "asserts": {
            "router_nontrivial": any(item["router_true"] for item in matrix.values())
            and any(item["router_false"] for item in matrix.values()),
            "validator_nontrivial": any(item["validator_true"] for item in matrix.values())
            and any(item["validator_false"] for item in matrix.values()),
            "correct_all_router_true": matrix["correct"]["router_true"] == matrix["correct"]["n"],
            "function_wrong_all_router_false": matrix["function_wrong"]["router_false"] == matrix["function_wrong"]["n"],
            "wrong_arguments_router_true_validator_false": matrix["function_correct_arguments_wrong"]["router_true"]
            == matrix["function_correct_arguments_wrong"]["n"]
            and matrix["function_correct_arguments_wrong"]["validator_false"]
            == matrix["function_correct_arguments_wrong"]["n"],
        },
    }


def build_signal_matrix(loaded_splits):
    result = {
        "signal_definitions": {
            "router_success": "ast_function_name(prediction) == ast_function_name(ground_truth)",
            "validator_success": "router_success and ast_arguments_match(prediction, ground_truth)",
            "call_success": "router_success and validator_success",
        },
        "splits": {},
    }
    for split, rows in loaded_splits.items():
        split_result = _matrix_for_rows(rows)
        split_result["n_episodes"] = len(rows)
        result["splits"][split] = split_result
    assert result["splits"]["multiple"]["asserts"]["router_nontrivial"] is True
    assert result["splits"]["multiple"]["asserts"]["validator_nontrivial"] is True
    assert result["splits"]["simple"]["asserts"]["validator_nontrivial"] is True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="A1 BFCL Multiple/Simple local signal matrix")
    parser.add_argument("--source-data-dir", default="../EDG-BFCL/deps/bfcl_eval_pkg/bfcl_eval/data")
    parser.add_argument("--raw-dir", default="data/raw/bfcl")
    parser.add_argument("--output", default="results/a1_signals.json")
    args = parser.parse_args(argv)
    ensure_raw_data(args.source_data_dir, args.raw_dir)
    result = build_signal_matrix(load_bfcl_splits(args.raw_dir))
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
