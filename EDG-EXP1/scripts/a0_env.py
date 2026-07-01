import argparse
import json
from pathlib import Path

from scripts.bfcl_common import (
    BFCL_VERSION,
    MULTIPLE_CATEGORIES,
    SIMPLE_CATEGORIES,
    ast_check,
    build_wrong_argument_prediction,
    concrete_arguments,
    ensure_raw_data,
    load_bfcl_splits,
    summarize_field_coverage,
    summarize_pool_sizes,
    write_json,
)


def _split_summary(rows):
    by_category = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], 0)
        by_category[row["category"]] = bucket + 1
    return {
        "n_episodes": len(rows),
        "categories": by_category,
        "function_pool_size": summarize_pool_sizes(rows),
        "field_coverage": summarize_field_coverage(rows),
    }


def _ast_self_check(rows):
    row = rows[0]
    gt = row["ground_truth_call"]
    known_correct = {"name": gt["name"], "arguments": concrete_arguments(gt)}
    known_wrong = build_wrong_argument_prediction(gt)
    return {
        "known_correct": ast_check(known_correct, gt),
        "known_wrong": ast_check(known_wrong, gt),
    }


def build_env_result(source_data_dir, raw_dir):
    copied = ensure_raw_data(source_data_dir, raw_dir)
    splits = load_bfcl_splits(raw_dir)
    self_check = {name: _ast_self_check(rows) for name, rows in splits.items()}
    for name, rows in splits.items():
        assert rows, name
        coverage = summarize_field_coverage(rows)
        assert all(value == len(rows) for value in coverage.values()), (name, coverage)
    assert min(len(row["function_pool"]) for row in splits["multiple"]) >= 2
    assert max(len(row["function_pool"]) for row in splits["simple"]) == 1
    assert all(check["known_correct"] is True for check in self_check.values())
    assert all(check["known_wrong"] is False for check in self_check.values())
    return {
        "bfcl_version": BFCL_VERSION,
        "source_data_dir": str(Path(source_data_dir)),
        "raw_dir": str(Path(raw_dir)),
        "categories": {
            "multiple": MULTIPLE_CATEGORIES,
            "simple": SIMPLE_CATEGORIES,
        },
        "copied_files": copied,
        "splits": {name: _split_summary(rows) for name, rows in splits.items()},
        "ast_self_check": self_check,
        "asserts": {
            "multiple_pool_min_ge_2": min(len(row["function_pool"]) for row in splits["multiple"]) >= 2,
            "simple_pool_max_eq_1": max(len(row["function_pool"]) for row in splits["simple"]) == 1,
            "ast_known_correct_true_each_split": all(check["known_correct"] is True for check in self_check.values()),
            "ast_known_wrong_false_each_split": all(check["known_wrong"] is False for check in self_check.values()),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="A0 BFCL Multiple/Simple environment probe")
    parser.add_argument("--source-data-dir", default="../EDG-BFCL/deps/bfcl_eval_pkg/bfcl_eval/data")
    parser.add_argument("--raw-dir", default="data/raw/bfcl")
    parser.add_argument("--output", default="results/a0_env.json")
    args = parser.parse_args(argv)
    result = build_env_result(args.source_data_dir, args.raw_dir)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
