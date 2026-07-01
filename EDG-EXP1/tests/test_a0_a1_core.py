import json
from pathlib import Path

from scripts.bfcl_common import (
    MULTIPLE_CATEGORIES,
    SIMPLE_CATEGORIES,
    ast_arguments_match,
    ast_check,
    build_wrong_argument_prediction,
    build_wrong_function_prediction,
    canonical_ground_truth,
    ensure_raw_data,
    load_bfcl_splits,
    router_success,
    validator_success,
)
from scripts.a0_env import build_env_result
from scripts.a1_signals import build_signal_matrix


SOURCE_DATA_DIR = Path("..") / "EDG-BFCL" / "deps" / "bfcl_eval_pkg" / "bfcl_eval" / "data"


def test_loads_simple_and_multiple_splits_with_function_pools(tmp_path):
    raw_dir = tmp_path / "bfcl"
    copied = ensure_raw_data(SOURCE_DATA_DIR, raw_dir)
    loaded = load_bfcl_splits(raw_dir)

    assert copied
    assert set(loaded) == {"multiple", "simple"}
    assert {row["category"] for row in loaded["multiple"]} == set(MULTIPLE_CATEGORIES)
    assert {row["category"] for row in loaded["simple"]} == set(SIMPLE_CATEGORIES)
    assert min(len(row["function_pool"]) for row in loaded["multiple"]) >= 2
    assert max(len(row["function_pool"]) for row in loaded["simple"]) == 1
    assert all(canonical_ground_truth(row)["name"] for rows in loaded.values() for row in rows)


def test_ast_check_handles_nested_live_multiple_arguments(tmp_path):
    raw_dir = tmp_path / "bfcl"
    ensure_raw_data(SOURCE_DATA_DIR, raw_dir)
    loaded = load_bfcl_splits(raw_dir)
    row = next(item for item in loaded["multiple"] if item["category"] == "live_multiple")
    gt = canonical_ground_truth(row)
    prediction = {"name": gt["name"], "arguments": row["ground_truth_concrete_arguments"]}

    assert ast_check(prediction, gt) is True
    assert router_success(prediction, gt) is True
    assert validator_success(prediction, gt) is True

    wrong_args = build_wrong_argument_prediction(gt)
    assert router_success(wrong_args, gt) is True
    assert ast_arguments_match(wrong_args, gt) is False
    assert validator_success(wrong_args, gt) is False

    wrong_function = build_wrong_function_prediction(row, gt)
    assert router_success(wrong_function, gt) is False
    assert validator_success(wrong_function, gt) is False


def test_a0_env_result_records_pool_distribution(tmp_path):
    raw_dir = tmp_path / "bfcl"
    result = build_env_result(SOURCE_DATA_DIR, raw_dir)

    assert result["splits"]["multiple"]["n_episodes"] == 1253
    assert result["splits"]["simple"]["n_episodes"] == 808
    assert result["splits"]["multiple"]["function_pool_size"]["min"] >= 2
    assert result["splits"]["simple"]["function_pool_size"]["max"] == 1
    assert result["ast_self_check"]["multiple"]["known_correct"] is True
    assert result["ast_self_check"]["multiple"]["known_wrong"] is False


def test_a1_signal_matrix_is_nontrivial_for_multiple_and_validator_for_simple(tmp_path):
    raw_dir = tmp_path / "bfcl"
    ensure_raw_data(SOURCE_DATA_DIR, raw_dir)
    loaded = load_bfcl_splits(raw_dir)
    result = build_signal_matrix(loaded)

    multiple = result["splits"]["multiple"]["asserts"]
    simple = result["splits"]["simple"]["asserts"]

    assert multiple["router_nontrivial"] is True
    assert multiple["validator_nontrivial"] is True
    assert simple["validator_nontrivial"] is True
    assert result["splits"]["multiple"]["matrix"]["function_wrong"]["router_false"] > 0
    json.dumps(result)
