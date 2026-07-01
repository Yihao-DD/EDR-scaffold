import json

from scripts.a2_collect_failures import (
    build_prompt,
    classify_failure,
    parse_model_output,
    summarize_records,
)
from scripts.a3_decouple import build_decoupled_result, decouple_failures


def test_parse_model_output_supports_json_and_call_strings():
    assert parse_model_output('{"name":"fn_a","arguments":{"x":1}}') == {
        "name": "fn_a",
        "arguments": {"x": 1},
    }
    assert parse_model_output('fn_a(x=1, y="z")') == {"name": "fn_a", "arguments": {"x": 1, "y": "z"}}


def test_prompt_includes_query_and_function_pool():
    prompt = build_prompt(
        {
            "query": "Need weather",
            "function_pool": [{"name": "get_weather", "parameters": {"type": "dict"}}],
        }
    )

    assert "Need weather" in prompt
    assert "get_weather" in prompt
    assert '{"name": "<function_name>", "arguments": {"<argument_name>": <argument_value>}}' in prompt


def test_summary_counts_by_split_and_failure_class():
    records = [
        {"split": "multiple", "category": "multiple", "call_success": True, "router_success": True, "validator_success": True, "query": "a", "ground_truth_call": {"accepted_arguments": {"x": [1]}}},
        {"split": "multiple", "category": "multiple", "call_success": False, "router_success": False, "validator_success": False, "query": "b c", "ground_truth_call": {"accepted_arguments": {}}},
        {"split": "simple", "category": "simple_python", "call_success": False, "router_success": True, "validator_success": False, "query": "d e f", "ground_truth_call": {"accepted_arguments": {"x": [1], "y": [2]}}},
    ]

    summary = summarize_records(records)

    assert summary["overall"]["n_total_episodes"] == 3
    assert summary["overall"]["n_failures"] == 2
    assert summary["by_split"]["multiple"]["router_fail"] == 1
    assert summary["by_split"]["simple"]["validator_fail"] == 1
    assert classify_failure(False, False) == "router_fail"
    assert classify_failure(True, False) == "validator_fail"


def test_decoupling_is_bijective_and_preserves_failure_distribution():
    failures = [
        {
            "episode_id": "e1",
            "split": "multiple",
            "failure_class": "router_fail",
            "ground_truth_call": {"name": "get_weather", "accepted_arguments": {"location": ["Paris"]}},
            "predicted_call": {"name": "get_time", "arguments": {"location": "Paris"}},
            "function_pool": [{"name": "get_weather"}, {"name": "get_time"}],
            "trace": {"harness_layer": "tool_interface", "error_label": "ParamError"},
        },
        {
            "episode_id": "e2",
            "split": "simple",
            "failure_class": "validator_fail",
            "ground_truth_call": {"name": "book_flight", "accepted_arguments": {"check_in_date": ["2026-01-01"]}},
            "predicted_call": {"name": "book_flight", "arguments": {"check_in_date": "bad"}},
            "function_pool": [{"name": "book_flight"}],
            "trace": {"harness_layer": "verification", "error_label": "SchemaError"},
        },
    ]

    decoupled, mapping = decouple_failures(failures)
    result = build_decoupled_result(failures)

    assert result["asserts"]["failure_class_distribution_preserved"] is True
    assert result["asserts"]["function_mapping_bijective"] is True
    assert result["asserts"]["field_mapping_bijective"] is True
    assert result["asserts"]["no_decoupled_function_name_equals_original"] is True
    assert decoupled[0]["ground_truth_call"]["name"].startswith("fn_")
    assert decoupled[1]["ground_truth_call"]["accepted_arguments"].keys() != failures[1]["ground_truth_call"]["accepted_arguments"].keys()
    assert mapping["function"]["get_weather"] == decoupled[0]["ground_truth_call"]["name"]
    json.dumps(result)
