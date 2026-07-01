from scripts.a0_bfcl_load import (
    ast_check,
    extract_ground_truth,
    parse_bfcl_call,
    summarize_category_rows,
)


def test_parse_bfcl_call_handles_name_and_arguments():
    call = parse_bfcl_call('get_weather(location="Paris", unit="celsius")')

    assert call["name"] == "get_weather"
    assert call["arguments"] == {"location": "Paris", "unit": "celsius"}


def test_ast_check_known_correct_and_wrong_cases():
    gt = {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}

    assert ast_check('get_weather(location="Paris", unit="celsius")', gt) is True
    assert ast_check('get_weather(location="Paris", unit="fahrenheit")', gt) is False
    assert ast_check('get_news(topic="Paris")', gt) is False


def test_extract_ground_truth_from_possible_answer_shape():
    row = {"id": "x", "ground_truth": ['get_weather(location="Paris")']}

    gt = extract_ground_truth(row)

    assert gt == {"name": "get_weather", "arguments": {"location": "Paris"}}


def test_summarize_category_rows_counts_required_fields():
    rows = [
        {
            "id": "1",
            "question": [[{"content": "What is weather?"}]],
            "function": [{"name": "get_weather", "parameters": {"properties": {"location": {"type": "string"}}}}],
            "ground_truth": ['get_weather(location="Paris")'],
        }
    ]

    summary = summarize_category_rows("simple", rows)

    assert summary["category"] == "simple"
    assert summary["n_episodes"] == 1
    assert summary["field_coverage"]["query"] == 1
    assert summary["field_coverage"]["ground_truth_function_name"] == 1
    assert summary["field_coverage"]["ground_truth_arguments"] == 1
