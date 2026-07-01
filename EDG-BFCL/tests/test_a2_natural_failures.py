from scripts.a2_natural_failures import (
    build_prompt,
    classify_failure,
    parse_model_output,
    summarize_records,
)


def test_parse_model_output_reads_json_call():
    parsed = parse_model_output('Here is the call: {"name":"get_weather","arguments":{"location":"Paris"}}')

    assert parsed == {"name": "get_weather", "arguments": {"location": "Paris"}}


def test_build_prompt_escapes_json_format_example():
    row = {"question": [[{"content": "What is weather?"}]], "function": [{"name": "get_weather"}]}

    prompt = build_prompt(row)

    assert '{"name": "<function_name>", "arguments": {"<argument_name>": <argument_value>}}' in prompt
    assert "What is weather?" in prompt


def test_parse_model_output_reads_function_call_string():
    parsed = parse_model_output('get_weather(location="Paris")')

    assert parsed == {"name": "get_weather", "arguments": {"location": "Paris"}}


def test_classify_failure_is_mechanical():
    assert classify_failure(False, False) == "router_fail"
    assert classify_failure(True, False) == "validator_fail"
    assert classify_failure(True, True) == "none"


def test_summarize_records_counts_failures():
    records = [
        {"category": "simple", "call_success": True, "router_success": True, "validator_success": True, "query": "short"},
        {"category": "simple", "call_success": False, "router_success": False, "validator_success": False, "query": "a bit longer"},
        {"category": "java", "call_success": False, "router_success": True, "validator_success": False, "query": "x"},
    ]

    summary = summarize_records(records)

    assert summary["n_total_episodes"] == 3
    assert summary["n_failures"] == 2
    assert summary["n_router_fail"] == 1
    assert summary["n_validator_fail"] == 1
    assert summary["by_category"]["simple"]["n_total"] == 2
    assert summary["by_category"]["java"]["n_failures"] == 1
