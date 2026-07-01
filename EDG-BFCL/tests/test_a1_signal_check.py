from scripts.a1_signal_check import (
    build_prediction_cases,
    call_success,
    router_success,
    validator_success,
)


def test_router_validator_and_call_signal_split():
    gt = {"name": "get_weather", "accepted_arguments": {"location": ["Paris"], "unit": ["celsius"]}}

    assert router_success('get_weather(location="Paris", unit="celsius")', gt) is True
    assert validator_success('get_weather(location="Paris", unit="celsius")', gt) is True
    assert call_success('get_weather(location="Paris", unit="celsius")', gt) is True

    assert router_success('get_weather(location="Paris", unit="fahrenheit")', gt) is True
    assert validator_success('get_weather(location="Paris", unit="fahrenheit")', gt) is False
    assert call_success('get_weather(location="Paris", unit="fahrenheit")', gt) is False

    assert router_success('get_news(location="Paris", unit="celsius")', gt) is False
    assert validator_success('get_news(location="Paris", unit="celsius")', gt) is False
    assert call_success('get_news(location="Paris", unit="celsius")', gt) is False


def test_build_prediction_cases_contains_four_classes():
    row = {
        "id": "x",
        "category": "simple_python",
        "ground_truth": [{"get_weather": {"location": ["Paris"], "unit": ["celsius"]}}],
    }

    cases = build_prediction_cases([row])
    case_types = {case["case_type"] for case in cases}

    assert {"correct", "function_correct_arguments_wrong", "function_wrong", "function_correct_arguments_partial"} <= case_types
