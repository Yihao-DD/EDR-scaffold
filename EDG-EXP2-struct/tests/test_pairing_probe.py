import json

from scripts.patch_pair_probe import (
    assert_no_info_padding,
    assert_three_way_parity,
    assert_iu_parity,
    build_information_bundle,
    build_pair,
    count_tokens,
    extract_patch_iu_ids,
    generate_iu_set,
    render_nl_padded_patch,
    render_nl_patch,
    render_struct_patch,
    select_probe_failures,
    summarize_pairs,
)


def _router_failure():
    return {
        "episode_id": "r1",
        "split": "multiple",
        "failure_class": "router_fail",
        "query": "Book a hotel room for tomorrow.",
        "function_pool": [
            {"name": "book_hotel", "description": "Book hotel rooms.", "parameters": {"properties": {"city": {"type": "string", "description": "City name"}}}},
            {"name": "book_flight", "description": "Book flights.", "parameters": {"properties": {"destination": {"type": "string"}}}},
        ],
        "ground_truth_call": {"name": "book_hotel", "accepted_arguments": {"city": ["Paris"]}},
        "predicted_call": {"name": "book_flight", "arguments": {"destination": "Paris"}},
    }


def _validator_failure():
    return {
        "episode_id": "v1",
        "split": "multiple",
        "failure_class": "validator_fail",
        "query": "Get weather in Paris in Celsius.",
        "function_pool": [
            {
                "name": "get_weather",
                "description": "Get current weather.",
                "parameters": {
                    "properties": {
                        "location": {"type": "string", "description": "Location name"},
                        "unit": {"type": "string", "description": "Temperature unit"},
                    }
                },
            }
        ],
        "ground_truth_call": {"name": "get_weather", "accepted_arguments": {"location": ["Paris"], "unit": ["celsius"]}},
        "predicted_call": {"name": "get_weather", "arguments": {"location": "Paris", "unit": "C"}},
    }


def _validator_failure_with_padding_collision():
    failure = _validator_failure()
    failure["episode_id"] = "v_keep"
    failure["query"] = "Keep the saved note concise."
    failure["function_pool"][0]["parameters"]["properties"]["keep"] = {
        "type": "string",
        "description": "Text to keep in the note",
    }
    failure["ground_truth_call"]["accepted_arguments"]["keep"] = ["saved note"]
    failure["predicted_call"]["arguments"]["keep"] = "note"
    return failure


def test_information_bundle_has_same_fields_for_router_pair():
    bundle = build_information_bundle(_router_failure())
    pair = build_pair(_router_failure())

    assert bundle["failure_class"] == "router_fail"
    assert "book_hotel" in pair["struct_patch"]
    assert "book_hotel" in pair["nl_patch"]
    assert pair["information_items"]["gt_function"] == "book_hotel"
    assert pair["struct_token_count"] > 0
    assert pair["nl_token_count"] > 0
    assert pair["token_count_delta_abs"] == abs(pair["struct_token_count"] - pair["nl_token_count"])
    assert pair["iu_parity"] is True
    assert pair["struct_iu_ids"] == pair["nl_iu_ids"]


def test_information_bundle_has_same_fields_for_validator_pair():
    pair = build_pair(_validator_failure())

    assert "location" in pair["struct_patch"]
    assert "location" in pair["nl_patch"]
    assert "unit" in pair["struct_patch"]
    assert "unit" in pair["nl_patch"]
    assert pair["information_items"]["gt_arguments"]["unit"] == ["celsius"]


def test_nl_padded_uses_safe_padding_when_generic_words_collide_with_iu_terms():
    pair = build_pair(_validator_failure_with_padding_collision())

    assert pair["three_way_parity"] is True
    assert pair["padding_no_info"] is True
    assert pair["iu_parity"] is True
    assert pair["struct_iu_ids"] == pair["nl_iu_ids"]


def test_router_iu_set_contains_intent_function_and_match_units():
    iu_set = generate_iu_set(_router_failure())
    iu_ids = {unit["id"] for unit in iu_set}

    assert "IU-intent" in iu_ids
    assert "IU-fn-book_hotel" in iu_ids
    assert "IU-fn-book_flight" in iu_ids
    assert "IU-match" in iu_ids


def test_validator_iu_set_contains_parameter_units_and_format_when_present():
    failure = _validator_failure()
    failure["function_pool"][0]["parameters"]["properties"]["unit"]["description"] = "Temperature unit format."
    iu_set = generate_iu_set(failure)
    iu_ids = {unit["id"] for unit in iu_set}

    assert "IU-param-location" in iu_ids
    assert "IU-param-unit" in iu_ids
    assert "IU-format-unit" in iu_ids


def test_validator_iu_set_does_not_treat_plain_type_as_format_unit():
    failure = _validator_failure()
    failure["function_pool"][0]["parameters"]["properties"]["unit"] = {
        "type": "integer",
        "description": "The number of units to request.",
    }
    iu_set = generate_iu_set(failure)
    iu_ids = {unit["id"] for unit in iu_set}

    assert "IU-format-location" not in iu_ids
    assert "IU-format-unit" not in iu_ids


def test_assert_iu_parity_rejects_mismatched_patch_metadata():
    pair = build_pair(_router_failure())
    assert assert_iu_parity(pair["struct_patch_record"], pair["nl_patch_record"]) is True

    bad_patch = dict(pair["nl_patch_record"])
    bad_patch["covered_iu_ids"] = bad_patch["covered_iu_ids"][:-1]
    try:
        assert_iu_parity(pair["struct_patch_record"], bad_patch)
    except AssertionError:
        pass
    else:
        raise AssertionError("mismatched IU metadata must fail")

    assert extract_patch_iu_ids(pair["struct_patch_record"]) == pair["struct_iu_ids"]


def test_render_nl_padded_patch_matches_struct_tokens_without_new_ius():
    failure = _router_failure()
    iu_set = generate_iu_set(failure)
    struct = render_struct_patch(iu_set, failure["failure_class"])
    nl = render_nl_patch(iu_set, failure["failure_class"])
    padded = render_nl_padded_patch(iu_set, failure["failure_class"], target_token_count=80)

    assert padded["family"] == "NL-padded"
    assert padded["covered_iu_ids"] == nl["covered_iu_ids"] == struct["covered_iu_ids"]
    assert padded["padding_text"]
    assert assert_no_info_padding(padded["padding_text"], iu_set) is True
    assert assert_three_way_parity(struct, nl, padded, target_token_count=80, tokenizer=None) is True


def test_no_info_padding_rejects_function_parameter_and_type_terms():
    failure = _validator_failure()
    iu_set = generate_iu_set(failure)

    for bad_padding in ["get_weather", "location", "string"]:
        try:
            assert_no_info_padding(bad_padding, iu_set)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{bad_padding} must be rejected as informative padding")


def test_select_probe_failures_stratifies_router_and_validator():
    failures = [_validator_failure() for _ in range(15)] + [_router_failure() for _ in range(15)]
    for index, failure in enumerate(failures):
        failure["episode_id"] = f"e{index}"

    selected = select_probe_failures(failures, n=20, min_per_class=10)

    assert len(selected) == 20
    assert sum(item["failure_class"] == "router_fail" for item in selected) == 10
    assert sum(item["failure_class"] == "validator_fail" for item in selected) == 10


def test_summarize_pairs_is_json_serializable():
    pairs = [build_pair(_router_failure()), build_pair(_validator_failure())]
    summary = summarize_pairs(pairs)

    assert summary["n_pairs"] == 2
    assert summary["struct_token_count"]["min"] > 0
    assert summary["nl_token_count"]["min"] > 0
    assert summary["mean_abs_delta"] >= 0
    assert summary["iu_parity_pass_rate"] == 1.0
    assert summary["three_way_parity_pass_rate"] == 1.0
    assert summary["nl_padded_to_struct_ratio"]["mean"] > 0
    assert summary["struct_to_nl_token_ratio"]["std"] >= 0
    assert "mean_abs_delta" in summary["by_failure_class"]["router_fail"]
    json.dumps(summary)


def test_count_tokens_has_regex_fallback():
    assert count_tokens("fn_A does weather.") >= 3
