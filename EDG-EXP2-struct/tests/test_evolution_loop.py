from scripts.evolution_loop import (
    PatchCandidate,
    compute_ci95,
    deterministic_split,
    evaluate_held_out,
    materialize_candidate,
    patch_relevant_to_episode,
    summarize_seed_results,
    update_validation_state,
)


def _failure(episode_id="e1", failure_class="validator_fail", gt_name="get_weather", pred_name="get_weather"):
    return {
        "episode_id": episode_id,
        "failure_class": failure_class,
        "query": "Get weather in Paris in Celsius.",
        "function_pool": [
            {"name": "get_weather", "description": "Get weather.", "parameters": {"properties": {"location": {"type": "string"}}}},
            {"name": "get_time", "description": "Get time.", "parameters": {"properties": {"location": {"type": "string"}}}},
        ],
        "ground_truth_call": {"name": gt_name, "accepted_arguments": {"location": ["Paris"]}},
        "predicted_call": {"name": pred_name, "arguments": {"location": "Paris"}},
    }


def test_deterministic_split_is_reproducible_and_disjoint():
    failures = [_failure(f"e{i}") for i in range(20)]

    first = deterministic_split(failures, seed=123)
    second = deterministic_split(failures, seed=123)
    train_ids = {item["episode_id"] for item in first["train"]}
    validation_ids = {item["episode_id"] for item in first["validation"]}
    held_out_ids = {item["episode_id"] for item in first["held_out"]}

    assert first == second
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(held_out_ids)
    assert validation_ids.isdisjoint(held_out_ids)
    assert sum(len(value) for value in first.values()) == 20


def test_materialize_candidate_uses_requested_family():
    candidate = materialize_candidate(_failure("e1", "router_fail", "get_weather", "get_time"), family="STRUCT")

    assert candidate.family == "STRUCT"
    assert candidate.component == "router"
    assert candidate.patch_text
    assert candidate.token_count > 0
    assert "IU-match" in candidate.iu_ids


def test_patch_relevance_uses_component_scope():
    router_patch = PatchCandidate(
        patch_id="p1",
        family="NL",
        component="router",
        trigger_episode_id="e1",
        target_function="get_weather",
        selected_function="get_time",
        parameter_names=[],
        patch_text="patch",
        token_count=1,
        iu_ids=["IU-match"],
    )
    validator_patch = PatchCandidate(
        patch_id="p2",
        family="NL",
        component="validator",
        trigger_episode_id="e2",
        target_function="get_weather",
        selected_function="get_weather",
        parameter_names=["location"],
        patch_text="patch",
        token_count=1,
        iu_ids=["IU-param-location"],
    )

    assert patch_relevant_to_episode(router_patch, _failure("v1", "router_fail", "get_weather", "get_time")) is True
    assert patch_relevant_to_episode(router_patch, _failure("v2", "router_fail", "other", "get_time")) is False
    assert patch_relevant_to_episode(validator_patch, _failure("v3", "validator_fail", "get_weather", "get_weather")) is True
    assert patch_relevant_to_episode(validator_patch, _failure("v4", "validator_fail", "other", "other")) is False


def test_held_out_without_relevant_patch_stays_baseline_failure_without_model_call():
    class FailingRunner:
        def generate(self, prompt, max_new_tokens=256):
            raise AssertionError("no-patch held-out episodes should not rerun the model")

    unrelated_patch = PatchCandidate(
        patch_id="p_unrelated",
        family="NL",
        component="validator",
        trigger_episode_id="e2",
        target_function="get_time",
        selected_function="get_time",
        parameter_names=["timezone"],
        patch_text="patch",
        token_count=1,
        iu_ids=["IU-param-timezone"],
    )

    records = evaluate_held_out("NL-evo", [_failure("held")], [unrelated_patch], FailingRunner())

    assert records == [
        {
            "episode_id": "held",
            "success": False,
            "failure_class": "validator_fail",
            "n_patches": 0,
            "patch_ids": [],
        }
    ]


def test_update_validation_state_accepts_gain_without_regression():
    current = {"v1": False, "v2": True}
    proposed = {"v1": True, "v2": True}
    rejected = {"v1": True, "v2": False}

    accepted, updated, stats = update_validation_state(current, proposed)
    assert accepted is True
    assert updated == proposed
    assert stats["fixed"] == 1
    assert stats["regressed"] == 0

    accepted, updated, stats = update_validation_state(current, rejected)
    assert accepted is False
    assert updated == current
    assert stats["regressed"] == 1


def test_summary_ci_and_seed_aggregation():
    values = [0.0, 0.5, 1.0, 0.5, 0.5]
    ci = compute_ci95(values)
    summary = summarize_seed_results(
        [
            {"method": "STRUCT-evo", "overall_pass_rate": value, "router_pass_rate": value, "validator_pass_rate": value}
            for value in values
        ]
    )

    assert ci["mean"] == 0.5
    assert ci["ci95"] > 0
    assert summary["STRUCT-evo"]["overall"]["mean"] == 0.5
