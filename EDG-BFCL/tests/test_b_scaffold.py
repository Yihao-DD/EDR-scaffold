import pytest

from gate_scaffold.arms_skeleton import (
    arm_no_update,
    arm_oracle_allocator,
    arm_ours_history_based,
    arm_random_allocator,
    arm_router_only,
    arm_uniform_split,
    arm_update_both,
    arm_validator_only,
)
from gate_scaffold.budget_meter import BudgetCaps, BudgetMeter, load_budget_config
from gate_scaffold.metrics_schema import AcceptedUpdateLog, ArmBudgetSeedMetrics
from gate_scaffold.patch_operators_skeleton import (
    apply_patch,
    estimate_utility,
    generate_router_patches,
    generate_validator_patches,
    rollback_patch,
)
from scripts.b0_split_failures import build_split_result, deterministic_split


def test_budget_meter_counts_and_flags_over_limit():
    meter = BudgetMeter(BudgetCaps(validation_calls=2, latency_ms=100, context_tokens=50))

    assert meter.add(validation_calls=1, latency_ms=40, context_tokens=20)["over_limit"] is False
    assert meter.add(validation_calls=1, latency_ms=70, context_tokens=10)["over_limit"] is True
    assert meter.totals.validation_calls == 2
    assert meter.totals.latency_ms == 110


def test_budget_config_has_three_tiers():
    config = load_budget_config()

    assert set(config) == {"loose", "medium", "tight"}
    assert all(item.validation_calls > 0 for item in config.values())


def test_all_arm_skeletons_raise_not_implemented():
    for func in [
        arm_no_update,
        arm_router_only,
        arm_validator_only,
        arm_uniform_split,
        arm_random_allocator,
        arm_update_both,
        arm_oracle_allocator,
        arm_ours_history_based,
    ]:
        with pytest.raises(NotImplementedError):
            func(None)


def test_patch_operator_skeletons_raise_not_implemented():
    for func in [
        generate_router_patches,
        generate_validator_patches,
        apply_patch,
        rollback_patch,
        estimate_utility,
    ]:
        with pytest.raises(NotImplementedError):
            func(None)


def test_metrics_schema_serializes_to_json_dicts():
    metrics = ArmBudgetSeedMetrics(
        arm="ours",
        budget="tight",
        seed=7,
        repair_success=0.1,
        validation_calls=2,
        latency_ms=10,
        context_tokens=20,
        regression_count=0,
        rollback_count=1,
        net_utility=0.05,
    )
    update = AcceptedUpdateLog(
        update_id="u1",
        component="router",
        trigger_failure_ids=["e1"],
        patch_type="alias",
        patch={"from": "x", "to": "y"},
        validation_cost={"calls": 1, "latency_ms": 2, "context_tokens": 3},
        validation_result={"fixed": 1, "regressed": 0, "net_gain": 1},
        accepted=True,
    )

    assert metrics.to_json_dict()["arm"] == "ours"
    assert update.to_json_dict()["validation_cost"]["calls"] == 1


def test_deterministic_split_is_reproducible_and_disjoint():
    ids = [f"e{i}" for i in range(20)]

    first = deterministic_split(ids, seed=123)
    second = deterministic_split(ids, seed=123)

    assert first == second
    assert set(first["train_update"]).isdisjoint(first["validation"])
    assert set(first["train_update"]).isdisjoint(first["held_out"])
    assert set(first["validation"]).isdisjoint(first["held_out"])
    assert sum(len(v) for v in first.values()) == len(ids)


def test_build_split_result_records_machine_asserts():
    result = build_split_result([f"e{i}" for i in range(8)], seed=321)

    assert result["n_failures"] == 8
    assert result["sizes"] == {"train_update": 4, "validation": 2, "held_out": 2}
    assert result["asserts"] == {
        "train_validation_disjoint": True,
        "train_held_out_disjoint": True,
        "validation_held_out_disjoint": True,
        "all_failures_accounted_for": True,
    }
